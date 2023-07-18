import argparse
from datetime import datetime

import wandb
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.nn.functional import sigmoid
import torch

from doseseg import (
    UNet2D,
    C2DModel,
    DualTrainer,
    SoftDiceLoss,
    get_segment_loader,
    unet_evaluation_fn,
)


def main(
        loss_configuration: str,
        lmbda: float,
        dose_prediction_model_path: str,
        dataset_path: str,
        split_config_path: str,
        batch_size: int,
        max_iter: int,
        learning_rate: float,
        weight_decay: float,
        use_wandb: bool,
):
    # initialize the run name
    run_name = 'dual_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # initialize the data loader
    loaders = get_segment_loader(
        path=dataset_path,
        split_config_path=split_config_path,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        batch_size_test=batch_size,
    )

    # initialize the trainer
    trainer = DualTrainer()
    trainer.setting.project_name = "UNet_dual"
    trainer.setting.output_dir = "./data/runs/" + run_name  # noqa
    trainer.setting.max_iter = max_iter
    trainer.setting.train_loader = loaders.get("train")
    trainer.setting.val_loader = loaders.get("valid")
    trainer.setting.test_loader = loaders.get("test")
    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.online_evaluation_function_val = unet_evaluation_fn
    trainer.setting.dose_model_path = dose_prediction_model_path
    trainer.setting.loss_lambda = lmbda

    # initialize the models, the optimizer, the lr scheduler, and set the GPU device
    trainer.setting.network = UNet2D(4, 1)
    trainer.setting.dose_network = C2DModel(
        in_ch=3,
        out_ch=1,
        list_ch_a=[-1, 16, 32, 64, 128, 256],
        list_ch_b=[-1, 32, 64, 128, 256, 512],
    )
    trainer.set_optimizer("Adam",
                          args={"lr": learning_rate, "weight_decay": weight_decay})

    trainer.set_lr_scheduler(lr_scheduler_type="cosine",
                             args={"T_max": args.max_iter, "eta_min": 1e-8, "last_epoch": -1})
    trainer.set_gpu_device([0])

    # initialize the loss function
    if loss_configuration.lower() == "doselo":
        trainer.setting.loss_fn_1 = BCEWithLogitsLoss(pos_weight=torch.Tensor([2.]).to(trainer.setting.device))
        trainer.setting.loss_fn_2 = MSELoss().to(trainer.setting.device)

        if lmbda == 0 or lmbda is None:
            raise ValueError("Lambda for DOSELO loss must be a positive value!")

    elif loss_configuration.lower() == "baseline":
        trainer.setting.loss_fn_1 = BCEWithLogitsLoss(pos_weight=torch.Tensor([2.]).to(trainer.setting.device))
        dice_args = {'apply_nonlinear': sigmoid, 'batch_dice': True, 'do_bg': True, 'smooth': 0.}
        trainer.setting.loss_fn_2 = SoftDiceLoss(**dice_args).to(trainer.setting.device)

        if lmbda != 0 or lmbda is None:
            trainer.setting.loss_lambda = 0
            print("Lambda for baseline loss must be 0 for the baseline!")

    else:
        raise ValueError("Invalid loss configuration!")

    # initialize wandb
    if use_wandb:
        wandb_config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'lr_scheduler': "cosine",
            'lr_scheduler_eta_min': 1e-8,
            'loss_fn_1': trainer.setting.loss_fn_1.__class__.__name__,
            'loss_fn_2': trainer.setting.loss_fn_2.__class__.__name__,
            'lambda_loss': trainer.setting.loss_lambda,
            'number_of_steps': max_iter,
            'batch_size': batch_size,
            'output_dir': run_name,
        }
        wb_logger = wandb.init(project='GTVDualPrediction',
                               name=run_name,
                               reinit=True,
                               config=wandb_config,
                               tags=['dual', 'dose-guided-segmentation'])
        wb_logger.define_metric('epoch')
        wb_logger.define_metric('mean_DICE', step_metric='epoch')
        wb_logger.define_metric('std_DICE', step_metric='epoch')
        wb_logger.define_metric('mean_HD95', step_metric='epoch')
        wb_logger.define_metric('std_HD95', step_metric='epoch')
        wb_logger.define_metric('min_HD95', step_metric='epoch')
        wb_logger.define_metric('max_HD95', step_metric='epoch')
        wb_logger.define_metric('mean_HD100', step_metric='epoch')
        wb_logger.define_metric('std_HD100', step_metric='epoch')
        wb_logger.define_metric('min_HD100', step_metric='epoch')
        wb_logger.define_metric('max_HD100', step_metric='epoch')
        wb_logger.define_metric('performance_index', step_metric='epoch')
        wb_logger.define_metric('val_loss', step_metric='epoch')


    # run the training
    trainer.run()

    # finish the training
    if trainer.setting.wandb_session is not None:
        trainer.setting.wandb_session.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_configuration", type=str, default="DOSELO",
                        help="the loss configuration (options: DOSELO (default), baseline)")
    parser.add_argument("--lmbda", type=float, default=0, help="lambda for dose loss (default: None)")
    parser.add_argument("--dose_prediction_model_path", type=str, help="the relative path to the dose prediction model",
                        default="./data/runs/dose_20230718-112501/best_val_evaluation_index.pkl")
    parser.add_argument("--dataset_path", type=str, default="./data/dataset/segment_dataset.h5",
                        help="dataset file (default: ./data/dataset/segment_dataset.h5)")
    parser.add_argument("--split_config_path", type=str, default="./data/split_config_segment/split_config_fold_0.json",
                        help="split config file path (default: ./data/split_config_segment/split_config_fold_0.json)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for training (default: 16)")
    parser.add_argument("--max_iter", type=int, default=1000, help="training iterations (default: 200000)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb (default: False)")

    args = parser.parse_args()

    main(args.loss_configuration, args.lmbda, args.dose_prediction_model_path, args.dataset_path,
         args.split_config_path, args.batch_size, args.max_iter, args.learning_rate, args.weight_decay,
         args.use_wandb)
