import argparse
from datetime import datetime

import wandb

from doseseg import (
    DosePredictorTrainer,
    C2DModel,
    SupervisionLoss,
    c2d_evaluation_fn,
    get_dose_loader
)


def main(
        dataset_path: str,
        split_config_path: str,
        batch_size: int,
        max_iter: int,
        learning_rate: float,
        weight_decay: float,
        use_wandb: bool
) -> None:
    # initialize the run name
    run_name = 'dose_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # initialize the data loader
    loaders = get_dose_loader(
        path=dataset_path,
        split_config_path=split_config_path,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        batch_size_test=batch_size,
        batch_size=batch_size,
    )

    # initialize the trainer
    trainer = DosePredictorTrainer()
    trainer.setting.project_name = "C2D"
    trainer.setting.output_dir = "./data/runs/" + run_name
    trainer.setting.max_iter = max_iter
    trainer.setting.train_loader = loaders.get("train")
    trainer.setting.val_loader = loaders.get("valid")
    trainer.setting.test_loader = loaders.get("test")
    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_fn_2 = SupervisionLoss()
    trainer.setting.online_evaluation_function_val = c2d_evaluation_fn

    # initialize the model, the optimizer, the lr scheduler, and set the GPU device
    trainer.setting.network = C2DModel(
        in_ch=3,
        out_ch=1,
        list_ch_a=[-1, 16, 32, 64, 128, 256],
        list_ch_b=[-1, 32, 64, 128, 256, 512],
    )
    trainer.set_optimizer(optimizer_type="Adam", lr=learning_rate, weight_decay=weight_decay)
    trainer.set_lr_scheduler(lr_scheduler_type="cosine", T_max=max_iter, eta_min=1e-7, last_epoch=-1)
    trainer.set_gpu_device([0])

    # initialize wandb (if needed)
    if use_wandb:
        wandb_config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'lr_scheduler': "cosine",
            'lr_scheduler_eta_min': 1e-7,
            'loss_fn': "L2 / MSE",
            'lambda_loss_fn': trainer.setting.loss_fn_2.lambda_,
            'number_of_steps': max_iter,
            'batch_size': batch_size,
            'output_dir': run_name
        }
        wb_logger = wandb.init(project='DosePredictor', name=run_name, reinit=True, config=wandb_config,
                               tags=['dose', 'dose_prediction', '2D'])
        wb_logger.define_metric('epoch')
        trainer.setting.wandb_session = wb_logger

    #  Start the training procedure
    trainer.run()

    # Finish the wandb session (if needed)
    if trainer.setting.wandb_session is not None:
        trainer.setting.wandb_session.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/dataset/dose_dataset.h5",
                        help="dataset file (default: ./data/dataset/dose_dataset.h5)")
    parser.add_argument("--split_config_path", type=str, default="./data/split_config_dose/split_config_fold_0.json",
                        help="split config file path (default: ./data/split_config_dose/split_config_fold_0.json)")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training (default: 16)")
    parser.add_argument("--max_iter", type=int, default=80000, help="training iterations (default: 80000)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="weight decay (default: 3e-5)")
    parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb (default: False)")

    args = parser.parse_args()

    main(args.dataset_path, args.split_config_path, args.batch_size, args.max_iter, args.learning_rate,
         args.weight_decay, args.use_wandb)
