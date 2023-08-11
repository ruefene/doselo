from abc import ABC, abstractmethod
import typing as t
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim


class TrainerSetting:

    def __init__(self):
        self.project_name = None
        self.output_dir = None

        self.max_iter = 999999999
        self.max_epoch = 999999999

        self.save_per_epoch = 999999999
        self.eps_train_loss = 0.01

        self.network = None
        self.dose_network = None
        self.dose_model_path = None
        self.device = None
        self.list_GPU_ids = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_type = None
        self.lr_scheduler_update_on_iter = False

        self.loss_fn_2 = None
        self.loss_fn_1 = None
        self.loss_fn_second = None
        self.loss_lambda = 0.5

        self.online_evaluation_function_val = None
        self.wandb_session = None


class TrainerLog:

    def __init__(self):
        self.iter = -1
        self.epoch = -1

        # Moving average loss
        self.moving_train_loss = None
        # Average train loss of an epoch
        self.average_train_loss = 99999999.0
        self.best_average_train_loss = 99999999.0
        # Evaluation index (higher is better)
        self.average_val_index = -99999999.0
        self.best_average_val_index = -99999999.0

        # Record changes in training loss
        self.list_average_train_loss_associate_iter = []
        # Record changes in validation evaluation index
        self.list_average_val_index_associate_iter = []
        # Record changes in learning rate
        self.list_lr_associate_iter = []

        # Save status of the training, e.g. best_train_loss, latest, best_val_evaluation_index
        self.save_status = []


class TrainerTime:

    def __init__(self):
        self.train_time_per_epoch = 0.0
        self.train_loader_time_per_epoch = 0.0

        self.val_time_per_epoch = 0.0
        self.val_loader_time_per_epoch = 0.0


class Trainer(ABC):

    def __init__(self):
        self.log = TrainerLog()
        self.setting = TrainerSetting()
        self.time = TrainerTime()

    def set_gpu_device(self, list_gpu_ids: t.Sequence[int]):
        self.setting.list_GPU_ids = list_gpu_ids
        num_devices = len(list_gpu_ids)

        # cpu only
        if list_gpu_ids[0] == -1:
            self.setting.device = torch.device("cpu")
        # single GPU
        elif num_devices == 1:
            self.setting.device = torch.device("cuda:" + str(list_gpu_ids[0]))
        # multi-GPU
        else:
            self.setting.device = torch.device("cuda:" + str(list_gpu_ids[0]))
            self.setting.network = nn.DataParallel(
                self.setting.network, device_ids=list_gpu_ids
            )

        self.setting.network.to(self.setting.device)

    def set_optimizer(self, optimizer_type: str, **args):
        if optimizer_type == "Adam":
            self.setting.optimizer = optim.Adam(
                self.setting.network.parameters(),
                lr=args["lr"],
                weight_decay=args["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        else:
            raise NotImplementedError()

    def set_lr_scheduler(self, lr_scheduler_type: str, **args):
        if lr_scheduler_type == "step":
            self.setting.lr_scheduler_type = "step"
            self.setting.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.setting.optimizer,
                milestones=args["milestones"],
                gamma=args["gamma"],
                last_epoch=args["last_epoch"])

        elif lr_scheduler_type == "cosine":
            self.setting.lr_scheduler_type = "cosine"
            self.setting.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.setting.optimizer,
                T_max=args["T_max"],
                eta_min=args["eta_min"],
                last_epoch=args["last_epoch"])

        elif lr_scheduler_type == "ReduceLROnPlateau":
            self.setting.lr_scheduler_type = "ReduceLROnPlateau"
            self.setting.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.setting.optimizer,
                mode="min",
                factor=args["factor"],
                patience=args["patience"],
                verbose=True,
                threshold=args["threshold"],
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08)

        else:
            raise NotImplementedError()

    def update_lr(self):
        # Update learning rate, only 'ReduceLROnPlateau' need use the moving train loss
        if self.setting.lr_scheduler_type == "ReduceLROnPlateau":
            self.setting.lr_scheduler.step(self.log.moving_train_loss)
        else:
            self.setting.lr_scheduler.step()

    def update_moving_train_loss(self, loss):
        if self.log.moving_train_loss is None:
            self.log.moving_train_loss = loss.item()
        else:
            self.log.moving_train_loss = ((1 - self.setting.eps_train_loss) * self.log.moving_train_loss +
                                          self.setting.eps_train_loss * loss.item())

    def update_average_statistics(self, loss, phase="train"):
        if phase == "train":
            self.log.average_train_loss = loss
            if loss < self.log.best_average_train_loss:
                self.log.best_average_train_loss = loss
                self.log.save_status.append("best_train_loss")
            self.log.list_average_train_loss_associate_iter.append([self.log.average_train_loss, self.log.iter])

        elif phase == "val":
            self.log.average_val_index = loss
            if loss > self.log.best_average_val_index:
                self.log.best_average_val_index = loss
                self.log.save_status.append("best_val_evaluation_index")
            self.log.list_average_val_index_associate_iter.append([self.log.average_val_index, self.log.iter])

    def print_log_to_file(self, txt, mode):
        with open(self.setting.output_dir + "/log.txt", mode) as log_:
            log_.write(txt)

        txt = txt.replace("\n", "")
        print(txt)


    def init_trainer(self, ckpt_file, list_gpu_ids, only_network=True):
        ckpt = torch.load(ckpt_file, map_location="cpu")

        self.setting.network.load_state_dict(ckpt["network_state_dict"])

        if not only_network:
            self.setting.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            self.setting.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.log = ckpt["log"]

        self.set_gpu_device(list_gpu_ids)

        # This for Adam
        if type(self.setting.optimizer).__name__ == "Adam":
            for key in self.setting.optimizer.state.items():
                key[1]["exp_avg"] = key[1]["exp_avg"].to(self.setting.device)
                key[1]["exp_avg_sq"] = key[1]["exp_avg_sq"].to(self.setting.device)
                key[1]["max_exp_avg_sq"] = key[1]["max_exp_avg_sq"].to(self.setting.device)

        self.print_log_to_file("==> Init training from " + ckpt_file + " successfully! \n", "a")

    def save_trainer(self, status="latest"):
        if len(self.setting.list_GPU_ids) > 1:
            network_state_dict = self.setting.network.module.state_dict()
        else:
            network_state_dict = self.setting.network.state_dict()

        optimizer_state_dict = self.setting.optimizer.state_dict()
        lr_scheduler_state_dict = self.setting.lr_scheduler.state_dict()

        ckpt = {
            "network_state_dict": network_state_dict,
            "lr_scheduler_state_dict": lr_scheduler_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "log": self.log,
        }

        torch.save(ckpt, self.setting.output_dir + "/" + status + ".pth.tar")
        self.print_log_to_file(
            "        ==> Saving " + status + " model successfully !\n", "a"
        )

    def val(self):
        time_start_val = time.time()
        self.setting.network.eval()

        val_index = self.setting.online_evaluation_function_val(self)
        self.update_average_statistics(val_index, phase="val")

        self.time.val_time_per_epoch = time.time() - time_start_val

        torch.cuda.empty_cache()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    def run(self):
        # initialize the run directory if not existing
        if not os.path.exists(self.setting.output_dir):
            os.mkdir(os.path.abspath(os.path.normpath(self.setting.output_dir)))

        # initialize the log file
        if self.log.iter == -1:
            self.print_log_to_file("Start training !\n", "w")
        else:
            self.print_log_to_file("Continue training !\n", "w")
        self.print_log_to_file(
            time.strftime("Local time: %H:%M:%S\n", time.localtime(time.time())), "a"
        )

        # Start training
        while (self.log.epoch < self.setting.max_epoch - 1) and (self.log.iter < self.setting.max_iter - 1):
            #
            time_start_this_epoch = time.time()
            self.log.epoch += 1

            # Print current learning rate
            if self.setting.wandb_session is not None:
                self.setting.wandb_session.log({"learning_rate": self.setting.optimizer.param_groups[0]["lr"]})
            self.print_log_to_file("Epoch: %d, iter: %d\n" % (self.log.epoch, self.log.iter), "a")
            self.print_log_to_file(
                "    Begin lr is %12.12f, %12.12f\n"
                % (
                    self.setting.optimizer.param_groups[0]["lr"],
                    self.setting.optimizer.param_groups[-1]["lr"],
                ),
                "a",
            )

            # Record initial learning rate for this epoch
            self.log.list_lr_associate_iter.append([self.setting.optimizer.param_groups[0]["lr"], self.log.iter])

            self.time.__init__()
            self.train()
            self.val()

            # If update learning rate per epoch
            if not self.setting.lr_scheduler_update_on_iter:
                self.update_lr()

            # Save training every "self.setting.save_per_epoch"
            if (self.log.epoch + 1) % self.setting.save_per_epoch == 0:
                self.log.save_status.append("iter_" + str(self.log.iter))
            self.log.save_status.append("latest")

            # Try save training
            if len(self.log.save_status) > 0:
                for status in self.log.save_status:
                    self.save_trainer(status=status)
                self.log.save_status = []

            self.print_log_to_file(
                "            Average train loss is             %12.12f,     best is           %12.12f\n"
                % (self.log.average_train_loss, self.log.best_average_train_loss),
                "a",
            )
            self.print_log_to_file(
                "            Average val evaluation index is   %12.12f,     best is           %12.12f\n"
                % (self.log.average_val_index, self.log.best_average_val_index),
                "a",
            )

            self.print_log_to_file(
                "    Train use time %12.5f\n" % self.time.train_time_per_epoch, "a"
            )
            self.print_log_to_file(
                "    Train loader use time %12.5f\n"
                % self.time.train_loader_time_per_epoch,
                "a",
            )
            self.print_log_to_file(
                "    Val use time %12.5f\n" % self.time.val_time_per_epoch, "a"
            )
            self.print_log_to_file(
                "    Total use time %12.5f\n" % (time.time() - time_start_this_epoch),
                "a",
            )
            self.print_log_to_file(
                "    End lr is %12.12f, %12.12f\n"
                % (
                    self.setting.optimizer.param_groups[0]["lr"],
                    self.setting.optimizer.param_groups[-1]["lr"],
                ),
                "a",
            )
            self.print_log_to_file(
                time.strftime("    time: %H:%M:%S\n", time.localtime(time.time())), "a"
            )

        self.print_log_to_file(
            "===============================> End successfully\n", "a"
        )


class DosePredictorTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def forward(self, input_, phase):
        time_start_load_data = time.time()
        # To device
        input_ = input_.to(self.setting.device)

        # Record time of moving input from cpu to gpu
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Forward
        if phase == "train":
            self.setting.optimizer.zero_grad()
        output = self.setting.network(input_)

        return output

    def backward(self, output, target):
        time_start_load_data = time.time()

        # Record time of moving target from cpu to gpu
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Optimize
        loss = self.setting.loss_fn_2(output, target)
        loss.backward()
        self.setting.optimizer.step()

        return loss

    def train(self):
        time_start_train = time.time()

        self.setting.network.train()
        sum_train_loss = 0.0
        count_iter = 0

        time_start_load_data = time.time()
        for batch_idx, batch in enumerate(self.setting.train_loader):

            images = np.stack(batch['images'], axis=0)
            oar = np.stack(batch['oar'], axis=0)
            ptv = np.stack(batch['ptv'], axis=0)
            input_ = [images, oar, ptv]
            input_ = torch.from_numpy(np.concatenate(input_, axis=1)).to(self.setting.device, dtype=torch.float)

            dose = np.stack(batch['dose'], axis=0)
            mask = np.stack(batch['mask'], axis=0)
            target = [torch.from_numpy(dose).to(self.setting.device, dtype=torch.float),
                      torch.from_numpy(mask).to(self.setting.device, dtype=torch.long)]

            if (self.setting.max_iter is not None) and (self.log.iter >= self.setting.max_iter - 1):
                break
            self.log.iter += 1

            # Record time of preparing data
            self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

            # Forward
            output = self.forward(input_, phase="train")

            # Backward
            loss = self.backward(output, target)

            # Used for counting average loss of this epoch
            if self.setting.wandb_session is not None:
                self.setting.wandb_session.log({"train_loss": loss.item()})
            sum_train_loss += loss.item()
            count_iter += 1

            self.update_moving_train_loss(loss)
            self.update_lr()

            # Print loss during the first epoch
            if self.log.epoch == 0:
                if self.log.iter % 10 == 0:
                    self.print_log_to_file(
                        "                Iter %12d       %12.5f\n"
                        % (self.log.iter, self.log.moving_train_loss),
                        "a",
                    )

            time_start_load_data = time.time()

        if count_iter > 0:
            average_loss = sum_train_loss / count_iter
            self.update_average_statistics(average_loss, phase="train")

        self.time.train_time_per_epoch = time.time() - time_start_train

        torch.cuda.empty_cache()


class DualTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def train(self):
        time_start_train = time.time()

        self.setting.network.train()
        self.setting.dose_network.eval()
        sum_train_loss = 0.0
        count_iter = 0

        time_start_load_data = time.time()
        for batch_idx, batch in enumerate(self.setting.train_loader):
            images = torch.from_numpy(np.stack(batch['images'], axis=0)).to(self.setting.device, dtype=torch.float)
            target_seg = torch.from_numpy(np.stack(batch['gtv'], axis=0)).to(self.setting.device, dtype=torch.float)

            if (self.setting.max_iter is not None) and (self.log.iter >= self.setting.max_iter - 1):
                break
            self.log.iter += 1

            # Record time of preparing data
            self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

            # Forward segmentation
            self.setting.optimizer.zero_grad()
            output_seg = self.setting.network(images)

            # SoftDice + BCE training
            if float(self.setting.loss_lambda) == 0. and \
                    'dice' in self.setting.loss_fn_2.__class__.__name__.lower():
                # BCE Loss
                loss_1 = self.setting.loss_fn_1(output_seg, target_seg)

                # Dice Loss
                loss_2 = self.setting.loss_fn_2(output_seg, target_seg)

                # Linear combination of the losses
                loss = loss_1 + loss_2

            # Dose Segmentation Loss (DOSELO) training
            elif float(self.setting.loss_lambda) > 0. and \
                    'dice' not in self.setting.loss_fn_2.__class__.__name__.lower():

                # dual forward pass on the dose predictor
                with torch.no_grad():
                    mask = torch.from_numpy(np.stack(batch['mask'], axis=0)).to(self.setting.device, dtype=torch.long)

                    ct = torch.from_numpy(np.stack(batch['ct'], axis=0)).to(self.setting.device, dtype=torch.float)
                    oar = torch.from_numpy(np.stack(batch['oar'], axis=0)).to(self.setting.device, dtype=torch.float)
                    gtv = torch.from_numpy(np.stack(batch['gtv'], axis=0)).to(self.setting.device, dtype=torch.float)

                    input_gt = torch.cat([ct, oar, gtv], dim=1).to(self.setting.device, dtype=torch.float)
                    output_dose_gt = self.setting.dose_network(input_gt)[1] * mask

                    gtv_torch = torch.ge(torch.sigmoid(output_seg), 0.5).to(self.setting.device, dtype=torch.float)
                    input_pred = torch.concat([ct, oar, gtv_torch], dim=1)
                    output_dose_pred = self.setting.dose_network(input_pred)[1] * mask

                # Loss BCE
                loss_1 = self.setting.loss_fn_1(output_seg, target_seg)

                # Dose loss
                loss_2 = self.setting.loss_fn_2(output_dose_pred, output_dose_gt)

                # Linear combination of the losses
                loss = loss_1 + self.setting.loss_lambda * loss_2

            else:
                raise NotImplementedError('The combination of loss functions and lambda value is not supported.')

            # Backward
            loss.backward()
            self.setting.optimizer.step()

            # Used for counting average loss of this epoch
            if self.setting.wandb_session is not None:
                self.setting.wandb_session.log({"train_loss": loss.item(),
                                                "train_loss_seg": loss_1.item(),
                                                "train_loss_dose": loss_2.item()})
            sum_train_loss += loss.item()
            count_iter += 1

            self.update_moving_train_loss(loss)
            self.update_lr()

            # Print loss during the first epoch
            if self.log.epoch <= 5:
                if self.log.iter % 10 == 0:
                    self.print_log_to_file(
                        "                Iter %12d       %12.5f\n"
                        % (self.log.iter, self.log.moving_train_loss),
                        "a",
                    )

            time_start_load_data = time.time()

        if count_iter > 0:
            average_loss = sum_train_loss / count_iter
            self.update_average_statistics(average_loss, phase="train")

        self.time.train_time_per_epoch = time.time() - time_start_train

        torch.cuda.empty_cache()

    def load_dose_model(self):
        model = self.setting.dose_network
        model.load_state_dict(torch.load(self.setting.dose_model_path).get('network_state_dict'))
        model.to(self.setting.device)
        model.eval()
        self.setting.dose_network = model

    def run(self):
        self.load_dose_model()
        super().run()
