import typing as t

import torch
import torch.nn as nn


class SupervisionLoss(nn.Module):

    def __init__(
            self,
            lambda_: float = 0.5,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.lambda_ = lambda_

    def forward(
            self,
            pred: t.List[torch.Tensor],
            gt: t.List[torch.Tensor]
    ) -> torch.Tensor:

        gt_dose = gt[0][gt[1] > 0]

        # linear combination of the loss terms
        loss = self.lambda_ * self.loss_fn(pred[0][gt[1] > 0], gt_dose) + self.loss_fn(pred[1][gt[1] > 0], gt_dose)
        return loss
