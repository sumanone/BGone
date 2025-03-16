import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from typing import List, Optional, Dict


class ContourLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(ContourLoss, self).__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: int = 10
    ) -> torch.Tensor:
        """
        target, pred: tensor of shape (B, C, H, W), where target[:,:,region_in_contour] == 1,
                        target[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        """
        # length term
        delta_r = (
            pred[:, :, 1:, :] - pred[:, :, :-1, :]
        )  # horizontal gradient (B, C, H-1, W)
        delta_c = (
            pred[:, :, :, 1:] - pred[:, :, :, :-1]
        )  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(
            torch.sqrt(delta_pred + epsilon)
        )  # eq.(11) in the paper, mean is used instead of sum.

        c_in = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in = torch.mean(
            pred * (target - c_in) ** 2
        )  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - pred) * (target - c_out) ** 2)
        region = region_in + region_out

        loss = weight * length + region

        return loss


class IoULoss(torch.nn.Module):
    def __init__(self) -> None:
        super(IoULoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)
        # return IoU/b
        return IoU


class StructureLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(StructureLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
        )
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class PatchIoULoss(torch.nn.Module):
    def __init__(self) -> None:
        super(PatchIoULoss, self).__init__()
        self.iou_loss = IoULoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        win_y, win_x = 64, 64
        iou_loss = 0.0
        for anchor_y in range(0, target.shape[0], win_y):
            for anchor_x in range(0, target.shape[1], win_y):
                patch_pred = pred[
                    :, :, anchor_y : anchor_y + win_y, anchor_x : anchor_x + win_x
                ]
                patch_target = target[
                    :, :, anchor_y : anchor_y + win_y, anchor_x : anchor_x + win_x
                ]
                patch_iou_loss = self.iou_loss(patch_pred, patch_target)
                iou_loss += patch_iou_loss
        return iou_loss


class ThrReg_loss(torch.nn.Module):
    def __init__(self) -> None:
        super(ThrReg_loss, self).__init__()

    def forward(
        self, pred: torch.Tensor, gt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))


class ClsLoss(nn.Module):
    """
    Auxiliary classification loss for each refined class output.
    """

    def __init__(self) -> None:
        super(ClsLoss, self).__init__()
        self.lambdas_cls = {"ce": 5.0}

        self.criterions_last = {"ce": nn.CrossEntropyLoss()}

    def forward(self, preds: List[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for _, pred_lvl in enumerate(preds):
            if pred_lvl is None:
                continue
            for criterion_name, criterion in self.criterions_last.items():
                loss += criterion(pred_lvl, gt) * self.lambdas_cls[criterion_name]
        return loss


class BiRefNetPixLoss(nn.Module):
    """
    Pixel loss for each refined map output.
    """

    def __init__(self) -> None:
        super(BiRefNetPixLoss, self).__init__()

        self.loss_weights = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            "bce": 30 * 1,  # high performance
            "iou": 0.5 * 1,  # 0 / 255
            "iou_patch": 0.5 * 0,  # 0 / 255, win_size = (64, 64)
            "mse": 150 * 0,  # can smooth the saliency map
            "triplet": 3 * 0,
            "reg": 100 * 0,
            "ssim": 10 * 1,  # help contours,
            "cnt": 5 * 0,  # help contours
            "structure": 5
            * 1,  # structure loss from codes of MVANet. A little improvement on DIS-TE[1,2,3], a bit more decrease on DIS-TE4.
        }

        self.losses: Dict[str, nn.Module] = {}
        if "bce" in self.loss_weights and self.loss_weights["bce"]:
            self.losses["bce"] = nn.BCELoss()
        if "iou" in self.loss_weights and self.loss_weights["iou"]:
            self.losses["iou"] = IoULoss()
        if "iou_patch" in self.loss_weights and self.loss_weights["iou_patch"]:
            self.losses["iou_patch"] = PatchIoULoss()
        if "ssim" in self.loss_weights and self.loss_weights["ssim"]:
            self.losses["ssim"] = SSIMLoss()
        if "mse" in self.loss_weights and self.loss_weights["mse"]:
            self.losses["mse"] = nn.MSELoss()
        if "reg" in self.loss_weights and self.loss_weights["reg"]:
            self.losses["reg"] = ThrReg_loss()
        if "cnt" in self.loss_weights and self.loss_weights["cnt"]:
            self.losses["cnt"] = ContourLoss()
        if "structure" in self.loss_weights and self.loss_weights["structure"]:
            self.losses["structure"] = StructureLoss()

    def compute_losses(
        self, scaled_preds: List[torch.Tensor], gt: torch.Tensor
    ) -> Dict[str, float]:
        losses = {}
        criterions_embedded_with_sigmoid = ["structure"]
        for prediction in scaled_preds:
            if prediction.shape != gt.shape:
                prediction = nn.functional.interpolate(
                    prediction, size=gt.shape[2:], mode="bilinear", align_corners=True
                )
            for criterion_name, criterion in self.losses.items():
                specific_loss = (
                    criterion(
                        (
                            prediction.sigmoid()
                            if criterion_name not in criterions_embedded_with_sigmoid
                            else prediction
                        ),
                        gt,
                    )
                    * self.loss_weights[criterion_name]
                )
                if criterion_name in losses:
                    losses[criterion_name] += specific_loss
                else:
                    losses[criterion_name] = specific_loss
        return losses

    def forward(
        self, scaled_preds: List[torch.Tensor], gt: torch.Tensor
    ) -> torch.Tensor:
        loss = 0.0
        losses = self.compute_losses(scaled_preds, gt)
        for criterion_name, criterion in self.losses.items():
            loss += losses[criterion_name]
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True) -> None:
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: Variable,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ssim = torch.mean(SSIM(x, y))
    return ssim
