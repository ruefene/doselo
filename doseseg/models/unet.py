"""Module for the network architecture."""

import torch
from torch import nn
import torch.nn.functional as func


# pylint: disable=invalid-name


class ConvBlock(nn.Module):
    """A convolution building block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, int]): Kernel size of the convolutional layers.
        padding (Tuple[int, int]): Padding of the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super().__init__()

        self.convBlk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=False)
        )

    def forward(self, inputs):
        """Forward pass of the convolution building block.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.convBlk(inputs)
        return x


class InConv(nn.Module):
    """An in-convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, int]): Kernel size of the convolutional layers.
        padding (Tuple[int, int]): Padding of the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, padding)

    def forward(self, inputs):
        """Forward pass of the in-convolution block.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(inputs)
        return x


class DownsamplingBlock(nn.Module):
    """A downsampling block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, int]): Kernel size of the convolutional layers.
        padding (Tuple[int, int]): Padding of the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, kernel_size, padding)
        )

    def forward(self, inputs):
        """Forward pass of the downsampling block.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.mpconv(inputs)
        return x


class UpsamplingBlock(nn.Module):
    """An upsampling block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, int]): Kernel size of the convolutional layers.
        padding (Tuple[int, int]): Padding of the convolutional layers.
        bilinear (bool): Whether to use a bilinear upsampling or a transpose convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), stride=(2, 2))
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, padding)

    # noinspection DuplicatedCode
    def forward(self, x1, x2):
        """Forward pass of the upsampling block.

        Args:
            x1 (torch.Tensor): Input tensor (the tensor to up-sample).
            x2 (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.bilinear:
            x1 = func.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]

        x1 = func.pad(x1, [diff_y // 2, x2.size()[3] - (x1.size()[3] + diff_y // 2),
                           diff_x // 2, x2.size()[2] - (x1.size()[2] + diff_x // 2)])
        x = torch.cat([x2, x1], dim=1)  # pylint: disable=no-member
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """An out-convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        """Forward pass of the out convolution block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv(x)
        return x


# noinspection DuplicatedCode
class UNet2D(nn.Module):
    """The 2D U-Net model / architecture.

    Args:
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
    """

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.in1 = InConv(n_channels, 64)
        self.down1 = DownsamplingBlock(64, 128)
        self.down2 = DownsamplingBlock(128, 256)
        self.down3 = DownsamplingBlock(256, 512)
        self.down4 = DownsamplingBlock(512, 512)
        self.up1 = UpsamplingBlock(1024, 256)
        self.up2 = UpsamplingBlock(512, 128)
        self.up3 = UpsamplingBlock(256, 64)
        self.up4 = UpsamplingBlock(128, 64)
        self.out1 = OutConv(64, n_classes)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x_1 = self.in1(x)
        x_2 = self.down1(x_1)
        x_3 = self.down2(x_2)
        x_4 = self.down3(x_3)
        x_5 = self.down4(x_4)
        xd_1 = self.up1(x_5, x_4)
        xd_2 = self.up2(xd_1, x_3)
        xd_3 = self.up3(xd_2, x_2)
        xd_4 = self.up4(xd_3, x_1)
        xd_5 = self.out1(xd_4)
        return xd_5
