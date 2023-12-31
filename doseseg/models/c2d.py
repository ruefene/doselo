import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size: int,
            stride: int,
            padding: int
    ) -> None:
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=True,
            ),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.single_conv(x)


class UpConv(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int
    ) -> None:
        super(UpConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv(x)
        return x


class Encoder(nn.Module):

    def __init__(
            self,
            in_ch: int,
            list_ch: t.List[int]
    ) -> None:
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_2 = nn.Sequential(
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_3 = nn.Sequential(
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_4 = nn.Sequential(
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_5 = nn.Sequential(
            SingleConv(list_ch[4], list_ch[5], kernel_size=3, stride=2, padding=1),
            SingleConv(list_ch[5], list_ch[5], kernel_size=3, stride=1, padding=1),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> t.List[torch.Tensor]:
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_5 = self.encoder_5(out_encoder_4)

        return [
            out_encoder_1,
            out_encoder_2,
            out_encoder_3,
            out_encoder_4,
            out_encoder_5,
        ]


class Decoder(nn.Module):

    def __init__(
            self,
            list_ch: t.List[int]
    ) -> None:
        super(Decoder, self).__init__()

        self.upconv_4 = UpConv(list_ch[5], list_ch[4])

        self.decoder_conv_4 = nn.Sequential(
            SingleConv(2 * list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[4], list_ch[4], kernel_size=3, stride=1, padding=1),
        )

        self.upconv_3 = UpConv(list_ch[4], list_ch[3])

        self.decoder_conv_3 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
        )

        self.upconv_2 = UpConv(list_ch[3], list_ch[2])

        self.decoder_conv_2 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
            SingleConv(list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
        )

        self.upconv_1 = UpConv(list_ch[2], list_ch[1])

        self.decoder_conv_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1)
        )

    def forward(
            self,
            out_encoder: t.List[torch.Tensor]
    ) -> torch.Tensor:
        (out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4, out_encoder_5,) = out_encoder

        out_decoder_4 = self.decoder_conv_4(
            torch.cat((self.upconv_4(out_encoder_5), out_encoder_4), dim=1)
        )

        out_decoder_3 = self.decoder_conv_3(
            torch.cat((self.upconv_3(out_decoder_4), out_encoder_3), dim=1)
        )

        out_decoder_2 = self.decoder_conv_2(
            torch.cat((self.upconv_2(out_decoder_3), out_encoder_2), dim=1)
        )

        out_decoder_1 = self.decoder_conv_1(
            torch.cat((self.upconv_1(out_decoder_2), out_encoder_1), dim=1)
        )

        return out_decoder_1


class BaseUNet(nn.Module):

    def __init__(
            self,
            in_ch: int,
            list_ch: t.List[int]
    ) -> None:
        super(BaseUNet, self).__init__()
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(list_ch)

        # init
        self.initialize()

    @staticmethod
    def init_conv_in(modules: t.Any) -> None:
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def initialize(self) -> None:
        self.init_conv_in(self.decoder.modules)
        self.init_conv_in(self.encoder.modules)

    def forward(
            self,
            x: torch.Tensor
    ) -> t.List[torch.Tensor]:
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)

        # Output is a list: [Output]
        return out_decoder


class C2DModel(nn.Module):

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            list_ch_a: t.List[int],
            list_ch_b: t.List[int]
    ) -> None:
        super(C2DModel, self).__init__()

        # list_ch records the number of channels in each stage, eg. [-1, 32, 64, 128, 256, 512]
        self.net_A = BaseUNet(in_ch, list_ch_a)
        self.net_B = BaseUNet(in_ch + list_ch_a[1], list_ch_b)

        self.conv_out_A = nn.Conv2d(
            list_ch_a[1], out_ch, kernel_size=1, padding=0, bias=True
        )
        self.conv_out_B = nn.Conv2d(
            list_ch_b[1], out_ch, kernel_size=1, padding=0, bias=True
        )

    def forward(self, x):
        out_net_a = self.net_A(x)
        out_net_b = self.net_B(torch.cat((out_net_a, x), dim=1))

        output_a = self.conv_out_A(out_net_a)
        output_b = self.conv_out_B(out_net_b)
        return [output_a + output_b, output_b]
