import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BrightnessTextureAttention(nn.Module):
    def __init__(self, in_channels):
        super(BrightnessTextureAttention, self).__init__()
        self.in_channels = in_channels

        self.brightness_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.texture_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        brightness_attn = torch.sigmoid(self.brightness_conv(x))

        texture_attn = torch.sigmoid(self.texture_conv(x))

        combined_attn = brightness_attn * texture_attn

        output = x * combined_attn

        return output

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return out

class FrequencyFFC(nn.Module):
    def __init__(self, num_channels):
        super(FrequencyFFC, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_real = torch.real(x)
        x_real = x_real.float()

        out = self.conv1(x_real)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        return out


class SpatialFrequencyBlock(nn.Module):
    def __init__(self, num_channels):
        super(SpatialFrequencyBlock, self).__init__()
        self.num_channels = num_channels

        self.frequency_ffc = FrequencyFFC(num_channels)

        self.final_conv = nn.Conv2d(num_channels * 2, num_channels, kernel_size=1)

    def forward(self, x):
        frequency_input = x.clone()
        frequency_input = torch.fft.fft2(frequency_input, dim=(-2, -1)).to(torch.complex64)

        frequency_output = self.frequency_ffc(frequency_input)

        frequency_output = torch.fft.ifft2(frequency_output, dim=(-2, -1)).real

        output = torch.cat([x, frequency_output], dim=1)

        output = self.final_conv(output)
        return output

class FusionModule(nn.Module):
    def __init__(self, in_channels):
        super(FusionModule, self).__init__()
        self.in_channels = in_channels
        self.brightness_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.texture_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        brightness_attn = torch.sigmoid(self.brightness_conv(x))
        texture_attn = torch.sigmoid(self.texture_conv(x))
        channel_attn = self.channel_attention(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        combined_attn = brightness_attn * texture_attn * spatial_attn

        output = x * combined_attn * channel_attn

        return output

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(num_channels, num_channels),
            nn.ReLU(),
            DepthwiseSeparableConv(num_channels, num_channels)
        )

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.depthwise(x)
        return out

class LIRSRN(nn.Module):
    def __init__(self, in_channels, num_channels, scale_factor):
        super(LIRSRN, self).__init__()
        self.scale_factor = scale_factor
        self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)

        self.fusion_module = FusionModule(num_channels)


        self.conv_mid = DepthwiseSeparableConv(num_channels, num_channels)
        self.conv_high = DepthwiseSeparableConv(num_channels, num_channels)
        self.spatial_frequency_block = SpatialFrequencyBlock(num_channels)

        self.res_block = ResidualBlock(num_channels)

        self.conv_out = nn.Conv2d(num_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.Tanh()


    def forward(self, x):
        out = self.conv_in(x)

        out = self.fusion_module(out)


        out = self.conv_mid(out)
        out = self.conv_high(out)
        out = self.spatial_frequency_block(out)

        out = self.res_block(out)

        out = self.conv_out(out)
        out = self.pixel_shuffle(out)
        out = self.activation(out)

        return out

model = LIRSRN(1, 64, 2)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

dummy_input = torch.randn(1, 1, 96, 96)
with torch.no_grad():
    multi_adds = total_params * dummy_input.size(2) * dummy_input.size(3) / 1e9
print(f"Multi-adds (G): {multi_adds} G")
