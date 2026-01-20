from torch import nn
import torch.nn.functional as functional
import torch

#Custom implementation of U-Net
class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.start = self.double_convolution(in_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 1024)
        self.upconv1 = self.upconvolutional_block(1024, 512)
        self.up1 = self.double_convolution(1024, 512)
        self.upconv2 = self.upconvolutional_block(512, 256)
        self.up2 = self.double_convolution(512, 256)

        self.upconv3 = self.upconvolutional_block(256, 128)
        self.up3 = self.double_convolution(256, 128)

        self.upconv4 = self.upconvolutional_block(128, 64)
        self.up4 = self.double_convolution(128, 64)
        self.finish1 = self.finish(64)


    def finish(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)
        )

    def double_convolution(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, padding=0, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, padding=0, kernel_size=3 ),
            nn.ReLU()
        )

    def down(self, input_channels, output_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            self.double_convolution(input_channels, output_channels)
        )
    
    def bottleneck(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=0),
            nn.ReLU()
        )
    
    def up(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(kernel_size=2, in_channels=input_channels, out_channels=output_channels, stride=2),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, padding=0, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, padding=0, kernel_size=3 ),
            nn.ReLU()
        )

    def upconvolutional_block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, padding=0, kernel_size=2, stride=2)
        )
    def count_difference(self, x1, x2):
        difference_height = x1.shape[2] - x2.shape[2]
        difference_width = x1.shape[3] - x2.shape[3]

        x2 = functional.pad(x2, [difference_width//2, difference_width - difference_width//2,
                                 difference_height//2, difference_height - difference_height//2])
        print(x2.shape)
        return x2


    def forward(self, x):
        start = self.start(x)
        print(f"First block shape: {start.shape}")
        down1 = self.down1(start)
        print(f"Second block shape: {down1.shape}")
        down2 = self.down2(down1)
        print(f"Third Blocl shape: {down2.shape}")
        down3 = self.down3(down2)
        print(down3.shape)
        down4 = self.down4(down3)
        print(down4.shape)
        up1 = self.upconv1(down4)
        up1 = self.count_difference(down3, up1)
        up1 = torch.cat([up1, down3], dim=1)
        up1 = self.up1(up1)
        print(f"Upconv1 shape: {up1.shape}")
        up2 = self.upconv2(up1)
        up2 = self.count_difference(down2, up2)
        up2 = torch.cat([up2, down2], dim=1)
        up2 = self.up2(up2)
        print(f"Upconv2 shape: {up2.shape}")
        up3 = self.upconv3(up2)
        up3 = self.count_difference(down1, up3)
        up3 = torch.cat([up3, down1], dim=1)
        up3 = self.up3(up3)
        print(f"Upconv3 shape: {up3.shape}")
        up4 = self.upconv4(up3)
        up4 = self.count_difference(start, up4)
        up4 = torch.cat([up4, start], dim=1)
        up4 = self.up4(up4)
        print(f"Upconv4 shape: {up4.shape}")
        output = self.finish1(up4)
        print(output)
        return output


