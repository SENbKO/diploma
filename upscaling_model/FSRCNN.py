import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, d, kernel_size=5, padding=2),
            nn.PReLU()
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )

        # Non-linear mapping
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(
                nn.Sequential(
                    nn.Conv2d(s, s, kernel_size=3, padding=1),
                    nn.PReLU()
                )
            )
        self.mapping = nn.Sequential(*mapping_layers)

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )

        # Deconvolution
        self.deconv = nn.ConvTranspose2d(
            d, 3,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x
