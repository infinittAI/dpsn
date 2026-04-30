from __future__ import annotations

import torch
from torch import nn


class StainNet(nn.Module):
    """
    StainNet neural network for stain normalization.

    This follows the original StainNet architecture from:
    https://github.com/khtao/StainNet

    The model is intentionally small: it uses 1x1 convolutions by default, so
    each output pixel is transformed based on its own RGB values rather than a
    spatial neighborhood. This makes it fast and suitable for patch-wise WSI
    inference.
    """

    def __init__(
        self,
        input_nc: int = 3,  # number of input channels
        output_nc: int = 3, # number of output channels
        n_layer: int = 3,   # total number of convolution layers
        n_channel: int = 32,# number of hidden feature channels used inside the network
        kernel_size: int = 1, # convolution kernel size
    ) -> None:
        super().__init__() # call parent constructor

        if input_nc <= 0:
            raise ValueError(f"input_nc must be > 0, got {input_nc}")
        if output_nc <= 0:
            raise ValueError(f"output_nc must be > 0, got {output_nc}")
        if n_layer < 2:
            raise ValueError(f"n_layer must be >= 2, got {n_layer}")
        if n_channel <= 0:
            raise ValueError(f"n_channel must be > 0, got {n_channel}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")

        # Same padding behavior as the original implementation:
        # padding=kernel_size // 2 preserves H/W for odd kernel sizes.
        padding = kernel_size // 2 # how much zero-padding to add around the image when doing convolution

        layers: list[nn.Module] = [ # first layer: Conv2d
            nn.Conv2d(
                input_nc,
                n_channel,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True), # then ReLU
        ]

        # Original StainNet adds hidden Conv + ReLU blocks for n_layer - 2.
        for _ in range(n_layer - 2): # add hidden conv blocks
            layers.extend(
                [
                    nn.Conv2d(
                        n_channel,
                        n_channel,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.append( # Final output layer
            nn.Conv2d(
                n_channel,
                output_nc,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )

        self.rgb_trans = nn.Sequential(*layers) # wrap everything using nn.Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run stain normalization on a batch of image tensors.

        Parameters
        ----------
        x:
            Tensor shaped [N, C, H, W]. The original StainNet inference code
            expects RGB inputs normalized to [-1, 1].

        Returns
        -------
        torch.Tensor
            Tensor shaped [N, output_nc, H, W].
        """
        return self.rgb_trans(x)
