import torch
import torch.nn as nn
from .restormer import * #TransformerBlock


class Fusion_Net(nn.Module):
    def __init__(
        self,
        dim=32,
        num_blocks=4,
        head=8,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
        output_mask=False
    ):
        super(Fusion_Net, self).__init__()

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ],
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ],
        )


        self.aggregator_1 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ],
        )

        self.aggregator_2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ],
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks)
            ],
            nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Sigmoid(),
        )
        if output_mask:
                self.decoder = nn.Sequential(
                    nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                    *[
                        TransformerBlock(
                            dim=dim,
                            num_heads=head,
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                        )
                        for _ in range(num_blocks)
                    ],
                    nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1, bias=bias),
                    nn.Sigmoid(),
                )

    def forward(self, features1, features2):
        """
        features1, features2: [B, 3C, H, W]
        Output: [B, 3, H, W]
        """

        fused = torch.cat([self.aggregator_1(features1),
                           self.aggregator_2(features2)],
                           dim=1)
        fused = self.decoder(fused)
        return fused

def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(2,48,64,64).astype(np.float32)).cuda()
    model = Fusion_Net(dim=16)
    model.cuda()
    y = model(x,x)
    print('output shape:', y.shape)


if __name__ == '__main__':
    unit_test()