import torch
import torch.nn as nn
import torch_dct  as dct
class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        # print("self.complex_weight = ",self.complex_weight.shape)  # [256, 64, 64, 2]
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        # freq = dct.dct_2d(x)
        # dct.idct_2d(x)
        # print("fre shapep = ",freq.shape)
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        # print("after shape = ",x.shape) # [2, 256, 64, 33]
        weight = torch.view_as_complex(self.complex_weight)   # [256,128,65]
        print("weight shape = ",weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        print("after shape = ",x.shape)
        x = x.reshape(B, C, H, W)

        return x

input = torch.randn(2,256,64,64)
model = FFParser(256)
out = model(input)
print(out.shape)