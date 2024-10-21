# --------------------------------------------------------
# MedFormer Library for Oceanic Forecasting
# Copyright (c) 2023-2024 CMCC Foundation
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# --------------------------------------------------------

# import torch.nn.functional as F
# from einops import rearrange
# from torch import nn
# import numpy as np
# import torch
# import math

# from Fires._utilities.decorators import export

# @export
# class Reshape(nn.Module):
#     def __init__(self, shape, contiguous=True, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.shape = shape
#         self.contiguous = contiguous

#     def forward(self, tensor:torch.Tensor):
#         if self.contiguous:
#             return tensor.contiguous().view(tensor.shape[0], *self.shape)
#         else:
#             return tensor.reshape(tensor.shape[0], *self.shape)


# @export
# class Permute(nn.Module):
#     def __init__(self, dims, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.dims = dims

#     def forward(self, tensor:torch.Tensor):
#         return tensor.permute(dims=(0, *self.dims))


# @export
# class Roll(nn.Module):
#     def __init__(self, shift, dims=(1,2,3), *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.shift = shift
#         self.dims = dims

#     def forward(self, tensor:torch.Tensor):
#         return tensor.roll(shifts=self.shift, dims=self.dims)


# @export
# class Concatenate(nn.Module):
#     def __init__(self, dim, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.dim = dim

#     def forward(self, tensors:list):
#         return torch.cat(tensors, dim=self.dim)


# @export
# class MoveAxis(nn.Module):
#     def __init__(self, src, dst, contiguous=True, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.source = src
#         self.destination = dst
#         self.contiguous = contiguous

#     def forward(self, tensor:torch.Tensor):
#         tensor = tensor.moveaxis(source=self.source, destination=self.destination)
#         if self.contiguous:
#             return tensor.contiguous()
#         return tensor


# @export
# class Expand(nn.Module):
#     def __init__(self, shape, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.shape = shape
    
#     def forward(self, tensor:torch.Tensor):
#         return tensor.expand(tensor.shape[0], *self.shape)


# @export
# class Padding(nn.Module):
#     def __init__(self, shape: tuple, patch_size: tuple, crop: bool = False, ch_last: bool = True) -> None:
#         super().__init__()
#         self.crop = crop
#         self.shape = shape
#         self.ch_last = ch_last
#         self.patch_size = patch_size

#     def forward(self, x: torch.Tensor):
#         if len(x.shape) == 5:
#             dims1 = (0,4,1,2,3)
#             dims2 = (0,2,3,4,1)
#         elif len(x.shape) == 4:
#             dims1 = (0,3,1,2)
#             dims2 = (0,2,3,1)
#         s, p = np.array(self.shape), np.array(self.patch_size)
#         mod = s % p
#         pad_amount = p - np.where(mod == 0, p, mod)
#         # left_pad, right_pad = np.floor(pad_amount / 2), np.ceil(pad_amount / 2)
#         left_pad, right_pad = np.array([0]*pad_amount.shape[0]), pad_amount
#         if self.crop:
#             left_pad, right_pad = -left_pad, -right_pad
#         padding = np.stack([left_pad, right_pad], axis=1).astype(np.int16)
#         padding = tuple(padding[::-1].flatten())
#         if self.ch_last:
#             x = x.permute(dims=dims1)
#             x = F.pad(x, padding)
#             x = x.permute(dims=dims2)
#             return x
#         else:
#             return F.pad(x, padding)


# @export
# class ClimePreprocess(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
    
#     def forward(self, inputs):
#         x3d, x2d, f2d, a2d = inputs

#         # process f2d and a2d and stack to x2d
#         with torch.no_grad():
#             data = [x2d]
#             if f2d.shape[1] != 0: 
#                 data += [f2d]
#             if a2d.shape[1] != 0:
#                 a2d = F.interpolate(a2d, size=(x2d.shape[2:]), mode='bilinear')
#                 data += [a2d]
#             x2d = torch.cat(data, dim=1)

#         return x3d, x2d


# @export
# class ClimaPreprocess2D(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, inputs):
#         x3d, x2d, a2d = inputs
#         # process f2d and a2d and stack to x2d
#         with torch.no_grad():
#             data = [x2d]
#             if a2d.shape[1] != 0:
#                 a2d = F.interpolate(a2d, size=(x2d.shape[2:]), mode='bilinear')
#                 data += [a2d]
#             x2d = torch.cat(data, dim=1)
#         # reshape x3D in 2D
#         B,C,Z,H,W = x3d.shape
#         x3d = torch.reshape(x3d, shape=(B,C*Z,H,W))
#         # concat the data in 2D
#         x = torch.cat([x3d, x2d], dim=1)
#         return x


# @export
# class ResidualClimaPreprocess2D(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, inputs):
#         x3d, x2d, a2d = inputs
#         with torch.no_grad():
#             if a2d.shape[1] != 0:
#                 atmos = F.interpolate(a2d, size=(x2d.shape[2:]), mode='bilinear')
#         # reshape x3D in 2D
#         B,C,Z,H,W = x3d.shape
#         x3d = torch.reshape(x3d, shape=(B,C*Z,H,W))
#         # concat the data in 2D
#         ocean = torch.cat([x3d, x2d], dim=1)
#         return ocean, atmos


# @export
# class ClimaPreprocess3D(nn.Module):
#     """
#     Pre-processes variables int 3D fashion (Time, Height, Width)
#     """
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, inputs):
#         x3d, x2d, a2d = inputs
#         # process f2d and a2d and stack to x2d
#         with torch.no_grad():
#             data = [x2d]
#             if a2d.shape[1] != 0:
#                 a2d = F.interpolate(a2d, size=(x2d.shape[2:]), mode='trilinear')
#                 data += [a2d]
#             x2d = torch.cat(data, dim=1)
#         # reshape x3D in 2D
#         B,C,T,Z,H,W = x3d.shape
#         x3d = x3d.permute(dims=(0,1,3,2,4,5)).reshape(B,C*Z,T,H,W)
#         # concat the data in 2D
#         x = torch.cat([x3d, x2d], dim=1)
#         return x


# @export
# class ResClimaPreprocess3D(nn.Module):
#     """
#     Pre-processes variables int 3D fashion (Time, Height, Width)
#     """
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, inputs):
#         x3d, x2d, a2d = inputs
#         # interpolate atmosphere
#         with torch.no_grad():
#             if a2d.shape[1] != 0:
#                 atmos = F.interpolate(a2d, size=(x2d.shape[2:]), mode='trilinear')
#         # reshape x3D in 2D
#         B,C,T,Z,H,W = x3d.shape
#         x3d = x3d.permute(dims=(0,1,3,2,4,5)).reshape(B,C*Z,T,H,W)
#         # concat the data in 2D
#         ocean = torch.cat([x3d, x2d], dim=1)
#         return ocean, atmos


# @export
# class AtmosInterp(nn.Module):
#     def __init__(self, mode='trilinear') -> None:
#         super().__init__()
#         self.mode = mode

#     def forward(self, x):
#         x3d, x2d, a2d = x
#         # interpolate atmosphere
#         with torch.no_grad():
#             if a2d.shape[1] != 0:
#                 a2d = F.interpolate(a2d, size=(x2d.shape[2:]), mode=self.mode)
#         return x3d, x2d, a2d


# @export
# class Preprocess3d(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x):
#         x3d, x2d, a2d = x
#         # reshape x3D in 2D
#         B,C,T,Z,H,W = x3d.shape
#         x3d = x3d.permute(dims=(0,1,3,2,4,5)).reshape(B,C*Z,T,H,W)
#         # concat the data in 2D
#         x = torch.cat([x3d, x2d, a2d], dim=1)
#         return x


# @export
# class Preprocess2d(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x):
#         x3d, x2d, a2d, s2d = x
#         # reshape x3D in 2D
#         B,C,Z,H,W = x3d.shape
#         x3d = x3d.reshape(B,C*Z,H,W)
#         # concat the data in 2D
#         x = torch.cat([x3d, x2d, a2d, s2d], dim=1)
#         return x


# @export
# class ClimaPostprocess2D(nn.Module):
#     def __init__(self, C3d, C2d, Z, H, W) -> None:
#         super().__init__()
#         self.C3d, self.C2d, self.Z, self.H, self.W = C3d, C2d, Z, H, W

#     def forward(self, x):
#         # separate 3D data from 2D data
#         x3d, x2d = x[:,:-self.C2d,:,:], x[:,-self.C2d:,:,:]
#         # separate C from Z in 3D variable
#         x3d = torch.reshape(x3d, shape=(-1, self.C3d, self.Z, self.H, self.W))
#         return x3d, x2d


# @export
# class Postprocess3d(nn.Module):
#     def __init__(self, C3d, C2d, Z, H, W) -> None:
#         super().__init__()
#         self.C3d, self.C2d, self.Z, self.H, self.W = C3d, C2d, Z, H, W

#     def forward(self, x):
#         B,C,T,H,W = x.shape
#         # separate 3D data from 2D data
#         x3d, x2d = x[:,:-self.C2d,:,:,:], x[:,-self.C2d:,:,:,:]
#         # separate C from Z in 3D variable
#         x3d = torch.reshape(x3d, shape=(B, self.C3d, self.Z, T, self.H, self.W)).transpose(2,3)
#         return x3d, x2d


# @export
# class PIClimaPostprocess2d(nn.Module):
#     def __init__(self, C3d, C2d, Z, H, W) -> None:
#         super().__init__()
#         self.C3d, self.C2d, self.Z, self.H, self.W = C3d, C2d, Z, H, W

#     def forward(self, x):
#         # separate 3D data from 2D data
#         x2d, x3d = x[:,:self.C2d], x[:,self.C2d:]
#         # separate 3D data from 2D data
#         # x3d, x2d = x[:,:-self.C2d,:,:], x[:,-self.C2d:,:,:]
#         # separate C from Z in 3D variable
#         x3d = torch.reshape(x3d, shape=(-1, self.C3d, self.Z, self.H, self.W))
#         return x3d, x2d


# @export
# class GaussianNoise(nn.Module):
#     """Gaussian noise regularizer.

#     Args:
#         sigma (float, optional): relative standard deviation used to generate the
#             noise. Relative means that it will be multiplied by the magnitude of
#             the value your are adding the noise to. This means that sigma can be
#             the same regardless of the scale of the vector.
#     """
#     def __init__(self, sigma=0.1):
#         super().__init__()
#         self.sigma = sigma

#     def forward(self, x: torch.Tensor):
#         if torch.is_grad_enabled() and self.sigma != 0:
#             noise = torch.sqrt(torch.as_tensor(self.sigma)) * torch.randn_like(x)
#             x = x + noise
#         return x


# @export
# class GroupNorm(nn.GroupNorm):
#     def __init__(self, dim: int, eps: float = 0.00001, affine: bool = True, device=None, dtype=None) -> None:
#         super().__init__(32, dim, eps, affine, device, dtype)

#     def forward(self, x: torch.Tensor):
#         x = x.transpose(1,2) # B, C, L
#         x = super().forward(x)
#         x = x.transpose(1,2) # B, L, C
#         return x


# @export
# def weights_init(init_type='gaussian'):
#     def init_fun(m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#             if init_type == 'gaussian':
#                 nn.init.normal_(m.weight, 0.0, 0.05)
#                 try:
#                     nn.init.normal_(m.bias, 0.0, 0.05)
#                 except:
#                     pass
#             elif init_type == 'xavier':
#                 nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
#             elif init_type == 'kaiming':
#                 nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
#             elif init_type == 'default':
#                 pass
#             else:
#                 assert 0, "Unsupported initialization: {}".format(init_type)
#             # if hasattr(m, 'bias') and m.bias is not None:
#             #     nn.init.constant_(m.bias, 0.0)

#     return init_fun


# @export
# class PatchifyModule(nn.Module):
#     def __init__(self, original_img_size, model_img_size) -> None:
#         super().__init__()
#         self.original_img_size = original_img_size
#         self.model_img_size = model_img_size
#         self.shapes_ratio = (self.original_img_size[0] // self.model_img_size[0], self.original_img_size[1] // self.model_img_size[1])

#     def forward(self, x: torch.Tensor):
#         B, C, T, H, W = x.shape
#         R1, R2 = self.shapes_ratio
#         x = torch.stack(x.split(split_size=[H//R1 for _ in range(R1)], dim=3), dim=1)
#         x = torch.stack(x.split(split_size=[W//R2 for _ in range(R2)], dim=5), dim=2)
#         x = x.view(B*R1*R2, C, T, H//R1, W//R2)
#         return x


# @export
# class UnpatchifyModule(nn.Module):
#     def __init__(self, original_img_size, model_img_size) -> None:
#         super().__init__()
#         self.original_img_size = original_img_size
#         self.model_img_size = model_img_size
#         self.shapes_ratio = (self.original_img_size[0] // self.model_img_size[0], self.original_img_size[1] // self.model_img_size[1])

#     def forward(self, x: torch.Tensor):
#         B_, C, H, W = x.shape
#         R1, R2 = self.shapes_ratio
#         B = B_//(R1*R2)
#         x = x.view(B, R1, R2, C, H, W)
#         x = torch.cat([x[:,i,] for i in range(R1)], dim=3)
#         x = torch.cat([x[:,j,] for j in range(R2)], dim=3)
#         return x


# @export
# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, x):
#         device = x.device
#         half_dim = self.dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
#         emb = x[:, None] * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb


# @export
# class TimeMlp(nn.Module):
#     def __init__(self, in_features: int, hidden_features: int=None, out_features: int=None) -> None:
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.dim = in_features
#         self.time_dim = out_features
#         self.sinusoidal_pos_embed = SinusoidalPosEmb(in_features)
#         self.linear1 = nn.Linear(in_features, hidden_features)
#         self.silu = nn.SiLU()
#         self.linear2 = nn.Linear(hidden_features, out_features)

#     def forward(self, t: torch.Tensor):
#         t = self.sinusoidal_pos_embed(t)
#         t = self.linear1(t)
#         t = self.silu(t)
#         t = self.linear2(t)
#         return t


# @export
# class TimeMlpShiftScale(nn.Module):
#     def __init__(self, in_features: int, out_features: int=None) -> None:
#         super().__init__()
#         out_features = out_features or in_features
#         self.dim = in_features
#         self.time_dim = out_features
#         self.silu = nn.SiLU()
#         self.linear = nn.Linear(in_features, out_features * 2)

#     def forward(self, t: torch.Tensor):
#         t = self.silu(t)
#         t = self.linear(t)
#         t = rearrange(t, "b t c -> b t 1 c")
#         scale_shift = t.chunk(2, dim=3)
#         return scale_shift


# @export
# class ShiftScale(nn.Module):
#     def __init__(self, dim, height=None, width=None) -> None:
#         super().__init__()
#         H, W = height or 1, width or 1
#         self.dim = dim
#         self.shift = nn.Parameter(torch.zeros(1,dim,1,H,W), requires_grad=True)
#         self.scale = nn.Parameter(torch.ones(1,dim,1,H,W), requires_grad=True)
    
#     def forward(self, x: torch.Tensor):
#         return x * self.scale + self.shift
