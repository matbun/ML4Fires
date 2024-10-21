# from Fires._layers.swin_utils import Padding
# from Fires._layers.swin_utils import Padding
# from Fires._utilities.decorators import export

# from timm.models.layers import to_2tuple, DropPath, trunc_normal_
# from typing import Optional, Any, Union, List, Tuple
# import torch.utils.checkpoint as checkpoint
# import torch.nn.functional as F
# from einops import rearrange
# import torch.nn as nn
# import numpy as np
# import torch



# @export
# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
#     return windows


# @export
# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
#     x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# @export
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# @export
# class TemporalPatchEmbed(nn.Module):
#     def __init__(self, 
#             img_size: Union[int,Tuple[int],List[int]], 
#             in_channels: int, 
#             patch_size: Optional[Union[int,Tuple[int],List[int]]]=4, 
#             time_instants: Optional[int]=1,
#             embed_dim: Optional[int]=96, 
#             norm_layer: Optional[Any]=None, 
#             use_bias: Optional[bool]=True):
#         """ Patch Embedding class similar to Swin-Transformer one.
#         Parameters
#         ----------
#             input_size (Union[int,Tuple[int],List[int]]): size of the input.
#             in_channels (int): number of input channels
#             patch_size (Optional[Union[int,Tuple[int],List[int]]], optional): Size of the patch for the patch embedding. Defaults to 4.
#             embed_dim (Optional[int], optional): Size of the output patch embedding. Defaults to 96.
#             norm_layer (Optional[Any], optional): Layer for normalization after the embedding. Defaults to None.

#         """
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         padded_input_size = get_padded_shape(img_size, patch_size)
#         patches_resolution = [padded_input_size[0] // patch_size[0], padded_input_size[1] // patch_size[1]]
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.input_size = img_size

#         self.pad = Padding(img_size, patch_size, ch_last=False)
#         self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=(time_instants,*patch_size), stride=(time_instants,*patch_size), bias=use_bias)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         _, _, _, H, W = x.shape
#         assert H == self.input_size[0] and W == self.input_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
#         x = self.pad(x)
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops



# @export
# class PatchUnembed(nn.Module):
#     def __init__(self, 
#             dim: int, 
#             out_channels: int, 
#             input_resolution: Union[Tuple[int],List[int]], 
#             output_resolution: Optional[Union[Tuple[int],List[int]]], 
#             upscale_factor: Optional[int]=4, 
#             norm_layer: Optional[Union[Any,nn.Module]]=nn.LayerNorm, 
#             ) -> None:
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_resolution = output_resolution
#         self.dim = dim
#         self.upscale_factor = upscale_factor
#         self.expand = nn.Linear(dim, (upscale_factor**2) * dim, bias=False)
#         self.norm = norm_layer(dim) if norm_layer is not None else None
#         self.conv = nn.Conv2d(dim, out_channels, kernel_size=1, bias=False)

#     def forward(self, x: torch.Tensor):
#         H, W = self.input_resolution
#         x = self.expand(x)
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.upscale_factor, p2=self.upscale_factor, c=C//(self.upscale_factor**2))
#         B, H, W, C = x.shape
#         x = x.view(B, H*W, C)
#         if self.norm is not None:
#             x = self.norm(x)
#         x = x.view(B, H, W, C).permute(0,3,1,2)
#         x = self.conv(x)
#         return x



# @export
# class PatchUnembedBilinear(nn.Module):
#     def __init__(self, 
#             dim: int, 
#             out_channels: int, 
#             input_resolution: Union[Tuple[int],List[int]], 
#             output_resolution: Optional[Union[Tuple[int],List[int]]], 
#             upscale_factor: Optional[int]=4, 
#             norm_layer: Optional[Union[Any,nn.Module]]=nn.LayerNorm, 
#             mode: Optional[str]='bilinear', 
#             ) -> None:
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_resolution = output_resolution
#         self.dim = dim
#         self.mode = mode
#         self.upscale_factor = upscale_factor
#         self.mode = mode
#         self.upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
#         self.proj = nn.Linear(dim, dim * 2, bias=False)
#         self.norm = norm_layer(dim * 2) if norm_layer is not None else nn.Identity()
#         self.out = nn.Linear(dim * 2, out_channels)

#     def forward(self, x: torch.Tensor):
#         H, W = self.input_resolution
#         _, L, _ = x.shape
#         assert L == H * W, "input feature has wrong size"
#         x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
#         # upsample
#         x = self.upsample(x)
#         H, W = x.shape[-2:]
#         # reshape x back
#         x = rearrange(x, "b c h w -> b (h w) c")
#         # project
#         x = self.proj(x)
#         x = self.norm(x)
#         x = self.out(x)
#         x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
#         return x



# @export
# class Downsample(nn.Module):
#     r""" Patch Merging Layer.

#     Parameters
#     ----------
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

#     """
#     def __init__(self, 
#             input_resolution: Union[Tuple[int],List[int]], 
#             dim: int, 
#             dim_factor: Optional[int] = 2, 
#             norm_layer: Optional[Union[Any,nn.Module]]=nn.LayerNorm, 
#             output_resolution: Optional[Union[Tuple[int],List[int]]] = None, 
#         ) -> None:
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_resolution = output_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(in_features=4 * dim, out_features=dim_factor * dim, bias=False)
#         self.norm = norm_layer(4 * dim) if norm_layer is not None else None
#         self.pad = Padding(input_resolution, patch_size=(2,2), ch_last=True)

#     def forward(self, x: torch.Tensor):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         # reshape the data 
#         x = x.view(B, H, W, C)                 # (B, H, W, C), before: (B, C, H, W)
#         # pad if needed
#         x = self.pad(x)                        # (B, H_, W_, C)
#         # get the new shape
#         _, H_, W_, _ = x.shape
#         # split H_ and W_ in 2 dimensions
#         x = x.view(B, H_//2, 2, W_//2, 2, C)   # (B, H_/2, 2, W_/2, 2, C)
#         # permute the order of the dimensions
#         x = x.permute(dims=(0,1,3,2,4,5))      # (B, H_/2, W_/2, 2, 2, C)
#         # aggregate last three dimensions together
#         x = x.contiguous().view(B, -1, 4 * C)  # (B, H_/2*W_/2, 4*C)
#         # normalize the tensor
#         if self.norm is not None:
#             x = self.norm(x)                   # (B, H_/2*W_/2, 2*C)
#         # halve the number of channels
#         x = self.reduction(x)                  # (B, H_/2*W_/2, 2*C)
#         return x                               # (B, H_/2*W_/2, 2*C)

#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"

#     def flops(self):
#         H, W = self.input_resolution
#         flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         flops += H * W * self.dim // 2
#         return flops



# @export
# class Upsample(nn.Module):
#     """ Reversed operation of Downsample."""
#     def __init__(self, 
#             input_resolution: Union[Tuple[int],List[int]], 
#             output_resolution: Optional[Union[Tuple[int],List[int]]], 
#             dim: int, 
#             dim_factor: Optional[int]=None, 
#             norm_layer: Optional[Union[Any,nn.Module]]=nn.LayerNorm, 
#         ) -> None:
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_resolution = output_resolution
#         self.expand = nn.Linear(dim, dim * 2 if dim_factor==2 else dim * 4, bias=False)
#         self.norm = norm_layer(dim // 2 if dim_factor==2 else dim) if norm_layer is not None else None
#         self.crop = Padding(output_resolution, patch_size=(2,2), crop=True, ch_last=True)

#     def forward(self, x: torch.Tensor):
#         B, L, C = x.shape                                   # (B, L, C)
#         H, W = self.input_resolution
#         assert L == H * W, "input feature has wrong size"
#         # increase the number of channels                    
#         x = self.expand(x)                                  # (B, L, 2C)
#         B, L, C = x.shape
#         x = x.view(B, H, W, C)
#         # reorder the dimension aggregating the channels into height and width
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
#         # pad
#         x = self.crop(x)                                    # (B, 2H_, 2W_, C/2)
#         # reshape x back
#         x = x.contiguous().view(B, -1, C//4)                # (B, 2H_*2W_, C/2)
#         # normalize
#         if self.norm is not None:
#             x = self.norm(x)                                # (B, 2H_*2W_, C/2)
#         return x                                            # (B, 2H_*2W_, C/2)



# @export
# class UpsampleBilinear(nn.Module):
#     """ Reversed operation of Downsample."""
#     def __init__(self, 
#             input_resolution: Union[Tuple[int],List[int]], 
#             output_resolution: Optional[Union[Tuple[int],List[int]]], 
#             dim: int, 
#             dim_factor: Optional[int]=None, 
#             norm_layer: Optional[Union[Any,nn.Module]]=nn.LayerNorm, 
#             mode: Optional[str]='bilinear', 
#         ) -> None:
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.output_resolution = output_resolution
#         self.mode = mode
#         self.upsample = nn.Upsample(scale_factor=2, mode=mode)
#         self.proj = nn.Linear(dim, dim // 2 if dim_factor==2 else dim, bias=False)
#         self.norm = norm_layer(dim // 2 if dim_factor==2 else dim) if norm_layer is not None else None
#         self.crop = Padding(output_resolution, patch_size=(2,2), crop=True, ch_last=True)

#     def forward(self, x: torch.Tensor):
#         B, L, C = x.shape
#         H, W = self.input_resolution
#         assert L == H * W, "input feature has wrong size"
#         # reshape x to use the upsample layer
#         x = x.view(B, H, W, C).permute(0,3,1,2)
#         # upsample
#         x = self.upsample(x)
#         B, C, H, W = x.shape
#         # reshape x back
#         x = x.permute(0,2,3,1)
#         # pad
#         x = self.crop(x)
#         # reshape x back
#         x = x.contiguous().view(B, -1, C)
#         # project
#         x = self.proj(x)
#         # normalize
#         if self.norm is not None:
#             x = self.norm(x)
#         return x



# @export
# class WindowAttention_v1(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.win_size = win_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.win_size[0])
#         coords_w = torch.arange(self.win_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.win_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.win_size}, num_heads={self.num_heads}'

#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops



# @export
# class WindowAttention_v2(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#         pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
#     """

#     def __init__(self, dim, win_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
#                  pretrained_window_size=[0, 0]):

#         super().__init__()
#         self.dim = dim
#         self.win_size = win_size  # Wh, Ww
#         self.pretrained_window_size = pretrained_window_size
#         self.num_heads = num_heads

#         self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

#         # mlp to generate continuous relative position bias
#         self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
#                                      nn.ReLU(inplace=True),
#                                      nn.Linear(512, num_heads, bias=False))

#         # get relative_coords_table
#         relative_coords_h = torch.arange(-(self.win_size[0] - 1), self.win_size[0], dtype=torch.float32)
#         relative_coords_w = torch.arange(-(self.win_size[1] - 1), self.win_size[1], dtype=torch.float32)
#         relative_coords_table = torch.stack(
#             torch.meshgrid([relative_coords_h,
#                             relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
#         if pretrained_window_size[0] > 0:
#             relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
#         else:
#             relative_coords_table[:, :, :, 0] /= (self.win_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (self.win_size[1] - 1)
#         relative_coords_table *= 8  # normalize to -8, 8
#         relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
#             torch.abs(relative_coords_table) + 1.0) / np.log2(8)

#         self.register_buffer("relative_coords_table", relative_coords_table)

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.win_size[0])
#         coords_w = torch.arange(self.win_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.win_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # bias=False)
#         # if qkv_bias:
#         #     self.q_bias = nn.Parameter(torch.zeros(dim))
#         #     self.v_bias = nn.Parameter(torch.zeros(dim))
#         # else:
#         #     self.q_bias = None
#         #     self.v_bias = None
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape

#         # qkv_bias = None
#         # if self.q_bias is not None:
#         #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
#         # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         # cosine attention
#         attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
#         logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(x.device)).exp()
#         attn = attn * logit_scale

#         relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
#         relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn1 = attn.view(B_ // nW, nW, self.num_heads, N, N)
#             mask1 = mask.unsqueeze(1).unsqueeze(0)
#             attn = attn1 + mask1
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.win_size}, ' \
#                f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops



# @export
# class SwinTransformerBlock_v1(nn.Module):
#     r""" Swin Transformer Block.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#         fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
#     """

#     def __init__(self, dim, input_resolution, num_heads, win_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  fused_window_process=False, sealand_attn_mask=None, pretrained_window_size=None):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.win_size = win_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         self.sealand_attn_mask = sealand_attn_mask
#         if self.input_resolution[0] <= self.win_size[0] and self.input_resolution[1] <= self.win_size[1]:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = (0,0)
#             self.win_size = self.input_resolution
#         assert 0 <= self.shift_size[0] < self.win_size[0], "shift_size must in 0-window_size"
#         assert 0 <= self.shift_size[1] < self.win_size[1], "shift_size must in 0-window_size"

#         self.pad = Padding(shape=input_resolution, patch_size=win_size)
#         self.crop = Padding(shape=input_resolution, patch_size=win_size, crop=True)

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention_v1(
#             dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         if self.shift_size[0] > 0:
#             # calculate attention mask for SW-MSA
#             H_, W_ = get_padded_shape(shape=input_resolution, patch_size=win_size)
#             img_mask = torch.zeros((1, H_, W_, 1))  # 1 H W 1
#             h_slices = (slice(0, -self.win_size[0]),
#                         slice(-self.win_size[0], -self.shift_size[0]),
#                         slice(-self.shift_size[0], None))
#             w_slices = (slice(0, -self.win_size[1]),
#                         slice(-self.win_size[1], -self.shift_size[1]),
#                         slice(-self.shift_size[1], None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.win_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.win_size[0] * self.win_size[1])
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask + sealand_attn_mask if sealand_attn_mask is not None else attn_mask
#         else:
#             # attn_mask = None
#             attn_mask = sealand_attn_mask if sealand_attn_mask is not None else None

#         self.register_buffer("attn_mask", attn_mask)
#         self.fused_window_process = fused_window_process

#     def forward(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         # pad
#         x = self.pad(x)
#         # get new data shape
#         B, H, W, C = x.shape

#         # cyclic shift
#         if self.shift_size[0] > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
#             # partition windows
#             x_windows = window_partition(shifted_x, self.win_size)  # nW*B, window_size, window_size, C
#         else:
#             shifted_x = x
#             # partition windows
#             x_windows = window_partition(shifted_x, self.win_size)  # nW*B, window_size, window_size, C

#         x_windows = x_windows.view(-1, self.win_size[0] * self.win_size[1], C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.win_size[0], self.win_size[1], C)

#         # reverse cyclic shift
#         if self.shift_size[0] > 0:
#             shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
#             x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
#         else:
#             shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
#             x = shifted_x

#         # cropping
#         x = self.crop(x)
#         # get new data shape
#         B, H, W, C = x.shape

#         x = x.view(B, H * W, C)
#         x = shortcut + self.drop_path(x)

#         # FFN
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.win_size / self.win_size
#         flops += nW * self.attn.flops(self.win_size * self.win_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops



# @export
# class SwinTransformerBlock_v2(nn.Module):
#     r""" Swin Transformer Block.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#         pretrained_window_size (int): Window size in pre-training.
#     """

#     def __init__(self, dim, input_resolution, num_heads, win_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, sealand_attn_mask=None):
#         super().__init__()
#         self.channels = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.win_size = win_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         self.sealand_attn_mask = sealand_attn_mask
#         if self.input_resolution[0] <= self.win_size[0] and self.input_resolution[1] <= self.win_size[1]:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = (0,0)
#             self.win_size = self.input_resolution
#         assert 0 <= self.shift_size[0] < self.win_size[0], "shift_size must in 0-window_size"
#         assert 0 <= self.shift_size[1] < self.win_size[1], "shift_size must in 0-window_size"

#         self.pad = Padding(shape=input_resolution, patch_size=win_size)
#         self.crop = Padding(shape=input_resolution, patch_size=win_size, crop=True)

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention_v2(
#             dim, win_size=self.win_size, num_heads=num_heads,
#             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
#             pretrained_window_size=to_2tuple(pretrained_window_size))

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         if self.shift_size[0] > 0:
#             # calculate attention mask for SW-MSA
#             H_, W_ = get_padded_shape(shape=input_resolution, patch_size=win_size)
#             img_mask = torch.zeros((1, H_, W_, 1))  # 1 H W 1
#             h_slices = (slice(0, -self.win_size[0]),
#                         slice(-self.win_size[0], -self.shift_size[0]),
#                         slice(-self.shift_size[0], None))
#             w_slices = (slice(0, -self.win_size[1]),
#                         slice(-self.win_size[1], -self.shift_size[1]),
#                         slice(-self.shift_size[1], None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.win_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.win_size[0] * self.win_size[1])
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#             attn_mask = attn_mask + sealand_attn_mask if sealand_attn_mask is not None else attn_mask
#         else:
#             # attn_mask = None
#             attn_mask = sealand_attn_mask if sealand_attn_mask is not None else None

#         self.register_buffer("attn_mask", attn_mask)

#     def forward(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = x.view(B, H, W, C)

#         # pad
#         x = self.pad(x)
#         # get new data shape
#         B, H, W, C = x.shape

#         # cyclic shift
#         if self.shift_size[0] > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = window_partition(shifted_x, self.win_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.win_size[0] * self.win_size[1], C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.win_size[0], self.win_size[1], C)
#         shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size[0] > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
#         else:
#             x = shifted_x

#         # cropping
#         x = self.crop(x)
#         # get new data shape
#         B, H, W, C = x.shape

#         x = x.view(B, H * W, C)
#         x = shortcut + self.drop_path(self.norm1(x))

#         # FFN
#         x = x + self.drop_path(self.norm2(self.mlp(x)))
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.channels}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.channels * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.win_size / self.win_size
#         flops += nW * self.attn.flops(self.win_size * self.win_size)
#         # mlp
#         flops += 2 * H * W * self.channels * self.channels * self.mlp_ratio
#         # norm2
#         flops += self.channels * H * W
#         return flops



# @export
# class BasicLayer(nn.Module):
#     def __init__(self, dim, input_resolution, depth, num_heads, win_size, dim_factor=2, output_resolution=None, 
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., use_checkpoint=False,
#                  drop_path=0., norm_layer=nn.LayerNorm, operation=None, pretrained_window_size=0, sealand_attn_mask=None, version=1):

#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.output_resolution = output_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         shift_sizes = [(0,0) if (i % 2 == 0) else (win_size[0] // 2, win_size[1] // 2) for i in range(depth)]
#         sealand_attn_masks = [sealand_attn_mask for i in range(depth)]

#         if version == 1:
#             swin_transformer_block = SwinTransformerBlock_v1
#         elif version == 2:
#             swin_transformer_block = SwinTransformerBlock_v2

#         # Construct basic blocks
#         self.blocks = nn.Sequential(*[
#             swin_transformer_block(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads, win_size=win_size,
#                                  shift_size=shift_sizes[i],
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer,
#                                  pretrained_window_size=pretrained_window_size, 
#                                  sealand_attn_mask=sealand_attn_masks[i], 
#                                  )
#             for i in range(depth)])

#         # downsampling-upsampling layer
#         if operation is not None:
#             self.operation = operation(input_resolution, dim=dim, norm_layer=norm_layer, output_resolution=output_resolution, dim_factor=dim_factor)
#         else:
#             self.operation = None

#     def forward(self, x: torch.Tensor):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.operation is not None:
#             x = self.operation(x)
#         return x

#     def _init_respostnorm(self):
#         for blk in self.blocks:
#             try:
#                 nn.init.constant_(blk.norm1.bias, 0)
#                 nn.init.constant_(blk.norm1.weight, 0)
#                 nn.init.constant_(blk.norm2.bias, 0)
#                 nn.init.constant_(blk.norm2.weight, 0)
#             except:
#                 pass
