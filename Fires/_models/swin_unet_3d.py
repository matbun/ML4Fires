# from Fires._layers.swin3d import BasicLayer, PatchEmbed, PatchUnembed, Downsample, Upsample
# from Fires._layers.swin_utils import ShiftScale, TimeMlpShiftScale
# from Fires._utilities.decorators import export

# from typing import Optional, Union, List, Tuple
# from timm.models.layers import trunc_normal_
# import torch.utils.checkpoint as checkpoint
# import torch.nn as nn
# import logging
# import torch


# @export
# class SwinUNet3DConfig():
#     def __init__(self, 
#                  img_size: Optional[Union[Tuple[int], List[int]]] = (224, 224), 
#                  patch_size: Optional[Union[Tuple[int], List[int]]] = 4, 
#                  win_size: Optional[Union[Tuple[int], List[int]]] = 8, 
#                  time_steps: Optional[int] = 1, # number of input time steps
#                  lead_times: Optional[int] = 1, # number of output time steps
#                  embed_dim: Optional[int] = 96, 
#                  depths: Optional[Union[Tuple[int], List[int]]] = [2,2,6,2], 
#                  num_heads: Optional[Union[Tuple[int], List[int]]] = [6,6,12,6], 
#                  pretrained_window_sizes: Optional[Union[Tuple[int], List[int]]] = [0,0,0,0], 
#                  dim_factor: Optional[int] = 1, 
#                  mlp_ratio: Optional[float] = 4., 
#                  qkv_bias: Optional[bool] = True, 
#                  unembed_bias: Optional[bool] = True, 
#                  upsample_bias: Optional[bool] = True, 
#                  drop_rate: Optional[float] = 0.0, 
#                  attn_drop_rate: Optional[float] = 0.0, 
#                  drop_path_rate: Optional[float] = 0.1, 
#                  norm_layer: Optional[nn.Module] = nn.LayerNorm, 
#                  ape: Optional[bool] = False, 
#                  use_checkpoint: Optional[bool] = False, 
#                  swin_block_version: Optional[int] = 2, 
#                  use_time_embed: Optional[bool] = False, 
#                  upsample_mode: Optional[str] = None, 
#                  unembed_mode: Optional[str] = None, 
#                  embed_skip: Optional[bool] = True, 
#                  embed_patch_norm: Optional[bool] = True, 
#                  unembed_patch_norm: Optional[bool] = True) -> None:
#         self.img_size = (time_steps, *img_size)
#         self.time_steps = time_steps
#         self.lead_times = lead_times
#         self.patch_size = patch_size
#         self.win_size = win_size
#         self.embed_dim = embed_dim
#         self.depths = depths
#         self.num_heads = num_heads
#         self.pretrained_window_sizes = pretrained_window_sizes
#         self.dim_factor = dim_factor
#         self.mlp_ratio = mlp_ratio
#         self.qkv_bias = qkv_bias
#         self.unembed_bias = unembed_bias
#         self.upsample_bias = upsample_bias
#         self.drop_rate = drop_rate
#         self.attn_drop_rate = attn_drop_rate
#         self.drop_path_rate = drop_path_rate
#         self.norm_layer = norm_layer
#         self.ape = ape
#         self.use_checkpoint = use_checkpoint
#         self.swin_block_version = swin_block_version
#         self.use_time_embed = use_time_embed
#         self.upsample_mode = upsample_mode
#         self.unembed_mode = unembed_mode
#         self.embed_skip = embed_skip
#         self.embed_patch_norm = embed_patch_norm
#         self.unembed_patch_norm = unembed_patch_norm
#         self.weights_std_config = {'linear': 0.02, 'conv': 0.03}
#         self._log_configuration()

#     def _log_configuration(self):
#         attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
#         values = [self.__getattribute__(attr) for attr in attributes]
#         logging.basicConfig(format="%(message)s")
#         logging.info(f'-------------- {self.__class__.__name__} --------------')
#         for attr, value in zip(attributes, values):
#             logging.info(f'  {attr} = {value}')
#         logging.info(f'-------------- {self.__class__.__name__} --------------')


# @export
# class SwinUNet3DModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, config: SwinUNet3DConfig) -> None:
#         super().__init__()
#         img_size = config.img_size
#         patch_size = config.patch_size
#         win_size = config.win_size
#         embed_dim = config.embed_dim
#         depths = config.depths
#         num_heads = config.num_heads
#         lead_times = config.lead_times
#         time_steps = config.time_steps
#         pretrained_window_sizes = config.pretrained_window_sizes
#         dim_factor = config.dim_factor
#         mlp_ratio = config.mlp_ratio
#         qkv_bias = config.qkv_bias
#         unembed_bias = config.unembed_bias
#         upsample_bias = config.upsample_bias
#         drop_rate = config.drop_rate
#         attn_drop_rate = config.attn_drop_rate
#         drop_path_rate = config.drop_path_rate
#         norm_layer = eval(config.norm_layer) if isinstance(config.norm_layer, str) else config.norm_layer
#         ape = config.ape
#         use_checkpoint = config.use_checkpoint
#         self.use_checkpoint = use_checkpoint
#         swin_block_version = config.swin_block_version
#         use_time_embed = config.use_time_embed
#         upsample_mode = config.upsample_mode
#         unembed_mode = config.unembed_mode
#         self.embed_skip = config.embed_skip
#         embed_patch_norm = config.embed_patch_norm
#         unembed_patch_norm = config.unembed_patch_norm
#         self.weights_std_config = config.weights_std_config

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.config = config
#         self.num_layers = len(depths)
#         self.ape = ape

#         # input scaling
#         # self.shift_scale = ShiftScale(dim=in_channels)

#         # time embedding
#         self.time_embed = nn.Embedding(367, embed_dim) if use_time_embed else None
#         self.time_mlp = TimeMlpShiftScale(embed_dim) if use_time_embed else None

#         # 3d patch embedding
#         self.patch_embed = PatchEmbed(img_size=img_size, in_channels=in_channels, time_steps=time_steps, lead_times=lead_times, 
#                                       patch_size=patch_size, embed_dim=embed_dim, 
#                                       norm_layer=norm_layer if embed_patch_norm else None, use_bias=True)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution

#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)

#         # positional dropout
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # add mirrored list
#         depths = depths + list(reversed(depths))
#         num_heads = num_heads + list(reversed(num_heads))
#         pretrained_window_sizes = pretrained_window_sizes + list(reversed(pretrained_window_sizes))

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

#         # create resolutions list
#         input_resolutions = [(patches_resolution[0], -(patches_resolution[1] // -(2 ** i)), -(patches_resolution[2] // -(2 ** i))) for i in range(self.num_layers)]
#         # mirror input_resolutions
#         input_resolutions = input_resolutions + list(reversed(input_resolutions))

#         # build model encoder layers
#         self.encoder = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * dim_factor ** i_layer), dim_factor=dim_factor, input_resolution=input_resolutions[i_layer], 
#                                depth=depths[i_layer], num_heads=num_heads[i_layer], win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
#                                operation=Downsample if i_layer < len(depths)//2-1 else None, norm_layer=norm_layer,
#                                pretrained_window_size=pretrained_window_sizes[i_layer], sealand_attn_mask=None, 
#                                use_checkpoint=use_checkpoint, version=swin_block_version)
#             self.encoder.append(layer)

#         # add skip connection for concatenation
#         self.skip_conn_concat = nn.ModuleList()
#         for i_layer in reversed(range(self.num_layers-1)):
#             layer = nn.Linear(int(embed_dim * dim_factor ** (i_layer+1))*2, int(embed_dim * dim_factor ** (i_layer+1)))
#             self.skip_conn_concat.append(layer)

#         # build model decoder layers
#         self.decoder = nn.ModuleList()
#         for i_layer in range(self.num_layers, self.num_layers*2):
#             if i_layer == self.num_layers:
#                 dim = int(embed_dim * dim_factor ** (self.num_layers * 2 - i_layer - 1))
#             else:
#                 dim = int(embed_dim * dim_factor ** (self.num_layers * 2 - i_layer))
#             layer = BasicLayer(dim=dim, dim_factor=dim_factor, input_resolution=input_resolutions[i_layer-1], 
#                                output_resolution=input_resolutions[i_layer], depth=depths[i_layer],
#                                num_heads=num_heads[i_layer], win_size=win_size, mode=upsample_mode, bias=upsample_bias, 
#                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                operation=Upsample if i_layer > self.num_layers else None, 
#                                norm_layer=norm_layer, pretrained_window_size=pretrained_window_sizes[i_layer], sealand_attn_mask=None, 
#                                use_checkpoint=use_checkpoint, version=swin_block_version)
#             self.decoder.append(layer)

#         self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
#         self.patch_unembed = PatchUnembed(dim=embed_dim, out_channels=out_channels, input_resolution=patches_resolution, mode=unembed_mode, 
#                                           output_resolution=(lead_times, *img_size[1:]), upscale_factor=patch_size[1], 
#                                           norm_layer=norm_layer if unembed_patch_norm else None, use_bias=unembed_bias)

#         # apply weight initialization
#         self.apply(self._init_weights)
#         for bly in self.encoder: bly._init_respostnorm()
#         for bly in self.decoder: bly._init_respostnorm()

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=self.weights_std_config['linear'])
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         # elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
#         #     trunc_normal_(m.weight, std=self.weights_std_config['conv'])
#         #     try: trunc_normal_(m.bias, std=self.weights_std_config['conv'])
#         #     except: pass
    
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

#     def time_embedding(self, x:torch.Tensor, days=None):
#         if days is not None and self.time_embed is not None:
#             shift, scale = self.time_mlp(self.time_embed(days[:,-1:]))
#             return x * scale + shift
#         else:
#             return x

#     def forward_features(self, x: torch.Tensor, days=None):
#         # embed with patch size
#         if self.use_checkpoint: x = checkpoint.checkpoint(self.patch_embed, x, use_reentrant=False)
#         else: x = self.patch_embed(x)
#         # time embedding
#         x = self.time_embedding(x, days)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         # store embedding
#         embed = x
#         x = self.pos_drop(x)
#         skips = []
#         for i,layer in enumerate(self.encoder):
#             # forward encoder
#             x = layer(x)
#             # add skip up to last encoder layer
#             if i != len(self.encoder)-1: 
#                 skips.append(x)
#         for i,layer in enumerate(self.decoder):
#             if i > 0:
#                 # add skip connection from the encoder
#                 x = torch.cat((x, skips[::-1][i-1]), dim=-1)
#                 x = self.skip_conn_concat[i-1](x)
#             # forward decoder
#             x = layer(x)
#         # concatenate patch embedding to x
#         if self.embed_skip:
#             x = x + embed
#         # normalize
#         x = self.norm(x)
#         # patch recovery
#         if self.use_checkpoint: x = checkpoint.checkpoint(self.patch_unembed, x, use_reentrant=False)
#         else: x = self.patch_unembed(x)
#         return x

#     def forward(self, x: torch.Tensor, days=None):
#         x = self.forward_features(x, days)
#         return x

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.encoder):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops
