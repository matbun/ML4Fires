# from Fires._layers.swin2d import TemporalPatchEmbed, PatchUnembed, PatchUnembedBilinear, Downsample, Upsample, UpsampleBilinear, BasicLayer

# from typing import Optional, Union, List, Tuple, Any
# from timm.models.layers import trunc_normal_, to_2tuple
# import torch.nn as nn
# import logging
# import torch


# class SwinUNetConfig():
#     def __init__(self, 
#                  img_size: Optional[Union[Tuple[int], List[int]]] = (224, 224), 
#                  patch_size: Optional[int] = 4, 
#                  win_size: Optional[int] = 8, 
#                  time_steps: Optional[int] = 1, 
#                  embed_dim: Optional[int] = 96, 
#                  depths: Optional[Union[Tuple[int], List[int]]] = [2,2,6,2], 
#                  num_heads: Optional[Union[Tuple[int], List[int]]] = [6,6,12,6], 
#                  pretrained_window_sizes: Optional[Union[Tuple[int], List[int]]] = [0,0,0,0], 
#                  dim_factor: Optional[int] = 1, 
#                  mlp_ratio: Optional[float] = 4., 
#                  qkv_bias: Optional[bool] = True, 
#                  drop_rate: Optional[float] = 0.0, 
#                  attn_drop_rate: Optional[float] = 0.0, 
#                  drop_path_rate: Optional[float] = 0.1, 
#                  norm_layer: Optional[nn.Module] = nn.LayerNorm, 
#                  ape: Optional[bool] = False, 
#                  use_checkpoint: Optional[bool] = False, 
#                  swin_block_version: Optional[int] = 2, 
#                  upsample_type: Optional[str] = 'standard', 
#                  embed_skip: Optional[bool] = True, 
#                  embed_patch_norm: Optional[bool] = True, 
#                  unembed_patch_norm: Optional[bool] = True) -> None:
#         self.img_size = img_size
#         self.patch_size = to_2tuple(patch_size)
#         self.win_size = to_2tuple(win_size)
#         self.time_steps = time_steps
#         self.embed_dim = embed_dim
#         self.depths = depths
#         self.num_heads = num_heads
#         self.pretrained_window_sizes = pretrained_window_sizes
#         self.dim_factor = dim_factor
#         self.mlp_ratio = mlp_ratio
#         self.qkv_bias = qkv_bias
#         self.drop_rate = drop_rate
#         self.attn_drop_rate = attn_drop_rate
#         self.drop_path_rate = drop_path_rate
#         self.norm_layer = norm_layer
#         self.ape = ape
#         self.use_checkpoint = use_checkpoint
#         self.swin_block_version = swin_block_version
#         self.upsample_type = upsample_type
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


# class SwinUNetModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, config: SwinUNetConfig) -> None:
#         super().__init__()
#         img_size = config.img_size
#         patch_size = config.patch_size
#         win_size = config.win_size
#         time_steps = config.time_steps
#         embed_dim = config.embed_dim
#         depths = config.depths
#         num_heads = config.num_heads
#         pretrained_window_sizes = config.pretrained_window_sizes
#         dim_factor = config.dim_factor
#         mlp_ratio = config.mlp_ratio
#         qkv_bias = config.qkv_bias
#         drop_rate = config.drop_rate
#         attn_drop_rate = config.attn_drop_rate
#         drop_path_rate = config.drop_path_rate
#         norm_layer = eval(config.norm_layer) if isinstance(config.norm_layer, str) else config.norm_layer
#         ape = config.ape
#         use_checkpoint = config.use_checkpoint
#         swin_block_version = config.swin_block_version
#         upsample_type = config.upsample_type
#         self.embed_skip = config.embed_skip
#         embed_patch_norm = config.embed_patch_norm
#         unembed_patch_norm = config.unembed_patch_norm
#         self.weights_std_config = config.weights_std_config

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.config = config
#         self.num_layers = len(depths)
#         self.ape = ape

#         # 3d patch embedding
#         self.patch_embed = TemporalPatchEmbed(img_size=img_size, in_channels=in_channels, patch_size=patch_size, 
#             time_instants=time_steps, embed_dim=embed_dim, norm_layer=norm_layer if embed_patch_norm else None, use_bias=True)
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
#         input_resolutions = [(-(patches_resolution[0] // -(2 ** i)), -(patches_resolution[1] // -(2 ** i))) for i in range(self.num_layers)]
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

#         if upsample_type=='standard':
#             UpsampleLayer = Upsample
#             PatchUnembedLayer = PatchUnembed
#         elif upsample_type=='bilinear':
#             UpsampleLayer = UpsampleBilinear
#             PatchUnembedLayer = PatchUnembedBilinear

#         # build model decoder layers
#         self.decoder = nn.ModuleList()
#         for i_layer in range(self.num_layers, self.num_layers*2):
#             if i_layer == self.num_layers:
#                 dim = int(embed_dim * dim_factor ** (self.num_layers * 2 - i_layer - 1))
#             else:
#                 dim = int(embed_dim * dim_factor ** (self.num_layers * 2 - i_layer))
#             layer = BasicLayer(dim=dim, dim_factor=dim_factor, input_resolution=input_resolutions[i_layer-1], 
#                                output_resolution=input_resolutions[i_layer], depth=depths[i_layer],
#                                num_heads=num_heads[i_layer], win_size=win_size,
#                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                operation=UpsampleLayer if i_layer > self.num_layers else None, 
#                                norm_layer=norm_layer, pretrained_window_size=pretrained_window_sizes[i_layer], 
#                                sealand_attn_mask=None, use_checkpoint=use_checkpoint, version=swin_block_version)
#             self.decoder.append(layer)

#         self.norm = norm_layer(embed_dim) if norm_layer is not None else None
#         self.patch_unembed = PatchUnembedLayer(dim=embed_dim, out_channels=out_channels, input_resolution=patches_resolution, 
#             output_resolution=(patches_resolution[0]*patch_size[0], patches_resolution[1]*patch_size[0]), upscale_factor=patch_size[0], 
#             norm_layer=norm_layer if unembed_patch_norm else None)

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
#         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#             trunc_normal_(m.weight, std=self.weights_std_config['conv'])
#             try: trunc_normal_(m.bias, std=self.weights_std_config['conv'])
#             except: pass
    
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

#     def forward_features(self, x: torch.Tensor):
#         # embed with patch size
#         x = self.patch_embed(x)
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
#         if self.norm is not None:
#             x = self.norm(x)
#         # patch recovery
#         x = self.patch_unembed(x)
#         return x

#     def forward(self, x: torch.Tensor):
#         x = self.forward_features(x)
#         return x

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.encoder):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops



# class SwinMlpUNetModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, config: SwinUNetConfig) -> None:
#         super().__init__()
#         img_size = config.img_size
#         patch_size = config.patch_size
#         win_size = config.win_size
#         time_steps = config.time_steps
#         embed_dim = config.embed_dim
#         depths = config.depths
#         num_heads = config.num_heads
#         pretrained_window_sizes = config.pretrained_window_sizes
#         dim_factor = config.dim_factor
#         mlp_ratio = config.mlp_ratio
#         qkv_bias = config.qkv_bias
#         drop_rate = config.drop_rate
#         attn_drop_rate = config.attn_drop_rate
#         drop_path_rate = config.drop_path_rate
#         norm_layer = eval(config.norm_layer) if isinstance(config.norm_layer, str) else config.norm_layer
#         ape = config.ape
#         use_checkpoint = config.use_checkpoint
#         swin_block_version = config.swin_block_version
#         embed_patch_norm = config.embed_patch_norm
#         unembed_patch_norm = config.unembed_patch_norm
#         self.weights_std_config = config.weights_std_config

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.config = config
#         self.ape = ape

#         # 3d patch embedding
#         self.patch_embed = TemporalPatchEmbed(img_size=img_size, in_channels=in_channels, patch_size=patch_size, 
#             time_instants=time_steps, embed_dim=embed_dim, norm_layer=norm_layer if embed_patch_norm else None, use_bias=True)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution

#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)

#         # positional dropout
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum([depths]))]  # stochastic depth decay rule

#         # resolution at transformer level
#         patches_resolution_transformer = (-(patches_resolution[0] // -(2 ** 1)), -(patches_resolution[1] // -(2 ** 1)))

#         # downsample layer
#         self.downsample = Downsample(input_resolution=patches_resolution, dim=embed_dim, 
#                                      dim_factor=dim_factor, norm_layer=None, 
#                                      output_resolution=patches_resolution_transformer)

#         # transformer backbone
#         self.transformer = BasicLayer(dim=embed_dim, dim_factor=dim_factor, input_resolution=patches_resolution_transformer, 
#                                depth=depths, num_heads=num_heads, win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, 
#                                operation=None, norm_layer=norm_layer,
#                                pretrained_window_size=pretrained_window_sizes, sealand_attn_mask=None, 
#                                use_checkpoint=use_checkpoint, version=swin_block_version)

#         # upsample layer
#         self.upsample = Upsample(input_resolution=patches_resolution_transformer, 
#                                  output_resolution=patches_resolution, dim=embed_dim, 
#                                  dim_factor=dim_factor, norm_layer=None)

#         # reverse operation of patch embedding
#         self.patch_unembed = PatchUnembed(dim=embed_dim, out_channels=out_channels, input_resolution=patches_resolution, 
#             output_resolution=(patches_resolution[0]*patch_size[0], patches_resolution[1]*patch_size[0]), upscale_factor=patch_size[0], 
#             norm_layer=norm_layer if unembed_patch_norm else None)

#         # apply weight initialization
#         self.apply(self._init_weights)
#         self.transformer._init_respostnorm()

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=self.weights_std_config['linear'])
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#             trunc_normal_(m.weight, std=self.weights_std_config['conv'])
#             try: trunc_normal_(m.bias, std=self.weights_std_config['conv'])
#             except: pass
    
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

#     def forward_features(self, x: torch.Tensor):
#         # embed with patch size
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#         # downsample mlp
#         x = self.downsample(x)
#         # swin transformer
#         x = self.transformer(x)
#         # upsample mlp
#         x = self.upsample(x)
#         # patch recovery
#         x = self.patch_unembed(x)
#         return x

#     def forward(self, x: torch.Tensor):
#         x = self.forward_features(x)
#         return x

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.transformer):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1]
#         flops += self.num_features * self.num_classes
#         return flops


# class SwinEncDecModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, config: SwinUNetConfig) -> None:
#         super().__init__()
#         img_size = config.img_size
#         patch_size = config.patch_size
#         win_size = config.win_size
#         time_steps = config.time_steps
#         embed_dim = config.embed_dim
#         depths = config.depths
#         num_heads = config.num_heads
#         pretrained_window_sizes = config.pretrained_window_sizes
#         dim_factor = config.dim_factor
#         mlp_ratio = config.mlp_ratio
#         qkv_bias = config.qkv_bias
#         drop_rate = config.drop_rate
#         attn_drop_rate = config.attn_drop_rate
#         drop_path_rate = config.drop_path_rate
#         norm_layer = eval(config.norm_layer) if isinstance(config.norm_layer, str) else config.norm_layer
#         ape = config.ape
#         use_checkpoint = config.use_checkpoint
#         swin_block_version = config.swin_block_version
#         embed_patch_norm = config.embed_patch_norm
#         unembed_patch_norm = config.unembed_patch_norm
#         self.weights_std_config = config.weights_std_config

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.config = config
#         self.num_layers = len(depths)
#         self.ape = ape

#         # 3d patch embedding
#         self.patch_embed = TemporalPatchEmbed(img_size=img_size, in_channels=in_channels, patch_size=patch_size, 
#             time_instants=time_steps, embed_dim=embed_dim, norm_layer=norm_layer if embed_patch_norm else None, use_bias=True)
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
#         input_resolutions = [(-(patches_resolution[0] // -(2 ** i)), -(patches_resolution[1] // -(2 ** i))) for i in range(self.num_layers)]
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

#         # build model decoder layers
#         self.decoder = nn.ModuleList()
#         for i_layer in range(self.num_layers, self.num_layers*2):
#             if i_layer == self.num_layers:
#                 dim = int(embed_dim * dim_factor ** (self.num_layers * 2 - i_layer - 1))
#             else:
#                 dim = int(embed_dim * dim_factor ** (self.num_layers * 2 - i_layer))
#             layer = BasicLayer(dim=dim, dim_factor=dim_factor, input_resolution=input_resolutions[i_layer-1], 
#                                output_resolution=input_resolutions[i_layer], depth=depths[i_layer],
#                                num_heads=num_heads[i_layer], win_size=win_size,
#                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                operation=Upsample if i_layer > self.num_layers else None, 
#                                norm_layer=norm_layer, pretrained_window_size=pretrained_window_sizes[i_layer], 
#                                sealand_attn_mask=None, use_checkpoint=use_checkpoint, version=swin_block_version)
#             self.decoder.append(layer)

#         self.norm = norm_layer(embed_dim) if norm_layer is not None else None
#         self.patch_unembed = PatchUnembed(dim=embed_dim, out_channels=out_channels, input_resolution=patches_resolution, 
#             output_resolution=(patches_resolution[0]*patch_size[0], patches_resolution[1]*patch_size[0]), upscale_factor=patch_size[0], 
#             norm_layer=norm_layer if unembed_patch_norm else None)

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
#         elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
#             trunc_normal_(m.weight, std=self.weights_std_config['conv'])
#             try: trunc_normal_(m.bias, std=self.weights_std_config['conv'])
#             except: pass
    
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

#     def forward_features(self, x: torch.Tensor):
#         # embed with patch size
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#         # forward encoder
#         for i,layer in enumerate(self.encoder):
#             x = layer(x)
#         # forward decoder
#         for i,layer in enumerate(self.decoder):
#             x = layer(x)
#         # normalize
#         if self.norm is not None:
#             x = self.norm(x)
#         # patch recovery
#         x = self.patch_unembed(x)
#         return x

#     def forward(self, x: torch.Tensor):
#         x = self.forward_features(x)
#         return x

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.encoder):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops
