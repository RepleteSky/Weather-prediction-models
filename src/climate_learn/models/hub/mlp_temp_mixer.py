# Standard library
import math
from functools import partial

# Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

# Third party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.mlp_mixer import PatchEmbed, DropPath, Mlp
from timm.layers import trunc_normal_


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self,
            dim,
            seq_len,
            history,
            mlp_ratio=(1.0, 0.5),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop=0.,
            drop_path=0.,
    ):
        super().__init__()
        self.tokens_dim, self.channels_dim= [int(x * dim) for x in mlp_ratio]
        self.temporals_dim = history
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, self.tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, self.channels_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp_temporals = mlp_layer(history, self.temporals_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(2, 3)).transpose(2, 3))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        x = x + self.drop_path(self.mlp_temporals(self.norm3(x).transpose(1, 3)).transpose(1, 3))
        return x

@register("mlp_temp_mixer")
class MlpTempMixer(nn.Module):

    def __init__(
            self,
            img_size,
            in_channels,
            out_channels,
            history,
            patch_size=16,
            depth=8,
            decoder_depth=2,
            embed_dim=512,
            mlp_ratio=(4.0, 0.5),
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            proj_drop_rate=0.1,
            drop_path_rate=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.history = history
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.patch_embed = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, self.in_channels, embed_dim) for i in range(history)]
        )
        self.num_patches = self.patch_embed[0].num_patches

        # Encoder
        self.blocks = nn.Sequential(*[
            MixerBlock(
                embed_dim,
                self.num_patches,
                self.history,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Reduction of time dimension
        self.blend = nn.Linear(self.num_patches*history, self.num_patches)
        # Decoder
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def blender(self, x):
        """
        x: (B, T, num_patches, embed_dim)
        return imgs: (B, num_patches, embed_dim)
        """
        if len(x.shape) == 4:  # x.shape = [B,T,num_patches,embed_dim]
            x = x.flatten(1, 2)
        x = self.blend(x.transpose(1, 2)).transpose(1, 2)
        return x

    def forward_encoder(self, x):
        embeds = []
        for i in range(self.history):
            embeds.append(self.patch_embed[i](x[:, i:i+1].squeeze(1)))
        x = torch.stack(embeds, dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        # x.shape = [B,T,in_channels,H,W]
        x = self.forward_encoder(x)
        # x.shape = [B,T,num_patches,embed_dim]
        # x = self.blender(x)
        x = x[:, 0].squeeze(1)
        # x.shape = [B,num_patches,embed_dim]
        x = self.head(x)
        # x.shape = [B,num_patches,embed_dim]
        preds = self.unpatchify(x)
        # preds.shape = [B,out_channels,H,W]
        return preds
