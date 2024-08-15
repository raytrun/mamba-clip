# Based on CLIP code bases
# Modified from github.com/openai/CLIP
# --------------------------------------------------------'

from collections import OrderedDict
import numpy as np
import timm
import torch
from torch import nn
import torchvision.transforms.functional_tensor as F_t

from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import (
    default_cfgs,
    build_model_with_cfg,
    checkpoint_filter_fn,
)

from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import simba


def get_att_mask(attention, ratio=0.5):
    bs = attention.shape[0]
    masks = torch.ones((bs, 49), dtype=torch.bool, device=attention.device)
    attention = attention.reshape((-1, 14, 14))
    attention = torch.nn.functional.interpolate(
        attention.unsqueeze(1), (7, 7), mode="bilinear"
    ).squeeze()
    attention = attention.reshape(bs, -1)
    N = int(attention.shape[1] * ratio)

    reservation = torch.argsort(attention, descending=True)
    reservation = reservation[:, : N + 1]
    masks = masks.scatter_(1, reservation, False)

    full_mask = torch.zeros((bs, 14, 14), dtype=torch.bool, device=attention.device)
    full_mask[:, 0::2, 0::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 0::2, 1::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 1::2, 0::2] = masks.reshape(bs, 7, 7)
    full_mask[:, 1::2, 1::2] = masks.reshape(bs, 7, 7)
    full_mask = full_mask.reshape(bs, -1)

    return full_mask


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        vision_model_ema: nn.Module,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.visual_e = vision_model_ema

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )
        self.transformer_e = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.token_embedding_e = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding_e = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.image_projection_e = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection_e = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.ln_final_e = LayerNorm(transformer_width)

        self.initialize_parameters()
        self._update_momentum()

    @torch.no_grad()
    def _sync_ema_params(self, ema, online, m=0):
        if isinstance(ema, nn.Module) and isinstance(online, nn.Module):
            for param_m, param_n in zip(ema.parameters(), online.parameters()):
                param_m.data = param_m.data * m + param_n.data * (1.0 - m)
                param_m.requires_grad = False
        else:
            ema.data = ema.data * m + online.data * (1.0 - m)
            ema.requires_grad = False

    @torch.no_grad()
    def _update_momentum(self, m=0):
        """Momentum update of the momentum encoder"""
        self._sync_ema_params(self.visual_e, self.visual, m)
        self._sync_ema_params(self.transformer_e, self.transformer, m)

        self._sync_ema_params(self.image_projection_e, self.image_projection, m)
        self._sync_ema_params(self.text_projection_e, self.text_projection, m)

        self._sync_ema_params(self.token_embedding_e, self.token_embedding, m)
        self._sync_ema_params(self.positional_embedding_e, self.positional_embedding, m)
        self._sync_ema_params(self.ln_final_e, self.ln_final, m)

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width**-0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, ema=False):
        if not ema:
            x = self.visual(image)
            x = x @ self.image_projection
        else:
            x = self.visual_e(image)
            x = x @ self.image_projection_e

        return x

    def encode_text(self, text, ema=False):
        if not ema:
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            x = self.token_embedding_e(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding_e
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer_e(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final_e(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = (
                x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                @ self.text_projection_e
            )

        return x

    def forward(self, image, text, momentum=0):
        self._update_momentum(momentum)
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logit_scale": self.logit_scale.exp(),
        }

def CLIP_Simba_L(**kwargs):
    from timm.models import create_model

    vision_model = create_model(
        "simba_l",
        pretrained=False,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=0.3,
        drop_block_rate=None,
        token_label=False,
    )
    vision_model_ema = create_model(
        "simba_l",
        pretrained=False,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=0.3,
        drop_block_rate=None,
        token_label=False,
    )

    model = CLIP(
        embed_dim=512,
        vision_width=512,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model


def CLIP_Simba_B(**kwargs):
    from timm.models import create_model

    vision_model = create_model(
        "simba_b",
        pretrained=False,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        token_label=False,
    )
    vision_model_ema = create_model(
        "simba_b",
        pretrained=False,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        token_label=False,
    )
    model = CLIP(
        embed_dim=512,
        vision_width=512,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model


def CLIP_Simba_S(**kwargs):
    from timm.models import create_model

    vision_model = create_model(
        "simba_s",
        pretrained=False,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        token_label=False,
    )
    vision_model_ema = create_model(
        "simba_s",
        pretrained=False,
        num_classes=0,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        token_label=False,
    )
    model = CLIP(
        embed_dim=512,
        vision_width=448,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model


def CLIP_VMamba_B(**kwargs):
    from vmamba import build_model
    from vmamba.config import get_model_config

    vmamba_model_name = "vssm_base_224"
    cfg = get_model_config(model_name=vmamba_model_name)
    vision_model = build_model(cfg)
    vision_model_ema = build_model(cfg)

    model = CLIP(
        embed_dim=512,
        vision_width=1024,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model


def CLIP_VMamba_T220(**kwargs):
    from vmamba import build_model
    from vmamba.config import get_model_config

    vmamba_model_name = "vssm_tiny_224_0220"
    cfg = get_model_config(model_name=vmamba_model_name)
    vision_model = build_model(cfg)
    vision_model_ema = build_model(cfg)

    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model


def CLIP_VMamba_S(**kwargs):
    from vmamba import build_model
    from vmamba.config import get_model_config

    vmamba_model_name = "vssm_small_224"
    cfg = get_model_config(model_name=vmamba_model_name)
    vision_model = build_model(cfg)
    vision_model_ema = build_model(cfg)

    model = CLIP(
        embed_dim=512,
        vision_width=768,
        vision_model=vision_model,
        vision_model_ema=vision_model_ema,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    )
    return model
