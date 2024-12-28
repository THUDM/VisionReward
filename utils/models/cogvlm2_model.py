"""
Here we modify clip_llama.py model to adapt the format of base_model.py
Specifically,
* support vision_ args
"""

from sat.model.official.llama_model import LLaMAModel
import json
import torch
import torch.nn.functional as F
from sat.model.base_model import BaseMixin
import math
import torch.nn as nn
from .mixins.visual_expert_llama3 import LlamaVisionExpertFCMixin, LlamaVisionExpertAttnMixin
from sat import mpu
import torch.nn.init as init
# from vision.openclip_vemb import Pix2StructOpenClipVembPos
# from timm.models.vision_transformer import PatchEmbed
# from vision.eva2_e import Eva2Encoder

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class GLU(nn.Module):
    def __init__(self, args, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, args.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(args.inner_hidden_size, args.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x

from .eva_clip_model import EVA2CLIPModel
import argparse
from copy import deepcopy
def override_dist_dtype_device_args(args, b={}):
    if args.mode == 'inference':
        minimal_args = argparse.Namespace(
            world_size=args.world_size,
            rank=args.rank,
            local_rank=args.local_rank,
            skip_init=args.skip_init,
            use_gpu_initialization=args.use_gpu_initialization,
            deepspeed=args.deepspeed,
            bf16=args.bf16,
            fp16=args.fp16,
            mode=args.mode,
            device=args.device
        )
    else:
        minimal_args = argparse.Namespace(
                world_size=args.world_size,
                rank=args.rank,
                local_rank=args.local_rank,
                skip_init=args.skip_init,
                use_gpu_initialization=args.use_gpu_initialization,
                deepspeed=args.deepspeed,
                bf16=args.bf16,
                fp16=args.fp16,
                mode=args.mode,
                checkpoint_activations=args.checkpoint_activations if not hasattr(args, 'vit_checkpoint_activations') else args.vit_checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                device=args.device,
                hidden_dropout=0.,
                attention_dropout=0.,
            )
    if hasattr(args, 'model_parallel_size'):
        b['model_parallel_size'] = args.model_parallel_size
    return argparse.Namespace(**deepcopy(b), **vars(minimal_args))

class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        vit_args = override_dist_dtype_device_args(args, args.eva_args)
        self.vit_model = EVA2CLIPModel(EVA2CLIPModel.get_args(**vars(vit_args)))
        self.in_features = 1792
        self.out_features = 1792
        self.conv = nn.Conv2d(in_channels=self.in_features, out_channels=self.out_features, kernel_size=2, stride=2)
        self.linear_proj = GLU(args, self.out_features)
        self.linear_proj.apply(init_weights)
        self.image_length = args.image_length
        self.boi = nn.Parameter(torch.ones(1, 1, args.hidden_size).float())
        self.eoi = nn.Parameter(torch.ones(1, 1, args.hidden_size).float())
        init.xavier_uniform_(self.boi)
        init.xavier_uniform_(self.eoi)

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        vision_inputs = {}
        for k in kw_args:
            if k.startswith('vision_') and k != 'vision_expert_mask':
                vision_inputs[k[7:]] = kw_args[k]
        if input_ids.shape[1] == 1 or not vision_inputs:
            # pre_image tokens are longer than the input_ids, or no vision inputs
            return self.transformer.word_embeddings(input_ids)
        image_emb = self.vit_model(**vision_inputs)[0]
        # print(f"image_emb:{image_emb.shape}")
        b, s, e = image_emb.shape
        grid_size = int(s**0.5)
        image_emb = image_emb.view(b, grid_size, grid_size, e).permute(0,3,1,2)
        image_emb = self.conv(image_emb)
        image_emb = image_emb.flatten(2).transpose(1, 2)

        image_emb = self.linear_proj(image_emb)
        # print(f"kw_args:{kw_args}")
        image_embed_mask = kw_args['image_embed_mask']
        # print(f"image_embed_mask:{image_embed_mask.shape}")
        # print(f"input_ids:{input_ids.shape}")
        word_embedding = self.transformer.word_embeddings(input_ids).clone()
        # # Yu Huang implement batch
        # batch_size = word_embedding.shape[0]  # 假设批次大小为word_embedding的第0维度
        # image_embed_mask = image_embed_mask.repeat(batch_size, 1)
        # # Yu Huang
        # print(f"word_embedding:{word_embedding.shape}")
        word_embedding[image_embed_mask.bool()] = torch.cat([self.boi.repeat(len(image_emb), 1, 1), image_emb, self.eoi.repeat(len(image_emb), 1, 1)], dim=1).reshape(-1, image_emb.shape[-1])
        return word_embedding.contiguous()


class VisualLlamaEVA(LLaMAModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))
        self.del_mixin("mlp")
        self.add_mixin("mlp", LlamaVisionExpertFCMixin(args.hidden_size, args.inner_hidden_size, args.num_layers, args.num_layers, frequency=1))
        self.del_mixin("rotary")
        self.add_mixin("rotary", LlamaVisionExpertAttnMixin(args.hidden_size, args.num_attention_heads, args.num_layers, args.num_layers, num_multi_query_heads=args.num_multi_query_heads, frequency=1))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VisualGLM', 'VisualGLM Configurations')
        group.add_argument('--image_length', type=int, default=256)
        group.add_argument('--image_size', type=int, default=224)
        group.add_argument('--eva_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)

    def forward(self, input_ids, **kwargs):
        if input_ids.shape[1] > 1 and 'vision_expert_mask' in kwargs and 'image_embed_mask' in kwargs:
            return super().forward(input_ids=input_ids, **kwargs)
        if "vision_expert_mask" in kwargs:
            kwargs.pop("vision_expert_mask")
        if "image_embed_mask" in kwargs:
            kwargs.pop("image_embed_mask")
        return super().forward(input_ids=input_ids, **kwargs)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 14
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x