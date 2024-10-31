import torch
import json
import os
import diffusers
from diffusers import CogVideoXPipeline

import torch
import torch.nn.functional as F
from torch import nn

# from models.attention_processor import Attention as Attention_
# diffusers.models.attention_processor.Attention = Attention_
# from models.cogvideox_transformer_3d import CogVideoXTransformer3DModel
# diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel = CogVideoXTransformer3DModelNew
# import accelerate

LAYER_ID = 0
def attention_forward(self, attn, hidden_states, encoder_hidden_states, attention_mask = None, image_rotary_emb = None, **kwargs):
    text_seq_length = encoder_hidden_states.size(1)

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
        if not attn.is_cross_attention:
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

    import torchvision
    # layer_id = kwargs['layer_id']
    global LAYER_ID
    layer_id = LAYER_ID
    LAYER_ID = LAYER_ID + 1
    if LAYER_ID == 42:
        LAYER_ID = 0
    print(f"layer_id: {layer_id}, q.shape: {query.shape}, k.shape: {key.shape}, v.shape: {value.shape}")
    attn_mat = (query[0].float().cpu() @ key[0].float().cpu().transpose(-1,-2) / query.shape[-1]**0.5).softmax(-1)
    size = 128
    attn_scaled = torchvision.transforms.Resize((size, size), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(attn_mat)
    if not os.path.exists("./metadata"):
        os.makedirs("./metadata")
    if attn_scaled.ndim != 4:
        attn_scaled = attn_scaled.unsqueeze(0)
    fname = f"./metadata/layer{layer_id}_attn_{size}_cond.pth"
    if os.path.exists(fname):
        res = torch.load(fname, map_location='cpu')
        res = torch.cat((res, attn_scaled), dim=0)
        torch.save(res, fname)
    else:
        torch.save(attn_scaled, fname)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return hidden_states, encoder_hidden_states


diffusers.models.attention_processor.CogVideoXAttnProcessor2_0.__call__ = attention_forward

prompt = "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

# pipe.transformer = None
# torch.cuda.empty_cache()

# config_path = "/root/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad/transformer/config.json"
# config = json.load(open(config_path, "r"))
# with accelerate.init_empty_weights():
# pipe.transformer = CogVideoXTransformer3DModel.from_config(config)

# cnt = 0
# for n_, v_ in pipe.transformer.named_parameters():
#     print(f"{n_:<100} : {v_.shape}")
#     cnt += v_.numel()
# print(cnt)

# from safetensors import safe_open
# ckpt_path = "/root/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad/transformer/"
# tensors = {}
# for i in [1,2]:
#     with safe_open(os.path.join(ckpt_path, f"diffusion_pytorch_model-0000{i}-of-00002.safetensors"), framework="pt", device=0) as f:
#         for k in f.keys():
#             tensors[k] = f.get_tensor(k)

# cnt2 = 0
# for n2, v2 in tensors.items():
#     cnt2 += v2.numel()
# print(cnt2)

# pipe.transformer = accelerate.load_checkpoint_and_dispatch(pipe.transformer, ckpt_path)
# pipe.transformer = pipe.transformer.to(torch.bfloat16)
# pipe.transformer.load_state_dict(tensors, strict=False)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

import pdb; pdb.set_trace();