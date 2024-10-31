
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel


def cogvideo_attn_forward(self, attn, hidden_states, encoder_hidden_states, attention_mask = None, image_rotary_emb = None, **kwargs):
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
)

def patch_cogvideo_attn(model):
    assert type(model.__class__) == CogVideoXTransformer3DModel
    for layer_id, block in enumerate(model.transformer_blocks):
        block.layer_id = layer_id
        block.attn1.layer_id = layer_id


    def _forward(self, attn, hidden_states, encoder_hidden_states, attention_mask = None, image_rotary_emb = None, **kwargs):


    