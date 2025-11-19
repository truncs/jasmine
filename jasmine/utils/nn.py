import math
from typing import Tuple, Callable

from flax import nnx
import jax
import jax.numpy as jnp
import einops


def _get_spatiotemporal_positional_encoding(d_model: int, max_len: int = 5000):
    """
    Creates a function that applies separate sinusoidal positional encodings to the temporal and spatial dimensions.
    """
    pe = jnp.zeros((max_len, d_model))
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def _encode(x: jax.Array) -> jax.Array:
        """
        Args:
            x: The input tensor of shape (Batch, Time, Space, Dimension).

        Returns:
            The input tensor with positional encodings added.
        """
        assert x.ndim == 4, f"Input must be 4-dimensional, but got shape {x.shape}"

        num_timesteps = x.shape[1]
        num_spatial_patches = x.shape[2]

        # Temporal positional encoding: (1, T, 1, D)
        temporal_pe = pe[jnp.newaxis, :num_timesteps, jnp.newaxis, :]
        x = x + temporal_pe

        # Spatial positional encoding: (1, 1, S, D)
        spatial_pe = pe[jnp.newaxis, jnp.newaxis, :num_spatial_patches, :]
        x = x + spatial_pe

        return x

    return _encode


def rope(x, ts=None, inverse=False, maxlen=4096):
    B, T, _, D = x.shape
    if ts is None:
        ts = jnp.ones(B, jnp.int32)[:, None] * jnp.arange(T)[None, :]  # [B, T]
    assert ts.shape == (B, T), (ts.shape, (B, T))
    if inverse:
        ts = -ts
    freq_exponents = (2.0 / D) * jnp.arange(D // 2)  # [D/2]
    timescale = maxlen ** freq_exponents
    radians = ts[:, :, None] / timescale[None, None, :]  # [B, T, D/2]
    radians = radians[..., None, :].astype(x.dtype)  # [B, T, 1, D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)  # [B, T, H, D/2]
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res


def rope2d(x_BTHWM):
    B, T, H, W, M = x_BTHWM.shape
    x1_BTHWI, x2_BTHWI = jnp.split(x_BTHWM, 2, axis=-1)

    I = x1_BTHWI.shape[-1]
    x1_BHNI = x1_BTHWI.swapaxes(1, 2).reshape((B, H, -1, I))
    x1_BHNI = rope(x1_BHNI, maxlen=H)
    x1_BTHWI = x1_BHNI.reshape((B, H, T, W, I)).swapaxes(1, 2)

    x2_BWNI = x2_BTHWI.swapaxes(1, 3).reshape((B, W, -1, I))
    x2_BWNI = rope(x2_BWNI, maxlen=W)
    x2_BTHWI = x2_BWNI.reshape((B, W, H, T, I)).swapaxes(1, 3)
    return jnp.concat([x1_BTHWI, x2_BTHWI], axis=-1)


class AxialSpatialBlock(nnx.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        spatial_causal: bool,
        sow_weights: bool,
        decode: bool,
    ):

        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.spatial_causal = spatial_causal
        self.sow_weights = sow_weights
        self.decode = decode

        self.spatial_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.qknorm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.qkv_proj = nnx.Linear(
            in_features=self.dim,
            out_features=self.dim*3,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

        self.spatial_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.dim,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=self.spatial_causal
            ),
            rngs=rngs,
            decode=self.decode,
        )

    @nnx.remat
    def __call__(self, x_BTHWM: jax.Array) -> jax.Array:
        # --- Spatial attention ---
        B, T, H, W, M = x_BTHWM.shape
        z_BTHWM = self.spatial_norm(x_BTHWM)

        q_BTHWM, k_BTHWM, v_BTHWM = jnp.split(self.qkv_proj(z_BTHWM), 3, -1)

        q_BTNM = rope2d(self.qknorm(q_BTHWM)).reshape((B, T, -1, M))
        k_BTNM = rope2d(self.qknorm(k_BTHWM)).reshape((B, T, -1, M))
        v_BTNM = v_BTHWM.reshape((B, T, -1, M))

        z_BTNM = self.spatial_attention(
            q_BTNM, k_BTNM, v_BTNM, sow_weights=self.sow_weights)
        x_BTHWM = x_BTHWM + z_BTNM.reshape((B, T, H, W, M))

        return x_BTHWM


class AxialBlock(nnx.Module):
    """Axial transformer block"""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        spatial_causal: bool,
        temporal_causal: bool,
        sow_weights: bool,
        sow_activations: bool,
        decode: bool,
    ):
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.spatial_causal = spatial_causal
        self.temporal_causal = temporal_causal
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations
        self.decode = decode


        _blocks = []

        for _ in range(3):
            _blocks.append(
                AxialSpatialBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    use_flash_attention=self.use_flash_attention,
                    spatial_causal=self.spatial_causal,
                    sow_weights=self.sow_weights,
                    decode=self.decode,
                    rngs=rngs
                )
            )

        self.spatial_blocks = nnx.List(_blocks)

        self.temporal_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.qknorm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.qkv_proj = nnx.Linear(
            in_features=self.dim,
            out_features=self.dim*3,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

        self.temporal_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.dim,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=self.temporal_causal
            ),
            rngs=rngs,
            decode=self.decode,
        )

        self.ffn_norm = nnx.LayerNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )
        self.ffn_dense1 = nnx.Linear(
            in_features=self.dim,
            out_features=self.ffn_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.ffn_dense2 = nnx.Linear(
            in_features=self.ffn_dim,
            out_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    @nnx.remat
    def __call__(self, x_BTHWM: jax.Array) -> jax.Array:
        # --- Spatial z ---
        B, T, H, W, M = x_BTHWM.shape
        for block in self.spatial_blocks:
            x_BTHWM = block(x_BTHWM)


        # --- Temporal attention ---
        x_BTNM = x_BTHWM.reshape((B, T, -1, M))
        x_BNTM = x_BTNM.swapaxes(1, 2)

        z_BNTM = self.temporal_norm(x_BNTM)

        q_BNTM, k_BNTM, v_BNTM = jnp.split(
            self.qkv_proj(z_BNTM), 3, -1)
        q_BNTM = self.qknorm(q_BNTM)
        k_BNTM = self.qknorm(k_BNTM)

        # import ipdb; ipdb.set_trace()
        q_BNTM = rope(q_BNTM)
        k_BNTM = rope(k_BNTM)

        z_BNTM = self.temporal_attention(
            q_BNTM, k_BNTM, v_BNTM, sow_weights=self.sow_weights)
        x_BNTM = x_BNTM + z_BNTM
        x_BTNM = x_BNTM.swapaxes(1, 2)

        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTND = self.ffn_dense1(z_BTNM)
        z_BTND = jax.nn.gelu(z_BTND)
        z_BTNM = self.ffn_dense2(z_BTND)
        x_BTNM = x_BTNM + z_BTNM
        if self.sow_activations:
            self.sow(nnx.Intermediate, "activations", x_BTNM)
        return x_BTNM.reshape((B, T, H, W, -1))


class AxialTransformer(nnx.Module):
    """
    Axial transformer

    Dimension keys:
        B: batch size
        T: number of frames
        N: number of patches per frame
        I: number of input features
        M: model dimension
        D: FFN dimension
        O: output dimension
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        ffn_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        spatial_causal: bool,
        temporal_causal: bool,
        decode: bool = False,
        sow_weights: bool = False,
        sow_activations: bool = False,
        sow_logits: bool = False,
        max_len: int = 5000,
    ):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.spatial_causal = spatial_causal
        self.temporal_causal = temporal_causal
        self.sow_logits = sow_logits
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations
        self.decode = decode
        self.input_norm1 = nnx.LayerNorm(
            num_features=self.input_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )
        self.input_dense = nnx.Linear(
            in_features=self.input_dim,
            out_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.input_norm2 = nnx.LayerNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # layer norm in full precision
            rngs=rngs,
        )

        self.pos_enc = _get_spatiotemporal_positional_encoding(
            self.model_dim, max_len=max_len
        )

        _blocks = []
        for _ in range(self.num_blocks):
            _blocks.append(
                AxialBlock(
                    dim=self.model_dim,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    use_flash_attention=self.use_flash_attention,
                    rngs=rngs,
                    spatial_causal=self.spatial_causal,
                    temporal_causal=self.temporal_causal,
                    sow_weights=self.sow_weights,
                    sow_activations=self.sow_activations,
                    decode=self.decode,
                )
            )
        self.blocks = nnx.List(_blocks)

        self.output_dense = nnx.Linear(
            in_features=self.model_dim,
            out_features=self.out_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(self, x_BTHWI: jax.Array) -> jax.Array:

        x_BTNI = self.input_norm1(x_BTHWI)
        x_BTHWM = self.input_dense(x_BTHWI)
        x_BTHWM = self.input_norm2(x_BTHWM)

        B, T, H, W, M = x_BTHWM.shape
        x_BTNM = x_BTHWM.reshape((B, T, H*W, M))
        # x_BTNM = self.pos_enc(x_BTNM)

        x_BTHWM = x_BTNM.reshape((B, T, H, W, M))
        for block in self.blocks:
            x_BTHWM = block(x_BTHWM)

        x_BTHWV = self.output_dense(x_BTHWM)
        if self.sow_logits:
            self.sow(nnx.Intermediate, "logits", x_BTNV)
        return x_BTHWV.reshape((B, T, H, W, -1))


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nnx.Module):
    """
    Dimension keys:
        D: B * T * N
        K: number of latents
        L: latent dimension
    """

    def __init__(
        self,
        latent_dim: int,
        num_latents: int,
        dropout: float,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.dropout = dropout
        self.dtype = dtype

        self.codebook = nnx.Param(
            normalize(
                nnx.initializers.normal(stddev=1)(
                    rngs.params(), (self.num_latents, self.latent_dim)
                )
            )
        )
        self.drop = nnx.Dropout(self.dropout, rngs=rngs)

    def __call__(
        self, x_DL: jax.Array, training: bool
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # --- Compute distances ---
        x_DL = x_DL.astype(self.dtype)
        codebook = self.codebook.value.astype(self.dtype)

        x_DL = normalize(x_DL)
        normalized_codebook_KL = normalize(codebook)
        distance_DK = -jnp.matmul(x_DL, normalized_codebook_KL.T)
        if training:
            distance_DK = self.drop(distance_DK)

        # --- Get indices and embeddings ---
        indices_D = jnp.argmin(distance_DK, axis=-1)
        z_DL = codebook[indices_D]

        # --- Straight through estimator ---
        z_q_DL = x_DL + jax.lax.stop_gradient(z_DL - x_DL)
        return z_q_DL, z_DL, x_DL, indices_D

    def get_codes(self, indices_E: jax.Array) -> jax.Array:
        return self.codebook[indices_E]


def _create_flash_attention_fn(use_flash_attention: bool, is_causal: bool) -> Callable:
    """
    Create an attention function that uses flash attention if enabled.

    flax.nnx.MultiHeadAttention provides tensors with shape (batch..., length, num_heads, head_dim),
    but jax.nn.dot_product_attention expects (batch, length, num_heads, head_dim). We reshape to
    ensure compatibility. cuDNN's flash attention additionally requires a sequence length that
    is a multiple of 4. We pad the sequence length to the nearest multiple of 4 and mask
    accordingly. Note that cuDNN requires the mask to be broadcast before calling the attention
    function due to strict shape checking.
    """

    def attention_fn(
        query_BTHD, key_BSHD, value_BSHD, bias=None, mask_B111=None, **kwargs
    ):
        implementation = "cudnn" if use_flash_attention else None

        def _merge_batch_dims(x):
            return einops.rearrange(x, "... l h k -> (...) l h k")

        def _pad(x, pad_size):
            return jnp.pad(x, ((0, 0), (0, pad_size), (0, 0), (0, 0)))

        original_shape = query_BTHD.shape
        T = query_BTHD.shape[-3]
        S = key_BSHD.shape[-3]

        # Pad to nearest multiple of 4
        Q = ((T + 3) // 4) * 4
        pad_size_Q = Q - T
        K = ((S + 3) // 4) * 4
        pad_size_K = K - S

        query_BQHD = _pad(_merge_batch_dims(query_BTHD), pad_size_Q)
        key_BKHD = _pad(_merge_batch_dims(key_BSHD), pad_size_K)
        value_BKHD = _pad(_merge_batch_dims(value_BSHD), pad_size_K)

        attention_mask = jnp.ones((Q, K), dtype=jnp.bool_)
        attention_mask = attention_mask.at[T:, :].set(False)
        attention_mask = attention_mask.at[:, S:].set(False)

        mask_11TS = attention_mask[jnp.newaxis, jnp.newaxis, :, :]

        bias_4d = (
            jnp.pad(
                _merge_batch_dims(bias),
                ((0, 0), (0, 0), (0, pad_size_Q), (0, pad_size_K)),
            )
            if bias is not None
            else None
        )

        # NOTE: jax.nn.dot_product_attention does not support dropout
        output_4d = jax.nn.dot_product_attention(
            query=query_BQHD,
            key=key_BKHD,
            value=value_BKHD,
            bias=bias_4d,
            mask=mask_11TS,
            implementation=implementation,
            is_causal=is_causal,
        )
        return output_4d[..., :T, :, :].reshape(original_shape)

    return attention_fn
