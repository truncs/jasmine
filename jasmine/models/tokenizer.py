from typing import Dict, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import jax

from jasmine.utils.preprocess import patchify, unpatchify
from jasmine.utils.nn import AxialTransformer, VectorQuantizer


class TokenizerVQVAE(nnx.Module):
    """
    ST-ViVit VQ-VAE

    Dimension keys:
        B: batch size
        T: sequence length
        N: number of patches per frame
        L: latent dimension
        D: B * T * N
        H: height
        W: width
        C: number of channels
        P: patch token dimension (patch_size^2 * C)
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        ffn_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        codebook_dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
    ):
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.codebook_dropout = codebook_dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.encoder = AxialTransformer(
            self.in_dim * self.patch_size**2,
            self.model_dim,
            self.ffn_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            rngs=rngs,
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
            self.dtype,
            rngs=rngs,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder = AxialTransformer(
            self.latent_dim,
            self.model_dim,
            self.ffn_dim,
            self.out_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            rngs=rngs,
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        H, W = batch["videos"].shape[2:4]
        videos_BTHWC = batch["videos"]
        outputs = self.vq_encode(videos_BTHWC, training)
        z_q_BTNL = outputs["z_q"]
        recon_BTHWC = self.decoder(z_q_BTNL)
        recon_BTHWC = recon_BTHWC.astype(jnp.float32)
        recon_BTHWC = nnx.sigmoid(recon_BTHWC)
        recon_BTHWC = recon_BTHWC.astype(self.dtype)
        recon_BTHWC = unpatchify(recon_BTHWC, self.patch_size, H, W)
        outputs["recon"] = recon_BTHWC
        return outputs

    def vq_encode(
        self, videos: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess + encode ---
        B, T = videos.shape[:2]
        patch_BTNP = patchify(videos, self.patch_size)
        N = patch_BTNP.shape[2]
        x_BTNL = self.encoder(patch_BTNP)

        # --- Vector quantize ---
        x_DL = x_BTNL.reshape(B * T * N, self.latent_dim)
        z_q_DL, z_DL, emb_DL, indices_D = self.vq(x_DL, training)
        z_q_BTNL = z_q_DL.reshape(B, T, N, self.latent_dim)
        indices_BTN = indices_D.reshape(B, T, N)
        return dict(z_q=z_q_BTNL, z=z_DL, emb=emb_DL, indices=indices_BTN)

    def decode(self, indices_BTN: jax.Array, video_hw: Tuple[int, int]) -> jax.Array:
        z_BTNL = self.vq.codebook[indices_BTN]
        recon_BTNP = self.decoder(z_BTNL)
        recon_BTNP = recon_BTNP.astype(jnp.float32)
        recon_BTNP = nnx.sigmoid(recon_BTNP)
        recon_BTNP = recon_BTNP.astype(self.dtype)
        return unpatchify(recon_BTNP, self.patch_size, *video_hw)


class TokenizerMAE(nnx.Module):
    """
    ST-ViVit MAE.

    Dimension keys:
        B: batch size
        T: sequence length
        N: number of patches per frame
        L: latent dimension
        D: B * T * N
        H: height
        W: width
        C: number of channels
        P: patch token dimension (patch_size^2 * C)
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        in_dim: int,
        model_dim: int,
        ffn_dim: int,
        latent_dim: int,
        num_latents: int,
        patch_size: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_mask_ratio: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
    ):
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_mask_ratio = max_mask_ratio
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.encoder = AxialTransformer(
            self.in_dim * self.patch_size**2,
            self.model_dim,
            self.ffn_dim,
            self.latent_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            rngs=rngs,
        )

        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder_num_latents = image_height // self.patch_size * image_width // self.patch_size
        
        self.decoder = AxialTransformer(
            self.latent_dim,
            self.model_dim,
            self.ffn_dim,
            self.out_dim,
            self.decoder_num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            rngs=rngs,
        )
        self.mask_patch = nnx.Param(
            nnx.initializers.lecun_uniform()(
                rngs.params(), (1, 1, 1, self.in_dim * self.patch_size**2)
            )
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        H, W = batch["videos"].shape[2:4]
        videos_BTHWC = batch["videos"]
        outputs = self.mask_and_encode(videos_BTHWC, batch["rng"], training)
        z_BTML = outputs["z"]

        _, recon_BTNC = self.decoder(z_BTML)
        B, T = recon_BTNC.shape[:2]
        recon_BTHWC = recon_BTNC.reshape(B, T, H//self.patch_size, W//self.patch_size, -1)
        recon_BTHWC = recon_BTHWC.astype(jnp.float32)
        recon_BTHWC = nnx.sigmoid(recon_BTHWC)
        recon_BTHWC = recon_BTHWC.astype(self.dtype)

        B, T, Hp, Wp, C = recon_BTHWC.shape
        recon_BTNC = recon_BTHWC.reshape((B, T, Hp*Wp, C))
        recon_BTHWC = unpatchify(recon_BTNC, self.patch_size, H, W)
        outputs["recon"] = recon_BTHWC
        return outputs

    def mask_and_encode(
        self, videos: jax.Array, rng: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess videos ---
        B, T = videos.shape[:2]
        patch_BTHWP = patchify(videos, self.patch_size)
        H, W = patch_BTHWP.shape[2:4]
        N = H*W

        patch_BTNP = patch_BTHWP.reshape((B, T, N, -1))

        # --- Randomly mask patches ---
        if training:
            _rng_prob, _rng_mask = jax.random.split(rng, 2)
            mask_prob = jax.random.uniform(
                _rng_prob, shape=(B * T,), minval=0, maxval=self.max_mask_ratio
            )
            mask = jax.vmap(
                lambda rng, prob: jax.random.bernoulli(rng, prob, (N,)),
                in_axes=(0, 0),
            )(jax.random.split(_rng_mask, B * T), mask_prob)
            mask_BTN = mask.reshape(B, T, N)
            patch_BTNP = jnp.where(
                mask_BTN[..., jnp.newaxis], self.mask_patch.value, patch_BTNP
            )
        else:
            mask_BTN = jnp.zeros((B, T, N), dtype=jnp.bool_)
        patch_BTHWP = patch_BTNP.reshape((B, T, H, W, -1))
        # --- Encode ---

        _, z_BTML = self.encoder(patch_BTHWP)
        # squeeze latents through tanh as described in Dreamer 4 section 3.1
        # z_BTHWL = nnx.tanh(z_BTHWL)
        z_BTML = nnx.tanh(z_BTML)
        outputs = dict(
            z=z_BTML,
            mask=mask_BTN,
            )
        return outputs

    def decode(
            self, z_BTNL: jax.Array, video_hw: Tuple[int, int]) -> jax.Array:
        recon_BTNP = self.decoder(z_BTNL)
        recon_BTNP = recon_BTNP.astype(jnp.float32)
        recon_BTNP = nnx.sigmoid(recon_BTNP)
        recon_BTNP = recon_BTNP.astype(self.dtype)
        return unpatchify(recon_BTNP, self.patch_size, *video_hw)
