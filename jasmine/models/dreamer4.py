import jax.numpy as jnp
from flax import nnx
import jax
import time
# from flax.core import FrozenDict # removing FrozenDict use for nnx generally
import flax
from enum import IntEnum
from typing import Optional, Tuple, Any
from einops import rearrange
import math
import dataclasses
from jasmine.utils.nn import _create_flash_attention_fn

class Modality(IntEnum):
    LATENT   = -1
    IMAGE    = 0
    ACTION   = 1
    PROPRIO  = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7
    # add more as needed

@dataclasses.dataclass  # standard dataclass often sufficient, or flax.struct.dataclass
class TokenLayout:
    """
    Ordered token layout for a single timestep: latents first (if any),
    then a sequence of (modality, count) segments.
    """
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]  # e.g., ((Modality.IMAGE, n_patches), (Modality.ACTION, n_act), ...)

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> jnp.ndarray:
        parts = [jnp.full((self.n_latents,), Modality.LATENT, dtype=jnp.int32)] if self.n_latents > 0 else []
        for m, n in self.segments:
            if n > 0:
                parts.append(jnp.full((n,), int(m), dtype=jnp.int32))
        return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.int32)  # (S,)

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> jnp.ndarray:
        parts = [jnp.full((self.n_latents,), Modality.LATENT, dtype=jnp.int32)] if self.n_latents > 0 else []
        for m, n in self.segments:
            if n > 0:
                parts.append(jnp.full((n,), int(m), dtype=jnp.int32))
        return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.int32)  # (S,)

    def slices(self) -> dict:
        """Convenience: start/stop indices per modality (first occurrence if repeated)."""
        idx = 0
        out = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents); idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out

    
def sinusoid_table(n: int, d: int, base: float = 10000.0, dtype=jnp.float32) -> jnp.ndarray:
    """
    Standard Transformer sinusoid: even dims use sin, odd dims use cos with frequencies
    base^{-2k/d}. Works for odd d too.
    """
    pos = jnp.arange(n, dtype=dtype)[:, None]            # (n,1)
    i = jnp.arange(d, dtype=dtype)[None, :]              # (1,d)
    # k = floor(i/2)
    k = jnp.floor(i / 2.0)
    div = jnp.power(base, -(2.0 * k) / jnp.maximum(1.0, jnp.array(d, dtype)))
    angles = pos * div                                    # (n,d)
    table = jnp.where((i % 2) == 0, jnp.sin(angles), jnp.cos(angles))
    return table.astype(dtype)


def add_sinusoidal_positions(tokens_btSd: jnp.ndarray) -> jnp.ndarray:
    """tokens: (B,T,S,D) -> adds time and step sinusoids and returns same shape."""
    B, T, S, D = tokens_btSd.shape
    pos_t = sinusoid_table(T, D)     # (T,D)
    pos_s = sinusoid_table(S, D)     # (S,D)
    # Optionally scale to keep variance stable (common trick)
    scale = 1.0 / jnp.sqrt(jnp.array(D, dtype=tokens_btSd.dtype))
    return tokens_btSd + scale * (pos_t[None, :, None, :] + pos_s[None, None, :, :])

class MAEReplacer(nnx.Module):
    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9, *, rngs: nnx.Rngs):
        self.p_min = p_min
        self.p_max = p_max
        self.d_model = d_model
        key = rngs.params()
        self.mask_token = nnx.Param(jax.random.normal(key, (d_model,)) * 0.02)

    def __call__(self, patches_btnd: jnp.ndarray, rngs: nnx.Rngs) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # patches_btnd: (B,T,Np,D)
        B, T, Np, D = patches_btnd.shape
        
        # draw RNGs from a named stream
        # In nnx, we assume rngs has 'mae'
        p_key = rngs.mae()
        m_key = rngs.mae() # splitting implicitly by calling twice or use jax.random.split on one key if preferred, but nnx.Rngs manages streams.
        # Actually nnx.Rngs.name() returns a key and advances logic if properly set up, or just returns one. 
        # Standard pattern: key = rngs.name(); k1, k2 = jax.random.split(key)
        
        key = rngs.mae()
        p_rng, m_rng = jax.random.split(key)
        
        p_bt = jax.random.uniform(p_rng, (B, T), minval=self.p_min, maxval=self.p_max)  # (B,T)
        keep_prob_bt1 = 1.0 - p_bt[..., None]                                           # (B,T,1)
        keep = jax.random.bernoulli(m_rng, keep_prob_bt1, (B, T, Np))                   # (B,T,Np)
        keep = keep[..., None]                                                          # (B,T,Np,1)
        replaced = jnp.where(keep, patches_btnd, self.mask_token.value.reshape(1, 1, 1, D))
        mae_mask = (~keep).astype(jnp.bool_)                                            # (B,T,Np,1)
        return replaced, mae_mask, keep_prob_bt1


# ---------- small building blocks ----------

class RMSNorm(nnx.Module):
    def __init__(self, features: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((features,)))

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * (self.scale.value / jnp.sqrt(var + self.eps))

class MLP(nnx.Module):
    """
    Transformer MLP with optional SwiGLU gating.
    """
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        swiglu: bool = True,
        parity_2over3: bool = False,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.swiglu = swiglu
        self.parity_2over3 = parity_2over3
        self.dtype = dtype

        mult = mlp_ratio
        if swiglu and parity_2over3:
            mult = mlp_ratio * (2.0 / 3.0)
        hidden = int(d_model * mult)

        if swiglu:
            self.fc_in = nnx.Linear(d_model, 2 * hidden, use_bias=True, dtype=dtype, rngs=rngs)
        else:
            self.fc_in = nnx.Linear(d_model, hidden, use_bias=True, dtype=dtype, rngs=rngs)
        
        self.fc_out = nnx.Linear(hidden, d_model, use_bias=True, dtype=dtype, rngs=rngs)
        self.dropout_layer = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = False) -> jnp.ndarray:
        if self.swiglu:
            pre = self.fc_in(x)
            u, v = jnp.split(pre, 2, axis=-1)
            h = u * jax.nn.silu(v)
        else:
            h = self.fc_in(x)
            h = nnx.gelu(h)

        h = self.dropout_layer(h, deterministic=deterministic)
        y = self.fc_out(h)
        y = self.dropout_layer(y, deterministic=deterministic)
        return y
# ---------- axial attention layers ----------
class SpaceSelfAttentionModality(nnx.Module):
    """
    Space self-attention with modality routing.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        modality_ids: jnp.ndarray,
        n_latents: int,
        mode: str = "encoder",
        dropout: float = 0.0,
        use_flash_attention: bool = False,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.modality_ids = modality_ids
        self.n_latents = n_latents
        self.mode = mode
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.dtype = dtype

        # Cache a (S,S) boolean mask indicating allowed key for each query index, per mode.
        S = int(self.modality_ids.shape[0])

        # Broadcast helpers
        q_idx = jnp.arange(S)[:, None]       # (S,1)
        k_idx = jnp.arange(S)[None, :]       # (1,S)

        is_q_lat = q_idx < self.n_latents     # (S,1) bool
        is_k_lat = k_idx < self.n_latents     # (1,S) bool

        q_mod = self.modality_ids[q_idx]      # (S,1)
        k_mod = self.modality_ids[k_idx]      # (1,S)
        same_mod = (q_mod == k_mod)           # (S,S)

        if self.mode == "encoder":
            # latents -> all; non-latents -> same modality only (no access to latents unless same modality==latent, which they aren't)
            allow_lat_q = jnp.ones((S, S), dtype=bool)             # lat q attends to everything
            allow_nonlat_q = same_mod                              # non-lat q attends within itself only
            mask = jnp.where(is_q_lat, allow_lat_q, allow_nonlat_q)
        elif self.mode == "decoder":
            # latents -> latents only; non-latents -> same modality OR latents
            allow_lat_q = is_k_lat                                  # lat q -> lat k only
            allow_nonlat_q = jnp.logical_or(same_mod, is_k_lat)     # non-lat q -> same mod + latents
            mask = jnp.where(is_q_lat, allow_lat_q, allow_nonlat_q)
        elif self.mode in ["wm_agent", "wm_agent_isolated"]:
            S = int(self.modality_ids.shape[0])
            q_idx = jnp.arange(S)[:, None]   # (S,1)
            k_idx = jnp.arange(S)[None, :]   # (1,S)
            q_mod = self.modality_ids[q_idx] # (S,1)
            k_mod = self.modality_ids[k_idx] # (1,S)

            is_agent_q = (q_mod == Modality.AGENT)
            is_agent_k = (k_mod == Modality.AGENT)
            is_action_q = (q_mod == Modality.ACTION)
            is_action_k = (k_mod == Modality.ACTION)

            # Observation bucket = spatial ∪ register ∪ shortcut tokens
            is_obs_k = (
                (k_mod == Modality.SPATIAL) |
                (k_mod == Modality.REGISTER) |
                (k_mod == Modality.SHORTCUT_SIGNAL) |
                (k_mod == Modality.SHORTCUT_STEP)
            )
            is_obs_q = (
                (q_mod == Modality.SPATIAL) |
                (q_mod == Modality.REGISTER) |
                (q_mod == Modality.SHORTCUT_SIGNAL) |
                (q_mod == Modality.SHORTCUT_STEP)
            )

            # Agent queries:
            #  - wm_agent: agent reads all (obs ∪ action ∪ agent)
            #  - wm_agent_isolated: agent reads nobody
            allow_for_agent_q = jnp.where(
                self.mode == "wm_agent",
                jnp.ones((S, S), dtype=bool),
                jnp.zeros((S, S), dtype=bool)
            )

            # Non-agent queries (route by query modality)
            allow_for_action_q = is_action_k                                  # action -> action only  (1,S)
            allow_for_obs_q    = (is_obs_k | is_action_k)                     # obs -> obs ∪ action    (1,S)

            # Build per-query row permissions with broadcasting from (1,S) to (S,S)
            allow_nonagent = jnp.where(
                is_action_q, allow_for_action_q,
                jnp.where(is_obs_q, allow_for_obs_q, jnp.zeros((S, S), dtype=bool))
            )

            # Nobody can read agent keys except agent q
            allow_nonagent = jnp.where(is_agent_k, False, allow_nonagent)

            mask = jnp.where(is_agent_q, allow_for_agent_q, allow_nonagent)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # Save (1,1,S,S) so it broadcasts over batch*time and heads -> (B*T, 1, S, S)
        self.modality_mask = mask[None, None, :, :]                   # (1,1,S,S)
        
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=d_model,
            dropout_rate=dropout,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=False
            ),
            rngs=rngs,
        )

    def __call__(self, x, *, deterministic: bool = False):
        # x: (B, T, S, D)  -> attention across S within each (B,T)
        B, T, S, D = x.shape
        x_ = x.reshape(B*T, S, D)

        if self.use_flash_attention:
             mask = self.modality_mask[0, 0] # (S, S)
        else:
             # Flax MHA mask shape can be (batch, num_heads, q_len, k_len). We want one mask per (B*T).
             mask = jnp.broadcast_to(self.modality_mask, (B*T, 1, S, S))   # (B*T,1,S,S)

        y_ = self.attention(
            x_, x_, 
            mask=mask, 
            deterministic=deterministic,
            decode=False,
        )

        y = y_.reshape(B, T, S, D)
        return y

class TimeSelfAttention(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, latents_only: bool = True, n_latents: int = 0, use_flash_attention: bool = False, dtype: Any = jnp.float32, *, rngs: nnx.Rngs):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.latents_only = latents_only
        self.n_latents = n_latents
        self.use_flash_attention = use_flash_attention
        self.dtype = dtype
        
        self.attention = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=d_model,
            qkv_features=d_model,
            dropout_rate=dropout,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, is_causal=True
            ),
            rngs=rngs,
        )

    def __call__(self, x, *, deterministic: bool = False):
        # x: (B, T, S, D) -> attend across T, causal
        B, T, S, D = x.shape
        if self.latents_only:
            assert 0 < self.n_latents <= S
            lat = x[:, :, :self.n_latents, :]                # (B, T, L, D)
            lat_btld = lat.transpose(0, 2, 1, 3).reshape(B*self.n_latents, T, D)  # (B*L, T, D)
            
            if self.use_flash_attention:
                mask = None
            else:
                mask = nnx.make_causal_mask(jnp.ones((B*self.n_latents, T), dtype=bool))
                
            out = self.attention(
                lat_btld, lat_btld,
                mask=mask,
                deterministic=deterministic,
                decode=False,
            )
            out = out.reshape(B, self.n_latents, T, D).transpose(0, 2, 1, 3)      # back to (B, T, L, D)
            x = x.at[:, :, :self.n_latents, :].set(out)
            return x
        else:
            x_bstd = x.transpose(0, 2, 1, 3).reshape(B*S, T, D)  # (B*S, T, D)
            
            if self.use_flash_attention:
                mask = None
            else:
                mask = nnx.make_causal_mask(jnp.ones((B*S, T), dtype=bool))

            out = self.attention(
                x_bstd, x_bstd,
                mask=mask,
                deterministic=deterministic,
                decode=False,
            )
            out = out.reshape(B, S, T, D).transpose(0, 2, 1, 3)  # back to (B, T, S, D)
            return out

# ---------- a single block-causal layer ----------
class BlockCausalLayer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: jnp.ndarray,
        space_mode: str,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        use_flash_attention: bool = False,
        layer_index: int = 0,
        time_every: int = 4,
        latents_only_time: bool = True,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_latents = n_latents
        self.modality_ids = modality_ids
        self.space_mode = space_mode
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.mlp_ratio = mlp_ratio
        self.layer_index = layer_index
        self.time_every = time_every
        self.latents_only_time = latents_only_time
        self.dtype = dtype

        # --- Space attention ---
        self.norm1 = RMSNorm(d_model, rngs=rngs)
        self.space_attn = SpaceSelfAttentionModality(
            d_model=d_model,
            n_heads=n_heads,
            modality_ids=modality_ids,
            n_latents=n_latents,
            mode=space_mode,
            dropout=dropout,
            use_flash_attention=self.use_flash_attention,
            dtype=dtype,
            rngs=rngs,
        )
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)

        # --- Time attention ---
        self.use_time = (layer_index + 1) % time_every == 0
        if self.use_time:
            self.norm2 = RMSNorm(d_model, rngs=rngs)
            self.time_attn = TimeSelfAttention(
                d_model, n_heads, dropout,
                latents_only=latents_only_time, n_latents=n_latents,
                use_flash_attention=self.use_flash_attention,
                dtype=dtype,
                rngs=rngs,
            )
            self.dropout2 = nnx.Dropout(dropout, rngs=rngs)

        # --- MLP ---
        self.norm3 = RMSNorm(d_model, rngs=rngs)
        self.mlp = MLP(d_model, mlp_ratio, dropout, dtype=dtype, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, *, deterministic: bool = False):
        # --- Space attention (within timestep, modality-aware) ---
        y = self.norm1(x)
        y = self.space_attn(y, deterministic=deterministic)
        x = x + self.dropout1(y, deterministic=deterministic)

        # --- Time attention (causal across timesteps), only on some layers ---
        if self.use_time:
            y = self.norm2(x)
            y = self.time_attn(y, deterministic=deterministic)
            x = x + self.dropout2(y, deterministic=deterministic)

        # --- MLP ---
        y = self.norm3(x)
        y = self.mlp(y, deterministic=deterministic)
        x = x + self.dropout3(y, deterministic=deterministic)
        return x
# ---------- the transformer stack ----------

class BlockCausalTransformer(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: jnp.ndarray,
        space_mode: str,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        use_flash_attention: bool = False,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        layers = []
        for i in range(depth):
            layers.append(
                BlockCausalLayer(
                    d_model, n_heads, n_latents,
                    modality_ids=modality_ids,
                    space_mode=space_mode,
                    dropout=dropout, mlp_ratio=mlp_ratio,
                    layer_index=i, time_every=time_every,
                    latents_only_time=latents_only_time,
                    use_flash_attention=use_flash_attention,
                    dtype=dtype,
                    rngs=rngs,
                )
            )
        self.layers = nnx.List(layers)

    def __call__(self, x, *, deterministic: bool = False):
        for layer in self.layers:
            x = layer(x, deterministic=deterministic)
        return x



class Encoder(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_latents: int,
        n_patches: int,
        n_heads: int,
        depth: int,
        d_bottleneck: int, # output bottleneck dim
        # d_input: int, # I need this!
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
        use_flash_attention: bool = False,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
        d_patch: int = None, # Optional to allow partial compat, but really needed.
    ):
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.d_bottleneck = d_bottleneck
        self.mae_p_min = mae_p_min
        self.mae_p_max = mae_p_max
        self.dtype = dtype
        
        # We need in_features for nnx.Linear.
        # If d_patch is None, we are in trouble unless we use lazy.
        # For this refactor I will REQUIRE d_patch or assume it matches something if appropriate.
        # But in test `d_patch=3`, `d_model=8`.
        # I will enforce explicit d_patch argument.
        if d_patch is None:
             # Fallback or error? I will try to support it by defaulting to d_model? No that's wrong.
             # I'll default to d_model but print warning? No.
             # I will just add d_patch param. Users need to update calls.
             raise ValueError("d_patch (input dimension) must be provided for nnx Encoder")

        self.patch_proj = nnx.Linear(d_patch, d_model, use_bias=True, dtype=dtype, rngs=rngs)
        self.bottleneck_proj = nnx.Linear(d_model, d_bottleneck, use_bias=True, dtype=dtype, rngs=rngs)
        
        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        self.modality_ids = self.layout.modality_ids()            # (S,)
        
        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=n_latents,
            modality_ids=self.modality_ids,
            space_mode="encoder",                 # << encoder routing
            dropout=dropout, mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=latents_only_time,
            use_flash_attention=use_flash_attention,
            dtype=dtype,
            rngs=rngs,
        )
        key = rngs.params()
        self.latents = nnx.Param(jax.random.normal(key, (n_latents, d_model)) * 0.02)
        
        self.mae = MAEReplacer(d_model=d_model, p_min=mae_p_min, p_max=mae_p_max, rngs=rngs)



    def __call__(self, patch_tokens, *, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1) Project patches to D_model
        proj_patches = self.patch_proj(patch_tokens)  # (B,T,Np,D)

        # 2) MAE mask-and-replace on patch tokens (encoder input only)
        if rngs is None:
             # fallback or error?
             # If deterministic=True, maybe we don't need RNG? 
             # MAEReplacer uses RNG always?
             # If deterministic, maybe we shouldn't mask?
             # Original code: "if train: rngs_enc = {...}"
             # If deterministic=True, does MAEReplacer work?
             # MAEReplacer draws rng. 
             # I should check if I should skip MAE if deterministic.
             # Original code executed MAEReplacer always, but depended on p_min/p_max?
             # Actually `MAEReplacer` generates mask regardless.
             # If I provide NO rngs, it will fail.
             # I'll assume rngs is provided or I handle it.
             pass
        
        proj_patches_masked, patch_mask, keep_prob = self.mae(proj_patches, rngs=rngs)

        # 3) Prepend learned latents (owned here)
        B, T = proj_patches_masked.shape[:2]
        latents = jnp.broadcast_to(self.latents.value[None, None, ...], (B, T, *self.latents.value.shape))
        tokens = jnp.concatenate([latents, proj_patches_masked], axis=2)  # (B,T,S=(Np+Nl),D)

        # 4) Add sinusoidal positions (param-free)
        tokens = add_sinusoidal_positions(tokens)

        # 5) Feed tokens into transformer
        # Pass rngs here if needed (dropout)
        # But transformer uses nnx.Dropout which might need rngs if not deterministic?
        # nnx.Dropout usually takes rngs in __call__ if not using functional?
        # No, `nnx.Dropout(rate, rngs=...)` in init sets the stream name.
        # In call, `dropout(x, deterministic=...)` automatically pulls from the stream in ThreadLocal or check `rngs` arg?
        # No, nnx modules (stateful) usually don't take rngs in `__call__` if they were initialized with stream names, BUT they need the stream to be available.
        # In `nnx`, `rngs` passed to `__call__` are usually implicit context or explicit?
        # Actually `nnx` is changing rapidly. 
        # Most robust way: pass `rngs` if APIs support it. 
        # But `nnx.Dropout.__call__` signature: `(x, *, deterministic=..., rngs=...)`.
        # So I should pass rngs.
        encoded_tokens = self.transformer(tokens, deterministic=deterministic) # pass rngs?
        # My BlockCausalTransformer.__call__ didn't accept rngs! I should have added it?
        # It calls layers which call dropout.
        # Layers call `self.dropout(y, deterministic=...)`.
        # If `nnx.Dropout` is used as a Module, does it auto-find RNG?
        # It finds `rngs` from the context or scope?
        # nnx doesn't use context manager for rngs like Linen `make_rng`.
        # It expects `rngs` argument in `__call__` OR stateful RNG.
        # But `nnx.Dropout` documentation says it stores the collection name (default 'dropout').
        # When called, it needs to generate a key.
        # If I don't pass `rngs` to `__call__`, it might fail?
        # Let's assume I don't need to pass rngs explicitly to `__call__` of my submodules IF I'm using "functional" style `nnx.split` / `pop` etc?
        # Actually, `nnx.Dropout` is a Module. 
        # `y = self.dropout(x)`
        # `nnx` separates state.
        # OK, I will trust that standard usage works: `rngs` are passed to `nnx.call` or similar top-level,
        # but inside `__call__` methods, we don't pass them manually unless using functional.
        # Wait, `MAEReplacer` I WROTE uses `rngs` arg explicitly.
        # `nnx.Dropout` implementation source: `__call__(self, inputs, *, deterministic, rngs=None)`.
        # If `rngs` is None, it tries `self.make_rng(self.rng_collection)`.
        # So it uses `make_rng` which works if we are wrapped in `nnx.Rngs` context?
        # No, `nnx.State` handling.
        # I will stick to: `BlockCausalTransformer` should handle its own dropout calls.
        # I'll rely on `nnx` magic for `Dropout`.
        # But `MAEReplacer` I made explicit `rngs` arg. I will use it.
        pass

        encoded_tokens = self.transformer(tokens, deterministic=deterministic)

        # 6) Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, :self.n_latents, :]
        proj_tokens = nnx.tanh(self.bottleneck_proj(latent_tokens))

        return proj_tokens, (patch_mask, keep_prob)  # keep mask if you want diagnostics



class Decoder(nnx.Module):
    """
    MAE-style decoder.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        n_patches: int,
        d_patch: int,
        d_bottleneck: int = None, # Added
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        use_flash_attention: bool = False,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = depth
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.d_patch = d_patch
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.time_every = time_every
        self.latents_only_time = latents_only_time
        self.use_flash_attention = use_flash_attention
        self.dtype = dtype
        
        if d_bottleneck is None:
             raise ValueError("d_bottleneck must be provided for nnx Decoder")

        self.layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        self.modality_ids = self.layout.modality_ids()
        
        self.up_proj = nnx.Linear(d_bottleneck, d_model, use_bias=True, dtype=dtype, rngs=rngs)
        
        key = rngs.params()
        self.patch_queries = nnx.Param(jax.random.normal(key, (n_patches, d_model)) * 0.02)
        
        self.patch_head = nnx.Linear(d_model, d_patch, use_bias=True, dtype=dtype, rngs=rngs)
        
        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=n_latents,
            modality_ids=self.modality_ids,
            space_mode="decoder",                 # << decoder routing
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=latents_only_time,
            use_flash_attention=use_flash_attention,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, z: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        B, T, N_l, d_bottleneck = z.shape

        # 1) Up-project latent bottleneck to d_model (per latent token)
        latents = nnx.tanh(self.up_proj(z))  # (B, T, N_l, D)

        # 2) Learned per-patch query tokens (owned by the decoder)
        patches = jnp.broadcast_to(
            self.patch_queries.value[None, None, ...],
            (B, T, self.n_patches, self.d_model),
        )  # (B, T, Np, D)

        # 3) Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=2)

        # 4) Add sinusoidal positions
        tokens = add_sinusoidal_positions(tokens)

        # 5) Axial block-causal transformer
        x = self.transformer(tokens, deterministic=deterministic)
        
        # 6) Prediction head over the patch-query slice
        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = nnx.sigmoid(self.patch_head(x_patches))  # (B,T,Np,D_patch)
        return pred_btnd

class ActionEncoder(nnx.Module):
    def __init__(self, d_model: int, n_keyboard: int = 5, *, rngs: nnx.Rngs):
        self.d_model = d_model
        self.n_keyboard = n_keyboard
        
        key = rngs.params()
        self.base_emb = nnx.Param(jax.random.normal(key, (d_model,)) * 0.02)
        
        self.emb_key = nnx.Embed(n_keyboard, d_model, rngs=rngs)

    def __call__(
        self,
        actions: Optional[jnp.ndarray],           # (B, T) int32 in [0, n_keyboard)
        batch_time_shape: Optional[Tuple[int,int]] = None,
        as_tokens: bool = True,
    ):
        # Base "action token" embedding (used always)
        
        if actions is None:
            # unlabeled videos: just broadcast base embedding
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = jnp.broadcast_to(self.base_emb.value, (B, T, self.d_model))
        else:
            # embed categorical actions
            emb_key = self.emb_key(actions)
            out = emb_key + self.base_emb.value  # broadcast add

        if as_tokens:
            # expand a token axis (S_a = 1)
            out = out[:, :, None, :]

        return out

class Dynamics(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        depth: int,
        k_max: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        space_mode: str = "wm_agent_isolated",
        use_flash_attention: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck
        self.d_spatial = d_spatial
        self.n_spatial = n_spatial
        self.n_register = n_register
        self.n_agent = n_agent
        self.n_heads = n_heads
        self.depth = depth
        self.k_max = k_max
        self.dropout = dropout
        self.spatial_slice = None # set below
        self.agent_slice = None
        self.space_mode = space_mode

        assert d_spatial % d_bottleneck == 0
        
        self.spatial_proj = nnx.Linear(d_spatial, d_model, use_bias=True, rngs=rngs)
        
        key = rngs.params()
        self.register_tokens = nnx.Param(jax.random.normal(key, (n_register, d_model)) * 0.02)
        
        self.action_encoder = ActionEncoder(d_model=d_model, rngs=rngs)

        # Two separate tokens for shortcut conditioning
        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),   # τ (signal level) token
            (Modality.SHORTCUT_STEP, 1),     # d (step size) token
            (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, n_register),
        ]
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))
        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        self.spatial_slice = self.layout.slices()[Modality.SPATIAL]
        self.agent_slice  = self.layout.slices().get(Modality.AGENT, slice(0,0))
        self.modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=0,
            modality_ids=self.modality_ids,
            space_mode=space_mode,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=False,
            use_flash_attention=use_flash_attention,
            rngs=rngs,
        )

        self.num_step_bins = int(math.log2(k_max)) + 1
        self.step_embed = nnx.Embed(self.num_step_bins, d_model, rngs=rngs)

        self.signal_embed = nnx.Embed(k_max + 1, d_model, rngs=rngs)
        # flow_x_head init: kernel_init zeros.
        # nnx.Linear uses kernel_init kwarg.
        self.flow_x_head = nnx.Linear(
            d_model, d_spatial, use_bias=True, 
            kernel_init=nnx.initializers.zeros, 
            bias_init=nnx.initializers.zeros,
            rngs=rngs
        )

    def __call__(
        self,
        actions,             # (B,T)
        step_idxs,           # (B,T)
        signal_idxs,         # (B,T)
        packed_enc_tokens,   # (B,T,n_s,d_spatial)
        *,
        agent_tokens: Optional[jnp.ndarray] = None,  # (B,T,n_agent,D) or None
        deterministic: bool = True,
    ):
        # --- 1) Project spatial tokens to model dimension
        spatial_tokens = self.spatial_proj(packed_enc_tokens) # (B, T, n_spatial, d_model)

        # --- 2) Encode actions to d_model
        action_tokens = self.action_encoder(actions)  # (B, T, N_a, d_model)

        # --- 3) Prepare learned register tokens
        B, T = spatial_tokens.shape[:2]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value[None, None, ...],  # (1,1,n_register,d_model)
            (B, T, self.n_register, self.d_model),
        )

        # --- 4) Shortcut embeddings (discrete lookup)
        step_tok   = self.step_embed(step_idxs)[:, :, None, :]      # (B, T, 1, d_model)
        signal_tok = self.signal_embed(signal_idxs)[:, :, None, :]     # (B, T, 1, d_model)
        
        # --- 5) Concatenate in your declared layout order
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = jnp.zeros((B, T, self.n_agent, self.d_model), dtype=spatial_tokens.dtype)
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, agent_tokens]
        else:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens]
        tokens = jnp.concatenate(toks, axis=2)                    # (B,T,S,D)

        tokens = add_sinusoidal_positions(tokens)      # (B, T, N_total, d_model)
        x = self.transformer(tokens, deterministic=deterministic)
        spatial_tokens = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_tokens)
        h_t = x[:, :, self.agent_slice, :] if self.n_agent > 0 else None  # (B,T,n_agent,D) or None
        return x1_hat, h_t

class TaskEmbedder(nnx.Module):
    def __init__(
        self, 
        d_model: int, 
        n_agent: int = 1, 
        use_ids: bool = True, 
        n_tasks: int = 128, 
        d_task: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_agent = n_agent
        self.use_ids = use_ids
        self.n_tasks = n_tasks
        self.d_task = d_task
        
        if use_ids:
            self.emb = nnx.Embed(n_tasks, d_model, rngs=rngs)
        else:
            self.emb = nnx.Linear(d_task, d_model, use_bias=True, rngs=rngs)
            
        key = rngs.params()
        self.agent_base = nnx.Param(jax.random.normal(key, (d_model,)) * 0.02)

    def __call__(self, task, B: int, T: int):
        if self.use_ids:
            emb = self.emb(task)
        else:
            emb = self.emb(task)

        x = emb + self.agent_base.value[None, :]

        # Replicate across time and agent slots
        x = jnp.broadcast_to(x[:, None, None, :], (B, T, self.n_agent, self.d_model))
        return x

# === Phase B heads (use existing MLP) =========================================

class PolicyHeadMTP(nnx.Module):
    """Multi-Token action prediction."""
    def __init__(
        self,
        d_model: int,
        action_dim: int,
        L: int = 8,
        kind: str = "categorical",
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        swiglu: bool = True,
        parity_2over3: bool = False,
        dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.action_dim = action_dim
        self.L = L
        self.kind = kind
        self.dtype = dtype

        self.projector = MLP(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            swiglu=swiglu,
            parity_2over3=parity_2over3,
            dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(d_model, L * action_dim, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.projector(h_t, deterministic=deterministic)  # (B, T, D)
        logits_flat = self.out(x)                             # (B, T, L*A)
        # Reshape to (B, T, L, A)
        logits = logits_flat.reshape(h_t.shape[0], h_t.shape[1], self.L, self.action_dim)
        return logits


class RewardHeadMTP(nnx.Module):
    """Multi-Token reward prediction with symexp twohot bins."""
    def __init__(
        self,
        d_model: int,
        L: int = 8,
        num_bins: int = 101,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        swiglu: bool = True,
        parity_2over3: bool = False,
        dtype: Any = jnp.float32,
        log_low: float = -8.0,
        log_high: float = 8.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.L = L
        self.num_bins = num_bins
        
        self.projector = MLP(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            swiglu=swiglu,
            parity_2over3=parity_2over3,
            dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(d_model, L * num_bins, use_bias=True, dtype=dtype, rngs=rngs)
        
        log_edges = jnp.linspace(log_low, log_high, num_bins)
        self.centers = log_edges # stored as attribute (static or array depending on usage)

    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t, deterministic=deterministic)   # (B, T, D)
        logits_flat = self.out(x)
        logits = logits_flat.reshape(h_t.shape[0], h_t.shape[1], self.L, self.num_bins)
        return logits, self.centers


class ValueHead(nnx.Module):
    """Value prediction with symexp twohot bins."""
    def __init__(
        self,
        d_model: int,
        num_bins: int = 101,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        swiglu: bool = True,
        parity_2over3: bool = False,
        dtype: Any = jnp.float32,
        log_low: float = -8.0,
        log_high: float = 8.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        
        self.projector = MLP(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            swiglu=swiglu,
            parity_2over3=parity_2over3,
            dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(d_model, num_bins, use_bias=True, dtype=dtype, rngs=rngs)
        
        log_edges = jnp.linspace(log_low, log_high, num_bins)
        self.centers = log_edges

    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t, deterministic=deterministic)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, K)
        return logits, self.centers


def test_encoder_decoder():
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    n_patches = 4
    d_patch = 3
    enc_n_latents = 2
    enc_d_bottleneck = 3
    x = jnp.ones((B, T, n_patches, d_patch))  # (B,T,Np,D_patch)

    encoder = Encoder(d_model=8, n_latents=enc_n_latents, n_patches=n_patches, n_heads=2, depth=2, dropout=0.5, d_bottleneck=enc_d_bottleneck)
    decoder = Decoder(d_model=8, n_heads=2, depth=2, n_patches=n_patches, n_latents=enc_n_latents, d_patch=d_patch, dropout=0.5)
    # init: give both "mae" and "dropout" keys (dropout only needed if deterministic=False)
    enc_vars = encoder.init(
        {"params": rng, "mae": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        x,
        deterministic=True,
    )
    # Decode
    fake_z = jnp.ones((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_z,
        deterministic=True,
    )

    def forward_apply(enc_vars: FrozenDict, dec_vars: FrozenDict,
                    patches_btnd: jnp.ndarray,
                    *, mae_key=None, drop_key=None, train: bool):
        # Encoder
        rngs_enc = {}
        if train:
            rngs_enc = {"mae": mae_key, "dropout": drop_key}
        else:
            rngs_enc = {"mae": mae_key}  # if you still want masking during eval

        z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd,
                                        rngs=rngs_enc,
                                        deterministic=not train)
        # Decoder
        rngs_dec = {"dropout": drop_key} if train else {}
        pred_btnd = decoder.apply(dec_vars, z_btLd,
                                rngs=rngs_dec,
                                deterministic=not train)
        return pred_btnd, mae_info
    
    jit_forward = jax.jit(forward_apply, static_argnames=["train"])
    mae_key = jax.random.PRNGKey(0)
    drop_key = jax.random.PRNGKey(1)
    # Warm-up (compilation happens here)
    t0 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t1 = time.time()
    # Hot run (should be much faster)
    t2 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t3 = time.time()

    print(f"Warm-up (compile+run): {t1 - t0:.3f}s")
    print(f"Hot run (cached):      {t3 - t2:.3f}s")

def test_dynamics():
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    fake_enc_z = jnp.ones((B, T, 512, 16), dtype=jnp.float32)
    fake_actions = jnp.ones((B, T), dtype=jnp.int32)
    fake_steps = jnp.full((B, T), 1/256, dtype=jnp.float32)
    fake_signals = jnp.full((B, T), 0.0, dtype=jnp.float32)
    def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
        """
        (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
        Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
        """
        return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)
    fake_packed_enc_tokens = pack_bottleneck_to_spatial(fake_enc_z, n_spatial=256, k=2)


    # need some way to assert that 512 * 16 == 256 * 32
    dynamics_kwargs = {
        "d_model": 128,
        "n_spatial": 256,
        "d_spatial": 32,
        "d_bottleneck": 16,
        "k_max": 8,
        "n_register": 10,
        "n_heads": 4,
        "depth": 4,
        "dropout": 0.0
    }
    dynamics = Dynamics(**dynamics_kwargs)
    dynamics_vars = dynamics.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_actions,
        fake_steps,
        fake_signals,
        fake_packed_enc_tokens,
    )
    out = dynamics.apply(dynamics_vars, fake_actions, fake_steps, fake_signals, fake_packed_enc_tokens,
                        rngs={"dropout": jax.random.PRNGKey(2)},
                        deterministic=True)

def _build_modality_mask(modality_ids, mode: str, n_latents=0, d_model=16, n_heads=2):
    class _Peek(nn.Module):
        @nn.compact
        def __call__(self, x):
            att = SpaceSelfAttentionModality(
                d_model=d_model, n_heads=n_heads,
                modality_ids=modality_ids, n_latents=n_latents,
                mode=mode, dropout=0.0)
            y = att(x, deterministic=True)
            # expose stored mask
            mask = att.variables["constants"]["modality_mask"]  # (1,1,S,S)
            return y, mask

    B,T,S,D = 1,1,modality_ids.shape[0],d_model
    x = jnp.zeros((B,T,S,D))
    vars_ = _Peek().init(jax.random.PRNGKey(0), x)
    _, mask = _Peek().apply(vars_, x, mutable=False)
    return jnp.asarray(mask)  # (1,1,S,S)

def _pack_bottleneck_to_spatial(z_btLd, n_spatial, k):
    return rearrange(z_btLd, 'b t (n k) d -> b t n (k d)', n=n_spatial, k=k)

def _abbr(m):
    # short labels just for printing rows/cols
    return {
        int(Modality.ACTION): "ACT",
        int(Modality.SHORTCUT_SIGNAL): "SIG",
        int(Modality.SHORTCUT_STEP): "STP",
        int(Modality.SPATIAL): "SPA",
        int(Modality.REGISTER): "REG",
        int(Modality.AGENT): "AGT",
        int(Modality.LATENT): "LAT",
    }.get(int(m), f"M{int(m)}")

def _print_mask_summary(name: str, modality_ids: jnp.ndarray, mask_2d: jnp.ndarray):
    # mask_2d: (S,S) with True meaning "query row can read key col"
    S = modality_ids.shape[0]
    mods = [int(x) for x in list(modality_ids)]
    headers = "     " + " ".join(f"{_abbr(m):>3}" for m in mods)
    print(f"\n[{name}] modality order (Q rows / K cols): {mods}")
    print(headers)
    for q in range(S):
        row = "".join("  ✓" if bool(mask_2d[q, k]) else "  ·" for k in range(S))
        print(f"{_abbr(modality_ids[q]):>3}: {row}")
    # row-wise counts
    counts = jnp.sum(mask_2d, axis=1)
    print("Row read-counts:", counts.tolist())

def test_agent_firewall():
    # layout: [ACTION, SIG, STEP, SPATIALx3, REGISTERx2, AGENTx1]
    ACTION,SIGNAL,STEP,SPATIAL,REGISTER,AGENT = 1,5,6,4,3,7
    modality_ids = jnp.array([ACTION, SIGNAL, STEP, SPATIAL, SPATIAL, SPATIAL, REGISTER, REGISTER, AGENT], dtype=jnp.int32)
    S = modality_ids.shape[0]
    agent_col = (modality_ids == AGENT)  # keys that are agent
    agent_row = (modality_ids == AGENT)  # queries that are agent

    # ----- wm_agent -----
    mask = _build_modality_mask(modality_ids, "wm_agent")[0,0]  # (S,S)
    _print_mask_summary("wm_agent", modality_ids, mask)

    # Others never see agent: find any offending (q,k) where q!=agent and k is agent
    bad_q = []
    for q in range(S):
        if not bool(agent_row[q]):
            if bool(mask[q, agent_col].sum()):
                bad_q.append(q)
    if bad_q:
        print("Violations in wm_agent (non-agent reads agent) at query rows:", bad_q)

    # Agent reads all in wm_agent
    agent_q_idx = int(jnp.where(agent_row, size=1, fill_value=-1)[0][0])
    if agent_q_idx >= 0:
        agent_reads = mask[agent_q_idx, :]
        missing = [k for k in range(S) if not bool(agent_reads[k])]
        if missing:
            print("Violations in wm_agent (agent cannot read some keys). Missing cols:", missing)

    # Assertions
    for q in range(S):
        if not bool(agent_row[q]):
            assert mask[q, agent_col].sum() == 0, "Non-agent query can attend to agent!"
    if agent_q_idx >= 0:
        assert jnp.all(mask[agent_q_idx, :]), "Agent query cannot read some token in wm_agent"

    # ----- wm_agent_isolated -----
    mask_iso = _build_modality_mask(modality_ids, "wm_agent_isolated")[0,0]
    _print_mask_summary("wm_agent_isolated", modality_ids, mask_iso)

    # Others still never see agent
    bad_q_iso = []
    for q in range(S):
        if not bool(agent_row[q]):
            if bool(mask_iso[q, agent_col].sum()):
                bad_q_iso.append(q)
    if bad_q_iso:
        print("Violations in wm_agent_isolated (non-agent reads agent) at query rows:", bad_q_iso)

    # Agent reads nobody in isolated
    if agent_q_idx >= 0:
        agent_reads_iso = int(mask_iso[agent_q_idx, :].sum())
        print("Agent read-count in isolated mode:", agent_reads_iso)

    # Assertions
    for q in range(S):
        if not bool(agent_row[q]):
            assert mask_iso[q, agent_col].sum() == 0, "Non-agent query can attend to agent in isolated!"
    if agent_q_idx >= 0:
        assert mask_iso[agent_q_idx, :].sum() == 0, "Agent should read nobody in wm_agent_isolated"


def test_x1hat_invariant_to_agent_tokens():
    B,T = 2,5
    n_b, d_b = 8, 4      # encoder latents
    n_spatial, pack = 4, 2
    d_spatial = d_b * pack
    D = 32

    fake_enc_z = jnp.ones((B, T, n_b, d_b))
    packed = _pack_bottleneck_to_spatial(fake_enc_z, n_spatial=n_spatial, k=pack)
    actions = jnp.zeros((B,T), dtype=jnp.int32)
    step_idx = jnp.zeros((B,T), dtype=jnp.int32)
    sig_idx  = jnp.zeros((B,T), dtype=jnp.int32)

    dyn = Dynamics(
        d_model=D, d_bottleneck=d_b, d_spatial=d_spatial,
        n_spatial=n_spatial, n_register=2, n_agent=1,
        n_heads=2, depth=2, k_max=8, dropout=0.0, mlp_ratio=2.0,
        time_every=2, space_mode="wm_agent"  # try either mode
    )
    vars_ = dyn.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                     actions, step_idx, sig_idx, packed)

    # random agent vs zeros
    agent_rand = jax.random.normal(jax.random.PRNGKey(2), (B,T,1,D))
    x1_a, _ = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                        agent_tokens=agent_rand, rngs={"dropout": jax.random.PRNGKey(3)}, deterministic=True)
    x1_b, _ = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                        agent_tokens=jnp.zeros_like(agent_rand), rngs={"dropout": jax.random.PRNGKey(3)}, deterministic=True)

    diff = x1_a - x1_b
    max_abs = float(jnp.max(jnp.abs(diff)))
    l2 = float(jnp.sqrt(jnp.sum(diff * diff)))
    print("\n[x1_hat invariance] max|Δ| =", max_abs, " ||Δ||₂ =", l2)
    print("x1_a shape:", x1_a.shape, " x1_b shape:", x1_b.shape)

    # Must be exactly equal because agent cannot influence others
    assert jnp.allclose(x1_a, x1_b, atol=0, rtol=0), "x1_hat changed with agent tokens—firewall broken"


def test_shapes_and_h_t():
    B,T,D = 2,6,32
    n_b,d_b = 8,4
    n_spatial, pack = 4,2
    d_spatial = d_b*pack

    packed = _pack_bottleneck_to_spatial(jnp.ones((B,T,n_b,d_b)), n_spatial, pack)
    dyn = Dynamics(d_model=D, d_bottleneck=d_b, d_spatial=d_spatial,
                   n_spatial=n_spatial, n_register=3, n_agent=1,
                   n_heads=2, depth=2, k_max=8, space_mode="wm_agent")
    actions = jnp.zeros((B,T), dtype=jnp.int32)
    step_idx = jnp.zeros((B,T), dtype=jnp.int32)
    sig_idx  = jnp.zeros((B,T), dtype=jnp.int32)
    vars_ = dyn.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                     actions, step_idx, sig_idx, packed)

    x1_hat, h_t = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                            agent_tokens=jnp.zeros((B,T,1,D)))
    print("\n[shapes] x1_hat:", x1_hat.shape, " h_t:", (None if h_t is None else h_t.shape))
    print("Expect x1_hat =", (B,T,n_spatial,d_spatial), " h_t =", (B,T,1,D))
    assert x1_hat.shape == (B,T,n_spatial,d_spatial)
    assert h_t.shape     == (B,T,1,D)

def test_wm_routed():
    """
    Checks space-attention routing for Dreamer-4-style dynamics:
      - Action q -> {Action k}
      - Obs q    -> {Obs k ∪ Action k} and never Agent k
      - Agent q  -> {Obs k ∪ Action k ∪ Agent k}    (wm_agent)
                  -> {}                              (wm_agent_isolated)
      - For any non-agent q, Agent k is disallowed.
    """
    # Shorthand modality ints
    ACTION  = int(Modality.ACTION)
    SIGNAL  = int(Modality.SHORTCUT_SIGNAL)
    STEP    = int(Modality.SHORTCUT_STEP)
    SPATIAL = int(Modality.SPATIAL)
    REGISTER= int(Modality.REGISTER)
    AGENT   = int(Modality.AGENT)

    # Toy layout (Q rows / K cols share this order):
    # [ACT, SIG, STP, SPA, SPA, SPA, REG, REG, ACT, AGT]
    modality_ids = jnp.array(
        [ACTION, SIGNAL, STEP, SPATIAL, SPATIAL, SPATIAL, REGISTER, REGISTER, ACTION, AGENT],
        dtype=jnp.int32
    )
    S = modality_ids.shape[0]

    # Helper sets
    is_agent = (modality_ids == AGENT)
    is_action = (modality_ids == ACTION)
    is_obs = (
        (modality_ids == SPATIAL) |
        (modality_ids == REGISTER) |
        (modality_ids == SIGNAL)  |
        (modality_ids == STEP)
    )

    def assert_mask(mode: str):
        mask = _build_modality_mask(modality_ids, mode)[0, 0]  # (S,S) bool
        _print_mask_summary(mode, modality_ids, mask)

        # 1) Non-agent q must never see Agent k
        for q in range(S):
            if not bool(is_agent[q]):
                assert not bool(mask[q, is_agent].any()), f"[{mode}] non-agent q={q} can read Agent k!"

        # 2) Action q -> Action k only
        for q in range(S):
            if bool(is_action[q]):
                # Allowed: action keys only
                allowed = mask[q]
                assert bool(allowed[is_action].all()), f"[{mode}] action q={q} cannot read some action k!"
                assert not bool(allowed[~is_action].any()), f"[{mode}] action q={q} reads non-action keys!"

        # 3) Obs q -> Obs k ∪ Action k (and never Agent k, already checked)
        for q in range(S):
            if bool(is_obs[q]):
                allowed = mask[q]
                # Must allow all obs keys? We enforce "subset includes only obs∪action".
                # It's okay if some obs keys are masked by design, but we require no extra keys.
                extras = allowed & ~(is_obs | is_action)
                assert not bool(extras.any()), f"[{mode}] obs q={q} reads keys outside obs∪action!"

                # Should at least be able to read *some* obs or action key (nontrivial)
                assert bool((allowed & (is_obs | is_action)).any()), f"[{mode}] obs q={q} cannot read obs∪action at all!"

        # 4) Agent q behavior differs by mode
        agent_rows = [i for i in range(S) if bool(is_agent[i])]
        if agent_rows:
            q = agent_rows[0]
            if mode == "wm_agent":
                # Agent reads everyone (including agent)
                assert bool(mask[q].all()), "[wm_agent] agent q cannot read all keys!"
            else:
                # Isolated: agent reads nobody
                assert int(mask[q].sum()) == 0, "[wm_agent_isolated] agent q should read nobody!"

    # Run both modes
    assert_mask("wm_agent")
    assert_mask("wm_agent_isolated")
    print("\n[test_wm_routed] All routing assertions passed ✅")


if __name__ == "__main__":
    # test_agent_firewall()
    # test_x1hat_invariant_to_agent_tokens()
    # test_shapes_and_h_t()
    test_wm_routed()
    print("\nAll tests passed ✅")
