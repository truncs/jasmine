import math
from typing import Dict

import einops
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp

from jasmine.models.dynamics import DynamicsMaskGIT, DynamicsCausal, DynamicsDiffusion
from jasmine.models.lam import LatentActionModel
from jasmine.models.tokenizer import TokenizerVQVAE, TokenizerMAE
from jasmine.baselines.diffusion.train_tokenizer_mae import Dreamer4TokenizerMAE, build_model, Args
from jasmine.models.dreamer4 import Dynamics, ActionEncoder


class GenieMaskGIT(nnx.Module):
    """GenieMaskGIT model"""

    def __init__(
        self,
        in_dim: int,
        tokenizer_dim: int,
        tokenizer_ffn_dim: int,
        latent_patch_dim: int,
        num_patch_latents: int,
        patch_size: int,
        tokenizer_num_blocks: int,
        tokenizer_num_heads: int,
        lam_dim: int,
        lam_ffn_dim: int,
        latent_action_dim: int,
        num_actions: int,
        lam_patch_size: int,
        lam_num_blocks: int,
        lam_num_heads: int,
        lam_co_train: bool,
        use_gt_actions: bool,
        dyna_type: str,
        dyna_dim: int,
        dyna_ffn_dim: int,
        dyna_num_blocks: int,
        dyna_num_heads: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        decode: bool,
        dropout: float = 0.0,
        mask_limit: float = 0.0,
    ):
        # --- Tokenizer ---
        self.in_dim = in_dim
        self.tokenizer_dim = tokenizer_dim
        self.tokenizer_ffn_dim = tokenizer_ffn_dim
        self.latent_patch_dim = latent_patch_dim
        self.num_patch_latents = num_patch_latents
        self.patch_size = patch_size
        self.tokenizer_num_blocks = tokenizer_num_blocks
        self.tokenizer_num_heads = tokenizer_num_heads
        # --- LAM ---
        self.lam_dim = lam_dim
        self.lam_ffn_dim = lam_ffn_dim
        self.latent_action_dim = latent_action_dim
        self.num_actions = num_actions
        self.lam_patch_size = lam_patch_size
        self.lam_num_blocks = lam_num_blocks
        self.lam_num_heads = lam_num_heads
        self.lam_co_train = lam_co_train
        self.use_gt_actions = use_gt_actions
        # --- Dynamics ---
        self.dyna_type = dyna_type
        self.dyna_dim = dyna_dim
        self.dyna_ffn_dim = dyna_ffn_dim
        self.dyna_num_blocks = dyna_num_blocks
        self.dyna_num_heads = dyna_num_heads
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.dropout = dropout
        self.mask_limit = mask_limit
        self.decode = decode
        self.tokenizer = TokenizerVQVAE(
            in_dim=self.in_dim,
            model_dim=self.tokenizer_dim,
            ffn_dim=self.tokenizer_ffn_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_patch_latents,
            patch_size=self.patch_size,
            num_blocks=self.tokenizer_num_blocks,
            num_heads=self.tokenizer_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
        )

        self.action_embed = nnx.Embed(
            self.num_actions,
            self.latent_action_dim, rngs=rngs
        )

        self.dynamics = DynamicsCausal(
            model_dim=self.dyna_dim,
            ffn_dim=self.dyna_ffn_dim,
            num_latents=self.num_patch_latents,
            latent_action_dim=self.latent_action_dim,
            num_blocks=self.dyna_num_blocks,
            num_heads=self.dyna_num_heads,
            dropout=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            decode=self.decode,
            rngs=rngs,
        )

    def __call__(
        self,
        batch: Dict[str, jax.Array],
    ) -> Dict[str, jax.Array]:
        videos_BTHWC = batch["videos"]
        H, W = videos_BTHWC.shape[2:4]
        latent_actions_BTm11L = None
        action_embeddings_BTm11L = None
        if self.use_gt_actions:
            assert self.action_embed is not None
            action_indices_E = None
            action_embeddings_BT1L = self.action_embed(batch["actions"]).reshape(
                *batch["actions"].shape[:2], 1, self.latent_action_dim
            )
            action_embeddings_BTm11L = action_embeddings_BT1L[:, :-1]
        else:
            assert self.lam is not None
            lam_outputs = self.lam.vq_encode(videos_BTHWC, training=False)
            z_q_BTm11L = lam_outputs["z_q"]
            action_indices_E = lam_outputs["indices"]
            latent_actions_BTm11L = jax.lax.cond(
                self.lam_co_train,
                lambda: z_q_BTm11L,
                lambda: jax.lax.stop_gradient(z_q_BTm11L),
            )

        tokenizer_outputs = self.tokenizer.vq_encode(videos_BTHWC, training=False)
        token_indices_BTN = tokenizer_outputs["indices"]

        latent_actions = (
            action_embeddings_BTm11L if self.use_gt_actions else latent_actions_BTm11L
        )
        assert isinstance(latent_actions, jax.Array)

        outputs = dict(
            video_tokens=jax.lax.stop_gradient(token_indices_BTN),
            latent_actions=latent_actions,
        )
        outputs["mask_rng"] = batch["rng"]
        dyna_logits_BTNV, dyna_mask = self.dynamics(outputs)
        outputs["token_logits"] = dyna_logits_BTNV
        outputs["mask"] = dyna_mask
        mle_indices_BTN = jnp.argmax(outputs["token_logits"], axis=-1)
        outputs["recon"] = self.tokenizer.decode(mle_indices_BTN, (H, W))

        if action_indices_E is not None:
            outputs["lam_indices"] = action_indices_E

        return outputs

    def sample(
        self,
        batch: Dict[str, jax.Array],
        seq_len: int,
        temperature: float = 1,
        sample_argmax: bool = False,
        maskgit_steps: int = 25,
    ) -> tuple[jax.Array, jax.Array]:
        if self.dyna_type == "maskgit":
            return self.sample_maskgit(
                batch, seq_len, maskgit_steps, temperature, sample_argmax
            )
        elif self.dyna_type == "causal":
            return self.sample_causal(batch, seq_len, temperature, sample_argmax)
        else:
            raise ValueError(f"Dynamics model type unknown: {self.dyna_type}")

    def sample_maskgit(
        self,
        batch: Dict[str, jax.Array],
        seq_len: int,
        steps: int = 25,
        temperature: float = 1,
        sample_argmax: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Autoregressively samples up to `seq_len` future frames, following Figure 8 of the Genie 1 paper (Bruce et al. (2022)).

        - Input frames are tokenized once.
        - Future frames are generated autoregressively in token space.
        - All frames are detokenized in a single pass.

        Note:
        - For interactive or step-wise sampling, detokenization should occur after each action.
        - To maintain consistent tensor shapes across timesteps, all current and future frames are decoded at every step.
        - Temporal causal structure is preserved by
            a) reapplying the mask before each decoding step.
            b) a temporal causal mask is applied within each ST-transformer block.

        Dimension keys:
            B: batch size
            T: number of input (conditioning) frames
            N: number of patches per frame
            M: model dimension
            S: sequence length
            H: height
            W: width
            E: B * (S - 1)
            P: S * N
        """
        assert isinstance(self.dynamics, DynamicsMaskGIT)
        # --- Encode videos and actions ---
        videos_BTHWC = batch["videos"]
        tokenizer_out = self.tokenizer.vq_encode(videos_BTHWC, training=False)
        token_idxs_BTN = tokenizer_out["indices"]
        B, T, N = token_idxs_BTN.shape
        pad_shape = (B, seq_len - T, N)
        pad = jnp.zeros(pad_shape, dtype=token_idxs_BTN.dtype)
        token_idxs_BSN = jnp.concatenate([token_idxs_BTN, pad], axis=1)
        init_logits_BSNV = jnp.zeros(
            shape=(*token_idxs_BSN.shape, self.num_patch_latents)
        )
        if self.use_gt_actions:
            assert self.action_embed is not None
            latent_actions_BT1L = self.action_embed(batch["actions"]).reshape(
                *batch["actions"].shape[:2], 1, self.latent_action_dim
            )
            latent_actions_BTm11L = latent_actions_BT1L[:, :-1]
            action_tokens_EL = latent_actions_BTm11L.reshape(-1, self.latent_action_dim)
        else:
            assert self.lam is not None
            latent_actions_E = batch["latent_actions"]
            action_tokens_EL = self.lam.vq.get_codes(latent_actions_E)

        # --- Extract submodule state ---
        dynamics_state = nnx.state(self.dynamics)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def maskgit_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            step: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            rng, token_idxs_BSN, logits_BSNV, mask_BSN, action_tokens_EL = carry
            S, N = token_idxs_BSN.shape[1:]
            L = action_tokens_EL.shape[-1]

            # We need to reconstruct the submodule inside scan body to prevent trace context mismatches
            dynamics_maskgit = DynamicsMaskGIT(
                model_dim=self.dyna_dim,
                ffn_dim=self.dyna_ffn_dim,
                num_latents=self.num_patch_latents,
                latent_action_dim=self.latent_action_dim,
                num_blocks=self.dyna_num_blocks,
                num_heads=self.dyna_num_heads,
                dropout=self.dropout,
                mask_limit=self.mask_limit,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                decode=self.decode,
                rngs=nnx.Rngs(0),
            )
            nnx.update(dynamics_maskgit, dynamics_state)

            # --- Construct + encode video ---
            vid_embed_BSNM = dynamics_maskgit.patch_embed(token_idxs_BSN)
            mask_token_111M = dynamics_maskgit.mask_token.value
            mask_expanded_BSN1 = mask_BSN[..., jnp.newaxis]
            vid_embed_BSNM = jnp.where(
                mask_expanded_BSN1, mask_token_111M, vid_embed_BSNM
            )

            # --- Predict transition ---
            action_tokens_BSm1L = jnp.reshape(action_tokens_EL, (B, S - 1, L))
            act_embed_BSm1M = dynamics_maskgit.action_up(action_tokens_BSm1L)
            act_embed_BSM = jnp.pad(act_embed_BSm1M, ((0, 0), (1, 0), (0, 0)))
            act_embed_BS1M = jnp.reshape(
                act_embed_BSM, (B, S, 1, act_embed_BSM.shape[-1])
            )
            vid_embed_BSNp1M = jnp.concatenate([act_embed_BS1M, vid_embed_BSNM], axis=2)
            unmasked_ratio = jnp.cos(jnp.pi * (step + 1) / (steps * 2))
            step_temp = temperature * (1.0 - unmasked_ratio)
            final_logits_BSNp1V = (
                dynamics_maskgit.transformer(vid_embed_BSNp1M) / step_temp
            )
            final_logits_BSNV = final_logits_BSNp1V[:, :, 1:]

            # --- Sample new tokens for final frame ---
            if sample_argmax:
                sampled_token_idxs_BSN = jnp.argmax(final_logits_BSNV, axis=-1)
            else:
                rng, _rng = jax.random.split(rng)
                sampled_token_idxs_BSN = jax.random.categorical(_rng, final_logits_BSNV)
            gather_fn = jax.vmap(jax.vmap(jax.vmap(lambda x, y: x[y])))
            final_token_probs_BSN = gather_fn(
                jax.nn.softmax(final_logits_BSNV), sampled_token_idxs_BSN
            )
            final_token_probs_BSN += ~mask_BSN
            # Update masked tokens and logits only
            token_idxs_BSN = jnp.where(mask_BSN, sampled_token_idxs_BSN, token_idxs_BSN)
            logits_BSNV = jnp.where(
                jnp.expand_dims(mask_BSN, -1), final_logits_BSNV, logits_BSNV
            )

            # --- Update mask ---
            num_unmasked_tokens = jnp.round(N * (1.0 - unmasked_ratio)).astype(int)
            final_token_probs_flat_BP = einops.rearrange(
                final_token_probs_BSN, "b s n -> b (s n)"
            )
            idx_mask_P = (
                jnp.arange(final_token_probs_flat_BP.shape[-1])
                <= N - num_unmasked_tokens
            )
            sorted_idxs_BP = jnp.argsort(final_token_probs_flat_BP, axis=-1)
            mask_update_fn = jax.vmap(lambda msk, ids: msk.at[ids].set(idx_mask_P))
            mask_flat_BP = einops.rearrange(mask_BSN, "b s n -> b (s n)")
            new_mask_flat_BP = mask_update_fn(mask_flat_BP, sorted_idxs_BP)
            new_mask_BSN = einops.rearrange(new_mask_flat_BP, "b (s n) -> b s n", n=N)

            new_carry = (
                rng,
                token_idxs_BSN,
                logits_BSNV,
                new_mask_BSN,
                action_tokens_EL,
            )
            return new_carry

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def generation_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array], step_t: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            rng, current_token_idxs_BSN, current_logits_BSNV = carry
            rng, step_rng = jax.random.split(rng)

            # Mask current frame (i.e., t == step_t)
            mask_S = jnp.arange(seq_len) == step_t
            mask_BSN = jnp.broadcast_to(
                mask_S[jnp.newaxis, :, jnp.newaxis], (B, seq_len, N)
            ).astype(bool)
            masked_token_idxs_BSN = current_token_idxs_BSN * ~mask_BSN
            masked_logits_BSNV = current_logits_BSNV * jnp.expand_dims(~mask_BSN, -1)

            # --- Initialize and run MaskGIT loop ---
            init_carry_maskgit = (
                step_rng,
                masked_token_idxs_BSN,
                masked_logits_BSNV,
                mask_BSN,
                action_tokens_EL,
            )
            final_carry_maskgit = maskgit_step_fn(init_carry_maskgit, jnp.arange(steps))
            updated_token_idxs_BSN = final_carry_maskgit[1]
            updated_logits_BSNV = final_carry_maskgit[2]
            new_carry = (rng, updated_token_idxs_BSN, updated_logits_BSNV)
            return new_carry

        # --- Run the autoregressive generation using jax.lax.scan ---
        initial_carry = (batch["rng"], token_idxs_BSN, init_logits_BSNV)
        timesteps_to_scan = jnp.arange(T, seq_len)
        final_carry = generation_step_fn(initial_carry, timesteps_to_scan)
        final_token_idxs_BSN = final_carry[1]
        final_logits_BSNV = final_carry[2]

        # --- Decode all tokens at once at the end ---
        H, W = batch["videos"].shape[2:4]
        final_frames_BSHWC = self.tokenizer.decode(
            final_token_idxs_BSN,
            video_hw=(H, W),
        )
        return final_frames_BSHWC, final_logits_BSNV

    def sample_causal(
        self,
        batch: Dict[str, jax.Array],
        seq_len: int,
        temperature: float = 1,
        sample_argmax: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Dimension keys:
            B: batch size
            T: number of input (conditioning) frames
            N: number of patches per frame
            M: model dimension
            S: sequence length
            H: height
            W: width
            E: B * (S - 1)
        """
        assert isinstance(self.dynamics, DynamicsCausal)
        # --- Encode videos and actions ---
        videos_BTHWC = batch["videos"]
        tokenizer_out = self.tokenizer.vq_encode(videos_BTHWC, training=False)
        token_idxs_BTN = tokenizer_out["indices"]
        B, T, N = token_idxs_BTN.shape
        pad_shape = (B, seq_len - T, N)
        pad = jnp.zeros(pad_shape, dtype=token_idxs_BTN.dtype)
        token_idxs_BSN = jnp.concatenate([token_idxs_BTN, pad], axis=1)
        logits_BSNV = jnp.zeros((*token_idxs_BSN.shape, self.num_patch_latents))
        dynamics_state = nnx.state(self.dynamics)

        if self.use_gt_actions:
            assert self.action_embed is not None
            latent_actions_BT1L = self.action_embed(batch["actions"]).reshape(
                *batch["actions"].shape[:2], 1, self.latent_action_dim
            )
            latent_actions_BTm11L = latent_actions_BT1L[:, :-1]
            action_tokens_EL = latent_actions_BTm11L.reshape(-1, self.latent_action_dim)
        else:
            assert self.lam is not None
            latent_actions_E = batch["latent_actions"]
            action_tokens_EL = self.lam.vq.get_codes(latent_actions_E)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def causal_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            step_n: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            rng, token_idxs_BSN, logits_BSNV, action_tokens_EL, step_t = carry
            S, N = token_idxs_BSN.shape[1:]
            L = action_tokens_EL.shape[-1]

            # We need to reconstruct the submodule inside scan body to prevent trace context mismatches
            dynamics_causal = DynamicsCausal(
                model_dim=self.dyna_dim,
                ffn_dim=self.dyna_ffn_dim,
                num_latents=self.num_patch_latents,
                latent_action_dim=self.latent_action_dim,
                num_blocks=self.dyna_num_blocks,
                num_heads=self.dyna_num_heads,
                dropout=self.dropout,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                decode=self.decode,
                rngs=nnx.Rngs(0),
            )
            nnx.update(dynamics_causal, dynamics_state)

            # --- Construct + encode video ---
            vid_embed_BSNM = dynamics_causal.patch_embed(token_idxs_BSN)

            # --- Predict transition ---
            action_tokens_BSm1L = jnp.reshape(action_tokens_EL, (B, S - 1, L))
            act_embed_BSm1M = dynamics_causal.action_up(action_tokens_BSm1L)
            act_embed_BSM = jnp.pad(act_embed_BSm1M, ((0, 0), (1, 0), (0, 0)))
            act_embed_BS1M = jnp.reshape(
                act_embed_BSM, (B, S, 1, act_embed_BSM.shape[-1])
            )
            vid_embed_BSNp1M = jnp.concatenate([act_embed_BS1M, vid_embed_BSNM], axis=2)
            final_logits_BTNp1V = (
                dynamics_causal.transformer(vid_embed_BSNp1M) / temperature
            )
            final_logits_BV = final_logits_BTNp1V[:, step_t, step_n, :]

            # --- Sample new tokens for final frame ---
            if sample_argmax:
                sampled_token_idxs_B = jnp.argmax(final_logits_BV, axis=-1)
            else:
                rng, _rng = jax.random.split(rng)
                sampled_token_idxs_B = jax.random.categorical(_rng, final_logits_BV)
            # Update next tokens only
            token_idxs_BSN = token_idxs_BSN.at[:, step_t, step_n].set(
                sampled_token_idxs_B
            )
            logits_BSNV = logits_BSNV.at[:, step_t, step_n].set(final_logits_BV)

            new_carry = (rng, token_idxs_BSN, logits_BSNV, action_tokens_EL, step_t)
            return new_carry

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def generation_step_fn(
            carry: tuple[jax.Array, jax.Array, jax.Array], step_t: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            rng, current_token_idxs_BSN, current_logits_BSNV = carry
            rng, step_rng = jax.random.split(rng)

            # --- Initialize and run causal loop ---
            init_carry_causal = (
                step_rng,
                current_token_idxs_BSN,
                current_logits_BSNV,
                action_tokens_EL,
                step_t,
            )
            final_carry_causal = causal_step_fn(init_carry_causal, jnp.arange(N))
            updated_token_idxs_BSN = final_carry_causal[1]
            updated_logits_BSNV = final_carry_causal[2]
            new_carry = (rng, updated_token_idxs_BSN, updated_logits_BSNV)
            return new_carry

        # --- Run the autoregressive generation using jax.lax.scan ---
        initial_carry = (batch["rng"], token_idxs_BSN, logits_BSNV)
        timesteps_to_scan = jnp.arange(T, seq_len)
        final_carry = generation_step_fn(initial_carry, timesteps_to_scan)
        final_token_idxs_BSN = final_carry[1]
        final_logits_BSNV = final_carry[2]

        # --- Decode all tokens at once at the end ---
        H, W = batch["videos"].shape[2:4]
        final_frames_BSHWC = self.tokenizer.decode(
            final_token_idxs_BSN,
            video_hw=(H, W),
        )
        return final_frames_BSHWC, final_logits_BSNV

    def vq_encode(self, batch: Dict[str, jax.Array], training: bool) -> jax.Array:
        # --- Preprocess videos ---
        assert self.lam is not None
        video_BTHWC = batch["videos"]
        lam_output: Dict[str, jax.Array] = self.lam.vq_encode(
            video_BTHWC, training=training
        )
        lam_indices_E = lam_output["indices"]
        return lam_indices_E


class GenieDiffusion(nnx.Module):
    """GenieDiffusion model"""

    def __init__(
        self,
        in_dim: int,
        image_height: int,
        image_width: int,
        image_channels: int,
        tokenizer_dim: int,
        tokenizer_ffn_dim: int,
        latent_patch_dim: int,
        num_patch_latents: int,
        patch_size: int,
        tokenizer_num_blocks: int,
        tokenizer_num_heads: int,
        latent_action_dim: int,
        num_actions: int,
        is_action_discrete: bool,
        dyna_dim: int,
        dyna_ffn_dim: int,
        dyna_num_blocks: int,
        dyna_num_heads: int,
        dyna_num_registers: int,
        dyna_num_agents: int,
        dyna_kmax: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        dropout: float = 0.0,
        diffusion_denoise_steps: int = 0,
        decode: bool = False,
    ):
        # --- Tokenizer ---
        self.in_dim = in_dim
        self.tokenizer_dim = tokenizer_dim
        self.tokenizer_ffn_dim = tokenizer_ffn_dim
        self.latent_patch_dim = latent_patch_dim
        self.num_patch_latents = num_patch_latents
        self.patch_size = patch_size
        self.tokenizer_num_blocks = tokenizer_num_blocks
        self.tokenizer_num_heads = tokenizer_num_heads
        # --- LAM ---
        self.latent_action_dim = latent_action_dim
        self.num_actions = num_actions
        # --- Dynamics ---
        self.dyna_dim = dyna_dim
        self.dyna_ffn_dim = dyna_ffn_dim
        self.dyna_num_blocks = dyna_num_blocks
        self.dyna_num_heads = dyna_num_heads
        self.dyna_num_registers = dyna_num_registers
        self.dyna_num_agents = dyna_num_agents
        self.dyna_kmax = dyna_kmax
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.dropout = dropout
        self.diffusion_denoise_steps = diffusion_denoise_steps
        self.decode = decode
        self.rngs = rngs

        self.dtype = dtype

        num_patches = (image_height // patch_size) * (image_width // patch_size)

        d_patch = in_dim * patch_size ** 2

        enc_kwargs = {
            "d_model": 512,
            "n_latents": num_patch_latents,
            "n_patches": num_patches,
            "n_heads": 8,
            "depth": 8,
            "dropout": 0.05,
            "d_bottleneck": latent_patch_dim,
            "mae_p_min": 0.0,
            "mae_p_max": 0.9,
            "time_every": 4,
            "d_patch": d_patch,
            "use_flash_attention": use_flash_attention,
            "dtype": dtype,
        }

        dec_kwargs = {
            "d_model": 512,
            "n_heads": 8,
            "n_patches": num_patches,
            "n_latents": num_patch_latents,
            "depth": 12,
            "d_patch": d_patch,
            "dropout": 0.05,
            "time_every": 4,
            "d_bottleneck": latent_patch_dim,
            "use_flash_attention": use_flash_attention,
            "dtype": dtype,
        }

        self.tokenizer = Dreamer4TokenizerMAE(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            in_dim=in_dim,
            encoder_kwargs=enc_kwargs,
            decoder_kwargs=dec_kwargs,
            dtype=dtype,
            rngs=rngs,
        )
        self.action_embed = ActionEncoder(
            d_model=latent_action_dim,
            n_keyboard=num_actions,
            is_discrete=is_action_discrete,
            rngs=rngs
        )

        self.dynamics = Dynamics(
            d_model=self.dyna_dim,
            d_bottleneck=latent_patch_dim,
            d_spatial=latent_patch_dim,
            n_spatial=num_patch_latents,
            n_register=dyna_num_registers,
            n_agent=dyna_num_agents,
            n_heads=self.dyna_num_heads,
            depth=self.dyna_num_blocks,
            k_max=dyna_kmax,
            dropout=self.dropout,
            dtype=dtype,
            use_flash_attention=self.use_flash_attention,
            rngs=rngs,
        )

    def __call__(
        self,
        batch: Dict[str, jax.Array],
        step_BT: jax.Array,
        sigma_BT: jax.Array,
    ) -> Dict[str, jax.Array]:
        videos_BTHWC = batch["videos"]
        H, W = videos_BTHWC.shape[2:4]

        assert self.action_embed is not None
        action_embeddings_BT1L = self.action_embed(batch["actions"]).reshape(
            *batch["actions"].shape[:2], 1, self.latent_action_dim
        )

        rng, _rng = jax.random.split(batch["rng"])
        batch['rng'] = _rng
        z_BTNL = self.encode(batch)
        z_BTNL = jax.lax.stop_gradient(z_BTNL)
        z_corrupt_BTNL = self.target(z_BTNL, sigma_BT)
        pred_latents_BTNL, denoise_t = self.dyn(
            z_corrupt_BTNL,
            action_embeddings_BT1L,
            step_BT, sigma_BT
        )

        return pred_latents_BTNL, denoise_t

    def encode(self, batch: Dict[str, jax.Array], *,
               rngs: nnx.Rngs) -> Dict[str, jax.Array]:

        tokenizer_outputs = self.tokenizer.mask_and_encode(
            batch, training=False, rngs=rngs,
        )
        return tokenizer_outputs['z']

    def dyn(self, z_BTNL: jax.Array, actions: jax.Array, step_BT, sigma_BT) -> jax.Array:

        action_embeddings_BT1L = self.action_embed(actions).reshape(
            *actions.shape[:2], 1, self.latent_action_dim
        )

        pred_latents_BTNL, denoise_t = self.dynamics(
            action_embeddings_BT1L,
            step_BT,
            sigma_BT,
            z_BTNL
        )
        return pred_latents_BTNL, denoise_t

    def target(self, z_BTNL: jax.Array, sigma_BT, *,
               rngs: nnx.Rngs) -> jax.Array:
        zbar_BTNL = rngs.normal(z_BTNL.shape, dtype=z_BTNL.dtype)

        noisy_latents_BTNL = (1.0 - sigma_BT)[..., None, None]*zbar_BTNL + sigma_BT[..., None, None]*z_BTNL

        return noisy_latents_BTNL

    def _make_tau_schedule(self, d):
        inv = int(round(1.0/float(d)))
        K = inv
        scale = self.dyna_kmax // K
        tau = [i/K for i in range(K)] + [1.0]
        tau_idx = [i*scale for i in range(K)] + [self.dyna_kmax]
        dt = 1.0 / K
        return jnp.Array(tau), jnp.Array(tau_idx), dt

    def sample(
        self,
        batch: Dict[str, jax.Array],
        context_len: int,
        diffusion_steps: int = 4,
        diffusion_corrupt_context_factor: float = 0.1,
    ) -> jax.Array:
        """
        Autoregressively samples up to `seq_len` future frames.

        - Input frames are tokenized once.
        - Future frames are generated autoregressively in latent space.
        - The context frames are corrupted by noise like in Dreamer4
          section 3.2.
        - All frames are decoded in a single pass.

        Dimension keys:
            B: batch size
            T: number of input (conditioning) frames
            N: number of patches per frame
            L: latent dimension
            S: sequence length
            H: height
            W: width
        """
        # --- Encode videos and actions ---
        videos_BTHWC = batch["videos"]
        H, W = videos_BTHWC.shape[2:4]
        rng, _rng_mask, _rng_noise_full = jax.random.split(batch["rng"], 3)
        tokenizer_outputs = self.tokenizer.mask_and_encode(
            videos_BTHWC, training=False, rngs=_rng_mask
        )
        token_latents_BTNL = tokenizer_outputs["z"]
        B, T, N, L = token_latents_BTNL.shape

        context_len = min(T-1, context_len)

        dynamics_state = nnx.state(self.dynamics)

        assert self.action_embed is not None
        latent_actions_BT1L = self.action_embed(batch["actions"]).reshape(
            *batch["actions"].shape[:2], 1, self.latent_action_dim
        )

        tau, tau_idx, dt = self._make_tau_schedule(1.0/diffusion_steps)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def denoise_step_fn(carry: tuple[jax.Array, jax.Array, jax.Array],
                            tau_param: tuple[jax.Array, jax.Array]):
            z_BPNL, latent_actions_BP1L, frame_idx = carry

            step_tau, step_tau_idx = tau_param

            dynamics = Dynamics(
                d_model=self.dyna_dim,
                d_bottleneck=self.latent_patch_dim,
                d_spatial=self.latent_patch_dim,
                n_spatial=self.num_patch_latents,
                n_register=self.dyna_num_registers,
                n_agent=self.dyna_num_agents,
                n_heads=self.dyna_num_heads,
                depth=self.dyna_num_blocks,
                k_max=self.dyna_kmax,
                dropout=self.dropout,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                rngs=self.rngs,
            )

            nnx.update(dynamics, dynamics_state)

            emax = int(round(math.log2(self.dyna_kmax)))
            step_idxs = jnp.full(*z_BPNL.shape[:2], emax)
            step_id_BT = step_idxs.at[:, -1].set(
                int(round(math.log2(diffusion_steps))))

            signal_id_BT = jnp.full(*z_BPNL.shape[:2], self.dyna_kmax - 1)
            signal_id_BT = signal_id_BT.at[:, -1].set(step_tau_idx)

            pred_BTNL, _ = dynamics(latent_actions_BP1L, step_id_BT,
                                    signal_id_BT, z_BPNL)
            z_cur_B1NL = z_BPNL[:, -1:, :, :]
            pred_cur_B1NL = pred_BTNL[:, -1:, :, :]

            denom = max(1e-4, 1.0 - step_tau)
            b = (z_cur_B1NL.float() - pred_cur_B1NL.float()) / denom
            z_cur_B1NL = (z_cur_B1NL.float() + b*dt).to(self.dtype)
            z_BTNL = z_BPNL.at[:, -1, :, :].set(z_cur_B1NL)
            new_carry = (z_BTNL, latent_actions_BP1L)
            return new_carry

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def autoregressive_step_fn(carry: tuple[jax.Array, jax.Array],
                                   frame_index: jax.Array):
            latents_BSNL, rng = carry
            rng, _rng_noise_context = jax.random.split(rng)

            token_B1NL = jax.random.normal(_rng_noise_context, (B, 1, N, L))
            latents_BSNL = jnp.concatentate([latents_BSNL, token_B1NL], axis=1)
            actions_BP1L = latent_actions_BT1L[:, frame_index+1]

            carry_denoise = (latents_BSNL, actions_BP1L, frame_index)

            final_carry_denoise = denoise_step_fn(
                carry_denoise, (tau, tau_idx))

            latents_BSNL = final_carry_denoise[0]

            new_carry = (latents_BSNL, rng)
            return new_carry

        rng, _rng_sample = jax.random.split(rng)

        token_latents_BSNL = token_latents_BTNL[:, :context_len]

        initial_carry = (token_latents_BSNL, _rng_sample)
        final_carry = autoregressive_step_fn(
            initial_carry, jnp.arange(context_len, T))
        final_latents_BSNL = final_carry[0]

        final_videos_BSHWC = self.tokenizer.decode(
            final_latents_BSNL, video_hw=(H, W))

        return final_videos_BSHWC


# FIXME (f.srambical): add conversion script for old checkpoints
def restore_genie_components(
    optimizer: nnx.ModelAndOptimizer,
    sharding: jax.sharding.NamedSharding,
    rng: jax.Array,
    tokenizer_type: str,
    args,
) -> nnx.ModelAndOptimizer:
    """Restore pre-trained Genie components"""
    rng_tokenizer, rng_lam = jax.random.split(rng)
    rngs_tokenizer = nnx.Rngs(rng_tokenizer)
    rngs_lam = nnx.Rngs(rng_lam)

    tx = optimizer.tx
    model = optimizer.model
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore,
        ocp.handlers.PyTreeCheckpointHandler
    )

    checkpoint_options = ocp.CheckpointManagerOptions(
        step_format_fixed_length=6,
    )
    tokenizer_checkpoint_manager = ocp.CheckpointManager(
        directory=args.tokenizer_checkpoint,
        options=checkpoint_options,
        handler_registry=handler_registry,
    )

    tokenizer_args = Args()
    tokenizer_args.image_height = args.image_height
    tokenizer_args.image_width = args.image_width
    tokenizer_args.max_mask_ratio = 0.0

    dummy_tokenizer, rng = build_model(tokenizer_args, rng)

    dummy_tokenizer_optimizer = nnx.ModelAndOptimizer(dummy_tokenizer, tx)
    dummy_tokenizer_optimizer_state = nnx.state(dummy_tokenizer_optimizer)
    abstract_sharded_tokenizer_optimizer_state = _create_abstract_sharded_pytree(
        dummy_tokenizer_optimizer_state, sharding
    )
    restored_tokenizer = tokenizer_checkpoint_manager.restore(
        step=tokenizer_checkpoint_manager.latest_step(),
        args=ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(  # type: ignore
                abstract_sharded_tokenizer_optimizer_state  # type: ignore
            ),
        ),
    )["model_state"]
    nnx.update(dummy_tokenizer_optimizer.model, restored_tokenizer.model)
    model.tokenizer = dummy_tokenizer_optimizer.model
    tokenizer_checkpoint_manager.close()

    # Reinitialize the optimizer states
    optimizer = nnx.ModelAndOptimizer(model, tx)
    return optimizer


def _create_abstract_sharded_pytree(
    pytree_template: nnx.GraphState, sharding_spec: jax.sharding.NamedSharding
) -> jax.Array:
    """Replaces arrays in a pytree with ShapeDtypeStructs having the given sharding."""

    def map_fn(leaf_template):
        if hasattr(leaf_template, "shape") and hasattr(leaf_template, "dtype"):
            return jax.ShapeDtypeStruct(
                leaf_template.shape, leaf_template.dtype, sharding=sharding_spec
            )
        return leaf_template

    return jax.tree_util.tree_map(map_fn, pytree_template)
