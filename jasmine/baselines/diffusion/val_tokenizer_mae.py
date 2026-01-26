import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.98")

from dataclasses import dataclass, field
from typing import cast, Optional

import einops
import itertools
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
import optax
import orbax.checkpoint as ocp
import numpy as np
import dm_pix as pix
import jax
import jax.numpy as jnp
import tyro
import wandb
import grain
import flax.nnx as nnx
import lpips_jax

from jasmine.models.tokenizer import TokenizerMAE
from jasmine.models.dreamer4 import Encoder, Decoder
from jasmine.utils.dataloader import get_dataloader
from jasmine.utils.preprocess import patchify, unpatchify
from jasmine.utils.train_utils import (
    count_parameters_by_component,
    print_mem_stats,
    print_compiled_memory_stats,
    print_compiled_cost_analysis,
)


@dataclass
class Args:
    # Experiment
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64
    data_dir: str = ""
    # Data
    batch_size: int = 48
    val_steps: int = 100
    # Tokenizer
    model_dim: int = 512
    ffn_dim: int = 2048
    latent_dim: int = 32
    num_latents: int = 128
    patch_size: int = 16
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    max_mask_ratio: float = 0.9
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True
    # Logging
    log: bool = True
    entity: str = ""
    project: str = ""
    name: str = "val_tokenizer_mae"
    tags: list[str] = field(default_factory=lambda: ["tokenizer", "mae", "val"])
    ckpt_dir: str = ""
    ckpt_path: Optional[str] = None
    restore_step: Optional[int] = None
    num_workers: int = 8
    prefetch_buffer_size: int = 1


class Dreamer4TokenizerMAE(nnx.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_dim: int,
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dtype = dtype

        self.encoder = Encoder(**encoder_kwargs, rngs=rngs)
        self.decoder = Decoder(**decoder_kwargs, rngs=rngs)

    def __call__(self, batch: dict, training: bool = True) -> dict:
        rngs = batch.get("rng", None)
        videos = batch["videos"]
        B, T, H, W, C = videos.shape
        
        patches = patchify(videos, self.patch_size) # (B, T, Hp, Wp, D)
        B, T, Hp, Wp, D = patches.shape
        patches_flat = patches.reshape(B, T, Hp*Wp, D)
        
        mae_rng = nnx.Rngs(mae=rngs) if rngs is not None else None
        
        z_latents, (mask, keep) = self.encoder(patches_flat, rngs=mae_rng)
        
        recon_patches_flat = self.decoder(z_latents)
        
        recon_videos = unpatchify(recon_patches_flat, self.patch_size, H, W)
        
        outputs = {
            "recon": recon_videos,
            "z": z_latents,
            "mask": mask
        }
        return outputs

def build_model(args: Args, rng: jax.Array) -> tuple[Dreamer4TokenizerMAE, jax.Array]:
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    
    num_patches = (args.image_height // args.patch_size) * (args.image_width // args.patch_size)
    d_patch = args.image_channels * args.patch_size ** 2

    enc_kwargs = {
        "d_model": 512, 
        "n_latents": args.num_latents, 
        "n_patches": num_patches, 
        "n_heads": 8, 
        "depth": 8, 
        "dropout": 0.0, # No dropout during eval
        "d_bottleneck": args.latent_dim,
        "mae_p_min": 0.0, 
        "mae_p_max": 0.9, 
        "time_every": 4,
        "d_patch": d_patch, 
        "use_flash_attention": args.use_flash_attention,
        "dtype": args.dtype,
    }
    
    dec_kwargs = {
        "d_model": 512, 
        "n_heads": 8, 
        "n_patches": num_patches, 
        "n_latents": args.num_latents, 
        "depth": 12,
        "d_patch": d_patch, 
        "dropout": 0.0, # No dropout during eval
        "time_every": 4,
        "d_bottleneck": args.latent_dim,
        "use_flash_attention": args.use_flash_attention,
        "dtype": args.dtype,
    }

    tokenizer = Dreamer4TokenizerMAE(
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        in_dim=args.image_channels,
        encoder_kwargs=enc_kwargs,
        decoder_kwargs=dec_kwargs,
        dtype=args.dtype,
        rngs=rngs,
    )
    return tokenizer, rng


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    return mesh, replicated_sharding, videos_sharding


def build_dataloader(args: Args, data_dir: str) -> grain.DataLoaderIterator:
    image_shape = (args.image_height, args.image_width, args.image_channels)
    array_record_files = [
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
        if x.endswith(".array_record")
    ]
    grain_dataloader = get_dataloader(
        array_record_files,
        args.seq_len,
        args.batch_size,
        *image_shape,
        num_workers=args.num_workers,
        prefetch_buffer_size=args.prefetch_buffer_size,
        seed=args.seed,
    )
    return grain_dataloader


def build_checkpoint_manager(args: Args) -> ocp.CheckpointManager:
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
    checkpoint_manager = ocp.CheckpointManager(
        args.ckpt_dir,
        handler_registry=handler_registry,
    )
    return checkpoint_manager


def restore_model_state(
    args: Args,
    checkpoint_manager: ocp.CheckpointManager,
    model: Dreamer4TokenizerMAE,
) -> int:
    restore_step = args.restore_step if args.restore_step is not None else checkpoint_manager.latest_step()
    if restore_step is None:
        print("No checkpoint found.")
        return 0

    # Get the current model state to use as a template for partial restore.
    # This ensures that any weights missing from the checkpoint remain 
    # as their current real arrays instead of becoming ShapeDtypeStructs.
    model_state = nnx.state(model)
    
    restore_args = ocp.args.Composite(
        model_state=ocp.args.PyTreeRestore(model_state, partial_restore=True),
    )
    
    restored = checkpoint_manager.restore(restore_step, args=restore_args)
    
    # Extract model state from common checkpoint structures
    if "model_state" in restored:
        model_state = restored["model_state"]
        # If it was saved with an optimizer (nnx.ModelAndOptimizer), 
        # the model state is under "model" key in the optimizer state.
        if "model" in model_state:
            model_state = model_state["model"]
    else:
        model_state = restored

    nnx.update(model, model_state)
    print(f"Restored model state from step {restore_step}")
    return restore_step


def restore_model_from_path(
    path: str,
    model: Dreamer4TokenizerMAE,
) -> None:
    # Get current model state as template
    model_state = nnx.state(model)
    
    # Try to load as a composite checkpoint (which is what manager.save creates)
    # We use a temporary CheckpointManager to handle the restoration from a specific path
    # by treating the parent of 'path' as the ckpt_dir and the basename as the 'step'.
    # This is a bit of a hack but works with the existing Orbax structure.
    
    parent_dir = os.path.dirname(os.path.abspath(path))
    target_name = os.path.basename(os.path.abspath(path))
    
    # Check if target_name is a number (step) or a literal folder name
    try:
        step = int(target_name)
        base_dir = parent_dir
    except ValueError:
        # If it's not an integer, we might need a different approach.
        # However, Orbax CheckpointManager usually expects integer steps.
        # If the user points to a folder like "best_model", we can try to
        # use StandardCheckpointer to restore from a literal path.
        checkpointer = ocp.StandardCheckpointer()
        
        # We need to know if it's a composite or a direct PyTree
        # Most of our checkpoints are composites with "model_state"
        try:
            # Try composite restore
            restore_args = ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(model_state, partial_restore=True),
            )
            restored = checkpointer.restore(path, args=restore_args)
            if "model_state" in restored:
                model_state = restored["model_state"]
                if "model" in model_state:
                    model_state = model_state["model"]
            else:
                model_state = restored
        except Exception:
            # Fallback to direct PyTree restore
            restore_args = ocp.args.PyTreeRestore(model_state, partial_restore=True)
            model_state = checkpointer.restore(path, args=restore_args)
        
        nnx.update(model, model_state)
        print(f"Restored model state from path: {path}")
        return

    # If it was an integer step, use the manager logic
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    manager = ocp.CheckpointManager(base_dir, handler_registry=handler_registry)
    
    restore_args = ocp.args.Composite(
        model_state=ocp.args.PyTreeRestore(model_state, partial_restore=True),
    )
    restored = manager.restore(step, args=restore_args)
    
    if "model_state" in restored:
        model_state = restored["model_state"]
        if "model" in model_state:
            model_state = model_state["model"]
    else:
        model_state = restored

    nnx.update(model, model_state)
    print(f"Restored model state from step {step} at {base_dir}")
    manager.close()


def main(args: Args) -> None:
    # Initialize JAX distributed if needed
    try:
        jax.distributed.initialize()
    except Exception:
        pass

    num_devices = jax.device_count()
    print(f"Running on {num_devices} devices.")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by "
            f"number of devices {num_devices}."
        )

    rng = jax.random.key(args.seed)

    # --- Initialize model ---
    model, rng = build_model(args, rng)

    _, params, _ = nnx.split(model, nnx.Param, ...)
    param_counts = count_parameters_by_component(params)

    if args.log and jax.process_index() == 0:
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.name,
            tags=args.tags,
            config=args,
        )
        wandb.config.update({"model_param_count": param_counts})

    print("Parameter counts:")
    print(param_counts)

    # --- Restore checkpoint ---
    if args.ckpt_path:
        restore_model_from_path(args.ckpt_path, model)
    elif args.ckpt_dir:
        checkpoint_manager = build_checkpoint_manager(args)
        restore_model_state(args, checkpoint_manager, model)
    else:
        print("No checkpoint provided. Running with random initialization.")

    # --- Mesh and Sharding ---
    _, replicated_sharding, videos_sharding = build_mesh_and_sharding(num_devices)
    
    # Shard model params
    model_state = nnx.state(model)
    sharded_model_state = jax.lax.with_sharding_constraint(model_state, replicated_sharding)
    nnx.update(model, sharded_model_state)

    # --- Dataloader ---
    val_iterator = build_dataloader(args, args.data_dir)
    dataloader_val = (
        {
            "videos": jax.make_array_from_process_local_data(
                videos_sharding, elem["videos"]
            ),
        }
        for elem in val_iterator
    )

    # LPIPS evaluator
    lpips_evaluator = lpips_jax.LPIPSEvaluator(replicate=False, net='alexnet')

    def tokenizer_loss_fn(
        model: Dreamer4TokenizerMAE, patch_size: int, inputs: dict, lpips_evaluator
    ) -> tuple[jax.Array, tuple[jax.Array, dict]]:
        # Avoid in-place mutation of the inputs dict
        videos_uint8 = inputs["videos"]
        gt = jnp.asarray(videos_uint8, dtype=jnp.float32) / 255.0
        
        # Prepare inputs for the model
        model_inputs = {**inputs, "videos": gt.astype(args.dtype)}
        
        outputs = model(model_inputs, training=False)
        recon = outputs["recon"].astype(jnp.float32)

        # Mask handling
        mask = outputs["mask"]
        P = patch_size
        H, W = gt.shape[2:4]
        hn, wn = H // P, W // P
        pixel_mask = mask.reshape(mask.shape[0], mask.shape[1], hn, wn, 1)
        pixel_mask = jnp.repeat(pixel_mask, P, axis=2)
        pixel_mask = jnp.repeat(pixel_mask, P, axis=3)
        pixel_mask = pixel_mask.astype(jnp.float32)
        vis_mask = 1.0 - pixel_mask

        # Masked MSE
        sq_err = jnp.square(gt - recon) * vis_mask
        mse = sq_err.sum() / jnp.maximum(vis_mask.sum() * gt.shape[-1], 1.0)

        # Masked LPIPS
        gt_masked = gt * vis_mask
        recon_masked = recon * vis_mask
        
        lpips = lpips_evaluator(jax.lax.collapse(gt_masked, 0, 2),
                                jax.lax.collapse(recon_masked, 0, 2)).mean()

        gt_clipped = gt.clip(0, 1).reshape(-1, *gt.shape[2:])
        recon_clipped = recon.clip(0, 1).reshape(-1, *recon.shape[2:])
        psnr = jnp.asarray(pix.psnr(gt_clipped, recon_clipped)).mean()
        ssim = jnp.asarray(pix.ssim(gt_clipped, recon_clipped)).mean()

        # Simple weighted sum for loss if desired, or just MSE + LPIPS
        loss = mse + 0.3 * lpips

        metrics = dict(
            mse=mse,
            lpips=lpips,
            psnr=psnr,
            ssim=ssim,
            loss=loss,
        )

        return loss, (recon, metrics)

    @nnx.jit(static_argnums=(1, 2,))
    def val_step(
        model: Dreamer4TokenizerMAE, lpips_evaluator, patch_size: int, inputs: dict
    ) -> tuple[jax.Array, jax.Array, dict]:
        model.eval()
        (loss, (recon, metrics)) = tokenizer_loss_fn(model, patch_size, inputs, lpips_evaluator)
        return loss, recon, metrics

    # --- Run Validation ---
    print(f"Starting validation for {args.val_steps} steps...")
    loss_per_step = []
    metrics_per_step = []
    last_batch = None
    last_recon = None
    
    for i, batch in enumerate(dataloader_val):
        if i >= args.val_steps:
            break
        
        rng, _rng_mask = jax.random.split(rng, 2)
        batch["rng"] = _rng_mask
        
        loss, recon, metrics = val_step(model, lpips_evaluator, args.patch_size, batch)
        
        loss_per_step.append(loss)
        metrics_per_step.append(metrics)
        last_batch = batch
        last_recon = recon
        
        if (i + 1) % 10 == 0:
            print(f"Validated {i + 1}/{args.val_steps} steps. Current MSE: {metrics['mse']:.6f}")

    # Aggregated metrics
    if not metrics_per_step:
        print("No validation steps performed.")
        return

    val_metrics = {
        f"val_{key}": np.mean([float(m[key]) for m in metrics_per_step])
        for key in metrics_per_step[0].keys()
    }
    val_metrics["val_loss"] = np.mean(loss_per_step)

    print("\nValidation Results:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.6f}")

    # --- Logging Images ---
    if args.log and jax.process_index() == 0 and last_batch is not None:
        wandb.log(val_metrics)
        
        gt_seq = last_batch["videos"][0].astype(jnp.float32) / 255.0
        recon_seq = last_recon[0].clip(0, 1)
        
        comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
        comparison_seq = einops.rearrange(
            comparison_seq * 255, "t h w c -> h (t w) c"
        )
        
        log_images = dict(
            val_image=wandb.Image(np.asarray(gt_seq[0])),
            val_recon=wandb.Image(np.asarray(recon_seq[0])),
            val_true_vs_recon=wandb.Image(
                np.asarray(comparison_seq.astype(np.uint8))
            ),
        )
        wandb.log(log_images)
        wandb.finish()

    if 'checkpoint_manager' in locals():
        checkpoint_manager.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
