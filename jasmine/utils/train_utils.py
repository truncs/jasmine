import jax
import optax
import flax.nnx as nnx
import operator

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh

from jasmine.utils.dataloader import get_dataloader

def get_lr_schedule(
    lr_schedule: str,
    init_lr: float,
    max_lr: float,
    decay_end: float,
    total_steps: int,
    warmup_steps: int,
    wsd_decay_steps: int,
) -> optax.Schedule:
    supported_schedules = ["wsd", "cos"]
    if lr_schedule == "cos":
        assert (
            warmup_steps <= total_steps
        ), "Warmup steps can't be greater than total steps."
        return optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,  # Note: decay_steps includes the warmup steps, so we need to pass total value
            end_value=decay_end,
        )
    elif lr_schedule == "wsd":
        assert (
            warmup_steps + wsd_decay_steps <= total_steps
        ), "Warmup and decay period is longer than total steps."
        schedules = [
            optax.linear_schedule(
                init_value=init_lr, end_value=max_lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(value=max_lr),
            optax.linear_schedule(
                init_value=max_lr, end_value=decay_end, transition_steps=wsd_decay_steps
            ),
        ]
        boundaries = [warmup_steps, total_steps - wsd_decay_steps]
        return optax.join_schedules(schedules, boundaries)
    else:
        raise ValueError(
            f"Learning rate schedule not supported. Please use one of {supported_schedules}"
        )


def _count_component(component_params):
    """Count total parameters in a component."""
    params_sizes = jax.tree.map(jax.numpy.size, component_params)
    total_parameters = jax.tree.reduce(operator.add, params_sizes)
    return total_parameters


def count_parameters_by_component(params):
    """Count parameters for each component of the model.

    Args:
        params: Model parameters from nnx.split(model, nnx.Param, ...)

    Returns:
        Dictionary with parameter counts for each component
    """
    component_names = list(params.keys())
    print(f"Counting all components: {component_names}")

    counts = {}
    total_params = 0

    for name in component_names:
        component_params = params[name]
        count = _count_component(component_params)
        counts[name] = count
        total_params += count

    counts["total"] = total_params
    return counts


def bytes_to_gb(num_bytes):
    return num_bytes / (1024**3)


def print_compiled_memory_stats(compiled_stats):
    """from: https://github.com/AI-Hypercomputer/maxtext/blob/b18829fbaa48aec7ac350a03e62248e24c6a76b2/MaxText/max_utils.py#L739"""
    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb
    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
    )


def print_compiled_cost_analysis(cost_stats):
    flops = float(cost_stats.get("flops", 0.0))
    bytes_accessed = float(cost_stats.get("bytes accessed", 0.0))
    gb = bytes_to_gb(bytes_accessed) if bytes_accessed else 0.0
    intensity = (flops / bytes_accessed) if bytes_accessed else float("nan")
    print(
        f"FLOPs: {flops:.3e}, Bytes: {bytes_accessed:.3e} ({gb:.1f} GB), "
        f"Intensity: {intensity:.1f} FLOPs/byte"
    )


def print_mem_stats(label: str):
    """from: https://github.com/AI-Hypercomputer/maxtext/blob/7898576359bacde81be25cb3038e348aac1f943b/MaxText/max_utils.py#L713"""
    print(f"\nMemstats: {label}:")
    try:
        for d in jax.local_devices():
            stats = d.memory_stats()
            used = round(stats["bytes_in_use"] / 2**30, 2)
            limit = round(stats["bytes_limit"] / 2**30, 2)
            print(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
    except (RuntimeError, KeyError, TypeError) as ex:
        print(f"\tMemstats unavailable, error: {ex}")


def build_optimizer(model: nnx.Module, lr_schedule: str, init_lr: float, max_lr: float, 
    decay_end: float, num_steps: int, warmup_steps: int, wsd_decay_steps: int, param_dtype: str) -> nnx.ModelAndOptimizer:
    lr_schedule = get_lr_schedule(
        lr_schedule,
        init_lr,
        max_lr,
        decay_end,
        num_steps,
        warmup_steps,
        wsd_decay_steps,
    )
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.9,
        weight_decay=1e-4,
        mu_dtype=param_dtype,  # moments in full precision
    )
    optimizer = nnx.ModelAndOptimizer(model, tx)
    return optimizer


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    return mesh, replicated_sharding, videos_sharding


def shard_optimizer_states(
    optimizer: nnx.ModelAndOptimizer, replicated_sharding: NamedSharding
) -> None:
    model_state = nnx.state(optimizer.model)
    model_sharded_state = jax.lax.with_sharding_constraint(
        model_state, replicated_sharding
    )
    nnx.update(optimizer.model, model_sharded_state)
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, replicated_sharding
    )
    nnx.update(optimizer, optimizer_sharded_state)


def build_dataloader(
    image_height: int,
    image_width: int,
    image_channels: int,
    seq_len: int,
    batch_size: int,
    data_dir: str,
    num_workers: int,
    prefetch_buffer_size: int,
    seed: int,
    num_epochs: Optional[int] = None,
) -> grain.DataLoaderIterator:
    image_shape = (image_height, image_width, image_channels)
    array_record_files = [
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
        if x.endswith(".array_record")
    ]
    grain_dataloader = get_dataloader(
        array_record_files,
        seq_len,
        # NOTE: We deliberately pass the global batch size
        # The dataloader shards the dataset across all processes
        batch_size,
        *image_shape,
        num_workers=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
        seed=seed,
        num_epochs=num_epochs,
    )
    return grain_dataloader


def build_checkpoint_manager(restore_ckpt: bool, save_ckpt: bool, checkpoint_dir: str) -> Optional[ocp.CheckpointManager]:
    if restore_ckpt or save_ckpt:
        handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        handler_registry.add(
            "model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
        )
        handler_registry.add(
            "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
        )
        handler_registry.add(
            "train_dataloader_state",
            grain.checkpoint.CheckpointSave,l
            cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
        )
        handler_registry.add(
            "train_dataloader_state",
            grain.checkpoint.CheckpointRestore,
            cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
        )
        checkpoint_options = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            max_to_keep=3,
            best_fn=lambda m: m["val_psnr"] if "val_psnr" in m else m["psnr"],
            best_mode="max",
            keep_period=keep_period,
            step_format_fixed_length=6,
            cleanup_tmp_directories=True,
        )
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=checkpoint_options,
            handler_registry=handler_registry,
        )
        return checkpoint_manager
    else:
        return None


def restore_checkpoint_if_needed(
    checkpoint_manager: Optional[ocp.CheckpointManager],
    optimizer: nnx.ModelAndOptimizer,
    train_iterator: grain.DataLoaderIterator,
    val_iterator: Optional[grain.DataLoaderIterator],
    restore_step: Optional[int] = None,
) -> tuple[
    int, nnx.ModelAndOptimizer, grain.DataLoaderIterator, grain.DataLoaderIterator
]:
    step = 0
    if checkpoint_manager and restore_step is None:
        restore_step = checkpoint_manager.latest_step()
        
    if restore_ckpt:
        assert checkpoint_manager is not None
        abstract_optimizer = nnx.eval_shape(lambda: optimizer)
        abstract_optimizer_state = nnx.state(abstract_optimizer)
        restore_args = ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(abstract_optimizer_state, partial_restore=True),  # type: ignore
            train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
            )
        if restore_step:
            restored = checkpoint_manager.restore(
                restore_step, args=restore_args)
            restored_optimizer_state = restored["model_state"]
            nnx.update(optimizer, restored_optimizer_state)
            train_iterator = restored["train_dataloader_state"]
        step = restore_step or 0
        print(f"Restored dataloader and model state from step {step}")
    return step, optimizer, train_iterator, val_iterator
