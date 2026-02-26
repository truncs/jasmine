import os
import numpy as np
from PIL import Image
import tyro
from dataclasses import dataclass
import json
import multiprocessing as mp
from data.jasmine_data.utils import save_chunks


@dataclass
class Args:
    input_path: str
    output_path: str
    env_name: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    multigame: bool = False
    original_fps: int = 60
    target_fps: int = 10
    target_width: int = 64
    chunk_size: int = 160
    chunks_per_file: int = 100


def preprocess_npz(input_dir, original_fps,
                   target_fps, chunk_size, target_width):
    print(f"Processing PNGs in {input_dir}")
    try:
        npz_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(".npz")],
        )

        if not npz_files:
            print(f"No PNG files found in {input_dir}")
            return []

        # Downsample indices
        n_total = len(npz_files)
        if original_fps == target_fps:
            selected_indices = np.arange(n_total)
        else:
            n_target = int(np.floor(n_total * target_fps / original_fps))
            selected_indices = np.linspace(0, n_total - 1, n_target, dtype=int)

        selected_files = [npz_files[i] for i in selected_indices]

        # Load images
        obs_chunks = []
        act_chunks = []
        reward_chunks = []

        for fname in selected_files:
            data = np.load(os.path.join(input_dir, fname))
            obs_current_chunk = data['image']
            act_current_chunk = data['action']
            reward_current_chunk = data['reward']

            obs_chunks.append(obs_current_chunk)
            act_chunks.append(act_current_chunk)
            reward_chunks.append(reward_current_chunk)

        return obs_chunks, act_chunks, reward_chunks
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")
        return ([], [])


def save_split(pool_args, chunks_per_file, output_path):
    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")
    obs_chunks = []
    act_chunks = []
    reward_chunks = []

    file_idx = 0
    results = []
    for bucket_idx in range(0, len(pool_args), num_processes):
        args_batch = pool_args[bucket_idx : bucket_idx + num_processes]
        with mp.Pool(processes=num_processes) as pool:
            for chunk in pool.starmap(preprocess_npz, args_batch):
                obs_chunks.append(chunk[0])
                act_chunks.append(chunk[1])
                reward_chunks.append(chunk[2])
        results_batch, file_idx, chunks, _ = save_chunks(
            file_idx, chunks_per_file, output_path, obs_chunks, act_chunks
        )
        results.extend(results_batch)

    if len(obs_chunks) > 0:
        print(
            f"Warning: Dropping {len(chunks)} chunks for consistent number of chunks per file.",
            "Consider changing the chunk_size and chunks_per_file parameters to prevent data-loss.",
        )

    print(f"Done processing files. Saved to {output_path}")
    return results


def main():
    args = tyro.cli(Args)
    print(f"Output path: {args.output_path}")
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert np.isclose(total_ratio, 1.0), "Ratios must sum to 1.0"

    directories = [
        os.path.join(args.input_path, d)
        for d in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, d))
    ]
    if args.multigame:
        episodes = [
            os.path.join(game, d) for game in directories for d in os.listdir(game)
        ]
    else:
        episodes = directories

    n_total = sum([len(os.listdir(episode)) for episode in episodes])
    n_train = round(n_total * args.train_ratio)
    n_val = round(n_total * args.val_ratio)

    pool_args_train = []
    pool_args_val = []
    pool_args_test = []

    train_counter = 0
    val_counter = 0
    np.random.shuffle(episodes)
    for episode in episodes:
        pool_arg = (
            episode,
            args.original_fps,
            args.target_fps,
            args.chunk_size,
            args.target_width,
        )
        n_frames = len(os.listdir(episode))
        if train_counter < n_train:
            pool_args_train.append(pool_arg)
            train_counter += n_frames
        elif val_counter < n_val:
            pool_args_val.append(pool_arg)
            val_counter += n_frames
        else:
            pool_args_test.append(pool_arg)

    train_episode_metadata = save_split(
        pool_args_train, args.chunks_per_file, os.path.join(args.output_path, "train")
    )
    val_episode_metadata = save_split(
        pool_args_val, args.chunks_per_file, os.path.join(args.output_path, "val")
    )
    test_episode_metadata = save_split(
        pool_args_test, args.chunks_per_file, os.path.join(args.output_path, "test")
    )

    # Calculate total number of chunks
    total_chunks = sum(
        ep["num_chunks"]
        for ep in train_episode_metadata + val_episode_metadata + test_episode_metadata
    )

    print("Done converting png to array_record files")

    print(f"Total number of chunks: {total_chunks}")

    metadata = {
        "env": args.env_name,
        "total_chunks": total_chunks,
        "avg_episode_len_train": np.mean(
            [ep["avg_seq_len"] for ep in train_episode_metadata]
        ),
        "avg_episode_len_val": np.mean(
            [ep["avg_seq_len"] for ep in val_episode_metadata]
        ),
        "avg_episode_len_test": np.mean(
            [ep["avg_seq_len"] for ep in test_episode_metadata]
        ),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Done.")


if __name__ == "__main__":
    main()
