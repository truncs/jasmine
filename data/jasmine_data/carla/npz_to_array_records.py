import os
import numpy as np
import pickle
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from array_record.python.array_record_module import ArrayRecordWriter

from PIL import Image
import tyro
import json
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
    original_fps: int = 10
    target_fps: int = 10
    target_width: int = 64
    chunk_size: int = 160
    chunks_per_file: int = 100
    fields: str = 'left_camera,action,reward,route_map'


def preprocess_npz(input_dir, original_fps,
                   target_fps, chunk_size, target_width, fields):
    print(f"Processing npzs in {input_dir}")
    try:
        npz_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(".npz")],
        )

        if not npz_files:
            print(f"No npz files found in {input_dir}")
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
        route_map_chunks = []
        first_person_chunks = []
        act_chunks = []

        for fname in selected_files:
            abs_fname = os.path.join(input_dir, fname)
            print(f'Processing file: {abs_fname}')
            data = np.load(abs_fname)

            is_terminal = data['is_terminal']
            terminal_idx = np.where(is_terminal == True)[0]
            terminal_idx = terminal_idx + 1
            obs_current_chunks = np.split(data['left_camera'], terminal_idx)
            route_map_chunk = np.split(data['route_map'], terminal_idx)
            first_person_chunk = np.split(data['first_person'], terminal_idx)
            act_current_chunks = np.split(data['action'], terminal_idx)

            obs_chunks.extend(obs_current_chunks)
            act_chunks.extend(act_current_chunks)
            route_map_chunks.extend(route_map_chunk)
            first_person_chunks.extend(first_person_chunk)

            print(f'Chunks for file {abs_fname} are {terminal_idx}')
        return obs_chunks, act_chunks, route_map_chunks, first_person_chunks
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")
        return ([], [], [], [])


def save_split(pool_args, chunks_per_file, output_path):
    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")
    obs_chunks = []
    act_chunks = []
    route_map_chunks = []
    first_person_chunks = []
    file_idx = 0
    results = []
    for bucket_idx in range(0, len(pool_args), num_processes):
        args_batch = pool_args[bucket_idx:bucket_idx + num_processes]
        with mp.Pool(processes=num_processes) as pool:
            for chunk in pool.starmap(preprocess_npz, args_batch):
                obs_chunks.extend(chunk[0])
                act_chunks.extend(chunk[1])
                route_map_chunks.extend(chunk[2])
                first_person_chunks.extend(chunk[3])
        results_batch, file_idx, chunks, _ = save_chunks(
            file_idx, chunks_per_file, output_path, obs_chunks, act_chunks, route_map_chunks, first_person_chunks,
        )
        results.extend(results_batch)

    if len(chunks) > 0 and len(chunks[list(chunks.keys())[0]]) > 0:
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


    assert args.fields is not None, 'Need to pass field names'

    args.fields = args.fields.split(',')

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
            args.fields,
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
