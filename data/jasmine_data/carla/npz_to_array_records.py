import os
import numpy as np
import tyro
import json
import multiprocessing as mp
import pickle
from array_record.python.array_record_module import ArrayRecordWriter
from dataclasses import dataclass
from PIL import Image


def save_chunks(file_idx, chunks_per_file, output_dir, obs_chunks, act_chunks=None, route_map_chunks=None, first_person_chunks=None):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    while len(obs_chunks) >= chunks_per_file:
        chunk_batch = obs_chunks[:chunks_per_file]
        obs_chunks = obs_chunks[chunks_per_file:]
        route_map_batch = route_map_chunks[:chunks_per_file]
        route_map_chunks = route_map_chunks[chunks_per_file:]
        first_person_batch = first_person_chunks[:chunks_per_file]
        first_person_chunks = first_person_chunks[chunks_per_file:]

        act_chunk_batch = None
        if act_chunks:
            act_chunk_batch = act_chunks[:chunks_per_file]
            act_chunks = act_chunks[chunks_per_file:]
        episode_path = os.path.join(output_dir, f"data_{file_idx:04d}.array_record")
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        seq_lens = []
        for idx, chunk in enumerate(chunk_batch):
            seq_len = chunk.shape[0]
            seq_lens.append(seq_len)
            chunk_record = {
                "raw_video": chunk.tobytes(),
                "sequence_length": seq_len,
                "route_map": route_map_batch[idx].tobytes(),
                "first_person": first_person_batch[idx].tobytes(),
            }
            if act_chunk_batch:
                assert len(chunk) == len(
                    act_chunk_batch[idx]
                ), f"Observation data length and action sequence length do not match: {len(chunk)} != {len(act_chunk_batch[idx])}"
                chunk_record["actions"] = act_chunk_batch[idx]
            writer.write(pickle.dumps(chunk_record))
        writer.close()
        file_idx += 1
        metadata.append(
            {
                "path": episode_path,
                "num_chunks": len(chunk_batch),
                "avg_seq_len": np.mean(seq_lens),
            }
        )
        print(f"Created {episode_path} with {len(chunk_batch)} video chunks")

    return metadata, file_idx, obs_chunks, act_chunks


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
        args_batch = pool_args[bucket_idx : bucket_idx + num_processes]
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
