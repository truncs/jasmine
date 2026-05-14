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
    original_fps: int = 60
    target_fps: int = 10
    target_width: int = 64
    chunk_size: int = 160
    chunks_per_file: int = 100
    fields: str = 'left_camera,action'


def preprocess_npz(input_dir, original_fps,
                   target_fps, chunk_size, target_width, fields):
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
        chunks = defaultdict(list)

        for fname in selected_files:
            data = np.load(os.path.join(input_dir, fname))

            is_terminal = data['is_terminal']
            terminal_idx = np.where(is_terminal == True)[0]
            terminal_idx = terminal_idx + 1


            for field in fields:
                chunk = np.split(data[field], terminal_idx)
                chunks[field].extend(chunk)

        return chunks
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")
        return chunks


def save_chunks(file_idx, chunks_per_file, output_dir, chunks):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    key = list(chunks.keys())[0]
    
    while len(chunks[key]) >= chunks_per_file:

        chunk_batch = {k: v[:chunks_per_file] for k, v in chunks.items()}
        chunks = {k: v[chunks_per_file:] for k, v in chunks.items()}

        episode_path = os.path.join(output_dir, f"data_{file_idx:04d}.array_record")
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        seq_lens = []
        for idx, chunk in enumerate(chunk_batch[key]):
            seq_len = chunk.shape[0]
            seq_lens.append(seq_len)

            chunk_record = {
                "sequence_length": seq_len,
            }

            for k, v in chunk_batch.items():
                if len(v[idx].shape) == 3:
                    chunk_record[key] = v[idx].tobytes()
                else:
                    chunk_record[key] = v[idx]

            writer.write(pickle.dumps(chunk_record))
        writer.close()
        file_idx += 1
        metadata.append(
            {
                "path": episode_path,
                "num_chunks": len(chunk_batch[key]),
                "avg_seq_len": np.mean(seq_lens),
            }
        )
        print(f"Created {episode_path} with {len(chunk_batch[key])} video chunks")

    return metadata, file_idx, chunks


def save_split(pool_args, chunks_per_file, output_path):
    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")

    chunks = defaultdict(list)
    
    file_idx = 0
    results = []
    for bucket_idx in range(0, len(pool_args), num_processes):
        args_batch = pool_args[bucket_idx:bucket_idx + num_processes]
        with mp.Pool(processes=num_processes) as pool:
            for chunk in pool.starmap(preprocess_npz, args_batch):
                for key, value in chunk.items():
                    chunks[key].extend(value)
        results_batch, file_idx, chunks = save_chunks(
            file_idx, chunks_per_file, output_path, chunks
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
