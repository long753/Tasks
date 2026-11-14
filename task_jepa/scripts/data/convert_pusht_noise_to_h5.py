"""Flatten a PushT noise split into an HDF5 dataset."""

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

import h5py
import hdf5plugin
import imageio.v3 as iio
import numpy as np
import torch

from stable_worldmodel.paths import StoragePaths, add_storage_arguments


def convert(src_dir: str | Path, out_path: str | Path) -> None:
    """Flatten one PushT split into the destination dataset."""
    src_dir = Path(src_dir)
    out_path = Path(out_path)
    states = torch.load(
        src_dir / 'states.pth', map_location='cpu', weights_only=False
    ).numpy()
    velocities = torch.load(
        src_dir / 'velocities.pth', map_location='cpu', weights_only=False
    ).numpy()
    rel_actions = torch.load(
        src_dir / 'rel_actions.pth', map_location='cpu', weights_only=False
    ).numpy()
    with (src_dir / 'seq_lengths.pkl').open('rb') as file:
        seq_lengths = [int(value) for value in pickle.load(file)]

    n_ep = len(seq_lengths)
    total = sum(seq_lengths)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as file:
        pixels = file.create_dataset(
            'pixels',
            (total, 224, 224, 3),
            dtype='uint8',
            chunks=(100, 224, 224, 3),
            **hdf5plugin.Blosc(),
        )
        state = file.create_dataset(
            'state', (total, 7), dtype='float32', chunks=(1000, 7)
        )
        block_ori = file.create_dataset(
            'block_ori', (total, 2), dtype='float32', chunks=(1000, 2)
        )
        proprio = file.create_dataset(
            'proprio', (total, 4), dtype='float32', chunks=(1000, 4)
        )
        action = file.create_dataset(
            'action', (total, 2), dtype='float32', chunks=(1000, 2)
        )
        episode_idx = file.create_dataset(
            'episode_idx',
            (total,),
            dtype='int64',
            chunks=(min(1000, total),),
        )
        step_idx = file.create_dataset(
            'step_idx', (total,), dtype='int64', chunks=(min(1000, total),)
        )
        ep_len = file.create_dataset(
            'ep_len', (n_ep,), dtype='int32', chunks=(min(1024, n_ep),)
        )
        ep_offset = file.create_dataset(
            'ep_offset', (n_ep,), dtype='int64', chunks=(min(1024, n_ep),)
        )

        offset = 0
        for index, length in enumerate(seq_lengths):
            frames = iio.imread(src_dir / 'obses' / f'episode_{index:03d}.mp4')
            assert frames.shape[0] == length, (
                f'ep {index}: {frames.shape[0]} frames != seq_len {length}'
            )
            row_slice = slice(offset, offset + length)
            pixels[row_slice] = frames
            state[row_slice] = np.concatenate(
                [states[index, :length, :5], velocities[index, :length, :2]],
                axis=1,
            )
            block_ori[row_slice] = np.stack(
                [
                    np.cos(states[index, :length, 4]),
                    np.sin(states[index, :length, 4]),
                ],
                axis=1,
            )
            proprio[row_slice] = np.concatenate(
                [states[index, :length, :2], velocities[index, :length, :2]],
                axis=1,
            )
            action[row_slice] = rel_actions[index, :length, :2] * 0.01
            episode_idx[row_slice] = index
            step_idx[row_slice] = np.arange(length)
            ep_len[index] = length
            ep_offset[index] = offset
            offset += length

    print(f'wrote {out_path}: {n_ep} episodes, {total} steps')


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Convert a PushT noise split into an HDF5 dataset.'
    )
    add_storage_arguments(parser)
    parser.add_argument('source', type=Path)
    parser.add_argument('dataset_name')
    args = parser.parse_args(argv)
    paths = StoragePaths.resolve(args.datasets, args.models, args.outputs)
    dataset_path = paths.datasets / Path(args.dataset_name).with_suffix('.h5')
    convert(args.source, dataset_path)


if __name__ == '__main__':
    main()
