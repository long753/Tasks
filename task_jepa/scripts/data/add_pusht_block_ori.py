"""Append the continuous PushT block-orientation target to HDF5 datasets."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from stable_worldmodel.paths import StoragePaths, add_storage_arguments


def add_block_ori(h5_path: str | Path) -> None:
    """Add ``block_ori`` to one dataset unless it already exists."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'r+') as file:
        if 'block_ori' in file:
            print(f'{h5_path.name}: block_ori already present, skipping')
            return
        theta = file['state'][:, 4].astype(np.float64)
        ori = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
        file.create_dataset(
            'block_ori',
            data=ori,
            chunks=(min(1000, len(ori)), 2),
        )

    with h5py.File(h5_path, 'r') as file:
        theta = file['state'][:, 4].astype(np.float64)
        cos, sin = file['block_ori'][:, 0], file['block_ori'][:, 1]
        assert np.allclose(cos**2 + sin**2, 1.0, atol=1e-5)
        recovered = np.arctan2(sin, cos)
        circular_err = np.abs(
            np.mod(recovered - theta + np.pi, 2 * np.pi) - np.pi
        ).max()
        assert circular_err < 1e-4, circular_err
    print(
        f'{h5_path.name}: wrote block_ori {ori.shape}, '
        f'max circular error {circular_err:.2e}'
    )


def _dataset_path(datasets: Path, dataset_name: str) -> Path:
    path = Path(dataset_name)
    if not path.is_absolute():
        path = datasets / path
    return path.with_suffix('.h5')


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Add PushT block orientation targets to datasets.'
    )
    add_storage_arguments(parser)
    parser.add_argument('dataset_names', nargs='+')
    args = parser.parse_args(argv)
    paths = StoragePaths.resolve(args.datasets, args.models, args.outputs)
    for dataset_name in args.dataset_names:
        add_block_ori(_dataset_path(paths.datasets, dataset_name))


if __name__ == '__main__':
    main()
