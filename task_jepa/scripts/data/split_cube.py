"""Split the Cube expert dataset into its train and validation datasets."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import h5py
import hdf5plugin

from stable_worldmodel.paths import StoragePaths, add_storage_arguments

SPLITS = {
    'cube_single_expert_train': (0, 9800),
    'cube_single_expert_val': (9800, 10000),
}
STEP = 3300


def split_cube(source: Path, datasets: Path) -> None:
    """Write the fixed Cube train/validation episode splits under ``datasets``."""
    datasets.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as src:
        ep_len, ep_offset = src['ep_len'][:], src['ep_offset'][:]
        for name, (episode_start, episode_end) in SPLITS.items():
            row_start = int(ep_offset[episode_start])
            row_end = int(
                ep_offset[episode_end - 1] + ep_len[episode_end - 1]
            )
            destination = datasets / f'{name}.h5'
            with h5py.File(destination, 'w-') as dst:
                dst['ep_len'] = ep_len[episode_start:episode_end]
                dst['ep_offset'] = (
                    ep_offset[episode_start:episode_end] - row_start
                )
                for key, dataset in src.items():
                    if key in ('ep_len', 'ep_offset'):
                        continue
                    kwargs = {}
                    if dataset.ndim == 4:
                        kwargs['compression'] = hdf5plugin.Blosc(
                            cname='lz4',
                            clevel=5,
                            shuffle=hdf5plugin.Blosc.SHUFFLE,
                        )
                    output = dst.create_dataset(
                        key,
                        shape=(row_end - row_start, *dataset.shape[1:]),
                        dtype=dataset.dtype,
                        chunks=dataset.chunks,
                        **kwargs,
                    )
                    for start in range(row_start, row_end, STEP):
                        block = dataset[start : min(start + STEP, row_end)]
                        if key == 'ep_idx':
                            block = block - episode_start
                        elif key == 'id':
                            block = block - row_start
                        output[
                            start - row_start : start - row_start + len(block)
                        ] = block
            print(
                f'wrote {name}.h5: {episode_end - episode_start} episodes, '
                f'{row_end - row_start} steps'
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Split the Cube expert dataset into train and validation.'
    )
    add_storage_arguments(parser)
    parser.add_argument(
        '--source',
        default='cube_single_expert.h5',
        help='Source dataset name or absolute path.',
    )
    args = parser.parse_args(argv)
    paths = StoragePaths.resolve(args.datasets, args.models, args.outputs)
    source = Path(args.source)
    if not source.is_absolute():
        source = paths.datasets / source
    split_cube(source, paths.datasets)


if __name__ == '__main__':
    main()
