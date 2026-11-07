"""Prepare Flickr8k and COCO captions from local ZIPs, without network access.

Paths are relative to the script's project root, not the working directory.
Images retain their original JPEG bytes; preprocessing belongs in the training loader.
"""

import argparse
import io
import json
import shutil
import stat
import tempfile
import zipfile
import zlib
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm


@dataclass
class Sample:
    filename: str
    captions: list[str]


@dataclass
class Split:
    source: str
    name: str
    archive: zipfile.ZipFile
    archive_dir: str
    image_dir: Path
    manifest_name: str
    samples: list[Sample]


def filename(value: str) -> str:
    """Accept only a single image filename, never an archive-supplied path."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or ":" in value
    ):
        raise ValueError(f"Unsafe image filename: {value!r}")
    return value


def member(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise FileNotFoundError(f"{archive.filename}: missing {name}") from exc
    if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"{archive.filename}: expected a regular file: {name}")
    return info


def read_text(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(member(archive, name)).decode("utf-8-sig")


def open_archive(stack: ExitStack, path: Path) -> zipfile.ZipFile:
    archive = stack.enter_context(zipfile.ZipFile(path))
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate ZIP member names")
    return archive


def flickr_splits(root: Path, images: zipfile.ZipFile, annotations: zipfile.ZipFile) -> list[Split]:
    captions = defaultdict(dict)
    for number, line in enumerate(read_text(annotations, "Flickr8k.token.txt").splitlines(), 1):
        if not line.strip():
            continue
        try:
            key, text = line.split("\t", 1)
            name, index_text = key.rsplit("#", 1)
            name = filename(name)
            index = int(index_text)
            if index not in range(5) or index in captions[name]:
                raise ValueError("invalid or duplicate caption index")
        except ValueError as exc:
            raise ValueError(f"Flickr8k.token.txt line {number}: {exc}") from exc
        captions[name][index] = text.strip()

    splits = []
    seen = set()
    for split in ("train", "dev", "test"):
        annotation = f"Flickr_8k.{split}Images.txt"
        names = [filename(line.strip()) for line in read_text(annotations, annotation).splitlines() if line.strip()]
        if not names:
            raise ValueError(f"{annotation}: empty split")
        samples = []
        for name in names:
            if name in seen:
                raise ValueError(f"{annotation}: duplicate image within or across splits: {name}")
            seen.add(name)
            texts = captions.get(name, {})
            if set(texts) != set(range(5)) or not all(texts.values()):
                raise ValueError(f"{annotation}: {name} needs five nonempty captions (0..4)")
            member(images, f"Flicker8k_Dataset/{name}")
            samples.append(Sample(name, [texts[index] for index in range(5)]))
        splits.append(
            Split(
                "flickr8k",
                split,
                images,
                "Flicker8k_Dataset",
                root / "data" / "flickr8k" / split / "images",
                f"flickr8k_{split}.jsonl",
                samples,
            )
        )
    return splits


def coco_splits(root: Path, train: zipfile.ZipFile, val: zipfile.ZipFile, annotations: zipfile.ZipFile) -> list[Split]:
    splits = []
    seen_ids = set()
    seen_names = set()
    for split, archive in (("train", train), ("val", val)):
        annotation = f"annotations/captions_{split}2017.json"
        with annotations.open(member(annotations, annotation)) as stream:
            data = json.load(io.TextIOWrapper(stream, encoding="utf-8-sig"))
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("images"), list)
            or not isinstance(data.get("annotations"), list)
        ):
            raise TypeError(f"{annotation}: expected images and annotations lists")
        images = {}
        for image in data["images"]:
            if not isinstance(image, dict) or type(image.get("id")) is not int:
                raise ValueError(f"{annotation}: image must have an integer id")
            image_id = image["id"]
            name = filename(image.get("file_name"))
            if image_id in seen_ids or name in seen_names:
                raise ValueError(
                    f"{annotation}: duplicate image ID or filename within or across splits: {image_id}, {name}"
                )
            seen_ids.add(image_id)
            seen_names.add(name)
            member(archive, f"{split}2017/{name}")
            images[image_id] = name
        if not images:
            raise ValueError(f"{annotation}: empty split")
        captions = defaultdict(list)
        for caption in data["annotations"]:
            if not isinstance(caption, dict) or type(caption.get("image_id")) is not int:
                raise ValueError(f"{annotation}: caption must have an integer image_id")
            image_id = caption["image_id"]
            if image_id not in images:
                raise ValueError(f"{annotation}: caption references missing image ID {image_id}")
            text = caption.get("caption")
            if not isinstance(text, str):
                raise TypeError(f"{annotation}: image {image_id} caption must be a string")
            if text.strip():
                captions[image_id].append(text.strip())
        samples = []
        for image_id in sorted(images):
            if not captions[image_id]:
                raise ValueError(f"{annotation}: image {image_id} ({images[image_id]}) has no nonempty captions")
            samples.append(Sample(images[image_id], captions[image_id]))
        splits.append(
            Split(
                "coco2017",
                split,
                archive,
                f"{split}2017",
                root / "data" / "coco" / f"{split}2017",
                f"coco_{split}2017.jsonl",
                samples,
            )
        )
    return splits


def matches_archive(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return checksum == info.CRC


def check_image(path: Path, context: str) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if min(image.size) <= 0:
                raise ValueError("empty image dimensions")
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid image {context}: {exc}") from exc


def prepare_image(archive: zipfile.ZipFile, name: str, destination: Path, root: Path) -> bool:
    info = member(archive, name)
    if not destination.resolve().is_relative_to((root / "data").resolve()) or destination.is_symlink():
        raise ValueError(f"Unsafe output image path: {destination}")
    context = f"{archive.filename}:{name}"
    if matches_archive(destination, info):
        check_image(destination, context)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            # Record the path before opening the ZIP member so entry failures are cleaned.
            temporary = Path(stream.name)
            with archive.open(info) as source:
                # Reading to EOF makes ZipFile check the member's CRC.
                shutil.copyfileobj(source, stream, length=1024 * 1024)
        check_image(temporary, context)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def prepare_data(root: Path, dataset: str = "all", limit: int = 0) -> None:
    """Validate all selected metadata, extract images, then publish the manifests.

    Failed extraction may leave verified images for the next run, but cannot
    replace any selected manifest until every selected split has completed.
    """
    if dataset not in {"all", "flickr8k", "coco"}:
        raise ValueError("dataset must be all, flickr8k, or coco")
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    root = Path(root).resolve()
    data_dir = root / "data"
    if limit:
        print(
            f"WARNING: --limit {limit} prepares only the first {limit} images per split; selected manifests will be replaced. Rerun without --limit before full training.",
            flush=True,
        )
    with ExitStack() as stack:
        splits = []
        if dataset in {"all", "flickr8k"}:
            images = open_archive(stack, data_dir / "Flickr8k_Dataset.zip")
            annotations = open_archive(stack, data_dir / "Flickr8k_text.zip")
            splits.extend(flickr_splits(root, images, annotations))
        if dataset in {"all", "coco"}:
            train = open_archive(stack, data_dir / "train2017.zip")
            val = open_archive(stack, data_dir / "val2017.zip")
            annotations = open_archive(stack, data_dir / "annotations_trainval2017.zip")
            splits.extend(coco_splits(root, train, val, annotations))
        manifest_dir = data_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        if not manifest_dir.resolve().is_relative_to(data_dir.resolve()):
            raise ValueError(f"Unsafe output manifest directory: {manifest_dir}")
        for split in splits:
            if (manifest_dir / split.manifest_name).is_symlink():
                raise ValueError(f"Refusing to replace symlink manifest: {split.manifest_name}")
        with tempfile.TemporaryDirectory(prefix=".prepare-data-", dir=manifest_dir) as staging:
            manifests = []
            for split in splits:
                count = min(limit, len(split.samples)) if limit else len(split.samples)
                temporary = Path(staging) / split.manifest_name
                written = 0
                caption_count = 0
                with temporary.open("w", encoding="utf-8") as stream:
                    for index in tqdm(range(count), desc=f"{split.source} {split.name}", unit="image"):
                        sample = split.samples[index]
                        path = split.image_dir / sample.filename
                        written += prepare_image(split.archive, f"{split.archive_dir}/{sample.filename}", path, root)
                        caption_count += len(sample.captions)
                        record = {
                            "source": split.source,
                            "split": split.name,
                            "image_id": f"{split.source}_{split.name}_{index:06d}",
                            "image_path": path.relative_to(root).as_posix(),
                            "captions": sample.captions,
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifests.append((temporary, manifest_dir / split.manifest_name))
                print(
                    f"{split.source} {split.name}: {count:,}/{len(split.samples):,} images, {caption_count:,} captions; {written:,} written, {count - written:,} verified and reused",
                    flush=True,
                )
            # No selected manifest changes while metadata or images are being processed.
            for temporary, target in manifests:
                temporary.replace(target)
                print(f"Prepared {target.relative_to(root)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "flickr8k", "coco"),
        default="all",
        help="Datasets to prepare (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="First N images per split; 0 means all. Replaces selected manifests.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    try:
        prepare_data(Path(__file__).resolve().parents[1], args.dataset, args.limit)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
