"""SLURM launcher for H-JEPA training, resume and per-epoch planning eval.

  python scripts/slurm/launch.py train --config-name cube_lewm --sweep crop_ab [--seeds 42,43,44] \
      [--grid level1.wm.history_size=3,7 ...] [--into <sweep_dir>] [hydra overrides ...]
  python scripts/slurm/launch.py resume <run_dir> [hydra overrides, e.g. trainer.max_epochs=100]
  python scripts/slurm/launch.py eval <sweep_dir|run_dir> [--epochs all|last|N,M] [eval.py overrides]
  common flags: --datasets D --models M --outputs O --partition P --account A --qos Q --time HH:MM:SS
      --gpus N --mem 200G --dry

train   One job per (grid cell x seed) in models/<env>/<sweep>_<YYYY-MM-DD_HH-MM>/<cell>/seed<S>.
        The worktree is copied once to <sweep_dir>/code; every job of the sweep, its resumes and --into
        additions use that snapshot.
resume  Relaunch one model run from its saved config.yaml; training restarts from lightning_resume/last.ckpt.
eval    One job per epoch checkpoint under the target. Results are in outputs/eval/<model-relative-run>/eval_epoch/
        and eval code is frozen to <sweep_dir>/eval_code on the first submission.
Logs are in outputs/slurm/<model-relative-run>/. Cluster settings are scripts/slurm/default.yaml < local.yaml < flags.
"""

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf

S = Path(__file__).resolve().parent
REPO = S.parents[1]
CONFIG_ROOT = REPO / "configs"
SET_BY_LAUNCHER = {"seed", "subdir", "output_model_name", "trainer.devices"}
LIVE = "PENDING,CONFIGURING,RUNNING,SUSPENDED,REQUEUED,REQUEUE_HOLD,RESIZING"  # not COMPLETING
EPOCH_RE = re.compile(r"_epoch_(\d+)_object\.ckpt$")


def _dict_config(value: object, description: str) -> DictConfig:
    if isinstance(value, DictConfig):
        return value
    raise TypeError(f"{description} must be a mapping.")


def die(msg: str) -> NoReturn:
    sys.exit(f"launch.py: ERROR: {msg}")


def storage_overrides(storage) -> list[str]:
    return [f"++{name}={value}" for name, value in storage.as_dict().items()]


def child_storage_args(storage) -> list[str]:
    return [item for name, value in storage.as_dict().items() for item in (f"--{name}", value)]


def model_relative(storage, path: Path) -> Path:
    try:
        return path.resolve().relative_to(storage.models)
    except ValueError:
        die(f"{path} must be within --models {storage.models}")


def output_path(storage, category: str, model_path: Path) -> Path:
    return storage.outputs / category / model_relative(storage, model_path)


def cluster(sub: str, a: argparse.Namespace) -> DictConfig:
    c = _dict_config(OmegaConf.load(S / "default.yaml"), "default cluster config")
    if (S / "local.yaml").exists():
        c = _dict_config(
            OmegaConf.merge(
                c,
                _dict_config(OmegaConf.load(S / "local.yaml"), "local cluster config"),
            ),
            "cluster config",
        )
    if a.partition and not a.account:
        die("--partition needs --account too (they are set together)")
    flags = {k: getattr(a, k) for k in ("partition", "account", "qos", "time", "mem") if getattr(a, k)}
    c = _dict_config(
        OmegaConf.merge(c, c.get(sub) or {}, flags, {"gpus_per_node": a.gpus} if a.gpus else {}),
        "cluster config",
    )
    keys = ("venv", "partition", "account", "qos", "gpus_per_node", "cpus_per_task", "mem", "time")
    if missing := [k for k in keys if c.get(k) in (None, "")]:
        die(f"cluster settings missing {missing}: set them in {S}/local.yaml")
    if not Path(c.venv, "bin/python").exists():
        die(f"venv {c.venv} has no bin/python")
    return c


def compose_cfg(config_dir: Path, name: str, overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    ov = [o for o in overrides if not o.lstrip("+~").startswith("hydra.")]
    try:
        with initialize_config_dir(str(config_dir), version_base=None):
            return _dict_config(compose(name, overrides=ov), f"config {config_dir}/{name}")
    except (HydraException, TypeError) as error:
        die(f"config {config_dir}/{name} does not compose with {ov}:\n  {type(error).__name__}: {error}")


def check_datasets(cfg: DictConfig, datasets: Path) -> None:
    for key in ("name", "val_name"):
        if value := OmegaConf.select(cfg, f"data.dataset.{key}", default=None):
            dataset = datasets / "droid" / value if value.endswith(".csv") else datasets / f"{value}.h5"
            dataset.exists() or die(f"data.dataset.{key}={value}: {dataset} not found")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def snapshot(dest: Path) -> None:
    dest.mkdir(parents=True)
    files = git("ls-files", "-co", "--exclude-standard")
    rsync = ["rsync", "-a", "--ignore-missing-args", "--files-from=-", ".", str(dest)]
    subprocess.run(rsync, cwd=REPO, input=files, text=True, check=True)
    for filename, args in (("GIT_COMMIT", ["rev-parse", "HEAD"]), ("GIT_STATUS", ["status", "--short"]),
                           ("UNCOMMITTED.diff", ["diff", "HEAD"])):  # fmt: skip
        (dest / filename).write_text(git(*args))
    print(f"code snapshot {dest} @ {git('rev-parse', '--short', 'HEAD').strip()}")


def find_snapshot(p: Path, name: str = "code") -> Path | None:  # the sweep's <name> code copy above p
    code = next((d / name for d in [p, *p.parents] if (d / name / "GIT_COMMIT").exists()), None)
    head = [git("rev-parse", "HEAD"), git("diff", "HEAD")]
    if code and [(code / f).read_text() for f in ("GIT_COMMIT", "UNCOMMITTED.diff")] != head:
        print(f"WARNING: {code} differs from the worktree; jobs run it, not your edits (mv it aside to refresh)")
    return code


def epoch_ckpts(d: Path) -> dict:
    return {int(m.group(1)): c for c in d.glob("*_epoch_*_object.ckpt") if (m := EPOCH_RE.search(c.name))}


def is_run(d: Path) -> bool:
    return (d / "config.yaml").exists() and bool(epoch_ckpts(d) or (d / "lightning_resume").exists())


def submit(c, name: str, log_dir: Path, gpus: int, code: Path, args: list, dry: bool) -> str | None:
    job = shlex.join([f"{c.venv}/bin/python", str(code / "scripts/slurm/launch.py"), "_job", str(code), *args])
    cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={c.partition}",
           f"--account={c.account}", f"--qos={c.qos}", f"--time={c.time}", "--nodes=1", "--requeue",
           "--open-mode=append", f"--ntasks-per-node={gpus}", f"--gpus-per-node={gpus}", f"--mem={c.mem}",
           f"--cpus-per-task={c.cpus_per_task}", *(f"--{key}={log_dir}/%x_%j.out" for key in ("output", "error")),
           f"--export=ALL,PYTHONPATH={code / 'src'}", f"--wrap={job}"]  # fmt: skip
    if dry:
        print("DRY", shlex.join(cmd))
        return "DRY"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}  # nested-submit leak
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    jid = result.stdout.strip().split(";")[0]
    if result.returncode or not jid.isdigit():
        print(f"SUBMIT FAILED {name}: rc={result.returncode} {result.stderr.strip()}")
        return None
    print(f"submitted {jid} {name}")
    return jid


def done_file(config: str, out: Path) -> Path:
    return out / ("eval.csv" if config.startswith("droid") else "metrics.yaml")


def run_job(code: str, mode: str, args: list) -> None:
    """Run snapshot train or eval commands with only its src package on PYTHONPATH."""
    code_root = Path(code)
    os.chdir(code_root)
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ.update(
        PYTHONPATH=str(code_root / "src"),
        PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        PYTHONDONTWRITEBYTECODE="1",
        HYDRA_FULL_ERROR="1",
        PYTHONFAULTHANDLER="1",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        WANDB_MODE="disabled" if mode == "eval" else os.environ.get("WANDB_MODE", "online"),
    )
    jid = os.environ["SLURM_JOB_ID"]
    print(f"{mode} code={code_root} job={jid} restart={os.environ.get('SLURM_RESTART_COUNT', '0')}", flush=True)
    if mode == "train":  # exec: srun gets the job's signals (preemption -> spt requeue handler)
        os.environ["MASTER_PORT"] = str(20000 + int(jid) % 20000)
        os.execvp("srun", ["srun", sys.executable, "scripts/train/train.py", *args])
    specs, overrides, returncode = args[: args.index("--")], args[args.index("--") + 1 :], 0
    for config, seed, checkpoint, results in (spec.split("|") for spec in specs):
        if not done_file(config, Path(results)).exists():
            returncode |= (
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/eval/eval.py",
                        "--config-name",
                        config,
                        f"seed={seed}",
                        f"policy={checkpoint}",
                        f"output.dir={results}",
                        f"hydra.run.dir={results}/hydra",
                        *overrides,
                    ],
                    check=False,
                ).returncode
                != 0
            )
    sys.exit(returncode)


def record(d: Path, entry: dict) -> None:
    f = d / "launch.json"
    log = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps([*log, {"time": f"{datetime.now():%F %T}", "argv": sys.argv, **entry}], indent=1))


def key_token(key: str) -> str:
    parts = key.lstrip("+~").split(".")
    lvl = "".join(f"l{m.group(1)}" for p in parts if (m := re.fullmatch(r"level(\d+)", p)))
    return lvl + "".join(w[0] for w in parts[-1].split("_") if w)


def safe(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", v).strip("-") or "x"


def cmd_train(a, overrides: list, storage) -> None:
    c = cluster("train", a)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.sweep):
        die(f"--sweep {a.sweep!r}: give a descriptive, filesystem-safe name")
    for override in overrides:
        if (key := override.split("=")[0].lstrip("+~")) in SET_BY_LAUNCHER or key.startswith("hydra."):
            die(f"{key} is set by launch.py")
    grid = [(key, values.split(",")) for key, _, values in (item.partition("=") for item in a.grid)]
    if bad := [key for key, values in grid if values == [""] or key.lstrip("+~") in SET_BY_LAUNCHER]:
        die(f"--grid {bad}: expected key=v1,v2 on keys launch.py does not set")
    tokens = [key_token(key) for key, _ in grid]
    if len(set(tokens)) < len(tokens):
        tokens = [safe(key.lstrip("+~")) + "-" for key, _ in grid]
    if a.into:
        sweep_dir = Path(a.into).resolve()
        if not (sweep_dir / "code/GIT_COMMIT").exists():
            die(f"--into {sweep_dir}: not a launch.py sweep dir (no code/GIT_COMMIT)")
    else:
        env = compose_cfg(
            CONFIG_ROOT / "train",
            a.config_name,
            [*storage_overrides(storage), *overrides],
        ).env
        sweep_dir = storage.models / env / f"{a.sweep}_{datetime.now():%Y-%m-%d_%H-%M}"
        if sweep_dir.exists():
            die(f"{sweep_dir} exists (launched this minute already?): wait a minute or use --into")
    rel = model_relative(storage, sweep_dir)
    jobs = []
    for combo in itertools.product(*[values for _, values in grid]):
        cell = "_".join(token + safe(value) for token, value in zip(tokens, combo)) or a.config_name
        for seed in [int(value) for value in a.seeds.split(",")]:
            run_dir = sweep_dir / cell / f"seed{seed}"
            job_overrides = [
                *overrides,
                *(f"{key}={value}" for (key, _), value in zip(grid, combo)),
                f"seed={seed}",
                f"subdir={rel}/{cell}/seed{seed}",
                f"output_model_name={cell}",
                f"trainer.devices={c.gpus_per_node}",
            ]
            if run_dir.exists():
                die(f"{run_dir} exists: a fresh train would silently resume it; use `resume` or a new sweep")
            cfg = compose_cfg(
                CONFIG_ROOT / "train",
                a.config_name,
                [*storage_overrides(storage), *job_overrides],
            )
            check_datasets(cfg, storage.datasets)
            if cfg.wandb.enabled:
                job_overrides += [f"++wandb.config.group={a.sweep}", f"++wandb.config.name={cell}_s{seed}"]
            jobs.append(
                (
                    cell,
                    seed,
                    run_dir,
                    [*job_overrides, f"hydra.run.dir={output_path(storage, '', run_dir) / 'hydra'}"],
                )
            )
    print(f"sweep {sweep_dir}: {len(jobs)} jobs ({len(jobs) // len(a.seeds.split(','))} cells x {a.seeds})")
    code = sweep_dir / "code"
    if a.into:
        find_snapshot(sweep_dir)
    elif not a.dry:
        snapshot(code)
    ids = {}
    for cell, seed, run_dir, job_overrides in jobs:
        args = ["train", *child_storage_args(storage), "--config-name", a.config_name, *job_overrides]
        ids[str(run_dir)] = submit(
            c,
            f"hj_{a.sweep}_{cell}_s{seed}",
            output_path(storage, "slurm", run_dir),
            c.gpus_per_node,
            code,
            args,
            a.dry,
        )
    if not a.dry:
        commit = (code / "GIT_COMMIT").read_text().strip()
        record(
            sweep_dir,
            {
                "cmd": "train",
                "config_name": a.config_name,
                "overrides": overrides,
                "grid": dict(grid),
                "seeds": a.seeds,
                "jobs": ids,
                "snapshot_commit": commit,
            },
        )


def cmd_resume(a, overrides: list, storage) -> None:
    c = cluster("resume", a)
    run_dir = Path(a.run_dir).resolve()
    model_relative(storage, run_dir)
    if below := [path for path in [*run_dir.glob("*"), *run_dir.glob("*/*")] if path.is_dir() and is_run(path)]:
        die(f"{run_dir} contains {len(below)} run dirs (a sweep dir?): resume one run dir")
    if not (run_dir / "lightning_resume/last.ckpt").exists() or not (run_dir / "config.yaml").exists():
        die(f"nothing to resume in {run_dir}: needs config.yaml and lightning_resume/last.ckpt (epoch ckpts are not)")
    saved = _dict_config(OmegaConf.load(run_dir / "config.yaml"), "saved run config")
    for filename in (run_dir / "launch.json", run_dir.parent.parent / "launch.json"):
        log = json.loads(filename.read_text()) if filename.exists() else []
        jid = next(
            (entry["jobs"][str(run_dir)] for entry in reversed(log) if str(run_dir) in entry.get("jobs", {})), None
        )
        queue = ["squeue", "-h", "-j", str(jid), "-t", LIVE, "-o", "%T"]
        if jid and (status := subprocess.run(queue, capture_output=True, text=True, check=False).stdout.strip()):
            die(f"job {jid} of {run_dir} is still {status}: it resumes by itself on requeue")
    if not (code := find_snapshot(run_dir)):
        die(f"no sweep code snapshot (code/GIT_COMMIT) above {run_dir}")
    devices = OmegaConf.select(saved, "trainer.devices")
    gpus = a.gpus or (devices if isinstance(devices, int) else int(c.gpus_per_node))
    job_overrides = [
        *overrides,
        f"trainer.devices={gpus}",
        f"hydra.run.dir={output_path(storage, '', run_dir) / 'hydra'}",
    ]
    compose_cfg(run_dir, "config", [*storage_overrides(storage), *job_overrides])
    jid = submit(
        c,
        f"hj_resume_{saved.output_model_name}_s{saved.seed}",
        output_path(storage, "slurm", run_dir),
        gpus,
        code,
        [
            "train",
            *child_storage_args(storage),
            "--config-path",
            str(run_dir),
            "--config-name",
            "config",
            *job_overrides,
        ],
        a.dry,
    )
    if not a.dry:
        record(run_dir, {"cmd": "resume", "overrides": overrides, "jobs": {str(run_dir): jid}, "code": str(code)})


def env_of(cfg: DictConfig) -> str:  # runs made before the train configs had an `env` key fall back
    pe = OmegaConf.select(cfg, "planning_eval.config_name", default=None) or ""
    ds = str(OmegaConf.select(cfg, "data.dataset.name", default=""))
    return cfg.get("env") or pe.split("_")[0] or ("droid" if ds.endswith(".csv") else die(f"no env for {ds}"))


def cmd_eval(a, overrides: list, storage) -> None:
    c = cluster("eval", a)
    root = Path(a.target).resolve()
    model_relative(storage, root)
    runs = [root] if is_run(root) else []
    for directory, directories, filenames in [] if runs else os.walk(root):
        if "config.yaml" in filenames and epoch_ckpts(Path(directory)):
            runs.append(Path(directory))
            directories.clear()
        directories[:] = sorted(
            name for name in directories if not name.startswith(("code", "hydra", ".", "slurm", "eval_", "lightning"))
        )
    if not runs:
        die(f"no run dirs (config.yaml + *_epoch_*_object.ckpt) under {root}")
    queue = subprocess.run(
        ["squeue", "--me", "-h", "-t", LIVE, "-o", "%j"],
        capture_output=True,
        text=True,
        check=False,
    )
    active, checked, plan, ndone = set(queue.stdout.split()), set(), [], 0
    for run_dir in runs:
        cfg = _dict_config(OmegaConf.load(run_dir / "config.yaml"), "saved run config")
        env, levels, seed, checkpoints = env_of(cfg), int(cfg.num_levels), int(cfg.seed), epoch_ckpts(run_dir)
        config_name, run = f"{env}_{'flat' if levels == 1 else f'l{levels}'}", str(cfg.output_model_name)
        run += "" if run.endswith(f"s{seed}") else f"_s{seed}"
        wanted = sorted(checkpoints) if a.epochs == "all" else [max(checkpoints)] if a.epochs == "last" else \
            [int(value) for value in a.epochs.split(",")]  # fmt: skip
        for epoch in wanted:
            results = output_path(storage, "eval", run_dir) / "eval_epoch" / f"epoch_{epoch}"
            name = f"ev_{run}_e{epoch}"
            done = done_file(config_name, results).exists()
            why = "missing" if epoch not in checkpoints else "done" if done else name in active and "active"
            if why:
                ndone += done
                print(f"SKIP {name}: {why}")
                continue
            if config_name not in checked:  # compose and ensure its eval tasks exist once per config
                eval_cfg = compose_cfg(
                    CONFIG_ROOT / "eval",
                    config_name,
                    [*storage_overrides(storage), f"seed={seed}", f"policy={checkpoints[epoch]}", *overrides],
                )
                tasks = eval_cfg.get("load_eval_trajs_path")
                try:
                    not tasks or Path(tasks).exists() or die(f"{config_name}: eval tasks {tasks} missing")
                except PermissionError:
                    print(f"WARNING: cannot stat eval tasks {tasks}")
                checked.add(config_name)
            plan.append(
                (
                    run_dir,
                    name,
                    f"{config_name}|{seed}|{checkpoints[epoch]}|{results}",
                    c.time_by_env.get(env, c.time_by_env.default),
                )
            )
    print(f"{len(runs)} runs, {ndone} evals done, {len(plan)} to run")
    if not plan:
        return
    if not (code := find_snapshot(root, "eval_code")):  # first eval freezes the worktree code
        code = next((path for path in [root, *root.parents] if (path / "code/GIT_COMMIT").exists()), root) / "eval_code"
        print(f"{'would snapshot' if a.dry else 'snapshotting'} the worktree to {code}")
        a.dry or snapshot(code)
    ids = {}
    for run_dir, name, spec, time_limit in plan:
        c.time = a.time or time_limit
        ids[name] = submit(
            c,
            name,
            output_path(storage, "slurm", run_dir),
            1,
            code,
            ["eval", spec, "--", *child_storage_args(storage), *overrides],
            a.dry,
        )
    if not a.dry:
        record(root, {"cmd": "eval", "epochs": a.epochs, "overrides": overrides, "jobs": ids, "code": str(code)})


def main() -> None:
    if sys.argv[1:2] == ["_job"]:
        return run_job(sys.argv[2], sys.argv[3], sys.argv[4:])
    from stable_worldmodel.paths import StoragePaths, add_storage_arguments

    common = argparse.ArgumentParser(add_help=False)
    for flag in ("--partition", "--account", "--qos", "--time", "--mem"):
        common.add_argument(flag)
    common.add_argument("--gpus", type=int, help="GPUs per node (train: trainer.devices)")
    common.add_argument("--dry", action="store_true", help="run every check, print the sbatch lines")
    add_storage_arguments(common)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--config-name", required=True)
    train.add_argument("--sweep", required=True)
    train.add_argument("--seeds", default="42")
    train.add_argument("--grid", action="append", default=[], help="key=v1,v2 (repeatable)")
    train.add_argument("--into", help="existing sweep dir to add jobs to")
    subparsers.add_parser("resume", parents=[common]).add_argument("run_dir")
    evaluate = subparsers.add_parser("eval", parents=[common])
    evaluate.add_argument("target")
    evaluate.add_argument("--epochs", default="all", help="all | last | N,M")
    args, overrides = parser.parse_known_args()
    if bad := [override for override in overrides if override.startswith("-") or "=" not in override]:
        die(f"unrecognized arguments {bad} (hydra overrides are key=value)")
    storage = StoragePaths.resolve(args.datasets, args.models, args.outputs)
    {"train": cmd_train, "resume": cmd_resume, "eval": cmd_eval}[args.cmd](args, overrides, storage)


if __name__ == "__main__":
    main()
