"""Train radar model."""

import json
import os
import time
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from deepradar import DeepRadar, config
from deepradar.pretrained import load_official_grt_base


class _SelectiveTrainModeCallback(Callback):
    """Keep frozen model branches in eval mode during selective training."""

    def __init__(self, trainable_module: str) -> None:
        super().__init__()
        self.trainable_module = trainable_module

    def _apply(self, model: torch.nn.Module) -> None:
        model.eval()
        model.get_submodule(self.trainable_module).train()
        # Lightning expects the root module to remain in training mode.
        model.training = True

    def on_train_start(self, trainer, pl_module) -> None:
        self._apply(pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._apply(pl_module)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        # Validation and other loops may recursively restore model.train().
        self._apply(pl_module)


class _FrozenModulesEvalCallback(Callback):
    """Keep explicitly frozen branches deterministic during training."""

    def __init__(self, modules: list[str]) -> None:
        super().__init__()
        self.modules = tuple(modules)

    def _apply(self, model: torch.nn.Module) -> None:
        for name in self.modules:
            model.get_submodule(name).eval()

    def on_train_start(self, trainer, pl_module) -> None:
        self._apply(pl_module)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._apply(pl_module)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        self._apply(pl_module)


def _batch_limit(value: str | None):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "default", "-1"}:
        return None
    if any(c in text for c in [".", "e"]):
        return float(text)
    return int(text)


def _metric_spec(value: str) -> tuple[str, str]:
    metric, sep, mode = value.rpartition(":")
    if not sep or mode not in {"min", "max"} or not metric:
        raise ValueError(
            f"Invalid checkpoint metric {value!r}; use metric/name:min|max.")
    return metric, mode


def _initialize_from_base_models(model, args) -> None:
    """Load optional base weights and apply encoder freezing afterwards."""
    if args.base_model is not None:
        base = DeepRadar.load_from_experiment(args.base_model, checkpoint=None)
        model.encoder.load_state_dict(base.encoder.state_dict())

        if args.load_decoder:
            decoder_params = {
                k: v for k, v in base.decoder.state_dict().items()
                if not k.startswith("unpatch")}
            model.decoder.load_state_dict(decoder_params, strict=False)

        if args.load_full_decoder:
            model.decoder.load_state_dict(
                base.decoder.state_dict(), strict=False)

    if args.official_base_model is not None:
        report = load_official_grt_base(
            model,
            args.official_base_model,
            elevation_index=args.official_elevation_index,
            load_decoder=not args.official_skip_decoder,
            decoder_head=args.official_decoder_head,
            min_loaded_fraction=args.official_min_loaded_fraction,
        )
        print("OFFICIAL_INIT " + json.dumps(report, indent=2), flush=True)

    if args.freeze:
        model.encoder.freeze()


def _parse():
    p = ArgumentParser(description="Train radar model.")

    g = p.add_argument_group("Path")
    g.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    g.add_argument(
        "-o", "--out", default="results", help="Root results directory.")

    g = p.add_argument_group("Training")
    g.add_argument(
        "-c", "--cfg", nargs='+', default=None,
        help="Training configuration; see `deepradar.config` for parsing "
        "rules. Must be specified unless resuming with "
        "`--checkpoint <checkpoint>`.")
    g.add_argument(
        "--cfg_dir", default="config", help="Configuration base directory.")
    g.add_argument(
        "-k", "--checkpoint", default=None,
        help="Checkpoint to load, if specified. Should have the structure "
        "`<folder>/checkpoints/<checkpoint>.ckpt`, where `folder` contains a "
        "`hparams.yaml` file.")
    g.add_argument(
        "--epochs", default=-1, type=int, help="Maximum number of epochs.")
    g.add_argument(
        "--max_time", default=None,
        help="Optional Lightning max_time limit, e.g. 00:03:45:00.")
    g.add_argument(
        "--metric", default="loss/val",
        help="Metric to watch for convergence (e.g. `loss/val`).")
    g.add_argument(
        "--metric_mode", default="min", choices=["min", "max"],
        help="Whether the monitored metric should be minimized or maximized.")
    g.add_argument(
        "--patience", default=3, type=int,
        help="Stop after this many validation checks with no improvement.")
    g.add_argument(
        "--find_unused", default=False, action='store_true',
        help="Find unused parameters during training; only necessary for some "
        "models with some underlying library bugs.")

    g = p.add_argument_group("Fine tuning")
    g.add_argument(
        "-b", "--base_model", default=None,
        help="Base model to load, if specified. Should be a "
        "experiment directory containing a `hparams.yaml` file and "
        "`checkpoints` directory.")
    g.add_argument(
        "-d", "--load_decoder", default=False, action='store_true',
        help="Load decoder weights (skipping `unpatch.*`) as well.")
    g.add_argument(
        "--load_full_decoder", default=False, action='store_true',
        help="Load decoder weights (including unpatch.*).")
    g.add_argument(
        "-f", "--freeze", action='store_true', default=False,
        help="Freeze encoder (i.e. don't allow tuning the encoder).")
    g.add_argument(
        "--freeze_decoder", action='store_true', default=False,
        help="Freeze decoder parameters after optional base-model loading.")
    g.add_argument(
        "--train_refiner_only", action='store_true', default=False,
        help="Freeze the encoder and base decoder, leaving only "
        "decoder.refiner trainable. Requires a refiner-enabled decoder.")
    g.add_argument(
        "--train_unpatch_only", action='store_true', default=False,
        help="Freeze the model except decoder.unpatch. Used for localized "
        "patch-boundary adaptation from a loaded checkpoint.")
    g.add_argument(
        "--official_base_model", default=None,
        help="Official NRDK-format GRT model directory to use for initialization.")
    g.add_argument(
        "--official_elevation_index", default=0, type=int,
        help=(
            "Official elevation bin to select when adapting the input patch. "
            "Use -1 only for explicit two-elevation diagnostics."))
    g.add_argument(
        "--official_skip_decoder", default=False, action='store_true',
        help="Only load official tokenizer/encoder weights, skipping decoder.")
    g.add_argument(
        "--official_decoder_head", default=None,
        help="Official task head to load, e.g. occ3d or semseg. If omitted, "
        "the unique decoder head in the checkpoint is inferred.")
    g.add_argument(
        "--official_min_loaded_fraction", default=0.95, type=float,
        help="Fail before training if official initialization covers less "
        "than this fraction of the requested encoder/decoder parameters.")

    g = p.add_argument_group("Logging")
    g.add_argument(
        "-n", "--name", default=None,
        help="Method name (for experiment tracking only).")
    g.add_argument(
        "-v", "--version", default=None,
        help="Experiment version (for experiment tracking only).")
    g.add_argument(
        "--val_interval", default=0.5, type=_batch_limit,
        help="Validation interval as a fraction of each epoch, or an integer number of training batches.")
    g.add_argument(
        "--log_example_interval", default=500, type=int,
        help="Interval to log example train images.")
    g.add_argument(
        "--log_interval", default=100, type=int,
        help="Logging interval for training statistics.")
    g.add_argument(
        "--num_checkpoints", default=-1, type=int,
        help="Number of checkpoints to save.")
    g.add_argument(
        "--extra_checkpoint_metric", action="append", default=[],
        help="Save an additional best checkpoint using metric:name mode, "
        "formatted as metric/name:min or metric/name:max. Repeatable.")
    g.add_argument(
        "--no_progress_bar", default=False, action='store_true',
        help="Disable Lightning/tqdm progress bars for non-interactive DDP logs.")
    g.add_argument(
        "--limit_train_batches", default=None, type=_batch_limit,
        help="Optional Lightning limit_train_batches as int batches or float fraction.")
    g.add_argument(
        "--limit_val_batches", default=None, type=_batch_limit,
        help="Optional Lightning limit_val_batches as int batches or float fraction.")
    g.add_argument(
        "--accumulate_grad_batches", default=1, type=int,
        help="Accumulate this many batches before each optimizer step. Useful "
        "for matching the original global batch size on a single GPU.")
    g.add_argument(
        "--precision", default="16-mixed",
        help="Lightning precision mode. AutoDL launchers explicitly select "
        "bf16-mixed for RTX 5090.")
    g.add_argument(
        "--seed", default=None, type=int,
        help="Optional reproducibility seed for model, data, and workers.")

    g = p.add_argument_group("Environment")
    g.add_argument(
        "-e", "--environment", default="local",
        help="Current system environment. Known options: local (generic "
        "system with possibly multiple GPUs), psc (bridges-2 @ PSC).")
    g.add_argument(
        "--workers", default=None, type=int,
        help="Number of dataloader workers. By default, the dataloader will "
        "use the number of (virtual) cores in the system.")

    return p


def _main(args):

    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    if args.environment == "psc":
        torch.multiprocessing.set_sharing_strategy('file_system')

    if args.cfg is None:
        if args.checkpoint is not None:
            experiment_dir = os.path.dirname(os.path.dirname(args.checkpoint))
            args.cfg = [os.path.join(experiment_dir, "hparams.yaml")]
        else:
            print(
                "Must specify a `config.yaml` file if not resuming training.")
            exit(1)
    else:
        args.cfg = [os.path.join(args.cfg_dir, c) for c in args.cfg]

    # Resume training
    model_cfg = config.load_config(*args.cfg)
    if args.checkpoint is None:
        model = DeepRadar(**model_cfg)
    else:
        model = DeepRadar.load_from_checkpoint(
            args.checkpoint, hparams_file=args.cfg[0])

    _initialize_from_base_models(model, args)

    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("FREEZE_DECODER enabled", flush=True)

    if args.train_refiner_only and args.train_unpatch_only:
        raise ValueError(
            "--train_refiner_only and --train_unpatch_only are mutually exclusive.")

    trainable_module = None
    if args.train_refiner_only:
        refiner = getattr(model.decoder, "refiner", None)
        if refiner is None:
            raise ValueError(
                "--train_refiner_only requires decoder.refiner to exist.")
        for param in model.parameters():
            param.requires_grad = False
        for param in refiner.parameters():
            param.requires_grad = True
        trainable = sum(
            param.numel() for param in model.parameters()
            if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        print(
            f"TRAIN_REFINER_ONLY enabled trainable={trainable} total={total}",
            flush=True,
        )
        trainable_module = "decoder.refiner"

    if args.train_unpatch_only:
        unpatch = getattr(model.decoder, "unpatch", None)
        if unpatch is None:
            raise ValueError(
                "--train_unpatch_only requires decoder.unpatch to exist.")
        for param in model.parameters():
            param.requires_grad = False
        for param in unpatch.parameters():
            param.requires_grad = True
        trainable = sum(
            param.numel() for param in model.parameters()
            if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        print(
            f"TRAIN_UNPATCH_ONLY enabled trainable={trainable} total={total}",
            flush=True,
        )
        trainable_module = "decoder.unpatch"

    # Metadata/logging-related config bypasses save_hyperparameters
    model.configure(log_interval=args.log_example_interval, num_examples=16)

    data = model.get_dataset(args.path, n_workers=args.workers)
    checkpoint = ModelCheckpoint(
        save_top_k=args.num_checkpoints, monitor=args.metric,
        mode=args.metric_mode, save_last=True, dirpath=None,
        filename="best-primary-{epoch:03d}-{step}",
        auto_insert_metric_name=False)
    extra_checkpoints = []
    for i, spec in enumerate(args.extra_checkpoint_metric):
        metric, mode = _metric_spec(spec)
        extra_checkpoints.append(ModelCheckpoint(
            save_top_k=1, monitor=metric, mode=mode, save_last=False,
            dirpath=None, filename=f"best-extra-{i}-{{epoch:03d}}-{{step}}",
            auto_insert_metric_name=False))
    stopping = EarlyStopping(
        monitor=args.metric, min_delta=0.0, patience=args.patience,
        mode=args.metric_mode)
    logger = TensorBoardLogger(
        args.out, name=args.name, version=args.version,
        default_hp_metric=False)
    accelerator = "gpu" if torch.cuda.is_available() else "auto"
    world_size = int(
        os.environ.get("WORLD_SIZE")
        or os.environ.get("SLURM_NTASKS")
        or "1")
    devices = torch.cuda.device_count() if torch.cuda.is_available() else "auto"
    strategy = (
        DDPStrategy(find_unused_parameters=args.find_unused)
        if world_size > 1 or (
            torch.cuda.is_available() and torch.cuda.device_count() > 1
        ) else "auto")
    callbacks = [checkpoint, *extra_checkpoints, stopping]
    if trainable_module is not None:
        callbacks.append(_SelectiveTrainModeCallback(trainable_module))
    else:
        frozen_modules = []
        if args.freeze:
            frozen_modules.append("encoder")
        if args.freeze_decoder:
            frozen_modules.append("decoder")
        if frozen_modules:
            callbacks.append(_FrozenModulesEvalCallback(frozen_modules))
    trainer_kwargs = {
        "logger": logger,
        "log_every_n_steps": args.log_interval,
        "callbacks": callbacks,
        "max_steps": -1,
        "max_epochs": args.epochs,
        "max_time": args.max_time,
        "val_check_interval": args.val_interval,
        "strategy": strategy,
        "accelerator": accelerator,
        "devices": devices,
        "precision": args.precision,
        "enable_progress_bar": not args.no_progress_bar,
        "accumulate_grad_batches": args.accumulate_grad_batches,
    }
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    trainer = L.Trainer(**trainer_kwargs)

    start = time.perf_counter()
    fit_kwargs = {"model": model, "datamodule": data}
    if args.checkpoint is not None:
        fit_kwargs["ckpt_path"] = args.checkpoint
    trainer.fit(**fit_kwargs)
    duration = time.perf_counter() - start

    with open(os.path.join(logger.log_dir, "meta.json"), 'w') as f:
        json.dump({
            "best": os.path.basename(checkpoint.best_model_path),
            "best_by_metric": {
                args.metric: os.path.basename(checkpoint.best_model_path),
                **{
                    cb.monitor: os.path.basename(cb.best_model_path)
                    for cb in extra_checkpoints
                },
            },
            "duration": duration
        }, f)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
