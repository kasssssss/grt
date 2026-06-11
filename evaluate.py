"""Evaluate radar model."""

import json
import os
import queue
import shutil
import threading
import time
from argparse import ArgumentParser

import numpy as np
from deepradar._compat import roverd
import torch

from deepradar import DeepRadar, config


def _parse():
    p = ArgumentParser(description="Evaluate radar model.")

    p.add_argument(
        "-p", "--path", default="data", help="Root dataset directory.")
    p.add_argument("-m", "--model", help="Path to model.")
    p.add_argument(
        "-t", "--traces", nargs='+', help="Traces to evaluate.",
        default=["eval[indoor,outdoor,bike]"])
    p.add_argument(
        "--cfg_dir", default="config", help="Configuration base directory.")
    p.add_argument(
        "-b", "--batch", default=16, type=int, help="Evaluation batch size.")
    p.add_argument(
        "-k", "--checkpoint", default=None,
        help="Override the default selected checkpoint; should be a file in "
        "{model}/checkpoints/.")
    p.add_argument(
        "-r", "--render", default=False, action='store_true',
        help="Render visualizations if specified.")
    p.add_argument(
        "--workers", default=None, type=int,
        help="Number of dataloader workers. By default, the dataloader will "
        "use the number of available CPUs.")

    return p


def evaluate(model, datamodule, trace, args, desc: str):
    """Evaluate a single trace."""
    out = os.path.join(args.model, "eval", trace + ".npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    consumers: list[threading.Thread] = []
    if args.render:
        render_dir = os.path.join(args.model, "eval", trace)
        shutil.rmtree(render_dir, ignore_errors=True)
        os.makedirs(render_dir, exist_ok=True)
        with open(os.path.join(render_dir, "meta.json"), "w") as f:
            json.dump({}, f)
        outd = roverd.sensors.SensorData(render_dir)

        def queue_iter(q: queue.Queue):
            while True:
                item = q.get()
                if item is None:
                    return
                yield item

        queues: dict[str, queue.Queue] = {}
        for objective in model.objectives:
            for name, fmt in objective.RENDER_CHANNELS.items():
                queues[name] = queue.Queue()
                channel = outd.create(name, fmt)
                kwargs = (
                    {"preset": 0}
                    if isinstance(channel, roverd.channels.LzmaFrameChannel)
                    else {})
                thread = threading.Thread(
                    target=channel.consume,
                    args=(queue_iter(queues[name]),),
                    kwargs=kwargs)
                thread.start()
                consumers.append(thread)

    else:
        queues = None  # type: ignore

    dataloader = datamodule.eval_dataloader(trace, batch_size=args.batch)
    res = model.evaluate(dataloader, desc=desc, outputs=queues)
    for thread in consumers:
        thread.join()
    np.savez(out, **res)


def _main(args):

    model = DeepRadar.load_from_experiment(
        args.model, checkpoint=args.checkpoint)
    model = torch.compile(model)
    datamodule = model.get_dataset(args.path, n_workers=args.workers)

    if len(args.traces) == 0:
        raise ValueError("Passed empty `-t [--traces]`.")

    _start = time.time()
    traces = config.load_config(
        *[os.path.join(args.cfg_dir, t) for t in args.traces]
    )["dataset"]["traces"]
    for i, trace in enumerate(traces):
        evaluate(
            model, datamodule, trace, args, f"[{i + 1}/{len(traces)}] {trace}")
    print("Finished evaluating: {:.01f}s".format(time.time() - _start))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    _main(_parse().parse_args())
