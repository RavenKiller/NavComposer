import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import action
import object
import scene
import summary
import speaker

from tools.config import get_config, print_config
from tools.registry import registry
from tools.tools import set_seed

import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


import torch.nn as nn
import torch


def count_parameters_in_object(obj):
    total = 0
    visited = set()

    def visit(o):
        nonlocal total
        if id(o) in visited:
            return
        visited.add(id(o))

        if isinstance(o, nn.Module):
            total += sum(p.numel() for p in o.parameters())
        elif isinstance(o, (list, tuple, set)):
            for item in o:
                visit(item)
        elif isinstance(o, dict):
            for item in o.values():
                visit(item)
        else:
            for attr_name in dir(o):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(o, attr_name)
                    visit(attr)
                except Exception:
                    continue

    visit(obj)
    return total


def list_available_modules(config):
    mapping = registry.get_mapping()
    for k, v in mapping.items():
        print(k, end=":\n")
        for k1 in v:
            print(f"  {k1}")


def get_key_frame_index(config):
    config.action_config.vo_config.post_process = 1
    config.speaker_config.use_enter_leave = False
    speaker_cls = registry.get_speaker(config.speaker_config.name)
    speaker = speaker_cls(config)

    for image_path in tqdm(list(Path("data/vlnce_traj_action/val_seen").glob("*/rgb"))):
        index_path = image_path.parent / "key_frame"
        os.makedirs(index_path, exist_ok=True)
        inst_gen = speaker.generate(image_path)
        info = speaker.get_detail(return_json=True)
        index = [int(v) for v in info["actions_compact"].keys()]
        if 0 not in index:
            index = [0] + index
        with open(index_path / "0.json", "w") as f:
            f.write(json.dumps(index))
    for image_path in tqdm(
        list(Path("data/vlnce_traj_action/val_unseen").glob("*/rgb"))
    ):
        index_path = image_path.parent / "key_frame"
        os.makedirs(index_path, exist_ok=True)
        inst_gen = speaker.generate(image_path)
        info = speaker.get_detail(return_json=True)
        index = [int(v) for v in info["actions_compact"].keys()]
        if 0 not in index:
            index = [0] + index
        with open(index_path / "0.json", "w") as f:
            f.write(json.dumps(index))


def run_test(config):
    speaker_cls = registry.get_speaker(config.speaker_config.name)
    speaker = speaker_cls(config)

    for image_path in tqdm(list(Path(config.run_folder).glob("*/rgb"))):
        for i in range(config.generate_num):
            inst_gen_path = image_path.parent / config.generate_alias
            if os.path.exists(inst_gen_path / f"{i}.txt") and config.skip_exist:
                print("skipping {} {}".format(image_path, i))
                continue
            os.makedirs(inst_gen_path, exist_ok=True)
            inst_gen = speaker.generate(image_path)
            print(speaker.get_detail())


def run(config):
    speaker_cls = registry.get_speaker(config.speaker_config.name)
    speaker = speaker_cls(config)
    num_params = count_parameters_in_object(speaker)
    print(f"Total parameters: {num_params:,}")
    max_mem = 0

    pbar = tqdm(list(Path(config.run_folder).glob("*/rgb")))
    for image_path in pbar:
        pbar.set_description(str(Path(*Path(image_path).parts[-3:])))
        for i in range(config.generate_num):
            inst_gen_path = image_path.parent / config.generate_alias
            if (
                os.path.exists(inst_gen_path / f"{i}.txt")
                and os.path.exists(inst_gen_path / f"{i}.info")
                and config.skip_exist
            ):
                print("skipping {} {}".format(image_path, i))
                continue
            os.makedirs(inst_gen_path, exist_ok=True)
            inst_gen = speaker.generate(image_path)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            max_mem = max(max_mem, meminfo.used)
            print("GPU Memory (MB) ", meminfo.used / 1024**2)
            # print("", flush=True)
            # print(inst_gen, flush=True)
            with open(inst_gen_path / f"{i}.txt", "w") as f:
                f.write(inst_gen)
            with open(inst_gen_path / f"{i}.info", "w") as f:
                f.write(speaker.get_detail())
    pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=False,
        help="The path of extra config file",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    config = get_config(config_file=args.config_file, opts=args.opts)
    set_seed(config.random_seed + 1)
    if config.log_config:
        print_config(config)

    if config.run_type == "generate":
        run(config)
    elif config.run_type == "generate_test":
        run_test(config)
    elif config.run_type == "keyframe":
        get_key_frame_index(config)
    elif config.run_type == "listmodules":
        list_available_modules(config)


if __name__ == "__main__":
    main()
