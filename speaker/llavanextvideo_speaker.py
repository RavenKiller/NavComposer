from PIL import Image
import numpy as np
from pathlib import Path
from itertools import groupby
from collections import OrderedDict
import random
import json
import natsort

import torch
import torch.distributed as dist

from tools.registry import registry
from tools.tools import diversify_action, diversify_element, im_resize

from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


def read_video_from_images(path, downsample=1):
    source = []
    for v in Path(path).glob("*.jpg"):
        source.append(v)
    source = natsort.natsorted(source, key=str)
    video = [np.array(Image.open(v)) for v in source[::downsample]]
    return np.stack(video)


@registry.register_speaker
class LlavaNextVideoSpeaker:
    def __init__(
        self,
        config,
        model=None,
    ):
        self.llavanextvideo_config = config.speaker_config.llavanextvideo_config
        self.device = torch.device(dist.get_rank())
        self.processor = LlavaNextVideoProcessor.from_pretrained(
            self.llavanextvideo_config.model_name
        )
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.llavanextvideo_config.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.system_prompt = self.llavanextvideo_config.system_prompt
        self.max_frames = config.speaker_config.max_frames
        self.fps = config.speaker_config.fps
        self.downsample = config.speaker_config.downsample
        self.info = {}

    def __call__(self, images):
        video = [np.array(Image.open(v)) for v in images]
        video = np.stack(video)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                    {"type": "video"},
                ],
            }
        ]
        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs_video = self.processor(
            text=prompt, videos=video, padding=True, return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(
            **inputs_video,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )
        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = response.split("ASSISTANT:")[1].strip()
        return response

    def generate(self, source):
        """Source can be a Path, str, list of image paths"""
        if isinstance(source, Path) or isinstance(source, str):
            p = Path(source)
            source = []
            for v in p.glob("*.jpg"):
                source.append(v)
        source = natsort.natsorted(source, key=str)
        source = source[: self.max_frames : self.downsample]
        res = self(source)
        self.info = {"instruction": res}
        return res

    def get_detail(self, return_json=False):
        if return_json:
            return self.info
        else:
            return json.dumps(self.info, indent=2)
