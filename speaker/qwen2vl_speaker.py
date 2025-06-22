from PIL import Image
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

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


@registry.register_speaker
class Qwen25VLSpeaker:
    def __init__(
        self,
        config,
        model=None,
    ):
        self.qwen2_config = config.speaker_config.qwen2_config
        self.device = torch.device(dist.get_rank())
        self.processor = AutoProcessor.from_pretrained(self.qwen2_config.model_name)
        if model:
            self.model = model
        else:
            if "Qwen2.5" in self.qwen2_config.model_name:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.qwen2_config.model_name,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.qwen2_config.model_name,
                    torch_dtype="auto",
                    device_map=self.device,
                )
        self.system_prompt = self.qwen2_config.system_prompt
        self.remove_introductory = self.qwen2_config.remove_introductory
        self.temperature = self.qwen2_config.temperature
        self.top_p = self.qwen2_config.top_p
        self.top_k = self.qwen2_config.top_k
        self.repetition_penalty = self.qwen2_config.repetition_penalty
        self.max_frames = config.speaker_config.max_frames
        self.fps = config.speaker_config.fps
        self.downsample = config.speaker_config.downsample
        self.info = {}

    def __call__(self, images):
        images = images[:: self.downsample]
        file_list = ["file://" + str(v) for v in images]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": file_list,
                        "fps": self.fps,
                    },
                    {"type": "text", "text": self.system_prompt},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # if self.remove_introductory:
        #     output_texts = [remove_introductory(v) for v in output_texts]
        return output_texts[0]

    def generate(self, source):
        """Source can be a Path, str, list of image paths"""
        if isinstance(source, Path) or isinstance(source, str):
            p = Path(source)
            source = []
            for v in p.glob("*.jpg"):
                source.append(v)
        source = natsort.natsorted(source, key=str)
        source = source[: self.max_frames]
        res = self(source)
        self.info = {"instruction": res}
        return res

    def get_detail(self, return_json=False):
        if return_json:
            return self.info
        else:
            return json.dumps(self.info, indent=2)
