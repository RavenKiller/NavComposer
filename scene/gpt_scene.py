import sys
import os
from PIL import Image
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from functools import partial
import openai
from itertools import groupby
from io import BytesIO
import base64
import tempfile
import re
import time
from collections import OrderedDict

from tools.tools import im_2_b64, remove_list_format


from tools.registry import registry


@registry.register_scene
class GPTScene:
    def __init__(self, config):
        self.gpt_config = config.scene_config.gpt_config
        self.OPENAI_API_KEY = self.gpt_config.openai_api_key
        self.BASE_URL = self.gpt_config.base_url
        self.client = openai.OpenAI(api_key=self.OPENAI_API_KEY, base_url=self.BASE_URL)
        self.temperature = self.gpt_config.temperature
        self.system_prompt = self.gpt_config.system_prompt
        self.model_name = self.gpt_config.model_name

    def process_single_image(self, s):
        if not isinstance(s, list):
            s = [s]
        error = True
        content = [
            {"type": "text", "text": self.system_prompt},
        ]
        for v in s:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{}".format(im_2_b64(v))
                    },
                }
            )
        while error:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    max_tokens=50,
                )
                error = False
                if (
                    len(response.choices) == 0
                    or response.choices[0].message.content is None
                ):
                    raise AssertionError("Encounter None response, {}".format(response))
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                AssertionError,
            ) as e:
                print(e)
                error = True
                time.sleep(10)
        return response.choices[0].message.content

    def __call__(self, images):
        return [self.process_single_image(v) for v in images]


if __name__ == "__main__":
    from tools.config import get_config, print_config

    config = get_config(config_file="./main.yaml")
    tmp = GPTScene(config)
    tmp.system_prompt = (
        "What scene is it? Please answer as short as possible in the phrase format."
    )
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/251/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
