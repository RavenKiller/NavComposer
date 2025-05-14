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

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.tools import im_2_b64, remove_list_format


from tools.registry import registry


@registry.register_object
class GPTObject:
    def __init__(self, config):
        self.gpt_config = config.object_config.gpt_config
        self.OPENAI_API_KEY = self.gpt_config.openai_api_key
        self.BASE_URL = self.gpt_config.base_url
        self.client = openai.OpenAI(api_key=self.OPENAI_API_KEY, base_url=self.BASE_URL)
        self.temperature = self.gpt_config.temperature
        self.system_prompt = self.gpt_config.system_prompt
        self.model_name = self.gpt_config.model_name

    def process_single_image(self, s):
        error = True
        while error:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.system_prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,{}".format(
                                            im_2_b64(s)
                                        )
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=50,
                )
                error = False
                if response.choices[0].message.content is None:
                    raise AssertionError("Encounter None response, {}".format(response))
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                openai.BadRequestError,
                AssertionError,
            ) as e:
                print(e)
                error = True
                time.sleep(5)
        # print(response.choices[0].message.content)
        return remove_list_format(response.choices[0].message.content)

    def __call__(self, images):
        return [self.process_single_image(v) for v in images]


if __name__ == "__main__":
    from tools.config import get_config, print_config
    from PIL import Image

    config = get_config(config_file="./main.yaml")
    # dist.init_process_group()
    tmp = GPTObject(config)

    tmp.question = "Describe the most significant object and its relative location in real-world as short as possible in the phrase format."
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/6995/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/734/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)

    tmp.question = "Describe the most significant object and its relative location in real-world as short as possible."
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/6995/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/734/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)

    tmp.question = "Describe the most significant object with its relative location as short as possible in the phrase format."
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/6995/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/734/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
