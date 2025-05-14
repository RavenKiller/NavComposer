from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from tools.tools import remove_list_format
import torch
import torch.distributed as dist


from tools.tools import im_2_b64, im_resize
from tools.registry import registry


@registry.register_object
class LlavaObject:
    def __init__(
        self,
        config,
        model=None,
        # model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        # question="Describe two objects and their locations in one sentence as short as possible. ",
        # device="cuda:2",
    ):
        self.llava_config = config.object_config.llava_config
        self.device = torch.device(dist.get_rank())
        self.processor = LlavaNextProcessor.from_pretrained(
            self.llava_config.model_name
        )
        if model:
            self.model = model
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.llava_config.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        self.question = self.llava_config.system_prompt

    def set_question(self, question):
        self.question = question
        return self

    def __call__(self, images, question=None):
        if question is None:
            question = self.question

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(images, [prompt] * len(images), return_tensors="pt").to(
            self.device, torch.float16
        )
        output = self.model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        return [
            remove_list_format(
                self.processor.decode(v, skip_special_tokens=True).split("[/INST]")[1]
            )
            for v in output
        ]


if __name__ == "__main__":
    from tools.config import get_config, print_config
    from PIL import Image

    config = get_config(config_file="./main.yaml")
    dist.init_process_group()
    tmp = LlavaObject(config)

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
