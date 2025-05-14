import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch.distributed as dist


from tools.registry import registry


@registry.register_object
class BLIP2Object:
    def __init__(self, config, model=None):
        self.blip2_config = config.object_config.blip2_config
        self.device = torch.device(dist.get_rank())
        self.processor = Blip2Processor.from_pretrained(self.blip2_config.model_name)
        if model:
            self.model = model
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.blip2_config.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        self.question = self.blip2_config.system_prompt

    def set_question(self, question):
        self.question = question
        return self

    def __call__(self, images, question=None):
        if question is None:
            question = self.question
        inputs = [
            self.processor(image, question, return_tensors="pt").to(
                self.device, torch.float16
            )
            for image in images
        ]
        pixel_values = torch.cat([v.pixel_values for v in inputs], dim=0)
        input_ids = torch.cat([v.input_ids for v in inputs], dim=0)
        out = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            do_sample=True,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.8,
            top_k=10,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=0.7,
        )

        results = [
            self.processor.decode(v, skip_special_tokens=True).strip() + "."
            for v in out
        ]
        return results


if __name__ == "__main__":
    from tools.config import get_config, print_config
    from PIL import Image

    config = get_config(config_file="./main.yaml")
    dist.init_process_group()
    tmp = BLIP2Object(config)

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
