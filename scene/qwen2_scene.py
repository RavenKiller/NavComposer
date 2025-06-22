from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info

import torch
import torch.distributed as dist


from tools.tools import im_2_b64, im_resize
from tools.registry import registry


@registry.register_scene
class Qwen2Scene:
    def __init__(
        self,
        config,
        model=None,
    ):
        self.qwen2_config = config.scene_config.qwen2_config
        self.device = torch.device(dist.get_rank())
        self.processor = AutoProcessor.from_pretrained(self.qwen2_config.model_name)
        self.run_precision = config.run_precision
        if model:
            self.model = model
        else:
            if "Qwen2.5" in self.qwen2_config.model_name:
                if self.run_precision == "bfloat16":
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.qwen2_config.model_name,
                        device_map=self.device,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                    )
                else:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.qwen2_config.model_name,
                        device_map=self.device,
                    )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.qwen2_config.model_name, device_map=self.device
                )
        self.question = self.qwen2_config.system_prompt
        self.temperature = self.qwen2_config.temperature - 0.1
        self.top_p = self.qwen2_config.top_p
        self.top_k = self.qwen2_config.top_k
        self.repetition_penalty = self.qwen2_config.repetition_penalty

    def set_question(self, question):
        self.question = question
        return self

    def __call__(self, images, question=None):
        if question is None:
            question = self.question
        if not isinstance(images[0], list):
            images = [[v] for v in images]
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "data:image/jpeg;base64,{}".format(im_2_b64(s)),
                        }
                        for s in ss
                    ]
                    + [{"type": "text", "text": question}],
                },
            ]
            for ss in images
        ]
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Batch Inference
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
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
        return output_texts


if __name__ == "__main__":
    from tools.config import get_config, print_config
    from PIL import Image

    config = get_config(config_file="./main.yaml")
    dist.init_process_group()
    tmp = Qwen2Scene(config)
    tmp.question = (
        "What room is it? Please answer as short as possible in the phrase format."
    )
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/6995/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/734/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)

    tmp.question = (
        "What scene is it? Please answer as short as possible in the phrase format."
    )
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/6995/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
    images = [Image.open("data/vlnce_traj_action_clean/val_seen/734/rgb/0_0.jpg")]
    res = tmp(images)
    print(res)
