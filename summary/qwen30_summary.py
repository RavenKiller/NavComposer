from typing import List, Optional

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import re
from tools.tools import im_2_b64
from tools.registry import registry


@registry.register_summary
class Qwen30Summary:
    def __init__(
        self,
        config,
    ):
        self.qwen30_config = config.summary_config.qwen30_config
        self.model_name = self.qwen30_config.model_name
        self.run_precision = config.run_precision
        self.enable_thinking = self.qwen30_config.enable_thinking
        if self.run_precision == "bfloat16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.temperature = self.qwen30_config.temperature
        self.system_prompt = self.qwen30_config.system_prompt

    def __call__(self, s):
        input_msg = ""
        if isinstance(s, str):
            input_msg = s
        elif isinstance(s, list):
            input_msg = "\n".join([f"{i} " + v for i, v in enumerate(s)])
        else:
            raise TypeError("Invalid input format")

        messages = [
            {
                "role": "user",
                "content": self.system_prompt + "\nHere is the list:\n" + input_msg,
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        res = content.replace('"', "")
        res = re.sub("^.*?:", "", res)
        res = res.strip()
        return res, thinking_content


if __name__ == "__main__":
    from tools.config import get_config, print_config

    config = get_config(config_file="./main.yaml")
    dist.init_process_group()
    tmp = Qwen30Summary(config)

    summary_str1 = "0. bear. Empty room. Empty mattress on the floor, located in a sparsely furnished room. \n1. step out. Hallway. White door at the left side of the room. \n2. left. Hallway. Long table positioned at the end of a narrow white hallway. \n3. advance to. Unfinished basement. Significant object: Table  \nLocation: Center of the room. \n4. swing left. Empty basement or storage room. Metal table located in the center of the room. \n5. go past. Basement or unfinished room. Long table against the wall in an empty room. \n6. veer. Basement. Empty room with gray carpet and a black table in the corner. \n7. move out. Basement. Empty basement space with neutral-colored walls. \n8. head right. Hallway. Significant object: Doorway at the end of the hallway. Location: White corridor. \n9. cross. Possible basement. Empty wall in a basement or unfinished room. \n10. take a left turn. Hallway. Long hallway with light-colored walls and a door at the end. \n11. move ahead. Hallway. Main hallway, located centrally with doors at the end. \n12. turn leftward. Hallway. White door at the end of the hallway. \n13. leave. Hallway. Significant object: Doorway at the end of the hall. Location: Center of the hallway. \n14. steer. Hallway or entryway. Open door in a hallway. \n15. keep going straight. Unfinished room. Open door leading to a renovated room. \n16. halt. Under construction room. Light fixture in the ceiling of an unfinished room. "

    summary_str2 = "0. turn. Living room. Large flat-screen TV on the wall. \n1. go. Living room. Tall vase with flowers on a table near the window. \n2. turn right. Entryway or foyer. Grand piano located in the entryway. \n3. move forward. Hallway. Staircase at the end of the hallway. \n4. turn left. Entryway. Staircase at the end of the hallway. \n5. get out. Hallway. Doorway leading to hallway. \n6. hold. Hallway. Significant object: Wall.  \nLocation: Hallway. "

    print(tmp(summary_str1))
    print("====")
    print(tmp(summary_str2))
