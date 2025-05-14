from typing import List, Optional

import fire
import re

import transformers
import torch
import torch.distributed as dist
from tools.registry import registry


@registry.register_summary
class Llama3Summary:
    def __init__(
        self,
        config,
    ):
        self.llama3_config = config.summary_config.llama3_config
        self.ckpt_dir = self.llama3_config.ckpt_dir
        self.tokenizer_path = self.llama3_config.tokenizer_path
        self.temperature = self.llama3_config.temperature
        self.top_p = self.llama3_config.top_p
        self.max_seq_len = self.llama3_config.max_seq_len
        self.max_batch_size = self.llama3_config.max_batch_size
        self.max_gen_len = self.llama3_config.max_gen_len

        self.model_id = self.llama3_config.hf_model_id

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=self.llama3_config.hf_token,
        )
        self.system_prompt = self.llama3_config.system_prompt
        # self.system_prompt = "You are professional in robotics and vision-and-language navigation research. Given a list of navigation action and observations, your task is to summarize the descriptions into a brief, fluent and concise robot navigation instruction as short as possible. You should skip unnecessary information that does not contribute to the navigation. Do not use list format."

    def __call__(self, s):
        input_msg = ""
        if isinstance(s, str):
            input_msg = s
        elif isinstance(s, list):
            input_msg = "\n".join([f"{i} " + v for i, v in enumerate(s)])
        else:
            raise TypeError("Invalid input format")

        dialogs = [
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_msg},
            ]
        ]

        results = self.pipeline(
            dialogs,
            max_new_tokens=self.max_gen_len,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )
        res = results[0][0]["generated_text"][-1]["content"].replace('"', "")
        res = re.sub("^.*?:", "", res)
        res = res.strip()
        return res


if __name__ == "__main__":
    from tools.config import get_config, print_config

    config = get_config(config_file="./main.yaml")
    dist.init_process_group()
    tmp = Llama3Summary(config)

    summary_str1 = "0. bear. Empty room. Empty mattress on the floor, located in a sparsely furnished room. \n1. step out. Hallway. White door at the left side of the room. \n2. left. Hallway. Long table positioned at the end of a narrow white hallway. \n3. advance to. Unfinished basement. Significant object: Table  \nLocation: Center of the room. \n4. swing left. Empty basement or storage room. Metal table located in the center of the room. \n5. go past. Basement or unfinished room. Long table against the wall in an empty room. \n6. veer. Basement. Empty room with gray carpet and a black table in the corner. \n7. move out. Basement. Empty basement space with neutral-colored walls. \n8. head right. Hallway. Significant object: Doorway at the end of the hallway. Location: White corridor. \n9. cross. Possible basement. Empty wall in a basement or unfinished room. \n10. take a left turn. Hallway. Long hallway with light-colored walls and a door at the end. \n11. move ahead. Hallway. Main hallway, located centrally with doors at the end. \n12. turn leftward. Hallway. White door at the end of the hallway. \n13. leave. Hallway. Significant object: Doorway at the end of the hall. Location: Center of the hallway. \n14. steer. Hallway or entryway. Open door in a hallway. \n15. keep going straight. Unfinished room. Open door leading to a renovated room. \n16. halt. Under construction room. Light fixture in the ceiling of an unfinished room. "

    summary_str2 = "0. turn. Living room. Large flat-screen TV on the wall. \n1. go. Living room. Tall vase with flowers on a table near the window. \n2. turn right. Entryway or foyer. Grand piano located in the entryway. \n3. move forward. Hallway. Staircase at the end of the hallway. \n4. turn left. Entryway. Staircase at the end of the hallway. \n5. get out. Hallway. Doorway leading to hallway. \n6. hold. Hallway. Significant object: Wall.  \nLocation: Hallway. "

    print(tmp(summary_str1))
    print(">")
    print(tmp(summary_str2))
