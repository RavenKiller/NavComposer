from typing import List, Optional

import fire
import re

import torch
import torch.distributed as dist
from transformers import pipeline
import huggingface_hub

from tools.registry import registry


@registry.register_summary
class Gemma2Summary:
    def __init__(
        self,
        config,
    ):
        self.gemma2_config = config.summary_config.gemma2_config
        self.model_name = self.gemma2_config.model_name
        self.max_new_tokens = self.gemma2_config.max_new_tokens

        # huggingface_hub.login(token = self.gemma2_config.hf_token)

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device(dist.get_rank()),
        )
        self.pipe.generation_config.update(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.05,
        )
        self.system = self.gemma2_config.system_prompt

    def __call__(self, s):
        input_msg = ""
        if isinstance(s, str):
            input_msg = s
        elif isinstance(s, list):
            input_msg = "\n".join([f"{i} " + v for i, v in enumerate(s)])
        else:
            raise TypeError("Invalid input format")

        dialogs = [
            {"role": "user", "content": self.system + "\n" + input_msg},
        ]

        results = self.pipe(dialogs, max_new_tokens=self.max_new_tokens)
        res = results[0]["generated_text"][-1]["content"].strip()
        res = re.sub("^.*?:", "", res)
        return res


if __name__ == "__main__":
    from tools.config import get_config, print_config

    config = get_config(config_file="./main.yaml")
    dist.init_process_group()
    tmp = Gemma2Summary(config)

    summary_str1 = "0. bear. Empty room. Empty mattress on the floor, located in a sparsely furnished room. \n1. step out. Hallway. White door at the left side of the room. \n2. left. Hallway. Long table positioned at the end of a narrow white hallway. \n3. advance to. Unfinished basement. Significant object: Table  \nLocation: Center of the room. \n4. swing left. Empty basement or storage room. Metal table located in the center of the room. \n5. go past. Basement or unfinished room. Long table against the wall in an empty room. \n6. veer. Basement. Empty room with gray carpet and a black table in the corner. \n7. move out. Basement. Empty basement space with neutral-colored walls. \n8. head right. Hallway. Significant object: Doorway at the end of the hallway. Location: White corridor. \n9. cross. Possible basement. Empty wall in a basement or unfinished room. \n10. take a left turn. Hallway. Long hallway with light-colored walls and a door at the end. \n11. move ahead. Hallway. Main hallway, located centrally with doors at the end. \n12. turn leftward. Hallway. White door at the end of the hallway. \n13. leave. Hallway. Significant object: Doorway at the end of the hall. Location: Center of the hallway. \n14. steer. Hallway or entryway. Open door in a hallway. \n15. keep going straight. Unfinished room. Open door leading to a renovated room. \n16. halt. Under construction room. Light fixture in the ceiling of an unfinished room. "

    summary_str2 = "0. turn. Living room. Large flat-screen TV on the wall. \n1. go. Living room. Tall vase with flowers on a table near the window. \n2. turn right. Entryway or foyer. Grand piano located in the entryway. \n3. move forward. Hallway. Staircase at the end of the hallway. \n4. turn left. Entryway. Staircase at the end of the hallway. \n5. get out. Hallway. Doorway leading to hallway. \n6. hold. Hallway. Significant object: Wall.  \nLocation: Hallway. "

    print(tmp(summary_str1))
    print("====")
    print(tmp(summary_str2))
