import openai
import time
from tools.registry import registry


@registry.register_summary
class GPTSummary:
    def __init__(self, config):
        self.gpt_config = config.summary_config.gpt_config
        self.OPENAI_API_KEY = self.gpt_config.openai_api_key
        self.BASE_URL = self.gpt_config.base_url
        self.client = openai.OpenAI(api_key=self.OPENAI_API_KEY, base_url=self.BASE_URL)
        self.temperature = self.gpt_config.temperature
        # self.system_prompt = "You are professional in robotic vision-and-language navigation research. You will receive a list of image descriptions from a navigation path. Every description contains the motion, the scene and the objects. Your task is to merge these descriptions into a fluent and concise robot navigation instruction as short as possible. You are allowed to skip unnecessary information that does not contribute to the navigation."
        self.system_prompt = self.gpt_config.system_prompt
        self.model_name = self.gpt_config.model_name

    def __call__(self, s):
        input_msg = ""
        if isinstance(s, str):
            input_msg = s
        elif isinstance(s, list):
            input_msg = "\n".join([f"{i} " + v for i, v in enumerate(s)])
        else:
            raise TypeError("Invalid input format")
        error = True
        while error:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": self.system_prompt
                            + "\nHere is the list:\n"
                            + input_msg,
                        },
                    ],
                    temperature=self.temperature,
                )
                error = False
                if response.choices[0].message.content is None:
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


if __name__ == "__main__":
    from tools.config import get_config, print_config

    config = get_config(config_file="./main.yaml")
    tmp = GPTSummary(config)

    summary_str1 = "0. bear. Empty room. Empty mattress on the floor, located in a sparsely furnished room. \n1. step out. Hallway. White door at the left side of the room. \n2. left. Hallway. Long table positioned at the end of a narrow white hallway. \n3. advance to. Unfinished basement. Significant object: Table  \nLocation: Center of the room. \n4. swing left. Empty basement or storage room. Metal table located in the center of the room. \n5. go past. Basement or unfinished room. Long table against the wall in an empty room. \n6. veer. Basement. Empty room with gray carpet and a black table in the corner. \n7. move out. Basement. Empty basement space with neutral-colored walls. \n8. head right. Hallway. Significant object: Doorway at the end of the hallway. Location: White corridor. \n9. cross. Possible basement. Empty wall in a basement or unfinished room. \n10. take a left turn. Hallway. Long hallway with light-colored walls and a door at the end. \n11. move ahead. Hallway. Main hallway, located centrally with doors at the end. \n12. turn leftward. Hallway. White door at the end of the hallway. \n13. leave. Hallway. Significant object: Doorway at the end of the hall. Location: Center of the hallway. \n14. steer. Hallway or entryway. Open door in a hallway. \n15. keep going straight. Unfinished room. Open door leading to a renovated room. \n16. halt. Under construction room. Light fixture in the ceiling of an unfinished room. "

    summary_str2 = "0. turn. Living room. Large flat-screen TV on the wall. \n1. go. Living room. Tall vase with flowers on a table near the window. \n2. turn right. Entryway or foyer. Grand piano located in the entryway. \n3. move forward. Hallway. Staircase at the end of the hallway. \n4. turn left. Entryway. Staircase at the end of the hallway. \n5. get out. Hallway. Doorway leading to hallway. \n6. hold. Hallway. Significant object: Wall.  \nLocation: Hallway. "

    print(tmp(summary_str1))
    print("====")
    print(tmp(summary_str2))
