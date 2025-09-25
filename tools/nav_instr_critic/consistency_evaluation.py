import json
import jsonlines
import argparse
from pathlib import Path
import re
import sys
import time
import numpy as np
from tqdm import tqdm
import random
import openai
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from functools import wraps
from tools.api_key import OPENAI_BASE_URL, OPENAI_API_KEY, HF_TOKEN


NONE_STR = "==NONE=="


def dict_to_str(a):
    if isinstance(a, dict):
        a = list(a.values())
    a = [v.lower().strip() if isinstance(v, str) else NONE_STR for v in a]
    res = "; ".join(a)
    if NONE_STR in res:
        res = ""
    return res


def remove_no_digits(s):
    return re.sub(r"\D", "", s)


PROMPT_MATCHER_COT = """You are tasked with evaluating the consistency between a generated navigation instruction and the {0} information along the navigation path. Assign a score between 0 and 10, where a higher score indicates better consistency.

Here is the information and instruction:

[Information]
{1}
[Instruction]
{2}

Please provide your evaluation and score. Write your final score in a new line surrounded by angle brackets, e.g., <5>.\n\n"""

PROMPT_MATCHER = """You are tasked with evaluating the consistency between a summarized navigation instruction and the {0} information along the navigation path. Assign a score between 0 and 10, where a higher score indicates better consistency.

Here is the information and instruction:

[Information]
{1}
[Instruction]
{2}

Please provide your final score in a new line surrounded by angle brackets, e.g., <5>.\n\n"""


PROMPT_MATCHER_LLAMA = """You are tasked with evaluating the consistency between a summarized navigation instruction and the {0} information along the navigation path. Assign a score between 0 and 10, where a higher score indicates better consistency.

Here is the information and instruction:

[Information]
{1}
[Instruction]
{2}

Please provide your final score between 0 and 10 on a new line, in the exact form like '<5>'. The score must be a number with no other characters or calculations, and it must be surrounded by angle brackets.\n\n"""


def extract_score(response, score_patterns=None, request=""):
    response_processed = response.replace('"', "").replace("*", "")
    response_processed = re.sub("^.*?:", "", response_processed)
    if score_patterns is None:
        score_patterns = [
            r"<(\d+\.?\d*?)>",
            r"(\d+\.?\d*?) out of 10",
            r"<(\d+\.?\d*?)",
            r"Score: (\d+\.?\d*?)",
            r"a score of (\d+\.?\d*?)",
            r"Overall Score \((\d+\.?\d*?)/10\)",
            r"<(\d+\.?\d*?)/30>",
            r"\s(\d+\.?\d*?)>",
        ]
    score = None
    for pattern in score_patterns:
        m = re.findall(pattern, response_processed)
        if m:
            score = m[-1]
            break
    if score is None:
        print("Fail to get score", flush=True)
        print("User: ", request, flush=True)
        print("Assistant: ", response, flush=True)
        score = 5
    score = float(score)
    if score > 10:
        print("Invalid score is set to 5 ", flush=True)
        print("User: ", request, flush=True)
        print("Assistant: ", response, flush=True)
        score = 5
    return score


class MatcherBase:
    def __init__(self, use_cot=True):
        if use_cot:
            self.system_matcher = PROMPT_MATCHER_COT
        else:
            self.system_matcher = PROMPT_MATCHER


class LLMMatcherQwen(MatcherBase):
    def __init__(self, use_bfloat16=False, use_cot=True, model_name=None):
        if model_name is None:
            self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        else:
            self.model_name = model_name
        self.use_bfloat16 = use_bfloat16
        self.tasks = []
        self.results = []
        self.results_file = "data/llmmatcher_results_qwen{}.json".format(time.time())
        self.batch_size = 32
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="cuda:0"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.backend = "qwen"
        super().__init__(use_cot)

    def prepare_task(self, info, inst, info_type, custom_id):
        s = self.system_matcher.format(info_type.lower(), info, inst)
        messages = [
            {"role": "user", "content": s},
        ]
        self.tasks.append(
            {
                "custom_id": custom_id,
                "info": info,
                "instruction": inst,
                "messages": messages,
            }
        )

    def upload_tasks(self):
        for start in tqdm(range(0, len(self.tasks), self.batch_size)):
            end = min(len(self.tasks), start + self.batch_size)
            batch_ids = [v["custom_id"] for v in self.tasks[start:end]]
            batch_messages = [v["messages"] for v in self.tasks[start:end]]
            batch_info = [v["info"] for v in self.tasks[start:end]]
            batch_inst = [v["instruction"] for v in self.tasks[start:end]]
            batch_text = [
                self.tokenizer.apply_chat_template(
                    v, tokenize=False, add_generation_prompt=True
                )
                for v in batch_messages
            ]
            model_inputs = self.tokenizer(
                batch_text, return_tensors="pt", padding=True, padding_side="left"
            ).to(self.model.device)

            generated_ids = self.model.generate(**model_inputs, max_new_tokens=2048)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            batch_response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            self.results.extend(
                [
                    {
                        "custom_id": batch_ids[i],
                        "info": batch_info[i],
                        "response": batch_response[i],
                        "instruction": batch_inst[i],
                    }
                    for i in range(len(batch_response))
                ]
            )
        for i in tqdm(range(len(self.results))):
            response = self.results[i]["response"]
            score = extract_score(response, request=self.results[i]["custom_id"])
            self.results[i]["score"] = score
        # print("Write results to: ", self.results_file)
        # with open(self.results_file, "w") as f:
        #     json.dump(self.results, f, indent=2)

    def wait_results(self):
        for i in range(0, len(self.results), 3):
            idx1 = self.results[i]["custom_id"].split("-")[0]
            idx2 = self.results[i + 1]["custom_id"].split("-")[0]
            idx3 = self.results[i + 2]["custom_id"].split("-")[0]
            assert idx1 == idx2 and idx2 == idx3, "Wrong results"
            score1 = self.results[i]["score"]
            score2 = self.results[i + 1]["score"]
            score3 = self.results[i + 2]["score"]
            yield score1, score2, score3

    def __call__(self, info, inst, info_type):
        s = self.system_matcher.format(info_type.lower(), info, inst)
        messages = [
            {"role": "user", "content": s},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=2048)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        score = extract_score(response, request=s)

        return score, response


class LLMMatcherLlama(MatcherBase):
    def __init__(self, use_bfloat16=False, use_cot=True):
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.use_bfloat16 = use_bfloat16
        self.hf_token = HF_TOKEN
        self.tasks = []
        self.results = []
        self.results_file = "data/llmmatcher_results_llama{}.json".format(time.time())
        self.batch_size = 32
        if use_bfloat16:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cuda:0",
                torch_dtype="bfloat16",
                token=self.hf_token,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="cuda:0", token=self.hf_token
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backend = "llama"
        self.system_matcher = PROMPT_MATCHER_LLAMA
        self.score_patterns = [
            r"<(\d+\.?\d*?)>",
            r"(\d+\.?\d*?) out of 10",
            r" (\d+\.?\d*?)>",
            r"<(\d+\.?\d*?) ",
            r"a score of (\d+\.?\d*?)",
            r"Overall Score \((\d+\.?\d*?)/10\)",
            r"<(\d+\.?\d*?)/30>",
            r"R(\d+\.?\d*?)>",
            r"= (\d+\.?\d*?)",
            r"≈ (\d+\.?\d*?)",
            r"(\d+\.?\d*?) out of ",
            r"score is (\d+\.?\d*?)",
            r"score is: (\d+\.?\d*?)",
            r"Total score (\d+\.?\d*?)",
            r"Total score: (\d+\.?\d*?)",
            r"Score: (\d+\.?\d*?)",
            r"score: (\d+\.?\d*?)",
            r"is (\d+\.?\d*?)",
        ]
        super().__init__(use_cot)

    def prepare_task(self, info, inst, info_type, custom_id):
        inst = inst.replace("\n", " ")
        s = self.system_matcher.format(info_type.lower(), info, inst)
        messages = [
            {"role": "user", "content": s},
        ]
        self.tasks.append(
            {
                "custom_id": custom_id,
                "info": info,
                "instruction": inst,
                "messages": messages,
            }
        )

    def upload_tasks(self):
        for start in tqdm(range(0, len(self.tasks), self.batch_size)):
            end = min(len(self.tasks), start + self.batch_size)
            batch_ids = [v["custom_id"] for v in self.tasks[start:end]]
            batch_messages = [v["messages"] for v in self.tasks[start:end]]
            batch_info = [v["info"] for v in self.tasks[start:end]]
            batch_inst = [v["instruction"] for v in self.tasks[start:end]]
            batch_text = [
                self.tokenizer.apply_chat_template(
                    v, tokenize=False, add_generation_prompt=True
                )
                for v in batch_messages
            ]
            model_inputs = self.tokenizer(
                batch_text, return_tensors="pt", padding=True, padding_side="left"
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            batch_response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            self.results.extend(
                [
                    {
                        "custom_id": batch_ids[i],
                        "info": batch_info[i],
                        "response": batch_response[i],
                        "instruction": batch_inst[i],
                    }
                    for i in range(len(batch_response))
                ]
            )
        for i in tqdm(range(len(self.results))):
            response = self.results[i]["response"]
            score = extract_score(
                response,
                score_patterns=self.score_patterns,
                request=self.results[i]["custom_id"],
            )
            self.results[i]["score"] = score
        print("Write results to: ", self.results_file)
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def wait_results(self):
        for i in range(0, len(self.results), 3):
            idx1 = self.results[i]["custom_id"].split("-")[0]
            idx2 = self.results[i + 1]["custom_id"].split("-")[0]
            idx3 = self.results[i + 2]["custom_id"].split("-")[0]
            assert idx1 == idx2 and idx2 == idx3, "Wrong results"
            score1 = self.results[i]["score"]
            score2 = self.results[i + 1]["score"]
            score3 = self.results[i + 2]["score"]
            yield score1, score2, score3

    def __call__(self, info, inst, info_type):
        s = self.system_matcher.format(info_type.lower(), info, inst)
        messages = [
            {"role": "user", "content": s},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        score = extract_score(response, request=s)

        return score, response


class LLMMatcherGPT(MatcherBase):
    def __init__(self, use_bfloat16=False, use_cot=True):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.model_name = "gpt-4o-mini"
        self.batch_id = ""
        self.check_interval = 5
        self.backend = "gpt"

        super().__init__(use_cot)

    def __call__(self, info, inst, info_type):
        s = self.system_matcher.format(info_type.lower(), info, inst)
        error = True
        while error:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": s},
                    ],
                )
                if response.choices[0].message.content is None:
                    raise AssertionError("Encounter None response, {}".format(response))
                error = False
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                AssertionError,
            ) as e:
                print(e)
                error = True
                time.sleep(5)
        response = response.choices[0].message.content

        score = extract_score(response, request=s)

        return score, response


class LLMMatcherDeepSeek(MatcherBase):
    def __init__(self, use_bfloat16=False, use_cot=True):
        self.client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.model_name = "deepseek-chat"
        self.batch_id = ""
        self.check_interval = 5
        self.backend = "deepseek"

        super().__init__(use_cot)

    def __call__(self, info, inst, info_type):
        s = self.system_matcher.format(info_type.lower(), info, inst)
        error = True
        while error:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": s},
                    ],
                )
                if response.choices[0].message.content is None:
                    raise AssertionError("Encounter None response, {}".format(response))
                error = False
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                AssertionError,
            ) as e:
                print(e)
                error = True
                time.sleep(5)
        response = response.choices[0].message.content

        score = extract_score(response, request=s)

        return score, response


def run_evaluation_matcher(matcher, folder, splits, inst_alias, tempfile=None):
    scores = {"scene": [], "object": [], "action": [], "average": []}
    info_list = []
    write_interval = 5
    if tempfile is not None:
        with open(tempfile, "r") as f:
            info_to_score = json.load(f)
        print("Read results from: ", tempfile)
    else:
        tempfile = "data/llmmatcher_results_{}{}.json".format(
            matcher.backend, time.time()
        )
        print("Write results to: ", tempfile)
        info_to_score = {}

    for split in splits:
        cur_folder = Path(folder) / split
        info_list.extend(list(cur_folder.glob("*/{}/*.info".format(inst_alias))))
    iter_bar = tqdm(info_list)
    for info_file in iter_bar:
        info_key = str(info_file).replace(str(folder), "")
        if info_key in info_to_score:
            action_score = info_to_score[info_key]["action"]
            scene_score = info_to_score[info_key]["scene"]
            object_score = info_to_score[info_key]["object"]
        else:
            with open(info_file, "r") as f:
                data = json.load(f)
            action_info = dict_to_str(data["actions_compact"])
            scene_info = dict_to_str(data["scenes"])
            object_info = dict_to_str(data["objects"])
            instruction = data["instruction"]

            action_score, action_response = matcher(
                action_info, instruction, info_type="Action"
            )
            scene_score, scene_response = matcher(
                scene_info, instruction, info_type="Scene"
            )
            object_score, object_response = matcher(
                object_info, instruction, info_type="Object"
            )
            info_to_score[info_key] = {
                "action": action_score,
                "action_info": action_info,
                "action_response": action_response,
                "scene": scene_score,
                "scene_info": scene_info,
                "scene_response": scene_response,
                "object": object_score,
                "object_info": object_info,
                "object_response": object_response,
                "instruction": instruction,
                "average": (scene_score + object_score + action_score) / 3,
            }
            if iter_bar.n % write_interval == 0:
                with open(tempfile, "w") as f:
                    json.dump(info_to_score, f, indent=2)

        iter_bar.set_postfix(
            action=action_score, scene=scene_score, object=object_score
        )
        scores["action"].append(action_score)
        scores["scene"].append(scene_score)
        scores["object"].append(object_score)
        scores["average"].append((action_score + scene_score + object_score) / 3)
    results = {k: np.mean(v) for k, v in scores.items()}
    return results


def run_evaluation_matcher_batch(matcher, folder, splits, inst_alias):
    scores = {"scene": [], "object": [], "action": [], "average": []}
    info_list = []
    for split in splits:
        cur_folder = Path(folder) / split
        info_list.extend(list(cur_folder.glob("*/{}/*.info".format(inst_alias))))
    iter_bar = tqdm(enumerate(info_list))
    for index, info_file in iter_bar:
        with open(info_file, "r") as f:
            data = json.load(f)
        action_info = dict_to_str(data["actions_compact"])
        scene_info = dict_to_str(data["scenes"])
        object_info = dict_to_str(data["objects"])
        instruction = data["instruction"]

        matcher.prepare_task(
            action_info,
            instruction,
            info_type="Action",
            custom_id=str(info_file).replace(str(folder), "") + "-Action",
        )
        matcher.prepare_task(
            scene_info,
            instruction,
            info_type="Scene",
            custom_id=str(info_file).replace(str(folder), "") + "-Scene",
        )
        matcher.prepare_task(
            object_info,
            instruction,
            info_type="Object",
            custom_id=str(info_file).replace(str(folder), "") + "-Object",
        )
    matcher.upload_tasks()
    iter_bar = tqdm(matcher.wait_results())
    for action_score, scene_score, object_score in iter_bar:
        iter_bar.set_postfix(
            action=action_score, scene=scene_score, object=object_score
        )
        scores["action"].append(action_score)
        scores["scene"].append(scene_score)
        scores["object"].append(object_score)
        scores["average"].append((scene_score + object_score + action_score) / 3)
    results = {k: np.mean(v) for k, v in scores.items()}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="matcher", required=False, help="Run mode"
    )
    parser.add_argument(
        "--backend", type=str, default="qwen", required=False, help="Which LLM is used"
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default="data/vlnce_traj_action_clean",
        required=False,
        help="Data folder",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default="val_seen",
        required=True,
        help="Evaluated splits",
    )
    parser.add_argument(
        "--inst_alias1",
        type=str,
        default="inst_vo_gpt_gpt",
        required=False,
        help="The test instruction folder name",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        default=False,
        required=False,
        help="Whether to use bfloat16",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        default=False,
        required=False,
        help="Whether to use CoT",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default="data/speaker_evaluation",
        required=False,
        help="The folder that saves json files",
    )
    args = parser.parse_args()
    print(args, flush=True)
    eval_splits_cat = "_".join(args.splits)
    if args.mode == "matcher":
        if args.backend == "gpt":
            matcher = LLMMatcherGPT(args.bfloat16, args.cot)
            res = run_evaluation_matcher(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        elif args.backend == "deepseek":
            matcher = LLMMatcherDeepSeek(args.bfloat16, args.cot)
            res = run_evaluation_matcher(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        elif args.backend == "llama":
            matcher = LLMMatcherLlama(args.bfloat16, args.cot)
            res = run_evaluation_matcher(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        elif args.backend == "qwenbatch":
            matcher = LLMMatcherQwen(args.bfloat16, args.cot)
            res = run_evaluation_matcher_batch(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        elif args.backend == "qwenlargebatch":
            matcher = LLMMatcherQwen(
                args.bfloat16, args.cot, model_name="Qwen/Qwen2.5-14B-Instruct"
            )
            res = run_evaluation_matcher_batch(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        elif args.backend == "qwenhugebatch":
            matcher = LLMMatcherQwen(
                args.bfloat16, args.cot, model_name="Qwen/Qwen2.5-32B-Instruct"
            )
            res = run_evaluation_matcher_batch(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        elif args.backend == "llamabatch":
            matcher = LLMMatcherLlama(args.bfloat16, args.cot)
            res = run_evaluation_matcher_batch(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        else:
            matcher = LLMMatcherQwen(args.bfloat16, args.cot)
            res = run_evaluation_matcher(
                matcher, args.folder, args.splits, args.inst_alias1
            )
        with open(
            Path(args.output_folder)
            / "{}_llmmatcher_{}_{}.json".format(
                args.inst_alias1, eval_splits_cat, args.backend
            ),
            "w",
        ) as f:
            json.dump(res, f, indent=2)
        print("Matcher {}: ".format(args.inst_alias1), res)
    elif args.mode == "comparator":
        raise NotImplementedError("Comparator is deprecated")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
