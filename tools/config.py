from typing import List, Optional
import copy
from typing import List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))


@dataclass
class VOConfig:
    feature: str = "SIFT"  # SIFT or ORB
    matcher_type: str = "FLANN"  # FLANN or BF
    camera_params: tuple = (
        224.0,
        224.0,
        112.0,
        112.0,
        111.0,
        111.0,
    )  # (width, height, fx, fy, cx, cy)
    angle_threshold: int = 7
    post_process: int = 2


@dataclass
class RNConfig:
    model_name: str = "microsoft/resnet-50"
    hidden_size: int = 2048
    ckpt_path: str = "data/model_weights/best_microsoft_resnet-50.pth"
    post_process: int = 2


@dataclass
class Dinov2Config:
    model_name: str = "facebook/dinov2-base"
    hidden_size: int = 768
    ckpt_path: str = "data/model_weights/best_dinov2_base.pth"
    post_process: int = 2


@dataclass
class BLIP2Config:
    model_name: str = "Salesforce/blip2-flan-t5-xl"
    system_prompt: str = ""


@dataclass
class DETRConfig:
    model_name: str = "facebook/detr-resnet-50"


@dataclass
class SWAGConfig:
    model_name: str = "vit_l16_in1k"
    resolution: int = 512
    batch_size: int = -1
    topk: int = 1
    imagenet_id_to_name: str = "data/model_weights/in_cls_idx.json"


@dataclass
class MAEConfig:
    model_name: str = "vit_base_patch16"
    ckpt_path: str = "data/model_weights/mae_tune_vit_base.pth"
    batch_size: int = -1
    topk: int = 1
    nb_classes: int = 1000
    global_pool: bool = True
    drop_path: float = 0.1
    places365_id_to_name: str = "data/model_weights/places365_cls_idx.json"


@dataclass
class GPTConfig:
    base_url: str = "base_url"
    openai_api_key: str = "api_key"
    model_name: str = "gpt-4o-mini"
    system_prompt: str = ""
    temperature: float = 0.7


@dataclass
class Llama3Config:
    ckpt_dir: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct/"
    tokenizer_path: str = (
        "llama-models/models/llama3_1/Meta-Llama-3.1-8B-Instruct/tokenizer.model"
    )
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 2048
    max_batch_size: int = 4
    max_gen_len: Optional[int] = 2048
    hf_token: str = "token"
    hf_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: str = ""


@dataclass
class LlavaConfig:
    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    system_prompt: str = ""


@dataclass
class Qwen2Config:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # "Qwen/Qwen2-VL-7B-Instruct"
    system_prompt: str = ""
    remove_introductory: bool = True
    stop_frame: bool = False
    stop_prompt: str = (
        "Describe the nearest object and its relative location in real-world as short as possible."
    )
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05


@dataclass
class Qwen25Config:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    system_prompt: str = ""
    temperature: float = 0.7


@dataclass
class Gemma2Config:
    model_name: str = "google/gemma-2-9b-it"
    max_new_tokens: int = 512
    hf_token: str = "token"
    system_prompt: str = ""


@dataclass
class LlavaNextVideoConfig:
    model_name: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    system_prompt: str = ""


@dataclass
class InternVideoConfig:
    model_name: str = "OpenGVLab/InternVideo2_5_Chat_8B"
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_frames: int = 500


@dataclass
class ActionConfig:
    name: str = "VOAction"
    vo_config: VOConfig = VOConfig()
    dinov2_config: Dinov2Config = Dinov2Config()
    rn_config: RNConfig = RNConfig()


@dataclass
class SceneConfig:
    name: str = "GPTScene"
    blip2_config: BLIP2Config = BLIP2Config(
        system_prompt="What scene is it? Please answer as short as possible in the phrase format."
    )
    gpt_config: GPTConfig = GPTConfig(
        system_prompt="What scene is it? Please answer as short as possible in the phrase format."
    )
    llava_config: LlavaConfig = LlavaConfig(
        system_prompt="What scene is it? Please answer as short as possible in the phrase format."
    )
    qwen2_config: Qwen2Config = Qwen2Config(
        system_prompt="What scene is it? Please answer as short as possible in the phrase format."
    )
    swag_config: SWAGConfig = SWAGConfig()
    mae_config: MAEConfig = MAEConfig()


@dataclass
class ObjectConfig:
    name: str = "GPTObject"
    blip2_config: BLIP2Config = BLIP2Config(
        system_prompt="Describe the most significant object and its relative location in real-world as short as possible in the phrase format."
    )
    detr_config: DETRConfig = DETRConfig()
    gpt_config: GPTConfig = GPTConfig(
        system_prompt="Describe the most significant object and its relative location in real-world as short as possible in the phrase format."
    )
    # gpt_config: GPTConfig = GPTConfig(
    #     system_prompt="Describe the room type and the most significant object with its location as short as possible in the phrase format."
    # )
    llava_config: LlavaConfig = LlavaConfig(
        system_prompt="Describe the most significant object and its relative location in real-world as short as possible in the phrase format."
    )
    qwen2_config: Qwen2Config = Qwen2Config(
        system_prompt="Describe the most significant object and its relative location in real-world as short as possible in the phrase format."
    )
    swag_config: SWAGConfig = SWAGConfig()


DEFAULT_PROMPT = "As an expert in robotics and vision-and-language navigation research, condense a list of observations and navigation actions into a brief instruction for robot navigation. Focus solely on essential information, omitting unnecessary details. The instruction must be as short as possible while remaining clear and fluent. Rephrase or restructure the content as needed, and do not use a list format. Avoid any introductory phrases and extra explanations."


@dataclass
class SummaryConfig:
    name: str = "GPTSummary"
    gpt_config: GPTConfig = GPTConfig(system_prompt=DEFAULT_PROMPT)
    llama3_config: Llama3Config = Llama3Config(system_prompt=DEFAULT_PROMPT)
    gemma2_config: Gemma2Config = Gemma2Config(system_prompt=DEFAULT_PROMPT)
    qwen25_config: Qwen25Config = Qwen25Config(system_prompt=DEFAULT_PROMPT)


@dataclass
class SpeakerConfig:
    name: str = "NavComposer"
    max_frames: int = 500
    batch_frames_limit: int = 500
    use_gt_action: bool = False
    diversify_level_action: int = (
        1  # 0: no, 1: double random, 2: uniform, 3: exponential decay
    )
    diversify_level_element: int = 0
    diversify_random_offset: float = 0.0
    compress_type: str = "middle"
    use_enter_leave: bool = True
    enter_leave_ratio: float = 0.5
    use_num_order: bool = False
    action_first: bool = True
    action_position: str = (
        "first"  # first, last, dynamic (for turning, last; otherwise, first)
    )
    image_resize: int = 0
    fps: float = 5.0
    downsample: int = 1
    time_file: str = "time_info.jsonl"

    # Use Qwen2-VL to describe video
    qwen2_config: Qwen2Config = Qwen2Config(
        system_prompt="Describe the navigation path from a first-person perspective video, aiming for a natural and smooth flow. Imagine you are instructing a robot to complete this path, so the description should contain necessary guidance of key actions and landmarks as short as possible. Do not use 'okay', 'sure', etc., in the first sentence."
    )
    # Use LlaVA-Next-Video to describe video
    llavanextvideo_config: LlavaNextVideoConfig = LlavaNextVideoConfig(
        system_prompt="Describe the navigation path from a first-person perspective video, aiming for a natural and smooth flow. Imagine you are instructing a robot to complete this path, so the description should contain necessary guidance of key actions and landmarks as short as possible. Do not use 'okay', 'sure', etc., in the first sentence. Limit the sentence length under 200 words."
    )
    # Use InternVideo to describe video
    internvideo_config: InternVideoConfig = InternVideoConfig(
        system_prompt="Describe the navigation path from a first-person perspective video, aiming for a natural and smooth flow. Imagine you are instructing a robot to complete this path, so the description should contain necessary guidance of key actions and landmarks as short as possible. Do not use 'okay', 'sure', etc., in the first sentence."
    )


@dataclass
class MainConfig:
    random_seed: int = 42
    run_type: str = "generate"  # generate, keyframe, listmodules
    run_folder: str = ""
    generate_num: int = 3
    generate_alias: str = "inst_navcomposer"
    reference_alias: str = "None"  # !! only used for ablation study
    run_precision: str = "float32"
    ablate_action: bool = False
    ablate_scene: bool = False
    ablate_object: bool = False
    skip_exist: bool = False
    log_config: bool = True
    action_config: ActionConfig = ActionConfig()
    scene_config: SceneConfig = SceneConfig()
    object_config: ObjectConfig = ObjectConfig()
    summary_config: SummaryConfig = SummaryConfig()
    speaker_config: SpeakerConfig = SpeakerConfig()


_C = OmegaConf.structured(MainConfig)


def get_config(config_file: str = None, opts: list = None):
    """Create a global config object, which merges the default config with config file and command line args.
    Args:
        config_file: a string path indicating the config file
        opts: options in a dot-list style, e.g., ["a.aa.aaa=1", "a.aa.bbb=2", "a.bb.aaa=3", "a.bb.bbb=4"]
    """
    config = _C.copy()
    file_conf = OmegaConf.create()
    opts_conf = OmegaConf.create()
    if config_file:
        file_conf = OmegaConf.load(config_file)
    if opts:
        opts_conf = OmegaConf.from_dotlist(opts)
    config = OmegaConf.merge(config, file_conf, opts_conf)
    return config


def print_config(config):
    print(OmegaConf.to_yaml(config))
