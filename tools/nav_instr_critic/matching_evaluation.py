##########################################
# Config System
##########################################
import copy
from typing import List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))


@dataclass
class CLIPMatcherConfig:
    use_clip_proj: bool = True
    use_bias_after_clip: bool = True
    video_transformer_layer: int = 1
    video_transformer_name: str = ""
    embed_dim: int = 768
    image_transformer_name: str = "ViT-B-16"
    image_transformer_pretrained: str = (
        "datacomp_xl_s13b_b90k"  # "laion2b_s34b_b88k" or "datacomp_xl_s13b_b90k"
    )
    text_transformer_name: str = "sentence-transformers/all-mpnet-base-v2"
    loss_type: str = "cosine"  # cosine, class, mse


@dataclass
class ModelConfig:
    clip_matcher: CLIPMatcherConfig = CLIPMatcherConfig()
    trainable_modules: List = (
        tuple()
    )  # store object reference path, e.g., "coca_model.visual.attn_pool"
    trainable_parameters: List = tuple()
    name: str = "CLIPMatcher"


@dataclass
class DataConfig:
    name: str = "MatcherLoader"
    data_path: str = "data"
    sources: Tuple[str] = ("vlnce_traj_action_clean",)
    image_alias: str = "rgb"
    depth_alias: str = "depth"
    text_alias: str = "inst"
    eval_splits: Tuple[str] = ("val_unseen",)
    eval_shuffle: bool = False

    image_size: int = 0  # !! resize all images, 0 for not resize
    colorjitter_eval: bool = False
    color_params: Tuple[float] = (
        0.5,  # brightness    [0,2]
        0.5,  # contrast      [0,2]
        0.5,  # saturation    [0,2]
        0.25,  # hue           [-0.5, 0.5]
        0.25,  # gaussian blur [0,2]
        15,  # rotation [0,30] int
        0.1,  # crop [0,1] float
        0.25,  # perspective [0,1]
    )

    use_key_frame: bool = False
    downsample_limit: int = 500  # downsample all trajectories into 500 steps
    use_downsample: int = 0
    limit_path_length: int = 300  # 0: no limit, >0: keep first limit_path_length frames
    use_episodes_orders: bool = True


@dataclass
class TrainerConfig:
    name: str = "MatcherTrainer"
    epochs: int = 20
    workers: int = 4
    batch_size: int = 8

    checkpoint_dir: str = "data/checkpoints/{}"
    eval_dir_alias: str = "eval"
    eval_output_folder: str = "data/checkpoints"
    eval_checkpoint: str = "data/model_weights/cm.pth"
    eval_dropout: bool = False
    eval_top_k: int = 3

    log_steps: int = -1
    eval_log_steps: int = 1
    log_group_metrics: bool = False
    log_full_metrics: bool = False
    feature_dir: str = "data/eval_features"

    use_wandb: bool = False


@dataclass
class MainConfig:
    random_seed: int = 42
    run_name: str = "cm"
    model_config: ModelConfig = ModelConfig()
    data_config: DataConfig = DataConfig()
    trainer_config: TrainerConfig = TrainerConfig()


_C = OmegaConf.structured(MainConfig)


def get_config(
    config_file: MainConfig = None, config_str: str = None, opts: list = None
):
    """Create a global config object, which merges the default config with config file and command line args.
    Args:
        config_file: a string path indicating the config file
        config_str: a yaml string
        opts: options in a dot-list style, e.g., ["a.aa.aaa=1", "a.aa.bbb=2", "a.bb.aaa=3", "a.bb.bbb=4"]
    """
    config = _C.copy()
    file_conf = OmegaConf.create()
    str_conf = OmegaConf.create()
    opts_conf = OmegaConf.create()
    if config_file:
        file_conf = OmegaConf.load(config_file)
    if config_str:
        str_conf = OmegaConf.create(config_str)
    if opts:
        opts_conf = OmegaConf.from_dotlist(opts)
    config = OmegaConf.merge(config, file_conf, str_conf, opts_conf)
    return config


##########################################
# Utils
##########################################
import collections
from typing import Any, Callable, DefaultDict, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import logging
from accelerate import Accelerator
import time
from accelerate.logging import MultiProcessAdapter

try:
    accelerator = Accelerator()
except RuntimeError:
    accelerator = Accelerator(cpu=True)


class BaseTrainer:
    @classmethod
    def from_config(cls, config):
        return cls(config=config)

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config):
        return cls(config=config)


class BaseLoader(DataLoader):
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Registry:
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(to_register, assert_type)
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("model", to_register, name, assert_type=BaseModel)

    @classmethod
    def register_trainer(cls, to_register: None = None, *, name: Optional[str] = None):
        return cls._register_impl("trainer", to_register, name, assert_type=BaseTrainer)

    @classmethod
    def register_dataloader(
        cls, to_register: None = None, *, name: Optional[str] = None
    ):
        return cls._register_impl(
            "dataloader", to_register, name, assert_type=BaseLoader
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        return cls._get_impl("model", name)

    @classmethod
    def get_trainer(cls, name: str) -> Type[BaseTrainer]:
        return cls._get_impl("trainer", name)

    @classmethod
    def get_dataloader(cls, name: str) -> Type[BaseTrainer]:
        return cls._get_impl("dataloader", name)


registry = Registry()


def get_obj_recur(obj, ref_path):
    if "." not in ref_path:
        if ref_path.isnumeric():
            return obj[int(ref_path)]
        else:
            return getattr(obj, ref_path)
    else:
        names = ref_path.split(".")
        now_path, next_path = names[0], ".".join(names[1:])
        return get_obj_recur(getattr(obj, now_path), next_path)


def change_trainable(model, modules, params, trainable=False):
    for module in modules:
        now_module = get_obj_recur(model, module)
        for param in now_module.parameters():
            param.requires_grad_(trainable)
    for param in params:
        now_param = get_obj_recur(model, param)
        now_param.requires_grad_(trainable)


def get_logger(
    name: str = "main_logger", log_level: str = "DEBUG", log_filename: str = None
):
    logger = logging.getLogger(name)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(log_level.upper())

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    if log_level is None:
        log_level = os.environ.get("ACCELERATE_LOG_LEVEL", None)
    if log_level is not None:
        logger.setLevel(log_level.upper())

    if log_filename is not None:
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.propagate = False
    return MultiProcessAdapter(logger, {})


# Generate a global logger
logger = get_logger(
    "MyLogger",
    log_level="DEBUG",
)


##########################################
# Model Definition
##########################################
import types
import os
import sys
import random
import torch
import open_clip
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig, AutoConfig

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import re


def mean_pooling_mask(features, attention_mask):
    attention_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(features.size()).float()
    )
    return torch.sum(features * attention_mask_expanded, 1) / torch.clamp(
        attention_mask_expanded.sum(1), min=1e-9
    )


@registry.register_model
class CLIPMatcher(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_clip_proj = config.model_config.clip_matcher.use_clip_proj
        self.video_transformer_layer = (
            config.model_config.clip_matcher.video_transformer_layer
        )
        self.use_video_transformer = (
            config.model_config.clip_matcher.video_transformer_layer != 0
        )
        self.video_transformer_name = (
            config.model_config.clip_matcher.video_transformer_name
        )
        self.embed_dim = config.model_config.clip_matcher.embed_dim
        self.loss_type = config.model_config.clip_matcher.loss_type

        self.image_transformer = open_clip.create_model(
            config.model_config.clip_matcher.image_transformer_name,
            pretrained=config.model_config.clip_matcher.image_transformer_pretrained,
        )
        self.image_transformer.transformer = (
            nn.Identity()
        )  # Delete the text transformer because of the limited 77 context

        if self.use_clip_proj:  # hidden_dim=512
            self.video_pre = nn.Linear(
                512,
                self.embed_dim,
                bias=config.model_config.clip_matcher.use_bias_after_clip,
            )
        else:  # hidden_dim=768
            self.video_pre = nn.Linear(
                768,
                self.embed_dim,
                bias=config.model_config.clip_matcher.use_bias_after_clip,
            )
        if self.use_video_transformer:
            if self.video_transformer_name:
                bert_config = BertConfig.from_pretrained(
                    config.model_config.clip_matcher.video_transformer_name
                )
                self.video_transformer = BertModel.from_pretrained(
                    config.model_config.clip_matcher.video_transformer_name
                )
            else:
                bert_config = BertConfig(num_hidden_layers=self.video_transformer_layer)
                self.video_transformer = BertModel(bert_config)
            self.video_transformer.embeddings.word_embeddings = (
                nn.Identity()
            )  # Non text
            for param in self.video_transformer.parameters():
                param.requires_grad_(True)
        else:
            bert_config = BertConfig(num_hidden_layers=1, hidden_size=self.embed_dim)

        # pooler in MPNet is used to select pos0 feature, linear+tanh, not used
        text_config = AutoConfig.from_pretrained(
            config.model_config.clip_matcher.text_transformer_name
        )
        self.text_transformer = AutoModel.from_pretrained(
            config.model_config.clip_matcher.text_transformer_name
        )

        for param in self.image_transformer.parameters():
            param.requires_grad_(False)
        for param in self.text_transformer.parameters():
            param.requires_grad_(False)
        # Trained parameters
        change_trainable(
            self,
            config.model_config.trainable_modules,
            config.model_config.trainable_parameters,
            True,
        )

        if self.use_clip_proj:
            pass
        else:
            self.image_transformer.visual.proj = None
        self.v_linear = nn.Linear(bert_config.hidden_size, self.embed_dim)
        self.t_linear = nn.Linear(text_config.hidden_size, self.embed_dim)

        ## for loss
        self.roll_size = 4
        self.class_head = nn.Sequential(nn.ReLU(), nn.Linear(2 * self.embed_dim, 1))
        self.logit_scale = nn.Parameter(torch.tensor(0.07), requires_grad=True)

    def single_score(self, v, t):
        return torch.dot(v, t) * torch.exp(self.logit_scale)

    def forward(self, batch):
        image = batch["image"]
        image_mask = batch["image_mask"]
        text = batch["text"]
        text_mask = batch["text_mask"]
        B = image.shape[0]
        T = image.shape[1]
        image = image.view(
            -1, image.shape[2], image.shape[3], image.shape[4]
        )  # (B*T, C, H, W)

        feat_video = self.image_transformer.encode_image(image)  # (B*T, D)
        feat_video = feat_video.view(B, T, -1)  # (B,T,D)

        feat_video = self.video_pre(feat_video)
        if self.use_video_transformer:
            outputs = self.video_transformer(
                inputs_embeds=feat_video, attention_mask=image_mask
            )  # (B,T,D), 0 is masked
            feat_video = outputs.last_hidden_state
        pooled_video = mean_pooling_mask(feat_video, image_mask)  # (B, D)
        pooled_video = self.v_linear(pooled_video)

        outputs = self.text_transformer(
            input_ids=text, attention_mask=text_mask
        )  # (B, L, D)
        feat_text = outputs.last_hidden_state
        pooled_text = mean_pooling_mask(feat_text, text_mask)  # (B, D)
        pooled_text = self.t_linear(pooled_text)

        if self.loss_type == "cosine":
            pooled_video = F.normalize(pooled_video, dim=1)
            pooled_text = F.normalize(pooled_text, dim=1)
            similarity_matrix = torch.matmul(pooled_video, pooled_text.t()) * torch.exp(
                self.logit_scale
            )
            targets = torch.arange(B).to(text.device)
            loss_h = F.cross_entropy(similarity_matrix, targets)
            loss_v = F.cross_entropy(similarity_matrix.T, targets)
            loss = (loss_h + loss_v) / 2
        elif self.loss_type == "class":
            fp = torch.cat((pooled_video, pooled_text), dim=1)
            fn = torch.cat(
                (
                    pooled_video,
                    torch.roll(pooled_text, self.roll_size, dims=0),
                ),
                dim=1,
            )
            dp = self.class_head(fp)
            dn = self.class_head(fn)
            logits_ = torch.cat((dp, dn), dim=0)
            targets_ = torch.cat((torch.ones_like(dp), torch.zeros_like(dn)), dim=0)
            loss = F.binary_cross_entropy_with_logits(logits_, targets_)

            # for evaluation
            with torch.no_grad():
                cat_embeddings = torch.cat(
                    [
                        pooled_video.unsqueeze(1).expand(-1, B, -1),
                        pooled_text.unsqueeze(0).expand(B, -1, -1),
                    ],
                    dim=2,
                )
                similarity_matrix = self.class_head(cat_embeddings).squeeze()
        elif self.loss_type == "mse":
            similarity_matrix = -torch.square(
                pooled_video.unsqueeze(1).expand(-1, B, -1)
                - pooled_text.unsqueeze(0).expand(B, -1, -1)
            ).mean(dim=2)
            targets = torch.arange(B).to(text.device)
            loss_h = F.cross_entropy(similarity_matrix, targets)
            loss_v = F.cross_entropy(similarity_matrix.T, targets)
            loss = (loss_h + loss_v) / 2

        return {
            "similarity": similarity_matrix,
            "pooled_video": pooled_video.detach(),
            "pooled_text": pooled_text.detach(),
            "loss": loss,
        }


##########################################
# Dataloader Definition
##########################################
import os
import sys
import functools
import random
from pathlib import Path
import numpy as np
import json
import hashlib
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import open_clip
from transformers import CLIPImageProcessor
from natsort import natsorted

from transformers import AutoTokenizer, AutoModel
from functools import partial


class ColorJitterWithPreprocess:
    def __init__(
        self,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        blur=0,
        rotation=0,
        crop=0,
        perspective=0,
        preprocess=None,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.blur = blur
        self.rotation = int(rotation)
        self.crop = int(crop)
        self.perspective = perspective

        self.use_augmentation = bool(
            self.brightness
            or self.contrast
            or self.saturation
            or self.hue
            or self.blur
            or self.rotation
            or self.crop
            or self.perspective
        )
        # logger.debug("Image augmentation: {}".format(self.use_augmentation))

        self.preprocess = preprocess

        self.brightness_factor = 0
        self.contrast_factor = 0
        self.saturation_factor = 0
        self.hue_factor = 0
        self.blur_factor = 0
        self.rotation_factor = 0
        self.crop_factor = 0
        self.perspective_factor = 0

    def get_perspective_params(self, width: int, height: int, distortion_scale: float):
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(
                torch.randint(
                    0, int(distortion_scale * half_width) + 1, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    0, int(distortion_scale * half_height) + 1, size=(1,)
                ).item()
            ),
        ]
        topright = [
            int(
                torch.randint(
                    width - int(distortion_scale * half_width) - 1, width, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    0, int(distortion_scale * half_height) + 1, size=(1,)
                ).item()
            ),
        ]
        botright = [
            int(
                torch.randint(
                    width - int(distortion_scale * half_width) - 1, width, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    height - int(distortion_scale * half_height) - 1, height, size=(1,)
                ).item()
            ),
        ]
        botleft = [
            int(
                torch.randint(
                    0, int(distortion_scale * half_width) + 1, size=(1,)
                ).item()
            ),
            int(
                torch.randint(
                    height - int(distortion_scale * half_height) - 1, height, size=(1,)
                ).item()
            ),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def random_factor(self):
        # Generate random color jitter parameters
        self.brightness_factor = random.uniform(
            max(0, 1 - self.brightness * 0.5), 1 + self.brightness
        )
        self.contrast_factor = random.uniform(
            max(0, 1 - self.contrast * 0.5), 1 + self.contrast
        )
        self.saturation_factor = random.uniform(
            max(0, 1 - self.saturation * 0.5), 1 + self.saturation
        )
        self.hue_factor = random.uniform(-self.hue, self.hue)

        # Generate other parameters
        self.blur_factor = random.uniform(0, self.blur)
        self.rotation_factor = random.randint(-self.rotation, self.rotation)
        self.crop_factor = random.uniform(0, self.crop)
        self.perspective_factor = random.uniform(0, self.perspective)

    def _image_augmentation(self, img):
        if self.use_augmentation:
            img = TF.adjust_brightness(img, self.brightness_factor)
            img = TF.adjust_contrast(img, self.contrast_factor)
            img = TF.adjust_saturation(img, self.saturation_factor)
            img = TF.adjust_hue(img, self.hue_factor)

            img = TF.gaussian_blur(img, 3, self.blur_factor)

            img = TF.rotate(img, self.rotation_factor)
            w, h = img.size
            new_w = int(w * (1 - self.crop_factor))
            new_h = int(h * (1 - self.crop_factor))
            img = TF.center_crop(img, (new_h, new_w))
            startpoints, endpoints = self.get_perspective_params(
                new_w, new_h, self.perspective_factor
            )
            img = TF.perspective(img, startpoints, endpoints)
        return img

    def __call__(self, img):
        img = self._image_augmentation(img)
        img = self.preprocess(img)
        return img


MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SCALE = 1000
TARGET_SIZE = 224


def im_resize(image, size):
    """Resize the shortest edge to `size`, keeping the aspect ratio unchanged."""
    if size == 0:
        return image
    w, h = image.size
    if w < h:
        w_new = size
        h_new = int(h * w_new / w)
    else:
        h_new = size
        w_new = int(w * h_new / h)
    return image.resize((w_new, h_new), Image.LANCZOS)


def exists_and_not_empty(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        return True
    else:
        return False


def identity_func(a, *args, **kwargs):
    return a


def collate_fn_pad_zero(batch, tokenizer=None):
    # Process different length of images
    batch_ret = {}
    key_list = batch[0].keys()
    for k in key_list:
        datas = []
        for data in batch:
            datas.append(data[k])
        B = len(datas)
        if isinstance(datas[0], torch.Tensor):
            if B > 1 and ("image" in k):
                # data shape is (T,C,H,W)
                max_len = max([data.shape[0] for data in datas])
                datas = [
                    F.pad(
                        data,
                        (
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            max_len - data.shape[0],
                        ),
                        "constant",
                        0,
                    )
                    for data in datas
                ]
                # datas shape is (B, max_len, C, H, W)
                datas = torch.stack(datas)
                # mask shape is (B,max_len). True means NO masked position, False means masked (full of zeros) position
                mask = torch.any(datas.reshape(B, max_len, -1), dim=2)
                batch_ret[k + "_mask"] = mask
                batch_ret[k] = datas
            else:
                batch_ret[k] = torch.stack(datas)
        elif None in datas:
            batch_ret[k] = None
        elif k == "text" and tokenizer is not None:
            encoded_input = tokenizer(
                datas, padding=True, truncation=True, return_tensors="pt"
            )
            batch_ret[k] = encoded_input.input_ids
            batch_ret[k + "_mask"] = encoded_input.attention_mask
        else:
            batch_ret[k] = datas
    return batch_ret


def _zero_pad(tensor, size):
    n = size - len(tensor) % size
    if n == size:
        return tensor
    else:
        z = torch.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
        return torch.cat((tensor, z), 0)


class MatcherDataset(Dataset):
    def __init__(self, config, splits=["train"], mode="train"):
        super().__init__()
        self.config = config
        self.is_validation_in_training = mode == "validation_in_train"
        self.mode = mode
        self.splits = splits
        self.data_path = Path(config.data_config.data_path)
        self.sources = config.data_config.sources
        self.use_key_frame = config.data_config.use_key_frame
        self.downsample_limit = config.data_config.downsample_limit
        self.use_downsample = config.data_config.use_downsample
        self.limit_path_length = config.data_config.limit_path_length
        self.use_episodes_orders = config.data_config.use_episodes_orders
        self.episodes = []
        self.episodes_insts = []

        self.image_alias = config.data_config.image_alias
        self.depth_alias = config.data_config.depth_alias
        self.text_alias = config.data_config.text_alias

        if self.use_episodes_orders:
            try:
                ## Load episode orders
                with open("data/model_weights/episodes_orders.json", "r") as f:
                    self.episodes = json.load(f)
                    self.episodes = [Path(v) for v in self.episodes]
                with open("data/model_weights/episodes_insts_orders.json", "r") as f:
                    self.episodes_insts = json.load(f)

                    def replace_inst_alias(p):
                        parts = list(p.parts)
                        parts[-2] = self.text_alias
                        return Path(*parts)

                    self.episodes_insts = [
                        replace_inst_alias(Path(v)) for v in self.episodes_insts
                    ]
            except FileNotFoundError:
                ## Load episodes by glob. Caution: Different episode orders may lead to different results
                for i, source in enumerate(self.sources):
                    for split in self.splits:
                        for episode in (self.data_path / source / split).glob("*"):
                            inst_paths = list((episode / self.text_alias).glob("*.txt"))
                            for inst_path in inst_paths:
                                self.episodes.append(episode)
                                self.episodes_insts.append(inst_path)
                os.makedirs("data/model_weights", exist_ok=True)
                with open("data/model_weights/episodes_orders.json", "w") as f:
                    json.dump([str(v) for v in self.episodes], f, indent=2)
                with open("data/model_weights/episodes_insts_orders.json", "w") as f:
                    json.dump([str(v) for v in self.episodes_insts], f, indent=2)
        else:
            for i, source in enumerate(self.sources):
                for split in self.splits:
                    for episode in (self.data_path / source / split).glob("*"):
                        inst_paths = list((episode / self.text_alias).glob("*.txt"))
                        for inst_path in inst_paths:
                            self.episodes.append(episode)
                            self.episodes_insts.append(inst_path)

        self.color_params = config.data_config.color_params
        self.colorjitter_eval = config.data_config.colorjitter_eval

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-16"
        )
        if mode == "train" or self.colorjitter_eval:
            raise AssertionError("Not for training!")
        else:
            self.transform = ColorJitterWithPreprocess(preprocess=preprocess_val)

    def read_image(self, index):
        episode = self.episodes[index]
        if (not self.image_alias) or not exists_and_not_empty(
            episode / self.image_alias
        ):
            return None
        image_path = episode / self.image_alias
        image_list = list(image_path.glob("*.jpg"))
        if not image_list:
            image_list = list(image_path.glob("*.png"))
        image_list = natsorted(image_list)
        image = []
        self.transform.random_factor()
        for img in image_list:
            image_t_m = Image.open(img)
            image_t_m = im_resize(image_t_m, self.config.data_config.image_size)
            image_t_m = self.transform(image_t_m)
            image.append(image_t_m)
        image = torch.stack(image)  # (T,C,H,W)

        # key frame
        if self.use_downsample:
            use_downsample = self.use_downsample
        else:
            use_downsample = 1 + len(image) // self.downsample_limit
        if use_downsample:
            image = image[::use_downsample]
        elif self.use_key_frame:
            key_frame_path = episode / "key_frame" / "0.json"
            with open(key_frame_path, "r") as f:
                key_frame_index = json.load(f)
            image = torch.index_select(
                image, 0, torch.tensor(key_frame_index, dtype=int)
            )
        if self.limit_path_length:
            image = image[: self.limit_path_length]
        return image

    def read_text(self, index):
        episode = self.episodes[index]
        if (not self.text_alias) or not exists_and_not_empty(episode / self.text_alias):
            return None
        with open(self.episodes_insts[index], "r") as f:
            text = f.read()

        return text

    def __getitem__(self, index):
        image = self.read_image(index)
        # depth = self.read_depth(index)
        text = self.read_text(index)

        infos = self.episodes[index].parts
        inst_id = self.episodes_insts[index].stem
        return {
            "image": image,  # (3,T,H,W)
            "text": text,  # (L) in training and (N_i, L) in evaluation
            "source": infos[-3],
            "split": infos[-2],
            "episode_id": infos[-1],
            "inst_id": inst_id,
        }

    def __len__(self):
        return len(self.episodes)


@registry.register_dataloader
class MatcherLoader(BaseLoader):
    def __init__(self, config, splits=["train"], mode="train", *args, **kwargs):
        self.config = config
        ds = MatcherDataset(config, splits, mode)
        # if config.trainer_config.batch_size > 1:
        kwargs["collate_fn"] = partial(collate_fn_pad_zero, tokenizer=ds.tokenizer)
        super().__init__(ds, *args, **kwargs)


##########################################
# Trainer Definition
##########################################
import os
import sys
import shutil

import time
from pathlib import Path
import json
import collections
from omegaconf import OmegaConf
import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.retrieval import (
    RetrievalRecall,
    RetrievalRPrecision,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalAUROC,
    RetrievalPrecision,
)
from natsort import natsorted
import open_clip
import transformers


def unwrap_model(model):
    return accelerator.unwrap_model(model)


SCALE_CONSTANT = 1.7


def CosineSimilarity(score, target, indexes):
    matched_scores = score[target]
    return (torch.mean(matched_scores) + SCALE_CONSTANT) / (2 * SCALE_CONSTANT)


class RetrievalMetrics:
    def __init__(self, top_k=1, prefix=""):
        self.top_k = top_k
        self.prefix = prefix

        self.metrics_func = {
            prefix + "recall_" + str(top_k): RetrievalRecall(top_k=top_k),
            prefix + "precision_" + str(top_k): RetrievalPrecision(top_k=top_k),
            prefix + "hitrate_" + str(top_k): RetrievalHitRate(top_k=top_k),
            prefix + "map_" + str(top_k): RetrievalMAP(top_k=top_k),
            prefix + "mrr_" + str(top_k): RetrievalMRR(top_k=top_k),
            prefix + "ndcg_" + str(top_k): RetrievalNormalizedDCG(top_k=top_k),
            prefix + "auroc_" + str(top_k): RetrievalAUROC(top_k=top_k),
            prefix + "rprecision": RetrievalRPrecision(),
            prefix + "similarity": RetrievalRPrecision(),  # NOT IMPLEMENTED
        }

    def __call__(self, score, target, indexes):
        results = {}
        for k, func in self.metrics_func.items():
            results[k] = func(score, target, indexes=indexes).item()
        return results

    def reduce(self, results):
        reduced_results = {k: [] for k in self.metrics_func.keys()}
        for res in results:
            for k in res.keys():
                reduced_results[k].append(res[k])
        for k in reduced_results.keys():
            reduced_results[k] = np.mean(reduced_results[k])
        return reduced_results


@registry.register_trainer
class MatcherTrainer(BaseTrainer):
    def __init__(self, config):
        logger.debug(OmegaConf.to_yaml(config), main_process_only=True)
        self.config = config

        ## Training
        self.epochs = config.trainer_config.epochs
        self.batch_size = config.trainer_config.batch_size
        self.workers = config.trainer_config.workers
        self.log_steps = config.trainer_config.log_steps
        self.eval_log_steps = config.trainer_config.eval_log_steps

        ## Directories
        self.checkpoint_dir = config.trainer_config.checkpoint_dir.format(
            config.run_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.eval_checkpoint = config.trainer_config.eval_checkpoint
        self.eval_dir_alias = config.trainer_config.eval_dir_alias
        self.eval_output_folder = config.trainer_config.eval_output_folder

        ## Evaluation
        self.eval_dropout = config.trainer_config.eval_dropout

        ## Placeholders
        self.model = None
        self.loader = None

        ## Parallelism
        self.is_main_process = accelerator.is_main_process
        self.device = accelerator.device
        self.process_index = accelerator.process_index
        self.num_processes = accelerator.num_processes

        ## Logging
        self.use_wandb = config.trainer_config.use_wandb

    def initiate(self, mode="eval"):
        ## Init model
        model_cls = registry.get_model(self.config.model_config.name)
        self.model = model_cls.from_config(self.config)
        self.model.to(self.device)
        if mode == "train":
            raise AssertionError("Not for training!")
        elif mode == "eval":
            self.model.eval()

        ## Init dataloader
        loader_cls = registry.get_dataloader(self.config.data_config.name)
        if mode == "train":
            raise AssertionError("Not for training!")
        elif mode == "eval":
            self.loader = loader_cls.from_config(
                self.config,
                splits=self.config.data_config.eval_splits,
                mode="validation",
                batch_size=self.batch_size,
                shuffle=self.config.data_config.eval_shuffle,
                num_workers=self.workers,
            )
            self.eval_num = len(self.loader.dataset)

        ## Check whether to load checkpoint
        if mode == "train":
            raise AssertionError("Not for training!")
        elif mode == "eval":
            if self.eval_checkpoint:
                ckpt = torch.load(self.eval_checkpoint)
                self.model.load_state_dict(ckpt["state_dict"])
            start_epoch = 0

        ## Prepare for parallel training
        if mode == "train":
            raise AssertionError("Not for training!")
        elif mode == "eval":
            self.model, self.loader = accelerator.prepare(self.model, self.loader)

        return start_epoch

    def log_model_info(self):
        params = sum(param.numel() for param in unwrap_model(self.model).parameters())
        params_t = sum(
            p.numel() for p in unwrap_model(self.model).parameters() if p.requires_grad
        )
        logger.debug(f"Model parameters: {params}. Trainable: {params_t}.")

    def eval(self):
        """Evaluate matching score"""
        logger.debug("Start evaluating")
        self.initiate(mode="eval")
        batch_bar = tqdm.tqdm(
            self.loader,
            total=len(self.loader),
            leave=False,
            dynamic_ncols=True,
            desc="Process {}/{}".format(self.process_index, self.num_processes - 1),
            position=self.process_index,
        )
        ## Feature generation
        if self.config.trainer_config.log_full_metrics:
            feature_dir = Path(self.config.trainer_config.feature_dir) / "_".join(
                self.config.data_config.eval_splits
            )
            # shutil.rmtree(feature_dir)
            os.makedirs(feature_dir, exist_ok=True)
        metric_tool = RetrievalMetrics(
            top_k=self.config.trainer_config.eval_top_k, prefix="v2t_"
        )
        batch_metrics = []
        with torch.no_grad():
            if self.eval_dropout:
                self.model.train()
            for batch in batch_bar:
                batch = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                outputs = self.model(batch)
                B = len(batch["text"])
                score = outputs["similarity"]
                target = torch.zeros((B, B), dtype=bool).to(score.device)
                for i in range(B):
                    for j in range(B):
                        if "_".join(
                            [
                                batch["source"][i],
                                batch["split"][i],
                                batch["episode_id"][i],
                            ]
                        ) == "_".join(
                            [
                                batch["source"][j],
                                batch["split"][j],
                                batch["episode_id"][j],
                            ]
                        ):
                            target[i, j] = True
                indexes = torch.arange(B).unsqueeze(1).expand(-1, B).to(score.device)
                batch_metrics.append(metric_tool(score, target, indexes))

                ## full metrics
                if self.config.trainer_config.log_full_metrics:
                    for i in range(len(batch["text"])):
                        v_feat = outputs["pooled_video"][i]
                        t_feat = outputs["pooled_text"][i]
                        source = batch["source"][i]
                        split = batch["split"][i]
                        episode_id = batch["episode_id"][i]
                        inst_id = batch["inst_id"][i]
                        v_feat_path = feature_dir / "@".join(
                            [source, split, episode_id, inst_id, "v.pt"]
                        )
                        t_feat_path = feature_dir / "@".join(
                            [source, split, episode_id, inst_id, "t.pt"]
                        )
                        torch.save(v_feat, v_feat_path)
                        torch.save(t_feat, t_feat_path)

            ## batch metrics
            batch_metrics = metric_tool.reduce(batch_metrics)

            ## full metrics
            if self.config.trainer_config.log_full_metrics:
                v_feat_paths = natsorted(list(feature_dir.glob("*@v.pt")))
                t_feat_paths = natsorted(list(feature_dir.glob("*@t.pt")))
                assert len(v_feat_paths) == len(t_feat_paths)
                N = len(v_feat_paths)
                score = torch.zeros((N, N), dtype=float)
                target = torch.zeros((N, N), dtype=bool)
                for i, v_feat_path in enumerate(tqdm.tqdm(v_feat_paths)):
                    for j, t_feat_path in enumerate(t_feat_paths):
                        v_feat = torch.load(v_feat_path)
                        t_feat = torch.load(t_feat_path)
                        # print(v_feat.device)
                        score[i, j] = unwrap_model(self.model).single_score(
                            v_feat, t_feat
                        )
                        v_name = v_feat_path.stem
                        v_name = v_name[:-2]
                        t_name = t_feat_path.stem
                        t_name = t_name[:-2]
                        target[i, j] = v_name == t_name
                indexes = torch.arange(N).unsqueeze(1).expand(-1, N)
                full_metrics = metric_tool(score, target, indexes=indexes)
            else:
                full_metrics = {}
            metrics = {"batch": batch_metrics, "full": full_metrics}

            # logger.info("Total evaluation: {}".format(N))
            eval_dir = Path(self.eval_output_folder)
            os.makedirs(eval_dir, exist_ok=True)
            filename = (
                self.eval_dir_alias
                + "_clipmatcher_"
                + "_".join(self.config.data_config.eval_splits)
                + ".json"
            )
            with open(eval_dir / filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            logger.info(
                "Process {}/{} Checkpoint: {} \nMetrics: {}".format(
                    self.process_index,
                    self.num_processes - 1,
                    self.eval_checkpoint,
                    json.dumps(metrics, indent=2),
                )
            )
        accelerator.wait_for_everyone()


##########################################
# Evaluation Entry
##########################################
"""The entry point. --mode and --config must be specified when running.
To run with only one GPU, just `python run.py --mode <> --config <>`.
To run wiht multiple GPUs, `accelerate launch run.py --mode <> --config <>`
"""

import os
import sys
import argparse
from typing import Optional, List
import random
import numpy as np

import torch
import open_clip

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Runs experiment given mode and config.

    Args:
        config_str: yaml string.
        mode: "eval".
        opts: list of strings of additional config options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["eval"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    config_str = """run_name: cm
trainer_config:
  name: MatcherTrainer
  eval_checkpoint: data/model_weights/cm.pth

  batch_size: 8

data_config:
  name: MatcherLoader
  eval_splits: 
    - val_unseen
  use_key_frame: False

model_config:
  name: CLIPMatcher
  trainable_modules: 
    - "text_transformer.encoder.layer.11"
    - "image_transformer.visual.transformer.resblocks.11"
    - "image_transformer.visual.ln_post"
  trainable_parameters:
    - "image_transformer.visual.proj"
  clip_matcher:
    video_transformer_layer: 1
    use_clip_proj: True
"""
    mode = args.mode
    opts = args.opts

    config = get_config(config_str=config_str, opts=opts)
    set_seed(config.random_seed)

    trainer_cls = registry.get_trainer(config.trainer_config.name)
    trainer = trainer_cls.from_config(config)

    if mode == "eval":
        trainer.eval()
    else:
        raise AssertionError("Not for training!")


if __name__ == "__main__":
    main()
