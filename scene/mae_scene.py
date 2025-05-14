from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import requests
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.registry import registry


def build_transform():
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    size = 224
    t = []
    t.append(
        transforms.Resize(
            size, interpolation=Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@registry.register_scene
class MAEScene:
    def __init__(self, config, model=None):
        self.device = torch.device(dist.get_rank())
        self.mae_config = config.scene_config.mae_config
        self.model_name = self.mae_config.model_name
        self.ckpt_path = self.mae_config.ckpt_path
        self.batch_size = self.mae_config.batch_size
        self.topk = self.mae_config.topk
        self.nb_classes = self.mae_config.nb_classes
        self.global_pool = self.mae_config.global_pool
        self.drop_path = self.mae_config.drop_path
        self.places365_id_to_name = {}
        with open(self.mae_config.places365_id_to_name, "r") as f:
            self.places365_id_to_name = {
                int(cls_id): name for cls_id, name in json.load(f).items()
            }
        self.actual_nb_classes = len(self.places365_id_to_name)
        if self.model_name == "vit_base_patch16":
            model_class = vit_base_patch16
        elif self.model_name == "vit_large_patch16":
            model_class = vit_large_patch16
        elif self.model_name == "vit_huge_patch14":
            model_class = vit_huge_patch14
        else:
            raise AssertionError("Invalid MAEScene.model_name")
        self.model = model_class(
            num_classes=self.nb_classes,
            drop_path_rate=self.drop_path,
            global_pool=self.global_pool,
        )
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_transform()

    #     self.transform = transforms.Compose([
    #         transforms.Resize(
    #             self.resolution,
    #             interpolation=transforms.InterpolationMode.BICUBIC,
    #         ),
    #         transforms.CenterCrop(self.resolution),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),
    #     ])
    def transform_image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        image = self.transform(image)
        return image

    def __call__(self, images):
        preds = []
        if self.batch_size == -1:
            batch_size = len(images)
        else:
            batch_size = self.batch_size
        for i in range(0, len(images), batch_size):
            cur_images = images[i : min(len(images), i + batch_size)]
            batch_image = [self.transform_image(image) for image in cur_images]
            batch_image = torch.stack(batch_image).to(self.device)
            with torch.no_grad():
                _, cur_preds = self.model(batch_image)[
                    :, : self.actual_nb_classes
                ].topk(self.topk)
            preds.append(cur_preds)
        preds = torch.cat(preds, dim=0)
        n, m = preds.shape
        res = []
        for i in range(n):
            cur_res = [
                self.places365_id_to_name[cls_id] for cls_id in preds[i].cpu().tolist()
            ]
            res.append(", ".join(cur_res))
        return res


if __name__ == "__main__":
    from tools.config import get_config, print_config

    dist.init_process_group()
    config = get_config(config_file="./main.yaml")
    tmp = MAEScene(config)
    images = [
        Image.open("data/places365_standard/train/airfield/00000012.jpg"),
        Image.open("data/places365_standard/train/valley/00000002.jpg"),
        Image.open("data/places365_standard/train/bedroom/00000004.jpg"),
    ]
    res = tmp(images)
    print(res)
