import torch
import requests
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

from tools.registry import registry


@registry.register_object
class SWAGObject:
    def __init__(self, config, model=None):
        self.device = torch.device(dist.get_rank())
        self.swag_config = config.object_config.swag_config
        self.model_name = (
            self.swag_config.model_name
        )  # torch.hub.load("facebookresearch/swag", model="vit_l16_in1k")
        self.resolution = self.swag_config.resolution  # resolution = 512
        self.batch_size = self.swag_config.batch_size
        self.topk = self.swag_config.topk
        self.imagenet_id_to_name = {}
        with open(self.swag_config.imagenet_id_to_name, "r") as f:
            # self.imagenet_id_to_name = json.load(f)
            self.imagenet_id_to_name = {
                int(cls_id): name for cls_id, (label, name) in json.load(f).items()
            }
        self.model = torch.hub.load("facebookresearch/swag", self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
                _, cur_preds = self.model(batch_image).topk(self.topk)
            preds.append(cur_preds)
        preds = torch.cat(preds, dim=0)
        n, m = preds.shape
        res = []
        for i in range(n):
            cur_res = [
                self.imagenet_id_to_name[cls_id] for cls_id in preds[i].cpu().tolist()
            ]
            res.append(", ".join(cur_res))
        return res


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.config import get_config, print_config

    dist.init_process_group()
    config = get_config(config_file="./main.yaml")
    tmp = SWAGObject(config)
    images = [
        Image.open("data/vlnce_traj_action_clean/val_seen/6797/rgb/16_0.jpg"),
        Image.open("data/vlnce_traj_action_clean/val_seen/6797/rgb/16_0.jpg"),
    ]
    res = tmp(images)
    print(res)
