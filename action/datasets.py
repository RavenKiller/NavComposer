import os
from pathlib import Path
import json
from tqdm import tqdm
import torch
import random
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor


#####################################################
# Tool
#####################################################
class BatchColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # hue is in [-0.5, 0.5], zero means no change
        # Others are in [0, \infty], one means no change
        self.brightness_factor = 0
        self.contrast_factor = 0
        self.saturation_factor = 0
        self.hue_factor = 0

    def __call__(self, img):
        if self.brightness or self.contrast or self.saturation or self.hue:
            img = TF.adjust_brightness(img, self.brightness_factor)
            img = TF.adjust_contrast(img, self.contrast_factor)
            img = TF.adjust_saturation(img, self.saturation_factor)
            img = TF.adjust_hue(img, self.hue_factor)
        return img

    def random_factor(self):
        # Generate random jitter parameters
        self.brightness_factor = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness
        )
        self.contrast_factor = random.uniform(
            max(0, 1 - self.contrast), 1 + self.contrast
        )
        self.saturation_factor = random.uniform(
            max(0, 1 - self.saturation), 1 + self.saturation
        )
        self.hue_factor = random.uniform(-self.hue, self.hue)


#####################################################
# Data
#####################################################
class ActionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        stride=1,
        folder="data/vlnce_traj_action",
        split="val_seen",
        model_name="facebook/dinov2-base",
        color_aug=False,
        action_aug=False,  # TODO: why action augmentation not work
        frame_aug=0,  # TODO: test effects
        need_infer=False,
    ):
        super().__init__()
        self.folder = Path(folder)
        self.split = split
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        self.image_pairs = []
        self.actions = []

        self.episode_images = {}
        self.episode_actions = {}

        traj_num = 0
        if isinstance(self.split, str):
            self.split = [self.split]
        for split in self.split:
            for traj_path in tqdm(
                list((self.folder / split).glob("*")), desc=f"Loading {split}"
            ):
                rgb_path = traj_path / "rgb"
                action_path = traj_path / "action"
                images = sorted(
                    list(rgb_path.glob("*.jpg")),
                    key=lambda x: int(x.name.split("_")[0].split(".")[0]),
                )
                action_file = action_path / "0.json"
                if not os.path.exists(action_file):
                    action_file = action_path / "action.json"

                if len(images) == 0 or (not os.path.exists(action_file)):
                    print("Incomplete episode {}".format(str(traj_path)))
                    continue
                with open(action_file, "r") as f:
                    actions_ = json.load(f)
                images = images[::stride]
                actions_ = actions_[::stride]
                if need_infer:
                    self.episode_images[str(rgb_path)] = images
                    self.episode_actions[str(action_path)] = actions_

                image_pairs = list(zip(images[:-1], images[1:]))
                actions = actions_[:-1]
                if frame_aug >= 2:  # merge continuous action within `frame_aug`
                    i = 0
                    while i < len(images) - 2:
                        j = i + frame_aug
                        if j < len(actions_) and len(set(actions_[i : j + 1])) == 1:
                            image_pairs.append((images[i], images[j]))
                            actions.append(actions_[i])
                        i = j + 1

                assert len(image_pairs) == len(
                    actions
                ), "{} images: {} actions: {} ".format(
                    str(traj_path), len(images), len(actions)
                )
                self.image_pairs.extend(image_pairs)
                self.actions.extend(actions)
                traj_num += 1

        # [1,2,3] to [0,1,2] !!
        self.actions = [v - 1 for v in self.actions]
        assert -1 not in self.actions
        # print("Split: {}, Traj num: {}, Pair num: {}".format(split, traj_num, len(self.image_pairs)))

        # Augmentation
        if color_aug:
            self.color_jitter = BatchColorJitter(0.5, 0.5, 0.5, 0.2)
        else:
            self.color_jitter = BatchColorJitter(0, 0, 0, 0)
        self.action_aug = action_aug

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        """Return image pairs and GT actions.
        Caution, the stop action IS DELETED, predicted action set is {0(move forward), 1(turn left), 2(turn right)}
        """
        image_pair = self.image_pairs[idx]
        action = self.actions[idx]
        first_image = Image.open(image_pair[0])
        second_image = Image.open(image_pair[1])
        self.color_jitter.random_factor()
        first_image = self.color_jitter(first_image)
        second_image = self.color_jitter(second_image)
        if "imagegpt" in self.model_name:
            first_image = self.processor(
                images=first_image, return_tensors="pt"
            ).input_ids.squeeze(0)
            second_image = self.processor(
                images=second_image, return_tensors="pt"
            ).input_ids.squeeze(0)
        else:
            first_image = self.processor(
                images=first_image, return_tensors="pt"
            ).pixel_values.squeeze(0)
            second_image = self.processor(
                images=second_image, return_tensors="pt"
            ).pixel_values.squeeze(0)
        first_image, second_image, action = self.action_reverser(
            first_image, second_image, action
        )

        return {
            "first_image": first_image,
            "second_image": second_image,
            "action": torch.tensor(action),
        }

    def action_reverser(self, first_image, second_image, action):
        if self.action_aug and action and random.random() < 0.5:
            if action == 1:
                action = 2
            elif action == 2:
                action = 1
            return (second_image, first_image, action)
        else:
            return (first_image, second_image, action)

    def get_episode_batch(self):
        """For episode evaluation
        The stop action is RESERVED, GT action set is {0(stop), 1(move forward), 2(turn left), 3(turn right)}
        """
        for (episode_rgb, images), (episode_action, action) in zip(
            self.episode_images.items(), self.episode_actions.items()
        ):
            images = torch.cat(
                [
                    self.processor(
                        images=Image.open(v), return_tensors="pt"
                    ).pixel_values
                    for v in images
                ],
                dim=0,
            )
            first_image = images[:-1]
            second_image = images[1:]

            yield {
                "episode": str(Path(episode_rgb).parent),
                "first_image": first_image,
                "second_image": second_image,
                "action": torch.tensor(action),
            }

    def episode_num(self):
        return len(self.episode_images)


class ActionDatasetForVO(torch.utils.data.Dataset):
    def __init__(
        self,
        stride=1,
        folder="data/vlnce_traj_action",
        split="val_seen",
        color_aug=False,
        need_infer=False,
    ):
        super().__init__()
        self.folder = Path(folder)
        self.split = split

        self.image_pairs = []
        self.actions = []

        self.episode_images = {}
        self.episode_actions = {}

        traj_num = 0
        if isinstance(self.split, str):
            self.split = [self.split]
        for split in self.split:
            for traj_path in (self.folder / split).glob("*"):
                rgb_path = traj_path / "rgb"
                action_path = traj_path / "action"
                images = sorted(
                    list(rgb_path.glob("*.jpg")),
                    key=lambda x: int(x.name.split("_")[0].split(".")[0]),
                )
                action_file = action_path / "0.json"
                if not os.path.exists(action_file):
                    action_file = action_path / "action.json"
                if len(images) == 0 or (not os.path.exists(action_file)):
                    print("Incomplete episode {}".format(str(traj_path)))
                    continue
                with open(action_file, "r") as f:
                    actions = json.load(f)
                images = images[::stride]
                actions = actions[::stride]
                if need_infer:
                    self.episode_images[str(rgb_path)] = images
                    self.episode_actions[str(action_path)] = actions

                image_pairs = list(zip(images[:-1], images[1:]))
                actions = actions[:-1]
                assert len(image_pairs) == len(
                    actions
                ), "{} images: {} actions: {} ".format(
                    str(traj_path), len(images), len(actions)
                )
                self.image_pairs.extend(image_pairs)
                self.actions.extend(actions)
                traj_num += 1

        # [1,2,3] to [0,1,2] !!
        self.actions = [v - 1 for v in self.actions]
        assert -1 not in self.actions
        # print("Split: {}, Traj num: {}, Pair num: {}".format(split, traj_num, len(self.image_pairs)))

        # Augmentation
        if color_aug:
            self.color_jitter = BatchColorJitter(0.5, 0.5, 0.5, 0.2)
        else:
            self.color_jitter = BatchColorJitter(0, 0, 0, 0)

    def __len__(self):
        return len(self.actions)

    def get_episode_batch(self):
        """For episode evaluation
        The stop action is RESERVED, GT action set is {0(stop), 1(move forward), 2(turn left), 3(turn right)}
        """
        for (episode_rgb, images), (episode_action, action) in zip(
            self.episode_images.items(), self.episode_actions.items()
        ):
            images = [Image.open(v) for v in images]
            first_image = images[:-1]
            second_image = images[1:]

            yield {
                "episode": str(Path(episode_rgb).parent),
                "first_image": first_image,
                "second_image": second_image,
                "action": action,
            }

    def episode_num(self):
        return len(self.episode_images)
