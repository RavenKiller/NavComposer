from PIL import Image
from pathlib import Path
from itertools import groupby
from collections import OrderedDict
import random
import json
import jsonlines
import time
import natsort
import torch

from tools.registry import registry
from tools.tools import diversify_action, diversify_element, im_resize, read_gt_action


@registry.register_speaker
class NavComposer:
    def __init__(self, config):
        self.speaker_config = config.speaker_config
        ap = registry.get_action(config.action_config.name)(config)
        sp = registry.get_scene(config.scene_config.name)(config)
        if (
            "Llava" in config.scene_config.name and "Llava" in config.object_config.name
        ) or (
            "Qwen2" in config.scene_config.name and "Qwen2" in config.object_config.name
        ):
            op = registry.get_object(config.object_config.name)(config, model=sp.model)
        else:
            op = registry.get_object(config.object_config.name)(config)
        ip = registry.get_summary(config.summary_config.name)(config)
        self.initialize(ap, sp, op, ip)

    def initialize(
        self,
        action_pipeline,
        scene_pipeline,
        object_pipeline,
        summary_pipeline,
    ):
        self.action_pipeline = action_pipeline
        self.scene_pipeline = scene_pipeline
        self.object_pipeline = object_pipeline
        self.summary_pipeline = summary_pipeline

        self.reset()

    @property
    def path_len(self):
        return len(self.images)

    def reset(self):
        """Reset all states"""
        self.current_source = ""
        self.images = []
        self.actions = []
        self.actions_compact = OrderedDict()
        self.scenes = OrderedDict()
        self.objects = OrderedDict()
        self.frame_description = ""
        self.info_str = ""
        self.lock = False

    def add_image(self, image):
        """Add a PIL image into the buffer"""
        assert (
            isinstance(image, Image.Image)
            or isinstance(image, Path)
            or isinstance(image, str)
        )
        if isinstance(image, Image.Image):
            self.images.append(im_resize(image, self.speaker_config.image_resize))
        else:
            self.images.append(
                im_resize(Image.open(image), self.speaker_config.image_resize)
            )

    def update_description(self):
        """Update action, scene and object description for new images"""
        assert self.path_len, "No images"
        for i in range(len(self.scenes), self.path_len):
            if i >= 1:
                if not self.speaker_config.use_gt_action:
                    self.actions.append(
                        self.action_pipeline([self.images[i - 1]], [self.images[i]])[0]
                    )
            # # In some case, scene and object are overlapped.
            self.scenes[i] = self.scene_pipeline([self.images[i]])[0]
            self.objects[i] = self.object_pipeline([self.images[i]])[0]
        return (self.actions, self.scenes, self.objects)

    def update_description_compact(self, level=1):
        """Update action, scene and object description for new images. Compact version: more than `level` same actions are downsampled to `level` actions.
        Caution, this function does not support dynamic image addtion.
        Args:
            level: A integer showing the compact level. Can be 1 (tail), 2 (middle and tail) or 3 (head, middle and tail). -1 means no compact
        """
        assert self.path_len, "No images"
        assert len(self.actions) == 0
        assert level in [1, 2, 3, -1]
        if level == -1:
            return self.update_description()
        if self.speaker_config.use_gt_action:
            assert isinstance(self.current_source, Path) or isinstance(
                self.current_source, str
            ), "When use GT action, the source must be folder!"
            self.actions = read_gt_action(Path(self.current_source).parent)
        else:
            self.actions = self.action_pipeline(self.images[:-1], self.images[1:])
        actions_stop = self.actions + ["stop"]
        assert (
            len(actions_stop) == self.path_len
        ), "The size of actions is not equal to that of images!"
        i = 0
        compact_indices = []
        while i < self.path_len:
            j = i + 1
            while j < self.path_len and actions_stop[i] == actions_stop[j]:
                j += 1
            if self.speaker_config.compress_type == "normal":
                if j - i >= level:
                    head = i
                    tail = j - 1
                    middle = (head + tail) // 2
                    if (level == 3) and (head not in compact_indices):
                        self.actions_compact[head] = actions_stop[head]
                        compact_indices.append(head)
                    if (level >= 2) and (middle not in compact_indices):
                        self.actions_compact[middle] = actions_stop[middle]
                        compact_indices.append(middle)
                    if level >= 1:
                        if (
                            level == 1
                            and self.speaker_config.use_enter_leave
                            and random.random() < self.speaker_config.enter_leave_ratio
                            and actions_stop[head] == "move forward"
                        ):
                            if (
                                tail - head >= 2
                            ):  # avoid one 'move forward' to be converted
                                x = random.random()
                                if x < 0.3334:  # enter
                                    tail_ = tail
                                    self.actions_compact[tail_] = "enter"
                                elif x < 0.6667:  # pass
                                    tail_ = middle
                                    self.actions_compact[tail_] = "pass"
                                else:  # leave
                                    tail_ = head
                                    self.actions_compact[tail_] = "leave"
                            else:
                                tail_ = tail
                                self.actions_compact[tail_] = "enter"
                        else:
                            tail_ = tail
                            self.actions_compact[tail_] = actions_stop[tail_]
                        compact_indices.append(tail_)
            elif self.speaker_config.compress_type == "middle":
                head = i
                tail = j - 1
                middle = (head + tail) // 2
                if (
                    self.speaker_config.use_enter_leave
                    and random.random() < self.speaker_config.enter_leave_ratio
                    and actions_stop[head] == "move forward"
                ):
                    if tail - head >= 2:  # avoid one 'move forward' to be converted
                        x = random.random()
                        if x < 0.3334:  # enter
                            tail_ = tail
                            self.actions_compact[tail_] = "enter"
                        elif x < 0.6667:  # pass
                            tail_ = middle
                            self.actions_compact[tail_] = "pass"
                        else:  # leave
                            tail_ = head
                            self.actions_compact[tail_] = "leave"
                    else:
                        tail_ = middle
                        self.actions_compact[tail_] = actions_stop[tail_]
                else:
                    tail_ = middle
                    self.actions_compact[tail_] = actions_stop[tail_]
                compact_indices.append(tail_)
            i = j
        compact_images = [self.images[idx] for idx in compact_indices]
        if len(compact_images) <= self.speaker_config.batch_frames_limit:
            compact_scenes = self.scene_pipeline(compact_images)
            compact_objects = self.object_pipeline(compact_images)
        else:
            print("May cause out of memory. Swith to single mode", flush=True)
            compact_scenes = [self.scene_pipeline([v])[0] for v in compact_images]
            compact_objects = [self.object_pipeline([v])[0] for v in compact_images]

        for i in range(len(compact_indices)):
            self.scenes[compact_indices[i]] = compact_scenes[i]
            self.objects[compact_indices[i]] = compact_objects[i]
        return (self.actions_compact, self.scenes, self.objects)

    def summarize_instruction(self):
        """Combine description to a complete navigation instruction"""
        if not len(self.actions_compact):
            actions_stop = self.actions + ["stop"]
        else:
            actions_stop = list(self.actions_compact.values())
        frame_description = []
        diversed_limit = random.random() + self.speaker_config.diversify_random_offset
        for i, (a, s, o) in enumerate(
            zip(actions_stop, self.scenes.values(), self.objects.values())
        ):
            desc = ""
            if self.speaker_config.use_num_order:
                desc += "{}. ".format(i)

            if self.speaker_config.action_first and a:
                a = a.strip()
                a = diversify_action(
                    a, diversed_limit, self.speaker_config.diversify_level_action
                )
                if a[-1] != ".":
                    a = a + "."
                desc += "{} ".format(a)
            if s:
                s = s.strip()
                s = diversify_element(
                    s, diversed_limit, self.speaker_config.diversify_level_element
                )
                if s[-1] != ".":
                    s = s + "."
                desc += "{} ".format(s)
            if o:
                o = o.strip()
                o = diversify_element(
                    o, diversed_limit, self.speaker_config.diversify_level_element
                )
                if o[-1] != ".":
                    o = o + "."
                desc += "{} ".format(o)
            if not self.speaker_config.action_first and a:
                a = a.strip()
                a = diversify_action(
                    a, diversed_limit, self.speaker_config.diversify_level_action
                )
                if a[-1] != ".":
                    a = a + "."
                desc += "{} ".format(a)
            frame_description.append(desc)
        # Remove redundant frames
        # frame_description = [key for key, _group in groupby(frame_description)]
        frame_description = "\n".join(frame_description)
        self.frame_description = frame_description
        instruction = self.summary_pipeline(frame_description)
        return instruction

    def get_detail(self, return_json=False):
        if return_json:
            return json.loads(self.info_str)
        else:
            return self.info_str

    def generate(self, source):
        """Source can be a Path, str, list of image paths or list of PIL images"""
        self.reset()
        self.current_source = source

        if isinstance(source, Path) or isinstance(source, str):
            p = Path(source)
            source = []
            for v in p.glob("*.jpg"):
                source.append(v)
            if not source:
                for v in p.glob("*.png"):
                    source.append(v)

        source = natsort.natsorted(source, key=str)
        source = source[
            : self.speaker_config.max_frames : self.speaker_config.downsample
        ]
        for v in source:
            self.add_image(v)
        self.update_description_compact()
        res = self.summarize_instruction()
        # print(res)
        info = {
            "actions": self.actions,
            "actions_compact": self.actions_compact,
            "scenes": self.scenes,
            "objects": self.objects,
            "summary": self.frame_description,
            "instruction": res,
        }
        self.info_str = json.dumps(info, indent=2)
        # print(self.info_str)
        return res


@registry.register_speaker
class NavComposerWithTime(NavComposer):
    def update_description_compact(self, level=1):
        """Update action, scene and object description for new images. Compact version: more than `level` same actions are downsampled to `level` actions.
        Caution, this function does not support dynamic image addtion.
        Args:
            level: A integer showing the compact level. Can be 1 (tail), 2 (middle and tail) or 3 (head, middle and tail). -1 means no compact
        """
        assert self.path_len, "No images"
        assert len(self.actions) == 0
        assert level in [1, 2, 3, -1]
        if level == -1:
            return self.update_description()
        if self.speaker_config.use_gt_action:
            assert isinstance(self.current_source, Path) or isinstance(
                self.current_source, str
            ), "When use GT action, the source must be folder!"
            self.actions = read_gt_action(Path(self.current_source).parent)
        else:
            ## actions time
            tic = time.time()
            self.actions = self.action_pipeline(self.images[:-1], self.images[1:])
            self.time_info["actions"] = time.time() - tic
        actions_stop = self.actions + ["stop"]
        assert (
            len(actions_stop) == self.path_len
        ), "The size of actions is not equal to that of images!"
        i = 0
        compact_indices = []
        while i < self.path_len:
            j = i + 1
            while j < self.path_len and actions_stop[i] == actions_stop[j]:
                j += 1
            if self.speaker_config.compress_type == "normal":
                if j - i >= level:
                    head = i
                    tail = j - 1
                    middle = (head + tail) // 2
                    if (level == 3) and (head not in compact_indices):
                        self.actions_compact[head] = actions_stop[head]
                        compact_indices.append(head)
                    if (level >= 2) and (middle not in compact_indices):
                        self.actions_compact[middle] = actions_stop[middle]
                        compact_indices.append(middle)
                    if level >= 1:
                        if (
                            level == 1
                            and self.speaker_config.use_enter_leave
                            and random.random() < self.speaker_config.enter_leave_ratio
                            and actions_stop[head] == "move forward"
                        ):
                            if (
                                tail - head >= 2
                            ):  # avoid one 'move forward' to be converted
                                x = random.random()
                                if x < 0.3334:  # enter
                                    tail_ = tail
                                    self.actions_compact[tail_] = "enter"
                                elif x < 0.6667:  # pass
                                    tail_ = middle
                                    self.actions_compact[tail_] = "pass"
                                else:  # leave
                                    tail_ = head
                                    self.actions_compact[tail_] = "leave"
                            else:
                                tail_ = tail
                                self.actions_compact[tail_] = "enter"
                        else:
                            tail_ = tail
                            self.actions_compact[tail_] = actions_stop[tail_]
                        compact_indices.append(tail_)
            elif self.speaker_config.compress_type == "middle":
                head = i
                tail = j - 1
                middle = (head + tail) // 2
                if (
                    self.speaker_config.use_enter_leave
                    and random.random() < self.speaker_config.enter_leave_ratio
                    and actions_stop[head] == "move forward"
                ):
                    if tail - head >= 2:  # avoid one 'move forward' to be converted
                        x = random.random()
                        if x < 0.3334:  # enter
                            tail_ = tail
                            self.actions_compact[tail_] = "enter"
                        elif x < 0.6667:  # pass
                            tail_ = middle
                            self.actions_compact[tail_] = "pass"
                        else:  # leave
                            tail_ = head
                            self.actions_compact[tail_] = "leave"
                    else:
                        tail_ = middle
                        self.actions_compact[tail_] = "enter"
                else:
                    tail_ = middle
                    self.actions_compact[tail_] = actions_stop[tail_]
                compact_indices.append(tail_)
            i = j
        compact_images = [self.images[idx] for idx in compact_indices]
        if len(compact_images) <= self.speaker_config.batch_frames_limit:
            ## scenes time
            tic = time.time()
            compact_scenes = self.scene_pipeline(compact_images)
            self.time_info["scenes"] = time.time() - tic
            ## objects time
            tic = time.time()
            compact_objects = self.object_pipeline(compact_images)
            self.time_info["objects"] = time.time() - tic
        else:
            print("May cause out of memory. Swith to single mode", flush=True)
            compact_scenes = [self.scene_pipeline([v])[0] for v in compact_images]
            compact_objects = [self.object_pipeline([v])[0] for v in compact_images]

        for i in range(len(compact_indices)):
            self.scenes[compact_indices[i]] = compact_scenes[i]
            self.objects[compact_indices[i]] = compact_objects[i]
        return (self.actions_compact, self.scenes, self.objects)

    def summarize_instruction(self):
        """Combine description to a complete navigation instruction"""
        if not len(self.actions_compact):
            actions_stop = self.actions + ["stop"]
        else:
            actions_stop = list(self.actions_compact.values())
        frame_description = []
        diversed_limit = random.random() + self.speaker_config.diversify_random_offset
        for i, (a, s, o) in enumerate(
            zip(actions_stop, self.scenes.values(), self.objects.values())
        ):
            desc = ""
            if self.speaker_config.use_num_order:
                desc += "{}. ".format(i)

            if self.speaker_config.action_first and a:
                a = a.strip()
                a = diversify_action(
                    a, diversed_limit, self.speaker_config.diversify_level_action
                )
                if a[-1] != ".":
                    a = a + "."
                desc += "{} ".format(a)
            if s:
                s = s.strip()
                s = diversify_element(
                    s, diversed_limit, self.speaker_config.diversify_level_element
                )
                if s[-1] != ".":
                    s = s + "."
                desc += "{} ".format(s)
            if o:
                o = o.strip()
                o = diversify_element(
                    o, diversed_limit, self.speaker_config.diversify_level_element
                )
                if o[-1] != ".":
                    o = o + "."
                desc += "{} ".format(o)
            if not self.speaker_config.action_first and a:
                a = a.strip()
                a = diversify_action(
                    a, diversed_limit, self.speaker_config.diversify_level_action
                )
                if a[-1] != ".":
                    a = a + "."
                desc += "{} ".format(a)
            frame_description.append(desc)
        # Remove redundant frames
        frame_description = [key for key, _group in groupby(frame_description)]
        frame_description = "\n".join(frame_description)
        self.frame_description = frame_description
        ## summary time
        tic = time.time()
        instruction = self.summary_pipeline(frame_description)
        self.time_info["summary"] = time.time() - tic
        return instruction

    def generate(self, source):
        """Source can be a Path, str, list of image paths or list of PIL images"""
        self.time_info = {}
        ## total time
        tic_total = time.time()
        self.reset()
        self.current_source = source

        if isinstance(source, Path) or isinstance(source, str):
            p = Path(source)
            source = []
            for v in p.glob("*.jpg"):
                source.append(v)
        source = natsort.natsorted(source, key=str)
        source = source[
            : self.speaker_config.max_frames : self.speaker_config.downsample
        ]
        for v in source:
            self.add_image(v)
        self.update_description_compact()
        res = self.summarize_instruction()
        # print(res)
        info = {
            "actions": self.actions,
            "actions_compact": self.actions_compact,
            "scenes": self.scenes,
            "objects": self.objects,
            "summary": self.frame_description,
            "instruction": res,
        }
        self.info_str = json.dumps(info, indent=2)
        # print(self.info_str)
        self.time_info["total"] = time.time() - tic_total
        self.time_info["steps"] = len(self.actions) + 1
        self.time_info["steps_compact"] = len(self.actions_compact)
        # print("==Time== ", self.time_info)
        with jsonlines.open(self.speaker_config.time_file, "a") as reader:
            reader.write(self.time_info)
        return res
