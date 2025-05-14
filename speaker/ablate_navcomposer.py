from PIL import Image
from pathlib import Path
from itertools import groupby
from collections import OrderedDict
import random
import json
import natsort
import torch

from tools.registry import registry
from tools.tools import diversify_action, diversify_element, im_resize


@registry.register_speaker
class AblateNavComposer:
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
        self.generate_alias = config.generate_alias
        self.reference_alias = config.reference_alias
        self.ablate_action = config.ablate_action
        self.ablate_scene = config.ablate_scene
        self.ablate_object = config.ablate_object
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
        self.images = []
        self.actions = []
        self.actions_compact = OrderedDict()
        self.scenes = OrderedDict()
        self.objects = OrderedDict()
        self.frame_description = ""
        self.info_str = ""
        self.lock = False

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
        instruction = self.summary_pipeline(frame_description)
        return instruction

    def get_detail(self, return_json=False):
        if return_json:
            return json.loads(self.info_str)
        else:
            return self.info_str

    def generate(self, source, idx=0):
        """Source must be a Path, str"""
        self.reset()

        source_str = str(source)
        ref_inst_path = source_str.replace(
            "/rgb", "/" + self.reference_alias + "/{}.txt".format(idx)
        )
        ref_info_path = source_str.replace(
            "/rgb", "/" + self.reference_alias + "/{}.info".format(idx)
        )
        with open(ref_info_path, "r") as f:
            ref_info = json.load(f)
        self.actions = ref_info["actions"]
        self.actions_compact = ref_info["actions_compact"]
        self.scenes = ref_info["scenes"]
        self.objects = ref_info["objects"]

        if self.ablate_action:
            for k in self.actions_compact:
                self.actions_compact[k] = "None"
        if self.ablate_scene:
            for k in self.scenes:
                self.scenes[k] = "None"
        if self.ablate_object:
            for k in self.objects:
                self.objects[k] = "None"
        if not self.speaker_config.use_enter_leave:
            for k, v in self.actions_compact.items():
                if v == "enter" or v == "leave" or v == "pass":
                    self.actions_compact[k] = "move forward"

        res = self.summarize_instruction()
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
