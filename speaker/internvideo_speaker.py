from PIL import Image
from pathlib import Path
from itertools import groupby
from collections import OrderedDict
import random
import json
import natsort
import time
import os
from fractions import Fraction
import av


import torch
import torch.distributed as dist

from tools.registry import registry
from tools.tools import diversify_action, diversify_element, im_resize

from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np
import random

### SETUP START ###
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


# video multi-round conversation
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_frame_indices(
    num_frames,
    vlen,
    sample="middle",
    fix_start=None,
    input_fps=1,
    min_num_frames=1,
    max_num_frames=-1,
    local_num_frames=8,
):
    if min_num_frames > vlen:
        if sample == "dynamic_fps1":
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen

    if sample == "dynamic_fps1":
        duration = float(vlen) / input_fps
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        sample = "middle"  # NOTE

        # logger.info(f"? is OK (img), duation={duration} frames={num_frames}!!!!")

    num_frames = max(min_num_frames, num_frames)

    # print(f"\033[0;31m vlen={vlen}, input_fps={input_fps} num_frames={num_frames} \033[0m")

    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = (
            1 / output_fps
        )  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError(f"Not support sample type: {sample}")

    return frame_indices


def read_frames_indices(
    vr,
    fps,
    num_frames,
    sample="rand",
    fix_start=None,
    min_num_frames=1,
    max_num_frames=-1,
    client=None,
    clip=None,
    local_num_frames=8,
):
    vlen = len(vr)
    duration = vlen / float(fps)

    if clip:
        start, end = clip
        start = max(0, start)
        end = min(duration - 0.1, end)
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        input_fps=fps,
        min_num_frames=min_num_frames,
        max_num_frames=max_num_frames,
        local_num_frames=local_num_frames,
    )

    return frame_indices


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def load_video(video_path, bound=None, input_size=448, max_num=1, max_num_frames=512):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = read_frames_indices(
        vr,
        fps,
        num_frames=max_num_frames,
        sample="dynamic_fps1",
        fix_start=None,
        min_num_frames=64,
        max_num_frames=max_num_frames,
        local_num_frames=8,
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def setup_video_chat(video_path, question, max_num_frames=512):
    pixel_values, num_patches_list = load_video(
        video_path, max_num_frames=max_num_frames
    )
    pixel_values = pixel_values.half().cuda()
    video_prefix = "".join(
        [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
    )
    question = video_prefix + question
    return pixel_values, num_patches_list, question


@registry.register_speaker
class InternVideoSpeaker:
    def __init__(
        self,
        config,
    ):
        self.internvideo_config = config.speaker_config.internvideo_config
        self.device = torch.device(dist.get_rank())

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.internvideo_config.model_name, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                self.internvideo_config.model_name, trust_remote_code=True
            )
            .half()
            .cuda()
        )
        # self.image_processor = self.model.get_vision_tower().image_processor

        self.system_prompt = self.internvideo_config.system_prompt
        self.temperature = self.internvideo_config.temperature
        self.top_p = self.internvideo_config.top_p
        self.top_k = self.internvideo_config.top_k
        self.max_frames = config.speaker_config.max_frames
        self.generation_config = dict(
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=256,
            top_p=self.top_p,
            num_beams=1,
        )

        self.fps = config.speaker_config.fps
        self.downsample = config.speaker_config.downsample
        self.info = {}

    def images_to_video(self, images):
        temp_video_path = "data/{}.mp4".format(time.time())

        images = [Image.open(v) for v in images]

        output = av.open(temp_video_path, "w")
        stream = output.add_stream("h264", rate=Fraction(str(self.fps)))
        stream.bit_rate = 8000000

        for i, img in enumerate(images):
            frame = av.VideoFrame.from_image(img)
            packet = stream.encode(frame)
            output.mux(packet)

        # flush
        packet = stream.encode(None)
        output.mux(packet)

        output.close()
        return temp_video_path

    def __call__(self, images):
        video_path = self.images_to_video(images)

        response = ""

        while not response:
            pixel_values, num_patches_list, prefix_question = setup_video_chat(
                video_path, self.system_prompt, max_num_frames=self.max_frames
            )

            response, chat_history = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=prefix_question,
                num_patches_list=num_patches_list,
                return_history=True,
                generation_config=self.generation_config,
            )
            response = response.strip()
            if not response:
                print("Empty response, repeat", flush=True)

        os.remove(video_path)
        return response

    def generate(self, source):
        """Source can be a Path, str, list of image paths"""
        if isinstance(source, Path) or isinstance(source, str):
            p = Path(source)
            source = []
            for v in p.glob("*.jpg"):
                source.append(v)
        source = natsort.natsorted(source, key=str)
        source = source[: self.max_frames]
        res = self(source)
        self.info = {"instruction": res}
        return res

    def get_detail(self, return_json=False):
        if return_json:
            return self.info
        else:
            return json.dumps(self.info, indent=2)
