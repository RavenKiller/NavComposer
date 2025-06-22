import os
import json
import random
import shutil
import tempfile
import subprocess
import re
from natsort import natsorted
from tqdm import tqdm

# --- Configuration ---
SOURCE_DATA_DIR = [
    "../../data/vlnce_traj_action_survey/val_unseen",
    "../../data/vlnce_traj_action_survey/val_seen",
]
OUTPUT_DIR = "./"
CANDIDATE_MODELS = [
    "inst_coca",
    "inst_envdrop",
    "inst_qwen25vl",
    "inst_vo_gpt_gpt_qwn_mid_update",
]
NUM_PATHS_TO_SELECT = 400
VIDEO_FPS = 5
# Provide the full path to your specific FFmpeg executable
FFMPEG_PATH = "/share/home/u19666033/hzt/ffmpeg/ffmpeg"


def create_video_from_images_ffmpeg(image_folder, output_video_path, fps):
    """Creates a web-compatible H.264 video using the specified FFmpeg command-line tool."""
    if not shutil.which(FFMPEG_PATH):
        print(f"\nFATAL ERROR: FFmpeg executable not found at '{FFMPEG_PATH}'.")
        return False

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    if not images:
        return False

    images = natsorted(images)

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, image_name in enumerate(images):
            source_path = os.path.join(image_folder, image_name)
            destination_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            shutil.copy(source_path, destination_path)

        image_pattern = os.path.join(temp_dir, "%05d.jpg")

        # Use -framerate for image sequences, which is more robust than -r
        command = [
            FFMPEG_PATH,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            image_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-loglevel",
            "error",
            output_video_path,
        ]

        result = subprocess.run(command)

        if result.returncode != 0:
            print(f"  - Error: FFmpeg failed to create video for {image_folder}.")
            return False

    return True


def get_random_instruction(instruction_folder):
    """Selects and reads a random instruction from a folder, with enhanced text cleaning."""
    instruction_files = [
        f for f in os.listdir(instruction_folder) if f.endswith(".txt")
    ]
    if not instruction_files:
        print(f"  - Warning: No instruction files found in {instruction_folder}.")
        return "(No instruction available)"

    chosen_file = random.choice(instruction_files)
    with open(
        os.path.join(instruction_folder, chosen_file), "r", encoding="utf-8"
    ) as f:
        text = f.read().strip()

        if not text:
            return ""

        text = re.sub(r"\s+([.,!?])", r"\1", text)

        sentences = re.split("([.!?])", text)
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        capitalized_sentences = [s.strip().capitalize() for s in sentences if s.strip()]

        cleaned_text = " ".join(capitalized_sentences)

        return cleaned_text


def process_data():
    """Main data processing function."""
    print("Starting to process survey data...")

    final_output_dir = OUTPUT_DIR
    output_videos_dir = os.path.join(final_output_dir, "videos")

    if os.path.exists(output_videos_dir):
        shutil.rmtree(output_videos_dir)
    os.makedirs(output_videos_dir)

    # Collect all valid paths from all source directories
    all_available_paths = []
    for source_dir in SOURCE_DATA_DIR:
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found, skipping: {source_dir}")
            continue
        for path_name in os.listdir(source_dir):
            full_path = os.path.join(source_dir, path_name)
            if os.path.isdir(os.path.join(full_path, "rgb")):
                # Store both the name and its base directory
                all_available_paths.append({"name": path_name, "base": source_dir})

    if len(all_available_paths) < NUM_PATHS_TO_SELECT:
        print(
            f"Warning: Found {len(all_available_paths)} total paths, which is less than the requested {NUM_PATHS_TO_SELECT}. Processing all available paths."
        )
        selected_paths = all_available_paths
    else:
        # Deduplicate paths by name before sampling, in case of name collisions across source dirs
        unique_paths = {p["name"]: p for p in all_available_paths}.values()
        if len(unique_paths) < NUM_PATHS_TO_SELECT:
            print(
                f"Warning: Found {len(unique_paths)} unique paths, which is less than the requested {NUM_PATHS_TO_SELECT}. Processing all unique paths."
            )
            selected_paths = list(unique_paths)
        else:
            selected_paths = random.sample(list(unique_paths), NUM_PATHS_TO_SELECT)
    print(f"Totally {len(all_available_paths)} paths")
    print(f"Processing {len(selected_paths)} randomly selected paths...")

    survey_data_json = {}

    for path_info in tqdm(selected_paths, desc="Processing paths"):
        path_name = path_info["name"]
        base_dir = path_info["base"]

        # Construct the full path to the source trajectory directory
        source_path_dir = os.path.join(base_dir, path_name)
        source_image_folder = os.path.join(source_path_dir, "rgb")
        output_video_file = os.path.join(output_videos_dir, f"{path_name}.mp4")
        source_image_folder = source_image_folder.replace(
            "vlnce_traj_action_survey", "vlnce_traj_action_vis"
        )  # For HR videos
        # if not create_video_from_images_ffmpeg(source_image_folder, output_video_file, VIDEO_FPS):
        #     print(f"Skipping path {path_name} due to video creation failure.")
        #     continue

        instructions_for_path = {
            model: (
                get_random_instruction(os.path.join(source_path_dir, model))
                if os.path.isdir(os.path.join(source_path_dir, model))
                else "(Model instruction missing)"
            )
            for model in CANDIDATE_MODELS
        }

        survey_data_json[path_name] = {
            "video": f"videos/{path_name}.mp4",
            "instructions": instructions_for_path,
        }

    json_output_path = os.path.join(final_output_dir, "data_best.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(survey_data_json, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete! Survey data is ready in: '{final_output_dir}'")


if __name__ == "__main__":
    process_data()
