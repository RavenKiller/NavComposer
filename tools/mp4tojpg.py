import cv2
import os


def extract_frames(input_video, output_dir, target_fps=1):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {input_video}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps

    print(
        f"Original FPS: {original_fps:.2f}, Total Frames: {total_frames}, Duration: {duration_sec:.2f} seconds"
    )

    os.makedirs(output_dir, exist_ok=True)

    frame_interval = int(original_fps / target_fps)
    saved_count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{saved_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_id += 1

    cap.release()
    print(f"Saved {saved_count} images to '{output_dir}'")


if __name__ == "__main__":
    video_path = "data/navtj_test/DJI_20250529_092216_8_null_video.mp4"
    output_folder = "data/navtj_test/DJI_20250529_092216_8/rgb"
    extract_frames(video_path, output_folder, target_fps=2)
