import cv2
import os
from pathlib import Path

# Folders you will use
BASE = Path(r"C:\Users\marya\Downloads\vision_project\data\UCF101\UCF-101")
OUT = Path(r"C:\Users\marya\Downloads\vision_project\raw_frames")

ACTIONS = [
    "BrushingTeeth",
    "Typing",
    "Punch",
    "WritingOnBoard"
]

SEQ_LEN = 15  # only 15 frames per video


def extract_only_15_frames(video_path, out_folder):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQ_LEN:
        return False  # skip very short videos

    # pick 15 evenly spaced frame indexes
    indexes = [int(i * total_frames / SEQ_LEN) for i in range(SEQ_LEN)]

    frames = []
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)

    cap.release()

    if len(frames) != SEQ_LEN:
        return False

    # save frames
    os.makedirs(out_folder, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(out_folder, f"frame{i}.jpg"), f)

    return True


def main():
    os.makedirs(OUT, exist_ok=True)

    for action in ACTIONS:
        action_in = os.path.join(BASE, action)
        action_out = os.path.join(OUT, action.replace(" ", "_"))
        os.makedirs(action_out, exist_ok=True)

        videos = [v for v in os.listdir(action_in)
                  if v.endswith((".avi", ".mp4"))]

        print(f"\nProcessing {action} ({len(videos)} videos)...")

        for v in videos:
            in_path = os.path.join(action_in, v)
            out_path = os.path.join(action_out, v.replace(".avi", "").replace(".mp4", ""))

            success = extract_only_15_frames(in_path, out_path)
            if success:
                print(f" Saved sequence from {v}")
            else:
                print(f" Skipped {v}")


if __name__ == "__main__":
    main()
