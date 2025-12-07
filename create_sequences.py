import os
import numpy as np
import cv2
from pathlib import Path

# This should match your step 1 output folder
BASE = Path(r"C:\Users\marya\Downloads\vision_project\raw_frames")

# Output folder for npz files
OUT = Path(r"C:\Users\marya\Downloads\vision_project\sequences")

ACTIONS = [
    "BrushingTeeth",
    "Typing",
    "Punch",
    "WritingOnBoard"
]

SEQ_LEN = 15

# Assign labels to each action
LABELS = {action: i for i, action in enumerate(ACTIONS)}


def create_one_sequence(folder_path):
    """Loads 15 frames from the folder and returns a numpy array."""
    frames = sorted(os.listdir(folder_path))

    # must have exactly 15 frames
    if len(frames) != SEQ_LEN:
        return None

    sequence = []
    for f in frames:
        img_path = os.path.join(folder_path, f)
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        sequence.append(img)

    return np.array(sequence)   # shape (15, 128, 128, 3)


def main():
    if not os.path.exists(OUT):
        os.makedirs(OUT)

    total_saved = 0

    for action in ACTIONS:
        print(f"\nProcessing {action}...")

        action_folder = os.path.join(BASE, action)
        label = LABELS[action]

        # each video is a folder containing frames
        video_folders = os.listdir(action_folder)

        for vf in video_folders:
            vf_path = os.path.join(action_folder, vf)

            if not os.path.isdir(vf_path):
                continue

            seq = create_one_sequence(vf_path)

            if seq is None:
                print(f" Skipped {vf} (not enough frames)")
                continue

            # save as .npz file
            out_path = os.path.join(OUT, f"{action}_{vf}.npz")
            np.savez_compressed(out_path, x=seq, y=label)
            print(f" Saved sequence: {out_path}")
            total_saved += 1

    print(f"\nTotal sequences saved: {total_saved}")


if __name__ == "__main__":
    main()
