"""
inspect_dataset.py
──────────────────
Quick sanity-check tool for collected HDF5 episodes.

Usage:
    python inspect_dataset.py --data-dir ./data/episodes
    python inspect_dataset.py --data-dir ./data/episodes --episode 3 --save-frames
"""

import argparse
import os
import glob
import numpy as np
import h5py
import cv2


def print_dataset_summary(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.h5")))
    if not files:
        print(f"No episodes found in {data_dir}")
        return

    total_frames = 0
    collisions = 0
    action_mins = np.full(3, np.inf)
    action_maxs = np.full(3, -np.inf)
    action_means = np.zeros(3)
    map_counts = {}

    print(f"\n{'='*55}")
    print(f"  Dataset: {data_dir}")
    print(f"  Episodes: {len(files)}")
    print(f"{'='*55}")

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            n = f["actions"].shape[0]
            total_frames += n
            actions = f["actions"][:]
            action_mins = np.minimum(action_mins, actions.min(axis=0))
            action_maxs = np.maximum(action_maxs, actions.max(axis=0))
            action_means += actions.mean(axis=0)

            town = str(f.attrs.get("town", "unknown"))
            map_counts[town] = map_counts.get(town, 0) + 1
            if f.attrs.get("collision", False):
                collisions += 1

    action_means /= len(files)

    print(f"  Total frames : {total_frames:,}")
    print(f"  Avg per ep   : {total_frames / len(files):.0f}")
    print(f"  Collisions   : {collisions} ({100*collisions/len(files):.1f}%)")
    print(f"\n  Action stats  [steer, throttle, brake]")
    print(f"    min  : {action_mins}")
    print(f"    max  : {action_maxs}")
    print(f"    mean : {action_means}")
    print(f"\n  Map distribution:")
    for town, count in sorted(map_counts.items()):
        print(f"    {town:12s}: {count} episodes")
    print()


def visualize_episode(data_dir: str, ep_idx: int, save_dir: str = None):
    fpath = os.path.join(data_dir, f"episode_{ep_idx:06d}.h5")
    if not os.path.exists(fpath):
        print(f"Episode file not found: {fpath}")
        return

    with h5py.File(fpath, "r") as f:
        print(f"\nEpisode {ep_idx}:")
        print(f"  Frames   : {f['actions'].shape[0]}")
        print(f"  Town     : {f.attrs.get('town')}")
        print(f"  Weather  : {f.attrs.get('weather')}")
        print(f"  Collision: {f.attrs.get('collision')}")

        actions = f["actions"][:]
        print(f"\n  Action summary:")
        print(f"    steer    mean={actions[:,0].mean():.3f}  std={actions[:,0].std():.3f}")
        print(f"    throttle mean={actions[:,1].mean():.3f}  std={actions[:,1].std():.3f}")
        print(f"    brake    mean={actions[:,2].mean():.3f}  std={actions[:,2].std():.3f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            rgb = f["observations/rgb_front"][:]
            step = max(1, len(rgb) // 20)   # sample 20 frames
            for i in range(0, len(rgb), step):
                frame = rgb[i]
                # Overlay action text
                a = actions[i]
                text = f"steer:{a[0]:.2f} thr:{a[1]:.2f} brk:{a[2]:.2f}"
                frame_bgr = frame[:, :, ::-1].copy()
                cv2.putText(frame_bgr, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                out_path = os.path.join(save_dir, f"ep{ep_idx:04d}_frame{i:05d}.jpg")
                cv2.imwrite(out_path, frame_bgr)
            print(f"\n  Saved sample frames → {save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    default="./data/episodes")
    p.add_argument("--episode",     type=int, default=None)
    p.add_argument("--save-frames", action="store_true")
    args = p.parse_args()

    print_dataset_summary(args.data_dir)
    if args.episode is not None:
        save = "./data/frame_samples" if args.save_frames else None
        visualize_episode(args.data_dir, args.episode, save_dir=save)
