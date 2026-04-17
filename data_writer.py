"""
data_writer.py
──────────────
Writes collected episodes to HDF5 files in a format compatible with OpenVLA
fine-tuning (and easily convertible to LeRobot / RLDS / HuggingFace datasets).

HDF5 layout per episode file:
  episode_XXXXXX.h5
  ├── observations/
  │   ├── rgb_front        [T, H, W, 3]  uint8
  │   ├── depth_front      [T, H, W, 1]  float32  (meters, log-scaled)
  │   ├── seg_front        [T, H, W, 1]  uint8    (class IDs)
  │   └── state            [T, 7]        float32  (x,y,z, pitch,yaw,roll, speed)
  ├── actions              [T, 3]        float32  (steer, throttle, brake)
  └── metadata             (attrs)
        town, weather, episode_id, fps, total_frames, collision, ...
"""

import os
import time
import numpy as np
import h5py
import cv2


class EpisodeBuffer:
    """Accumulates one episode in RAM before flushing to disk."""

    def __init__(self):
        self.rgb_frames: list[np.ndarray] = []
        self.depth_frames: list[np.ndarray] = []
        self.seg_frames: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.meta: dict = {}

    def add_step(
        self,
        rgb: np.ndarray,        # (H, W, 3)  uint8
        depth: np.ndarray,      # (H, W)     float32
        seg: np.ndarray,        # (H, W)     uint8
        state: np.ndarray,      # (7,)       float32
        action: np.ndarray,     # (3,)       float32
    ):
        self.rgb_frames.append(rgb)
        self.depth_frames.append(depth)
        self.seg_frames.append(seg)
        self.states.append(state)
        self.actions.append(action)
        self.timestamps.append(time.time())

    def __len__(self):
        return len(self.rgb_frames)


class DataWriter:
    """Saves EpisodeBuffers to HDF5 on disk."""

    def __init__(self, output_dir: str, fps: int = 20, compress: bool = True):
        self.output_dir = output_dir
        self.fps = fps
        self.compress = compress
        os.makedirs(output_dir, exist_ok=True)
        self._episode_count = self._count_existing()

    def _count_existing(self) -> int:
        files = [f for f in os.listdir(self.output_dir) if f.endswith(".h5")]
        return len(files)

    def save_episode(self, buf: EpisodeBuffer) -> str:
        """Write buffer to disk. Returns path of saved file."""
        if len(buf) == 0:
            return ""

        ep_id = self._episode_count
        self._episode_count += 1
        fname = os.path.join(self.output_dir, f"episode_{ep_id:06d}.h5")

        rgb_arr    = np.stack(buf.rgb_frames,   axis=0)    # (T, H, W, 3)
        depth_arr  = np.stack(buf.depth_frames, axis=0)    # (T, H, W)
        seg_arr    = np.stack(buf.seg_frames,   axis=0)    # (T, H, W)
        state_arr  = np.stack(buf.states,       axis=0)    # (T, 7)
        action_arr = np.stack(buf.actions,      axis=0)    # (T, 3)

        comp = dict(compression="lzf") if self.compress else {}

        with h5py.File(fname, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("rgb_front",   data=rgb_arr,                   **comp)
            obs.create_dataset("depth_front", data=depth_arr[..., np.newaxis].astype(np.float32), **comp)
            obs.create_dataset("seg_front",   data=seg_arr[..., np.newaxis].astype(np.uint8),     **comp)
            obs.create_dataset("state",       data=state_arr,                 **comp)
            f.create_dataset("actions", data=action_arr, **comp)

            # Metadata
            f.attrs["episode_id"]    = ep_id
            f.attrs["total_frames"]  = len(buf)
            f.attrs["fps"]           = self.fps
            f.attrs["action_space"]  = "steer,throttle,brake"
            f.attrs["action_min"]    = np.array([-1.0, 0.0, 0.0])
            f.attrs["action_max"]    = np.array([ 1.0, 1.0, 1.0])
            for k, v in buf.meta.items():
                try:
                    f.attrs[k] = v
                except Exception:
                    f.attrs[k] = str(v)

        print(f"  ✓ Saved episode {ep_id:06d} — {len(buf)} frames → {fname}")
        return fname


# ─── Image processing helpers ─────────────────────────────────────────────────

def carla_image_to_rgb(carla_img) -> np.ndarray:
    """Convert carla.Image (BGRA) → (H,W,3) RGB uint8."""
    arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_img.height, carla_img.width, 4))
    return arr[:, :, :3][..., ::-1].copy()   # BGRA → RGB


def carla_depth_to_meters(carla_img) -> np.ndarray:
    """
    Decode CARLA depth image to float32 meters.
    CARLA encodes depth in RGB channels as:
      depth_m = (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    """
    arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_img.height, carla_img.width, 4)).astype(np.float32)
    depth = (arr[:, :, 2] + arr[:, :, 1] * 256.0 + arr[:, :, 0] * 256.0 * 256.0)
    depth /= (256.0 ** 3 - 1.0)
    depth *= 1000.0   # convert to meters
    return depth.astype(np.float32)


def carla_seg_to_classid(carla_img) -> np.ndarray:
    """Extract semantic class IDs (R channel) from segmentation image."""
    arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_img.height, carla_img.width, 4))
    return arr[:, :, 2].copy()   # R channel = class ID in CARLA seg


def build_state_vector(ctrl_info: dict) -> np.ndarray:
    """
    Construct a (7,) state vector from controller telemetry.
    [x, y, z, pitch, yaw, roll, speed_ms]
    """
    x, y, z   = ctrl_info["location"]
    p, w, r   = ctrl_info["rotation"]
    speed     = ctrl_info["speed_ms"]
    return np.array([x, y, z, p, w, r, speed], dtype=np.float32)
