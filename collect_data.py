"""
collect_data.py
───────────────
Main CARLA data collection script for OpenVLA fine-tuning.

Usage:
    python collect_data.py \
        --episodes 500 \
        --steps-per-episode 600 \
        --output-dir ./data/episodes \
        --maps Town01 Town03 Town05 \
        --n-vehicles 40 \
        --n-walkers 20 \
        --host 127.0.0.1 \
        --port 2000 \
        --seed 42

Action space: 3D continuous [steer ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]]
"""

import argparse
import random
import sys
import time
import traceback
from collections import deque
from threading import Event

import carla
import numpy as np
from tqdm import tqdm

from sensor_config import attach_sensors, destroy_sensors
from autopilot_controller import (
    AutopilotController,
    randomize_weather,
    spawn_npc_traffic,
    TOWN_MAPS,
)
from data_writer import (
    DataWriter,
    EpisodeBuffer,
    carla_image_to_rgb,
    carla_depth_to_meters,
    carla_seg_to_classid,
    build_state_vector,
)


# ─── Global sensor data store (written by callbacks, read by main loop) ───────
_sensor_data: dict = {}
_collision_flag = Event()
_lane_violation_count = 0


def _on_rgb(image):
    _sensor_data["rgb"] = image

def _on_depth(image):
    _sensor_data["depth"] = image

def _on_seg(image):
    _sensor_data["seg"] = image

def _on_lidar(data):
    _sensor_data["lidar"] = data

def _on_collision(event):
    _collision_flag.set()

def _on_lane_invasion(event):
    global _lane_violation_count
    _lane_violation_count += 1


# ─── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    client: carla.Client,
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    writer: DataWriter,
    args: argparse.Namespace,
    ep_idx: int,
) -> bool:
    """
    Run one data collection episode.
    Returns True if episode was saved successfully.
    """
    global _sensor_data, _lane_violation_count
    _sensor_data = {}
    _collision_flag.clear()
    _lane_violation_count = 0

    # ── Randomize map every N episodes ───────────────────────────────────────
    if ep_idx % args.map_switch_every == 0:
        town = random.choice(args.maps)
        print(f"\n[Episode {ep_idx}] Loading map: {town}")
        world = client.load_world(town)
        world.set_pedestrians_cross_factor(0.1)
        time.sleep(3.0)   # let map load
    else:
        town = world.get_map().name.split("/")[-1]

    # ── Sync mode: deterministic fixed timestep ───────────────────────────────
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(70.0)

    # ── Weather ───────────────────────────────────────────────────────────────
    weather = randomize_weather(world)
    weather_name = str(weather)

    # ── Spawn ego vehicle ─────────────────────────────────────────────────────
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.lincoln.mkz_2020")
    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", "255,255,255")

    spawn_points = world.get_map().get_spawn_points()
    ego_spawn = random.choice(spawn_points)

    ego_vehicle = None
    sensors = {}
    npc_vehicles, npc_walkers = [], []

    try:
        ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn)
        world.tick()  # register actor

        # ── Sensors ───────────────────────────────────────────────────────────
        sensors = attach_sensors(world, ego_vehicle)
        sensors["rgb_front"].listen(_on_rgb)
        sensors["depth_front"].listen(_on_depth)
        sensors["seg_front"].listen(_on_seg)
        sensors["lidar"].listen(_on_lidar)
        sensors["collision"].listen(_on_collision)
        sensors["lane_invasion"].listen(_on_lane_invasion)

        # ── NPC traffic ───────────────────────────────────────────────────────
        npc_vehicles, npc_walkers = spawn_npc_traffic(
            world, client,
            n_vehicles=args.n_vehicles,
            n_walkers=args.n_walkers,
            tm_port=args.tm_port,
        )

        # ── Autopilot ─────────────────────────────────────────────────────────
        controller = AutopilotController(
            ego_vehicle,
            perturb_prob=args.perturb_prob,
            perturb_std=args.perturb_std,
        )
        controller.enable_autopilot(tm_port=args.tm_port)

        # Warmup: let autopilot take over before recording
        for _ in range(args.warmup_steps):
            world.tick()
            time.sleep(0.001)

        # ── Data collection loop ──────────────────────────────────────────────
        buf = EpisodeBuffer()
        buf.meta = {
            "town": town,
            "weather": weather_name,
            "episode_index": ep_idx,
            "fps": args.fps,
            "spawn_x": ego_spawn.location.x,
            "spawn_y": ego_spawn.location.y,
        }

        min_speed_for_recording = 0.5   # m/s — skip standing-still frames
        consecutive_stopped = 0

        pbar = tqdm(range(args.steps_per_episode), desc=f"Ep {ep_idx:04d}", leave=False)
        for step in pbar:
            world.tick()   # advance simulation one step

            # ── Wait for all sensor callbacks to fire ─────────────────────────
            timeout = 0.1
            t0 = time.time()
            while not all(k in _sensor_data for k in ["rgb", "depth", "seg"]):
                if time.time() - t0 > timeout:
                    break
                time.sleep(0.001)

            if not all(k in _sensor_data for k in ["rgb", "depth", "seg"]):
                continue   # drop frame if sensors didn't respond

            # ── Abort on collision ────────────────────────────────────────────
            if _collision_flag.is_set():
                buf.meta["terminated_by"] = "collision"
                buf.meta["collision"] = True
                break

            # ── Get control action ────────────────────────────────────────────
            ctrl_info = controller.get_control()

            # Skip near-stationary frames (parked state, red light hold etc.)
            speed = ctrl_info["speed_ms"]
            if speed < min_speed_for_recording:
                consecutive_stopped += 1
                if consecutive_stopped > 60:   # ~3s stopped → end episode
                    buf.meta["terminated_by"] = "stuck"
                    break
                continue
            else:
                consecutive_stopped = 0

            # ── Process sensor data ───────────────────────────────────────────
            rgb_frame   = carla_image_to_rgb(_sensor_data["rgb"])
            depth_frame = carla_depth_to_meters(_sensor_data["depth"])
            seg_frame   = carla_seg_to_classid(_sensor_data["seg"])
            state_vec   = build_state_vector(ctrl_info)
            action_vec  = ctrl_info["action"]   # [steer, throttle, brake]

            buf.add_step(
                rgb=rgb_frame,
                depth=depth_frame,
                seg=seg_frame,
                state=state_vec,
                action=action_vec,
            )

            pbar.set_postfix(
                speed=f"{speed:.1f}m/s",
                steer=f"{ctrl_info['steer']:.2f}",
                frames=len(buf),
            )

        # ── Save if episode is long enough ────────────────────────────────────
        buf.meta["lane_violations"] = _lane_violation_count
        if not buf.meta.get("collision"):
            buf.meta["collision"] = False
        if not buf.meta.get("terminated_by"):
            buf.meta["terminated_by"] = "completed"

        min_frames = 30
        if len(buf) >= min_frames:
            writer.save_episode(buf)
            return True
        else:
            print(f"  ✗ Episode {ep_idx} too short ({len(buf)} frames) — discarded")
            return False

    except Exception as e:
        print(f"\n[ERROR] Episode {ep_idx} failed: {e}")
        traceback.print_exc()
        return False

    finally:
        # ── Cleanup actors ────────────────────────────────────────────────────
        destroy_sensors(sensors)
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()

        # Destroy NPC vehicles
        client.apply_batch([
            carla.command.DestroyActor(v) for v in npc_vehicles
        ])

        # Destroy NPC walkers + controllers
        walker_ids = [w["id"] for w in npc_walkers]
        ctrl_ids = [w["controller"] for w in npc_walkers if "controller" in w]
        all_walker_actors = world.get_actors(walker_ids + ctrl_ids)
        for actor in all_walker_actors:
            if "controller" in actor.type_id:
                actor.stop()
        client.apply_batch([
            carla.command.DestroyActor(a.id) for a in all_walker_actors
        ])

        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)

        world.tick()


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CARLA VLA Data Collector")
    p.add_argument("--host",               default="127.0.0.1")
    p.add_argument("--port",          type=int, default=2000)
    p.add_argument("--tm-port",       type=int, default=8000,  help="Traffic Manager port")
    p.add_argument("--episodes",      type=int, default=500,   help="Total episodes to collect")
    p.add_argument("--steps-per-episode", type=int, default=600, help="Max frames per episode at target FPS")
    p.add_argument("--fps",           type=int, default=20)
    p.add_argument("--warmup-steps",  type=int, default=30,    help="Frames before recording starts")
    p.add_argument("--output-dir",    default="./data/episodes")
    p.add_argument("--maps",          nargs="+", default=["Town01", "Town02", "Town03", "Town05"])
    p.add_argument("--map-switch-every", type=int, default=20, help="Switch map every N episodes")
    p.add_argument("--n-vehicles",    type=int, default=40)
    p.add_argument("--n-walkers",     type=int, default=20)
    p.add_argument("--perturb-prob",  type=float, default=0.05, help="Steer perturbation probability")
    p.add_argument("--perturb-std",   type=float, default=0.08, help="Steer perturbation std dev")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  CARLA VLA Data Collector")
    print(f"  Target: {args.episodes} episodes × {args.steps_per_episode} steps @ {args.fps} FPS")
    print(f"  Maps: {args.maps}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # ── Connect to CARLA ──────────────────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    server_ver = client.get_server_version()
    client_ver = client.get_client_version()
    print(f"\nConnected  server={server_ver}  client={client_ver}")

    traffic_manager = client.get_trafficmanager(args.tm_port)
    world = client.load_world(random.choice(args.maps))
    time.sleep(3.0)

    writer = DataWriter(output_dir=args.output_dir, fps=args.fps)

    # ── Collection loop ───────────────────────────────────────────────────────
    success = 0
    fail = 0
    pbar = tqdm(range(args.episodes), desc="Episodes")

    for ep_idx in pbar:
        ok = run_episode(
            client=client,
            world=world,
            traffic_manager=traffic_manager,
            writer=writer,
            args=args,
            ep_idx=ep_idx,
        )
        if ok:
            success += 1
        else:
            fail += 1

        # Reload world reference (may have been switched in run_episode)
        world = client.get_world()

        pbar.set_postfix(ok=success, fail=fail)

    print(f"\n✓ Done. Saved {success}/{args.episodes} episodes → {args.output_dir}")
    print(f"  Failed/skipped: {fail}")


if __name__ == "__main__":
    main()
