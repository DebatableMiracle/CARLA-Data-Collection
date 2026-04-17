"""
sensor_config.py
────────────────
Defines the full sensor suite attached to the ego vehicle.

Action space used (3D continuous control):
  - steer   : [-1.0,  1.0]   (left ← 0 → right)
  - throttle: [ 0.0,  1.0]
  - brake   : [ 0.0,  1.0]

All three are stored every frame so OpenVLA can learn to predict
the full VehicleControl tuple from the RGB observation.
"""

import carla

# ─── Image dimensions ─────────────────────────────────────────────────────────
IMG_W = 640
IMG_H = 480
IMG_FOV = 90   # degrees

# ─── Sensor tick (seconds). 0.0 = every simulation step ──────────────────────
SENSOR_TICK = 0.0


def attach_sensors(world, vehicle):
    """
    Spawns and attaches all sensors to `vehicle`.
    Returns a dict: { sensor_key: carla.Sensor }
    """
    bp_lib = world.get_blueprint_library()
    sensors = {}

    # ── 1. Front RGB camera (primary observation for VLA) ─────────────────────
    rgb_bp = bp_lib.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(IMG_W))
    rgb_bp.set_attribute("image_size_y", str(IMG_H))
    rgb_bp.set_attribute("fov", str(IMG_FOV))
    rgb_bp.set_attribute("sensor_tick", str(SENSOR_TICK))
    rgb_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    sensors["rgb_front"] = world.spawn_actor(
        rgb_bp, rgb_transform, attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid
    )

    # ── 2. Semantic segmentation camera (ground truth, optional) ──────────────
    seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
    seg_bp.set_attribute("image_size_x", str(IMG_W))
    seg_bp.set_attribute("image_size_y", str(IMG_H))
    seg_bp.set_attribute("fov", str(IMG_FOV))
    seg_bp.set_attribute("sensor_tick", str(SENSOR_TICK))
    sensors["seg_front"] = world.spawn_actor(
        seg_bp, rgb_transform, attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid
    )

    # ── 3. Depth camera (useful for 3D scene understanding) ───────────────────
    depth_bp = bp_lib.find("sensor.camera.depth")
    depth_bp.set_attribute("image_size_x", str(IMG_W))
    depth_bp.set_attribute("image_size_y", str(IMG_H))
    depth_bp.set_attribute("fov", str(IMG_FOV))
    depth_bp.set_attribute("sensor_tick", str(SENSOR_TICK))
    sensors["depth_front"] = world.spawn_actor(
        depth_bp, rgb_transform, attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid
    )

    # ── 4. LiDAR (optional — disable if disk space is tight) ──────────────────
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("range", "50")
    lidar_bp.set_attribute("points_per_second", "100000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("sensor_tick", str(SENSOR_TICK))
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.4))
    sensors["lidar"] = world.spawn_actor(
        lidar_bp, lidar_transform, attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid
    )

    # ── 5. Collision sensor (to detect crashes → end episode) ─────────────────
    col_bp = bp_lib.find("sensor.other.collision")
    sensors["collision"] = world.spawn_actor(
        col_bp, carla.Transform(), attach_to=vehicle
    )

    # ── 6. Lane invasion sensor ───────────────────────────────────────────────
    lane_bp = bp_lib.find("sensor.other.lane_invasion")
    sensors["lane_invasion"] = world.spawn_actor(
        lane_bp, carla.Transform(), attach_to=vehicle
    )

    return sensors


def destroy_sensors(sensors):
    for s in sensors.values():
        if s is not None and s.is_alive:
            s.destroy()
