"""
autopilot_controller.py
────────────────────────
Wraps CARLA's built-in autopilot and adds:
  - Random gaussian noise perturbations (improves policy generalization)
  - Episode-level variance: speed limits, weather, maps
  - Clean control signal recording for VLA supervision

Action space:
  action = [steer, throttle, brake]   ← 3D continuous
"""

import random
import numpy as np
import carla


# Weather presets to cycle across episodes
WEATHER_PRESETS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.ClearNight,
]

# Available maps (make sure Additional Maps are installed)
TOWN_MAPS = [
    "Town01", "Town02", "Town03", "Town04",
    "Town05", "Town06", "Town07", "Town10HD",
]


class AutopilotController:
    """
    Controls the ego vehicle using CARLA's built-in autopilot
    and records the ground-truth (steer, throttle, brake) action.

    Perturbation mode adds small Gaussian noise to steer every N frames
    to create recovery demonstrations — critical for imitation learning.
    """

    def __init__(
        self,
        vehicle,
        perturb_prob: float = 0.05,    # probability of injecting perturbation
        perturb_std: float = 0.08,     # steer noise std when perturbing
        perturb_duration: int = 8,     # frames to hold perturbation
    ):
        self.vehicle = vehicle
        self.perturb_prob = perturb_prob
        self.perturb_std = perturb_std
        self.perturb_duration = perturb_duration

        self._perturb_countdown = 0
        self._perturb_steer = 0.0

    def enable_autopilot(self, tm_port: int = 8000):
        """Enable CARLA traffic manager autopilot."""
        self.vehicle.set_autopilot(True, tm_port)

    def get_control(self) -> dict:
        """
        Read the current control applied to the vehicle by autopilot.
        Optionally override steer with perturbation.
        Returns a dict with the 3D action and metadata.
        """
        ctrl = self.vehicle.get_control()
        steer = float(ctrl.steer)
        throttle = float(ctrl.throttle)
        brake = float(ctrl.brake)

        is_perturbed = False

        # Perturbation logic — creates recovery scenarios
        if self._perturb_countdown > 0:
            steer = np.clip(steer + self._perturb_steer, -1.0, 1.0)
            self._perturb_countdown -= 1
            is_perturbed = True
        elif random.random() < self.perturb_prob:
            self._perturb_steer = float(np.random.normal(0, self.perturb_std))
            self._perturb_countdown = self.perturb_duration
            steer = np.clip(steer + self._perturb_steer, -1.0, 1.0)
            is_perturbed = True

        # Apply (potentially perturbed) control back to vehicle
        new_ctrl = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=ctrl.hand_brake,
            reverse=ctrl.reverse,
        )
        self.vehicle.apply_control(new_ctrl)

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        return {
            # ── 3D action (the VLA supervision target) ──────────────────────
            "action": np.array([steer, throttle, brake], dtype=np.float32),
            # ── metadata ─────────────────────────────────────────────────────
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "speed_ms": speed_ms,
            "location": (transform.location.x, transform.location.y, transform.location.z),
            "rotation": (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            "is_perturbed": is_perturbed,
            "reverse": ctrl.reverse,
            "hand_brake": ctrl.hand_brake,
        }


def randomize_weather(world):
    """Pick a random weather preset and apply it."""
    preset = random.choice(WEATHER_PRESETS)
    world.set_weather(preset)
    return preset


def spawn_npc_traffic(world, client, n_vehicles=40, n_walkers=20, tm_port=8000):
    """
    Spawn NPC vehicles and walkers to make the scene realistic.
    Returns (vehicle_list, walker_list, walker_controller_list) for cleanup.
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # ── Vehicles ─────────────────────────────────────────────────────────────
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [v for v in vehicle_bps if int(v.get_attribute("number_of_wheels")) >= 4]
    npc_vehicles = []
    batch = []
    for i, sp in enumerate(spawn_points[:n_vehicles]):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        batch.append(carla.command.SpawnActor(bp, sp)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)))

    for response in client.apply_batch_sync(batch, True):
        if not response.error:
            npc_vehicles.append(response.actor_id)

    # ── Walkers ───────────────────────────────────────────────────────────────
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walker_spawn_points = []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc:
            walker_spawn_points.append(carla.Transform(loc))

    walker_batch = []
    for sp in walker_spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        walker_batch.append(carla.command.SpawnActor(bp, sp))

    walkers = []
    for response in client.apply_batch_sync(walker_batch, True):
        if not response.error:
            walkers.append({"id": response.actor_id})

    # Spawn walker AI controllers
    ctrl_bp = bp_lib.find("controller.ai.walker")
    ctrl_batch = [
        carla.command.SpawnActor(ctrl_bp, carla.Transform(), w["id"])
        for w in walkers
    ]
    for i, response in enumerate(client.apply_batch_sync(ctrl_batch, True)):
        if not response.error:
            walkers[i]["controller"] = response.actor_id

    world.tick()  # Let spawn settle

    # Start walker controllers
    walker_ids = [w["id"] for w in walkers]
    ctrl_ids = [w.get("controller") for w in walkers if "controller" in w]
    all_actor_ids = walker_ids + ctrl_ids
    all_actors = world.get_actors(all_actor_ids)

    for actor in all_actors:
        if "controller" in actor.type_id:
            actor.start()
            actor.go_to_location(world.get_random_location_from_navigation())
            actor.set_max_speed(1.0 + random.random())

    return npc_vehicles, walkers
