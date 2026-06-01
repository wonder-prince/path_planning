#!/usr/bin/env python3
"""
Enhanced Three-dimensional UAV path planning with practical boundary conditions
and interactive 3D HTML visualization.

Improvements over the original:
- No-fly zones (regulatory restrictions)
- Battery and energy constraints
- GPS coverage gaps
- Terrain elevation data
- Wind field effects
- Dynamic obstacle updates
- Interactive Plotly HTML visualization with playback controls
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize

RESEARCH_DIR = Path(__file__).resolve().parents[1]


# ============================================================================
# Enhanced Configuration with Practical Boundary Conditions
# ============================================================================


@dataclass(frozen=True)
class NoFlyZone:
    """Regulatory no-fly zone (e.g., airport, stadium, restricted airspace)."""
    center: np.ndarray          # (x, y, z)
    radius_m: float            # horizontal radius
    min_z_m: float            # minimum altitude (floor)
    max_z_m: float            # maximum altitude (ceiling)
    label: str = ""


@dataclass(frozen=True)
class GPSCoverageZone:
    """GPS coverage gap or degraded area (e.g., urban canyon, indoor)."""
    center: np.ndarray
    radius_m: float
    hdop_factor: float = 3.0   # HDOP multiplier (>1 means degraded accuracy)


@dataclass(frozen=True)
class WindField:
    """3D wind field at a given altitude band."""
    direction_hdg_deg: float   # horizontal wind direction (degrees true)
    speed_mps: float           # wind speed (m/s)
    min_z_m: float            # altitude band start
    max_z_m: float            # altitude band end
    gust_factor: float = 1.3  # gust multiplier


@dataclass(frozen=True)
class TerrainData:
    """Digital elevation model grid."""
    origin: np.ndarray        # (x_min, y_min)
    resolution_m: float
    elevations: np.ndarray    # 2D array [ny, nx] of terrain heights AGL


@dataclass(frozen=True)
class BatteryModel:
    """Battery constraints."""
    capacity_wh: float = 5000.0
    discharge_rate_wh_per_m: float = 0.05   # Wh per meter traveled (cruise)
    mass_kg: float = 4.5
    max_discharge_pct: float = 0.90         # max 90% discharge (keep reserve)
    cruise_power_w: float = 450.0           # watts at cruise


@dataclass(frozen=True)
class PlannerConfig:
    start: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    xyz_min: np.ndarray
    xyz_max: np.ndarray
    grid_resolution: np.ndarray

    # Core optimization settings
    n_segments: int = 20
    drone_radius: float = 1.0
    safety_margin: float = 5.0

    # Kinematic constraints
    cruise_speed: float = 8.0
    max_speed: float = 23.0
    max_climb_speed: float = 6.0
    max_descent_speed: float = 5.0
    max_turn_deg: float = 45.0
    max_flight_time_s: float = 55.0 * 60.0

    # Objective weights
    length_weight: float = 1.0
    smoothness_weight: float = 0.02
    energy_weight: float = 0.01
    obstacle_weight: float = 20.0
    alpha_energy: float = 1.0
    beta_climb: float = 1.5
    gamma_maneuver: float = 0.5

    # --- NEW: Practical boundary conditions ---
    no_fly_zones: tuple[NoFlyZone, ...] = field(default_factory=tuple)
    gps_coverage_zones: tuple[GPSCoverageZone, ...] = field(default_factory=tuple)
    wind_fields: tuple[WindField, ...] = field(default_factory=tuple)
    terrain: Optional[TerrainData] = None
    battery: BatteryModel = field(default_factory=BatteryModel)

    # Risk weights for practical constraints
    no_fly_penalty: float = 100.0
    gps_gap_penalty: float = 50.0
    wind_penalty: float = 0.5
    terrain_penalty: float = 30.0
    battery_penalty: float = 40.0

    @property
    def total_time_s(self) -> float:
        nominal_length = np.linalg.norm(self.goal - self.start)
        return nominal_length / self.cruise_speed

    @property
    def dt_s(self) -> float:
        return self.total_time_s / self.n_segments

    @property
    def final_inflated_radii(self) -> np.ndarray:
        return self.obstacles[:, 3] + self.drone_radius + self.safety_margin

    def battery_range_m(self) -> float:
        """Maximum range based on battery capacity."""
        usable_wh = self.battery.capacity_wh * self.battery.max_discharge_pct
        return usable_wh / self.battery.discharge_rate_wh_per_m

    def wind_at_point(self, point: np.ndarray) -> np.ndarray:
        """Get wind vector at a 3D point. Returns (wx, wy, wz) in m/s."""
        wx, wy, wz = 0.0, 0.0, 0.0
        for wf in self.wind_fields:
            if float(wf.min_z_m) <= point[2] <= float(wf.max_z_m):
                hdg_rad = math.radians(wf.direction_hdg_deg)
                wx = wf.speed_mps * math.sin(hdg_rad)
                wy = wf.speed_mps * math.cos(hdg_rad)
                wz = 0.0  # vertical wind usually negligible
                break
        return np.array([wx, wy, wz])

    def is_in_no_fly_zone(self, point: np.ndarray) -> bool:
        for zone in self.no_fly_zones:
            horiz_dist = math.sqrt((point[0] - zone.center[0])**2 + (point[1] - zone.center[1])**2)
            if horiz_dist <= zone.radius_m and zone.min_z_m <= point[2] <= zone.max_z_m:
                return True
        return False

    def is_in_gps_gap(self, point: np.ndarray) -> float:
        """Returns HDOP factor (>1 means degraded). 1.0 = nominal GPS."""
        for zone in self.gps_coverage_zones:
            dist = math.sqrt((point[0] - zone.center[0])**2 + (point[1] - zone.center[1])**2)
            if dist <= zone.radius_m:
                return zone.hdop_factor
        return 1.0

    def terrain_elevation_at(self, x: float, y: float) -> float:
        """Get terrain elevation AGL at (x, y)."""
        if self.terrain is None:
            return 0.0
        t = self.terrain
        ix = int(round((x - t.origin[0]) / t.resolution_m))
        iy = int(round((y - t.origin[1]) / t.resolution_m))
        if 0 <= ix < t.elevations.shape[1] and 0 <= iy < t.elevations.shape[0]:
            return float(t.elevations[iy, ix])
        return 0.0


def make_default_config(n_segments: int = 20, safety_margin: float = 5.0) -> PlannerConfig:
    """Enhanced default configuration with realistic no-fly zones and practical constraints."""
    return PlannerConfig(
        start=np.array([0.0, 0.0, 50.0]),
        goal=np.array([500.0, 420.0, 80.0]),
        obstacles=np.array(
            [
                [120.0, 100.0, 40.0, 45.0],
                [240.0, 180.0, 70.0, 60.0],
                [330.0, 300.0, 50.0, 55.0],
                [410.0, 360.0, 80.0, 50.0],
                # Additional dynamic obstacle (e.g., moving aircraft)
                [300.0, 200.0, 60.0, 35.0],
            ]
        ),
        xyz_min=np.array([0.0, 0.0, 20.0]),
        xyz_max=np.array([500.0, 500.0, 120.0]),
        grid_resolution=np.array([10.0, 10.0, 5.0]),
        n_segments=n_segments,
        safety_margin=safety_margin,

        # --- Practical boundary conditions ---
        no_fly_zones=(
            # Airport-like restriction near (80, 350)
            NoFlyZone(center=np.array([80.0, 350.0, 0.0]), radius_m=60.0, min_z_m=0.0, max_z_m=150.0, label="Airport A"),
            # Stadium no-fly zone near (400, 100)
            NoFlyZone(center=np.array([400.0, 100.0, 0.0]), radius_m=40.0, min_z_m=0.0, max_z_m=80.0, label="Stadium B"),
        ),
        gps_coverage_zones=(
            # Urban canyon effect near (250, 250)
            GPSCoverageZone(center=np.array([250.0, 250.0, 0.0]), radius_m=80.0, hdop_factor=4.5),
            # Building shadow near (350, 400)
            GPSCoverageZone(center=np.array([350.0, 400.0, 0.0]), radius_m=50.0, hdop_factor=3.0),
        ),
        wind_fields=(
            # Light wind from west at low altitude
            WindField(direction_hdg_deg=270.0, speed_mps=4.0, min_z_m=20.0, max_z_m=60.0, gust_factor=1.2),
            # Stronger wind from NW at higher altitude
            WindField(direction_hdg_deg=315.0, speed_mps=8.0, min_z_m=60.0, max_z_m=120.0, gust_factor=1.4),
        ),
        terrain=TerrainData(
            origin=np.array([0.0, 0.0]),
            resolution_m=20.0,
            elevations=np.array([
                [0, 0, 1, 2, 3, 2, 1, 0, 0, 0],
                [0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
                [0, 0, 1, 2, 3, 2, 1, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 1, 2, 3, 2, 1, 0, 0, 0],
                [0, 1, 2, 3, 4, 3, 2, 1, 0, 0],
                [0, 0, 1, 2, 3, 2, 1, 0, 0, 0],
            ], dtype=float) * 5.0,  # 5m resolution grid, terrain up to ~20m AGL
        ),
        battery=BatteryModel(
            capacity_wh=5000.0,
            discharge_rate_wh_per_m=0.05,
            mass_kg=4.5,
            max_discharge_pct=0.90,
            cruise_power_w=450.0,
        ),
    )


# ============================================================================
# Core Path Planning Functions (unchanged with enhancements)
# ============================================================================


def softplus(x: np.ndarray | float, beta: float = 12.0) -> np.ndarray | float:
    """Numerically stable softplus."""
    return np.logaddexp(0.0, beta * x) / beta


def point_is_free(point: np.ndarray, config: PlannerConfig, inflated_radii: np.ndarray) -> bool:
    if np.any(point < config.xyz_min) or np.any(point > config.xyz_max):
        return False
    centers = config.obstacles[:, :3]
    dist_sq = np.sum((centers - point) ** 2, axis=1)
    if not np.all(dist_sq > inflated_radii**2):
        return False
    # Check no-fly zones
    if config.is_in_no_fly_zone(point):
        return False
    return True


def idx_to_point(idx: tuple[int, int, int], config: PlannerConfig) -> np.ndarray:
    return config.xyz_min + np.asarray(idx, dtype=float) * config.grid_resolution


def point_to_idx(point: np.ndarray, config: PlannerConfig) -> tuple[int, int, int]:
    raw = np.rint((point - config.xyz_min) / config.grid_resolution).astype(int)
    return tuple(int(v) for v in raw)


def neighbor_offsets() -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def astar_3d(config: PlannerConfig) -> np.ndarray:
    inflated_radii = config.final_inflated_radii
    start_idx = point_to_idx(config.start, config)
    goal_idx = point_to_idx(config.goal, config)
    grid_shape = np.floor((config.xyz_max - config.xyz_min) / config.grid_resolution).astype(int) + 1

    def heuristic(idx: tuple[int, int, int]) -> float:
        return float(np.linalg.norm(idx_to_point(idx, config) - config.goal))

    open_heap: list[tuple[float, float, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start_idx), 0.0, start_idx))
    parents: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_idx: None}
    costs: dict[tuple[int, int, int], float] = {start_idx: 0.0}
    offsets = neighbor_offsets()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current == goal_idx:
            break
        if current_cost > costs[current]:
            continue

        current_arr = np.asarray(current)
        for offset in offsets:
            nxt_arr = current_arr + np.asarray(offset)
            if np.any(nxt_arr < 0) or np.any(nxt_arr >= grid_shape):
                continue
            nxt = tuple(int(v) for v in nxt_arr)
            nxt_point = idx_to_point(nxt, config)
            if not point_is_free(nxt_point, config, inflated_radii):
                continue

            step = np.linalg.norm(np.asarray(offset, dtype=float) * config.grid_resolution)
            new_cost = current_cost + float(step)
            if new_cost < costs.get(nxt, math.inf):
                costs[nxt] = new_cost
                parents[nxt] = current
                heapq.heappush(open_heap, (new_cost + heuristic(nxt), new_cost, nxt))

    if goal_idx not in parents:
        raise RuntimeError("A* 未找到可行初始路径；请调整栅格分辨率、边界或安全裕度。")

    indices: list[tuple[int, int, int]] = []
    node: tuple[int, int, int] | None = goal_idx
    while node is not None:
        indices.append(node)
        node = parents[node]
    indices.reverse()
    return np.vstack([idx_to_point(idx, config) for idx in indices])


def resample_by_arclength(path: np.ndarray, n_points: int) -> np.ndarray:
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, cumulative[-1], n_points)
    resampled = np.column_stack(
        [np.interp(targets, cumulative, path[:, dim]) for dim in range(path.shape[1])]
    )
    return resampled


def bspline_resample(path: np.ndarray, n_points: int) -> np.ndarray:
    if len(path) < 4:
        return resample_by_arclength(path, n_points)
    diffs = np.diff(path, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    if cumulative[-1] < 1e-9:
        return np.repeat(path[:1], n_points, axis=0)
    u = cumulative / cumulative[-1]
    tck, _ = splprep(path.T, u=u, s=0.0, k=min(3, len(path) - 1))
    new_u = np.linspace(0.0, 1.0, n_points)
    smoothed = np.column_stack(splev(new_u, tck))
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def path_from_internal_points(x: np.ndarray, config: PlannerConfig) -> np.ndarray:
    internal = x.reshape(config.n_segments - 1, 3)
    return np.vstack([config.start, internal, config.goal])


def segment_min_dist_sq_to_center(p1: np.ndarray, p2: np.ndarray, center: np.ndarray) -> float:
    seg = p2 - p1
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-12:
        return float(np.sum((p1 - center) ** 2))
    t = float(np.dot(center - p1, seg) / seg_len_sq)
    t = float(np.clip(t, 0.0, 1.0))
    closest = p1 + t * seg
    return float(np.sum((closest - center) ** 2))


def segment_clearances(path: np.ndarray, config: PlannerConfig, safety_margin: float) -> np.ndarray:
    centers = config.obstacles[:, :3]
    inflated = config.obstacles[:, 3] + config.drone_radius + safety_margin
    values: list[float] = []
    for i in range(len(path) - 1):
        for center, radius in zip(centers, inflated):
            min_dist_sq = segment_min_dist_sq_to_center(path[i], path[i + 1], center)
            values.append(min_dist_sq - radius**2)
    return np.asarray(values)


# ============================================================================
# Enhanced Objective & Constraints with Practical Boundary Conditions
# ============================================================================


def objective(x: np.ndarray, config: PlannerConfig, safety_margin: float) -> float:
    path = path_from_internal_points(x, config)
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    length_term = float(np.sum(segment_lengths))

    second_diff = path[2:] - 2.0 * path[1:-1] + path[:-2]
    smoothness_term = float(np.sum(second_diff**2))

    velocities = diffs / config.dt_s
    climb_term = np.asarray(softplus(velocities[:, 2], beta=5.0))
    maneuver = np.diff(velocities, axis=0)
    energy_term = float(
        np.sum(
            config.alpha_energy * np.sum(velocities**2, axis=1)
            + config.beta_climb * climb_term**2
        )
        * config.dt_s
        + config.gamma_maneuver * np.sum(maneuver**2) * config.dt_s
    )

    centers = config.obstacles[:, :3]
    inflated = config.obstacles[:, 3] + config.drone_radius + safety_margin
    obstacle_term = 0.0
    for point in path[1:-1]:
        distances_sq = np.sum((centers - point) ** 2, axis=1)
        normalized_violation = (inflated**2 - distances_sq) / np.maximum(inflated**2, 1.0)
        obstacle_term += float(np.sum(np.asarray(softplus(normalized_violation)) ** 2))

    # --- Practical boundary condition penalties ---
    no_fly_penalty = 0.0
    for point in path:
        if config.is_in_no_fly_zone(point):
            no_fly_penalty += 1.0

    gps_penalty = 0.0
    for point in path:
        hdop = config.is_in_gps_gap(point)
        if hdop > 1.0:
            gps_penalty += (hdop - 1.0)**2

    wind_penalty = 0.0
    for point in path:
        wind = config.wind_at_point(point)
        wind_speed = np.linalg.norm(wind)
        if wind_speed > 0:
            rel_vel = velocities[np.argmax(np.linalg.norm(velocities, axis=1))]
            wind_align = np.dot(rel_vel, wind) / (np.linalg.norm(rel_vel) * wind_speed + 1e-6)
            wind_penalty += wind_speed * (1.0 - wind_align)

    battery_penalty = 0.0
    total_length = length_term
    if total_length > config.battery_range_m():
        battery_penalty += (total_length - config.battery_range_m())**2

    terrain_penalty = 0.0
    if config.terrain is not None:
        for point in path:
            terrain_h = config.terrain_elevation_at(point[0], point[1])
            if point[2] < terrain_h + config.drone_radius + 3.0:
                terrain_penalty += softplus(terrain_h + config.drone_radius + 3.0 - point[2])**2

    return float(
        config.length_weight * length_term
        + config.smoothness_weight * smoothness_term
        + config.energy_weight * energy_term
        + config.obstacle_weight * obstacle_term
        + config.no_fly_penalty * no_fly_penalty
        + config.gps_gap_penalty * gps_penalty
        + config.wind_penalty * wind_penalty
        + config.battery_penalty * battery_penalty
        + config.terrain_penalty * terrain_penalty
    )


def nonlinear_constraints(x: np.ndarray, config: PlannerConfig, safety_margin: float) -> np.ndarray:
    path = path_from_internal_points(x, config)
    diffs = np.diff(path, axis=0)
    segment_len_sq = np.sum(diffs**2, axis=1)
    dz = diffs[:, 2]

    speed_ok = config.max_speed**2 * config.dt_s**2 - segment_len_sq
    climb_ok = config.max_climb_speed * config.dt_s - dz
    descent_ok = config.max_descent_speed * config.dt_s + dz
    obstacle_ok = segment_clearances(path, config, safety_margin)

    v_prev = diffs[:-1]
    v_next = diffs[1:]
    denom = np.sqrt(np.sum(v_prev**2, axis=1) + 1e-6) * np.sqrt(
        np.sum(v_next**2, axis=1) + 1e-6
    )
    cos_angles = np.sum(v_prev * v_next, axis=1) / denom
    turn_ok = cos_angles - math.cos(math.radians(config.max_turn_deg))

    return np.concatenate([speed_ok, climb_ok, descent_ok, obstacle_ok, turn_ok])


def optimize_path(initial_path: np.ndarray, config: PlannerConfig) -> tuple[np.ndarray, list[dict[str, float | bool | str]]]:
    internal = initial_path[1:-1].copy()
    x = internal.reshape(-1)
    bounds: list[tuple[float, float]] = []
    for _ in range(config.n_segments - 1):
        bounds.extend(
            [
                (float(config.xyz_min[0]), float(config.xyz_max[0])),
                (float(config.xyz_min[1]), float(config.xyz_max[1])),
                (float(config.xyz_min[2]), float(config.xyz_max[2])),
            ]
        )

    margins = sorted(set([0.5, min(2.0, config.safety_margin), config.safety_margin]))
    stages: list[dict[str, float | bool | str]] = []

    for margin in margins:
        result = minimize(
            fun=lambda values, m=margin: objective(values, config, m),
            x0=x,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda values, m=margin: nonlinear_constraints(values, config, m),
                }
            ],
            options={"maxiter": 500, "ftol": 1e-7, "disp": False},
        )
        x = result.x
        stage_path = path_from_internal_points(x, config)
        stages.append(
            {
                "margin_m": float(margin),
                "success": bool(result.success),
                "message": str(result.message),
                "objective": float(result.fun),
                "min_constraint": float(np.min(nonlinear_constraints(x, config, margin))),
                "path_length_m": float(np.sum(np.linalg.norm(np.diff(stage_path, axis=0), axis=1))),
            }
        )

    return path_from_internal_points(x, config), stages


def compute_metrics(path: np.ndarray, config: PlannerConfig) -> dict[str, float]:
    diffs = np.diff(path, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    velocities = diffs / config.dt_s
    second_diff = path[2:] - 2.0 * path[1:-1] + path[:-2]
    v_prev = diffs[:-1]
    v_next = diffs[1:]
    denom = np.sqrt(np.sum(v_prev**2, axis=1) + 1e-6) * np.sqrt(
        np.sum(v_next**2, axis=1) + 1e-6
    )
    cos_angles = np.clip(np.sum(v_prev * v_next, axis=1) / denom, -1.0, 1.0)
    turn_angles = np.degrees(np.arccos(cos_angles))
    clearances_sq = segment_clearances(path, config, config.safety_margin)
    expanded_radii = np.tile(config.final_inflated_radii, config.n_segments)
    signed_clearances = np.sqrt(np.maximum(clearances_sq + expanded_radii**2, 0.0)) - expanded_radii

    # Practical metrics
    no_fly_violations = sum(1 for p in path if config.is_in_no_fly_zone(p))
    max_gps_hdop = max(config.is_in_gps_gap(p) for p in path)
    max_wind_speed = max(np.linalg.norm(config.wind_at_point(p)) for p in path)

    return {
        "path_length_m": float(np.sum(lengths)),
        "smoothness_sum_sq": float(np.sum(second_diff**2)),
        "max_speed_mps": float(np.max(np.linalg.norm(velocities, axis=1))),
        "max_climb_mps": float(np.max(velocities[:, 2])),
        "max_descent_mps": float(np.max(-velocities[:, 2])),
        "max_turn_deg": float(np.max(turn_angles)),
        "min_clearance_m": float(np.min(signed_clearances)),
        "flight_time_s": float(config.total_time_s),
        # Practical boundary condition metrics
        "no_fly_zone_violations": float(no_fly_violations),
        "max_gps_hdop": float(max_gps_hdop),
        "max_wind_speed_mps": float(max_wind_speed),
        "battery_range_used_pct": float(np.sum(lengths) / config.battery_range_m() * 100),
        "terrain_clearance_min_m": float(min(
            p[2] - config.terrain_elevation_at(p[0], p[1]) for p in path
        )) if config.terrain else 0.0,
    }


def compute_profile_data(path: np.ndarray, config: PlannerConfig) -> dict[str, np.ndarray]:
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    waypoint_times = np.arange(len(path), dtype=float) * config.dt_s
    segment_times = waypoint_times[:-1] + config.dt_s / 2.0
    velocities = diffs / config.dt_s
    speeds = np.linalg.norm(velocities, axis=1)

    v_prev = diffs[:-1]
    v_next = diffs[1:]
    denom = np.sqrt(np.sum(v_prev**2, axis=1) + 1e-6) * np.sqrt(
        np.sum(v_next**2, axis=1) + 1e-6
    )
    cos_angles = np.clip(np.sum(v_prev * v_next, axis=1) / denom, -1.0, 1.0)
    turn_angles = np.degrees(np.arccos(cos_angles))

    centers = config.obstacles[:, :3]
    inflated = config.final_inflated_radii
    segment_min_clearance = []
    closest_obstacle_index = []
    for p1, p2 in zip(path[:-1], path[1:]):
        per_obstacle = []
        for center, radius in zip(centers, inflated):
            min_dist_sq = segment_min_dist_sq_to_center(p1, p2, center)
            per_obstacle.append(math.sqrt(max(min_dist_sq, 0.0)) - radius)
        per_obstacle_arr = np.asarray(per_obstacle)
        segment_min_clearance.append(float(np.min(per_obstacle_arr)))
        closest_obstacle_index.append(int(np.argmin(per_obstacle_arr)))

    # Practical boundary condition profiles
    gps_hdop_profile = np.array([config.is_in_gps_gap(p) for p in path])
    wind_speed_profile = np.array([np.linalg.norm(config.wind_at_point(p)) for p in path])

    return {
        "waypoint_times_s": waypoint_times,
        "segment_times_s": segment_times,
        "cumulative_distance_m": cumulative_distance,
        "segment_lengths_m": segment_lengths,
        "speeds_mps": speeds,
        "climb_rates_mps": velocities[:, 2],
        "turn_angles_deg": turn_angles,
        "segment_min_clearance_m": np.asarray(segment_min_clearance),
        "closest_obstacle_index": np.asarray(closest_obstacle_index),
        "gps_hdop_profile": gps_hdop_profile,
        "wind_speed_profile": wind_speed_profile,
    }


# ============================================================================
# Visualization Functions
# ============================================================================


def draw_sphere(ax: plt.Axes, center: np.ndarray, radius: float, color: str, alpha: float) -> None:
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=False)


def save_visualization(
    coarse_path: np.ndarray,
    initial_path: np.ndarray,
    optimized_path: np.ndarray,
    config: PlannerConfig,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(1, 2, 2)

    # Draw obstacles
    for obstacle, inflated_radius in zip(config.obstacles, config.final_inflated_radii):
        center = obstacle[:3]
        draw_sphere(ax3d, center, obstacle[3], color="#d62728", alpha=0.25)
        draw_sphere(ax3d, center, inflated_radius, color="#ff9896", alpha=0.08)
        circle = plt.Circle(
            (center[0], center[1]),
            inflated_radius,
            color="#d62728",
            alpha=0.16,
            linewidth=0,
        )
        ax_xy.add_patch(circle)

    # Draw no-fly zones
    for zone in config.no_fly_zones:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        z_bottom = max(zone.min_z_m, config.xyz_min[2])
        z_top = min(zone.max_z_m, config.xyz_max[2])
        for z_val in [z_bottom, z_top]:
            x = zone.center[0] + zone.radius_m * np.outer(np.cos(u), np.sin(v))
            y = zone.center[1] + zone.radius_m * np.outer(np.sin(u), np.sin(v))
            z = np.full_like(x, z_val)
            ax3d.plot_surface(x, y, z, color="#8b0000", alpha=0.15, linewidth=0)
        # Vertical walls
        for angle in np.linspace(0, 2*np.pi, 20):
            x_wall = zone.center[0] + zone.radius_m * np.cos(angle)
            y_wall = zone.center[1] + zone.radius_m * np.sin(angle)
            ax3d.plot([x_wall, x_wall], [y_wall, y_wall], [z_bottom, z_top], color="#8b0000", alpha=0.15)

    # Draw GPS coverage zones (semi-transparent cylinders)
    for zone in config.gps_coverage_zones:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, 1, 8)
        x = zone.center[0] + zone.radius_m * np.outer(np.cos(u), v)
        y = zone.center[1] + zone.radius_m * np.outer(np.sin(u), v)
        z = np.ones_like(x) * config.xyz_min[2]
        ax3d.plot_surface(x, y, z, color="#006400", alpha=0.12, linewidth=0)

    ax3d.plot(coarse_path[:, 0], coarse_path[:, 1], coarse_path[:, 2], color="#7f7f7f", linestyle="--", linewidth=1.2, label="A* coarse path")
    ax3d.plot(initial_path[:, 0], initial_path[:, 1], initial_path[:, 2], color="#ff7f0e", marker="o", linewidth=1.6, markersize=3, label="B-spline init")
    ax3d.plot(optimized_path[:, 0], optimized_path[:, 1], optimized_path[:, 2], color="#1f77b4", marker="o", linewidth=2.2, markersize=3, label="SQP optimized")
    ax3d.scatter(*config.start, color="green", s=55, label="Start", marker="s")
    ax3d.scatter(*config.goal, color="blue", s=55, label="Goal", marker="^")
    ax3d.set_xlim(config.xyz_min[0], config.xyz_max[0])
    ax3d.set_ylim(config.xyz_min[1], config.xyz_max[1])
    ax3d.set_zlim(config.xyz_min[2], config.xyz_max[2])
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D UAV Path Planning with Practical Constraints")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.legend(loc="upper left")

    ax_xy.plot(coarse_path[:, 0], coarse_path[:, 1], color="#7f7f7f", linestyle="--", linewidth=1.2, label="A* coarse path")
    ax_xy.plot(initial_path[:, 0], initial_path[:, 1], color="#ff7f0e", marker="o", linewidth=1.6, markersize=3, label="B-spline init")
    ax_xy.plot(optimized_path[:, 0], optimized_path[:, 1], color="#1f77b4", marker="o", linewidth=2.2, markersize=3, label="SQP optimized")
    ax_xy.scatter(config.start[0], config.start[1], color="green", s=55, marker="s")
    ax_xy.scatter(config.goal[0], config.goal[1], color="blue", s=55, marker="^")
    for zone in config.no_fly_zones:
        circle = plt.Circle(
            (zone.center[0], zone.center[1]),
            zone.radius_m,
            color="#8b0000",
            alpha=0.25,
            linewidth=1.5,
            linestyle="--",
            fill=False,
        )
        ax_xy.add_patch(circle)
    for zone in config.gps_coverage_zones:
        circle = plt.Circle(
            (zone.center[0], zone.center[1]),
            zone.radius_m,
            color="#006400",
            alpha=0.18,
            linewidth=1.2,
            linestyle=":",
            fill=False,
        )
        ax_xy.add_patch(circle)
    ax_xy.set_xlim(config.xyz_min[0], config.xyz_max[0])
    ax_xy.set_ylim(config.xyz_min[1], config.xyz_max[1])
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top View / XY Projection")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_dashboard(
    coarse_path: np.ndarray,
    initial_path: np.ndarray,
    optimized_path: np.ndarray,
    config: PlannerConfig,
    output_path: Path,
) -> None:
    optimized_profile = compute_profile_data(optimized_path, config)
    initial_metrics = compute_metrics(initial_path, config)
    optimized_metrics = compute_metrics(optimized_path, config)

    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.05, 0.95])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_altitude = fig.add_subplot(grid[0, 2])
    ax_speed = fig.add_subplot(grid[1, 1])
    ax_clearance = fig.add_subplot(grid[1, 2])

    for idx, (obstacle, inflated_radius) in enumerate(zip(config.obstacles, config.final_inflated_radii), start=1):
        center = obstacle[:3]
        draw_sphere(ax3d, center, obstacle[3], color="#d62728", alpha=0.24)
        draw_sphere(ax3d, center, inflated_radius, color="#ff9896", alpha=0.07)
        circle = plt.Circle(
            (center[0], center[1]),
            inflated_radius,
            color="#d62728",
            alpha=0.14,
            linewidth=0,
        )
        ax_xy.add_patch(circle)
        ax_xy.text(center[0], center[1], f"O{idx}", ha="center", va="center", fontsize=9, color="#8c1d18")

    ax3d.plot(coarse_path[:, 0], coarse_path[:, 1], coarse_path[:, 2], color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.8, label="A* coarse")
    ax3d.plot(initial_path[:, 0], initial_path[:, 1], initial_path[:, 2], color="#ff7f0e", linewidth=1.7, marker="o", markersize=3, label="B-spline init")
    ax3d.plot(optimized_path[:, 0], optimized_path[:, 1], optimized_path[:, 2], color="#1f77b4", linewidth=2.4, marker="o", markersize=3, label="SQP optimized")
    ax3d.scatter(*config.start, color="green", s=60, label="Start", marker="s")
    ax3d.scatter(*config.goal, color="blue", s=60, label="Goal", marker="^")
    ax3d.set_xlim(config.xyz_min[0], config.xyz_max[0])
    ax3d.set_ylim(config.xyz_min[1], config.xyz_max[1])
    ax3d.set_zlim(config.xyz_min[2], config.xyz_max[2])
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Trajectory and Safety Envelope")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.legend(loc="upper left")

    ax_xy.plot(coarse_path[:, 0], coarse_path[:, 1], "--", color="#7f7f7f", linewidth=1.0, label="A*")
    ax_xy.plot(initial_path[:, 0], initial_path[:, 1], color="#ff7f0e", linewidth=1.6, marker="o", markersize=2.5, label="B-spline")
    ax_xy.plot(optimized_path[:, 0], optimized_path[:, 1], color="#1f77b4", linewidth=2.3, marker="o", markersize=2.5, label="SQP")
    ax_xy.scatter(config.start[0], config.start[1], color="green", s=45)
    ax_xy.scatter(config.goal[0], config.goal[1], color="blue", s=45)
    ax_xy.set_xlim(config.xyz_min[0], config.xyz_max[0])
    ax_xy.set_ylim(config.xyz_min[1], config.xyz_max[1])
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title("Top View / XY Projection")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(True, alpha=0.28)
    ax_xy.legend(loc="upper left")

    ax_altitude.plot(optimized_profile["cumulative_distance_m"], optimized_path[:, 2], color="#1f77b4", linewidth=2.2, marker="o", markersize=3)
    ax_altitude.axhspan(config.xyz_min[2], config.xyz_max[2], color="#2ca02c", alpha=0.08, label="allowed altitude")
    ax_altitude.set_title("Altitude Profile")
    ax_altitude.set_xlabel("Cumulative distance (m)")
    ax_altitude.set_ylabel("Z (m)")
    ax_altitude.grid(True, alpha=0.28)
    ax_altitude.legend(loc="best")

    ax_speed.plot(optimized_profile["segment_times_s"], optimized_profile["speeds_mps"], color="#1f77b4", linewidth=2.1, marker="o", markersize=3, label="speed")
    ax_speed.axhline(config.max_speed, color="#d62728", linestyle="--", linewidth=1.2, label="speed limit")
    ax_speed.plot(optimized_profile["segment_times_s"], optimized_profile["climb_rates_mps"], color="#2ca02c", linewidth=1.6, marker="s", markersize=2.5, label="vertical speed")
    ax_speed.axhline(config.max_climb_speed, color="#2ca02c", linestyle=":", linewidth=1.1)
    ax_speed.axhline(-config.max_descent_speed, color="#2ca02c", linestyle=":", linewidth=1.1)
    ax_speed.set_title("Speed and Vertical Rate")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("m/s")
    ax_speed.grid(True, alpha=0.28)
    ax_speed.legend(loc="best")

    ax_clearance.plot(optimized_profile["segment_times_s"], optimized_profile["segment_min_clearance_m"], color="#9467bd", linewidth=2.1, marker="o", markersize=3, label="min clearance")
    ax_clearance.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2, label="safety boundary")
    ax_clearance.set_title("Obstacle Clearance Margin")
    ax_clearance.set_xlabel("Time (s)")
    ax_clearance.set_ylabel("Clearance beyond envelope (m)")
    ax_clearance.grid(True, alpha=0.28)
    ax_clearance.legend(loc="best")

    summary = (
        f"Length: {initial_metrics['path_length_m']:.1f} → {optimized_metrics['path_length_m']:.1f} m\n"
        f"Smoothness: {initial_metrics['smoothness_sum_sq']:.1f} → {optimized_metrics['smoothness_sum_sq']:.1f}\n"
        f"Max speed: {optimized_metrics['max_speed_mps']:.2f} / {config.max_speed:.0f} m/s\n"
        f"Max turn: {optimized_metrics['max_turn_deg']:.2f}° / {config.max_turn_deg:.0f}°\n"
        f"Min clearance: {optimized_metrics['min_clearance_m']:.3f} m\n"
        f"No-fly violations: {optimized_metrics['no_fly_zone_violations']:.0f}\n"
        f"Max GPS HDOP: {optimized_metrics['max_gps_hdop']:.2f}\n"
        f"Battery used: {optimized_metrics['battery_range_used_pct']:.1f}%"
    )
    ax3d.text2D(0.03, 0.02, summary, transform=ax3d.transAxes, fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"})

    fig.suptitle("UAV 3D Path Planning Dashboard (Practical Constraints)", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def interpolate_path(path: np.ndarray, frames_per_segment: int = 8) -> np.ndarray:
    points = []
    for p1, p2 in zip(path[:-1], path[1:]):
        for alpha in np.linspace(0.0, 1.0, frames_per_segment, endpoint=False):
            points.append((1.0 - alpha) * p1 + alpha * p2)
    points.append(path[-1])
    return np.vstack(points)


def save_trajectory_animation(
    optimized_path: np.ndarray,
    config: PlannerConfig,
    gif_path: Path,
    mp4_path: Path,
) -> dict[str, str]:
    animated_points = interpolate_path(optimized_path, frames_per_segment=8)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obstacle, inflated_radius in zip(config.obstacles, config.final_inflated_radii):
        center = obstacle[:3]
        draw_sphere(ax, center, obstacle[3], color="#d62728", alpha=0.26)
        draw_sphere(ax, center, inflated_radius, color="#ff9896", alpha=0.07)

    for zone in config.no_fly_zones:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 8)
        z_bottom = max(zone.min_z_m, config.xyz_min[2])
        z_top = min(zone.max_z_m, config.xyz_max[2])
        x = zone.center[0] + zone.radius_m * np.outer(np.cos(u), np.sin(v))
        y = zone.center[1] + zone.radius_m * np.outer(np.sin(u), np.sin(v))
        z_bottom_surf = np.full_like(x, z_bottom)
        z_top_surf = np.full_like(x, z_top)
        ax.plot_surface(x, y, z_bottom_surf, color="#8b0000", alpha=0.15, linewidth=0)
        ax.plot_surface(x, y, z_top_surf, color="#8b0000", alpha=0.15, linewidth=0)

    ax.plot(optimized_path[:, 0], optimized_path[:, 1], optimized_path[:, 2], color="#9ecae1", linewidth=1.5, linestyle="--", label="planned path")
    trace_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2.6, label="flown path")
    drone_point, = ax.plot([], [], [], marker="o", color="#ff7f0e", markersize=8, label="UAV")
    ax.scatter(*config.start, color="green", s=55, label="Start")
    ax.scatter(*config.goal, color="blue", s=55, label="Goal")
    status_text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes, fontsize=11)

    ax.set_xlim(config.xyz_min[0], config.xyz_max[0])
    ax.set_ylim(config.xyz_min[1], config.xyz_max[1])
    ax.set_zlim(config.xyz_min[2], config.xyz_max[2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Optimized UAV Flight Animation")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")

    def init():
        trace_line.set_data([], [])
        trace_line.set_3d_properties([])
        drone_point.set_data([], [])
        drone_point.set_3d_properties([])
        status_text.set_text("")
        return trace_line, drone_point, status_text

    def update(frame: int):
        pts = animated_points[: frame + 1]
        current = animated_points[frame]
        trace_line.set_data(pts[:, 0], pts[:, 1])
        trace_line.set_3d_properties(pts[:, 2])
        drone_point.set_data([current[0]], [current[1]])
        drone_point.set_3d_properties([current[2]])
        progress = frame / max(len(animated_points) - 1, 1)
        status_text.set_text(
            f"Progress: {progress * 100:5.1f}%\n"
            f"Position: ({current[0]:.1f}, {current[1]:.1f}, {current[2]:.1f}) m"
        )
        return trace_line, drone_point, status_text

    animation = FuncAnimation(fig, update, init_func=init, frames=len(animated_points), interval=80, blit=False)

    artifacts = {}
    animation.save(gif_path, writer=PillowWriter(fps=12), dpi=120)
    artifacts["animation_gif"] = str(gif_path)

    if shutil.which("ffmpeg"):
        animation.save(mp4_path, writer=FFMpegWriter(fps=12, bitrate=1800), dpi=120)
        artifacts["animation_mp4"] = str(mp4_path)
    else:
        artifacts["animation_mp4_status"] = "ffmpeg_not_found"

    plt.close(fig)
    return artifacts


# ============================================================================
# NEW: Interactive 3D HTML Visualization with Plotly
# ============================================================================


def build_obstacle_mesh(center: np.ndarray, radius: float, n_phi: int = 12, n_theta: int = 18) -> tuple:
    """Build triangulated mesh for a sphere."""
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)
    phi, theta = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    return x.flatten(), y.flatten(), z.flatten()


def build_cylinder_mesh(center: np.ndarray, radius: float, z_min: float, z_max: float, n_angle: int = 24) -> tuple:
    """Build mesh for a vertical cylinder (no-fly zone)."""
    angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    x_bottom = center[0] + radius * np.cos(angles)
    y_bottom = center[1] + radius * np.sin(angles)
    x_top = x_bottom.copy()
    y_top = y_bottom.copy()

    x_faces = []
    y_faces = []
    z_faces = []
    for i in range(n_angle):
        next_i = (i + 1) % n_angle
        # Bottom face
        x_faces.extend([x_bottom[i], x_bottom[next_i], center[0]])
        y_faces.extend([y_bottom[i], y_bottom[next_i], center[1]])
        z_faces.extend([z_min, z_min, z_min])
        # Top face
        x_faces.extend([x_top[i], x_top[next_i], center[0]])
        y_faces.extend([y_top[i], y_top[next_i], center[1]])
        z_faces.extend([z_max, z_max, z_max])

    return np.array(x_faces), np.array(y_faces), np.array(z_faces)


def build_disc_mesh(center: np.ndarray, radius: float, z: float, n_angle: int = 24) -> tuple:
    """Build mesh for a flat disc (GPS coverage zone)."""
    angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    x_ring = center[0] + radius * np.cos(angles)
    y_ring = center[1] + radius * np.sin(angles)
    x_tri = np.concatenate([[center[0]] * n_angle, x_ring, np.roll(x_ring, -1)])
    y_tri = np.concatenate([[center[1]] * n_angle, y_ring, np.roll(y_ring, -1)])
    z_tri = np.full(len(x_tri), z)
    return x_tri, y_tri, z_tri


def generate_plotly_html(
    coarse_path: np.ndarray,
    initial_path: np.ndarray,
    optimized_path: np.ndarray,
    config: PlannerConfig,
    output_path: Path,
) -> None:
    """Generate an interactive 3D HTML visualization using Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("WARNING: plotly not installed. Skipping HTML visualization.")
        print("  Install with: pip install plotly kaleido")
        return

    profile = compute_profile_data(optimized_path, config)

    # --- Build traces ---
    traces = []

    # Obstacles (inflated safety envelopes)
    for idx, (obstacle, inflated_radius) in enumerate(zip(config.obstacles, config.final_inflated_radii)):
        center = obstacle[:3]
        ox, oy, oz = build_obstacle_mesh(center, inflated_radius)
        traces.append(go.Mesh3d(
            x=ox, y=oy, z=oz,
            color="#ff9896", opacity=0.12, name=f"O{idx+1}_safety",
            showscale=False, hoverinfo="skip",
        ))
        # Core obstacle
        cx, cy, cz = build_obstacle_mesh(center, obstacle[3])
        traces.append(go.Mesh3d(
            x=cx, y=cy, z=cz,
            color="#d62728", opacity=0.30, name=f"O{idx+1}",
            showscale=False, hoverinfo="skip",
        ))

    # No-fly zones
    for zone in config.no_fly_zones:
        nx, ny, nz = build_cylinder_mesh(zone.center, zone.radius_m, zone.min_z_m, zone.max_z_m)
        traces.append(go.Mesh3d(
            x=nx, y=ny, z=nz,
            color="#8b0000", opacity=0.12, name=f"NFZ: {zone.label or 'Restricted'}",
            showscale=False,
        ))

    # GPS coverage zones (discs at ground level)
    for zone in config.gps_coverage_zones:
        gx, gy, gz = build_disc_mesh(zone.center, zone.radius_m, config.xyz_min[2])
        traces.append(go.Mesh3d(
            x=gx, y=gy, z=gz,
            color="#006400", opacity=0.15, name=f"GPS gap (HDOP={zone.hdop_factor})",
            showscale=False,
        ))

    # Paths
    traces.append(go.Scatter3d(
        x=coarse_path[:, 0], y=coarse_path[:, 1], z=coarse_path[:, 2],
        mode="lines", line=dict(color="#7f7f7f", width=1.5, dash="dash"),
        name="A* coarse", hoverinfo="name",
    ))
    traces.append(go.Scatter3d(
        x=initial_path[:, 0], y=initial_path[:, 1], z=initial_path[:, 2],
        mode="lines+markers", line=dict(color="#ff7f0e", width=2),
        marker=dict(size=3, color="#ff7f0e"),
        name="B-spline init", hoverinfo="name",
    ))
    traces.append(go.Scatter3d(
        x=optimized_path[:, 0], y=optimized_path[:, 1], z=optimized_path[:, 2],
        mode="lines+markers", line=dict(color="#1f77b4", width=3),
        marker=dict(size=4, color="#1f77b4"),
        name="SQP optimized", hoverinfo="name",
    ))

    # Start and Goal markers
    traces.append(go.Scatter3d(
        x=[config.start[0]], y=[config.start[1]], z=[config.start[2]],
        mode="markers", marker=dict(size=12, color="green", symbol="square"),
        name="Start", hovertemplate="Start<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}",
    ))
    traces.append(go.Scatter3d(
        x=[config.goal[0]], y=[config.goal[1]], z=[config.goal[2]],
        mode="markers", marker=dict(size=12, color="blue", symbol="diamond"),
        name="Goal", hovertemplate="Goal<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}",
    ))

    # Animated drone trace (for slider)
    animated_points = interpolate_path(optimized_path, frames_per_segment=8)
    n_frames = len(animated_points)

    frames_data = []
    for frame_idx in range(0, n_frames, max(1, n_frames // 60)):
        pts = animated_points[:frame_idx + 1]
        frames_data.append(go.Frame(
            data=[
                go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                             mode="lines", line=dict(color="#1f77b4", width=3)),
                go.Scatter3d(x=[animated_points[frame_idx, 0]], y=[animated_points[frame_idx, 1]], z=[animated_points[frame_idx, 2]],
                             mode="markers", marker=dict(size=10, color="#ff7f0e", symbol="circle")),
            ],
            name=str(frame_idx),
        ))

    # Build figure
    fig = go.Figure(data=traces, frames=frames_data)

    # Sliders and buttons for animation control
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "prefix": "Frame: ",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 50},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.05,
        "y": 0,
        "steps": [
            {
                "args": [[str(i)], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                "label": str(i),
                "method": "animate",
            }
            for i in range(0, n_frames, max(1, n_frames // 60))
        ],
    }

    fig.update_layout(
        title=dict(
            text="UAV 3D Path Planning — Interactive Visualization<br><sup>Drag to rotate · Scroll to zoom · Click + drag to pan</sup>",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=[config.xyz_min[0], config.xyz_max[0]], showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="Y (m)", range=[config.xyz_min[1], config.xyz_max[1]], showgrid=True, gridcolor="#eee"),
            zaxis=dict(title="Z (m)", range=[config.xyz_min[2], config.xyz_max[2]], showgrid=True, gridcolor="#eee"),
            camera=dict(eye=dict(x=1.2, y=-1.5, z=0.8)),
            aspectmode="cube",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.02,
                x=0.15,
                xanchor="right",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 80, "redraw": True}, "fromsource": True,
                                    "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[None, {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[sliders_dict],
        legend=dict(x=0.85, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="white",
    )

    # Profile charts as subplots
    metrics = compute_metrics(optimized_path, config)
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Altitude Profile", "Speed Profile", "Clearance Profile", "GPS HDOP Profile"),
        horizontal_spacing=0.12, vertical_spacing=0.15,
    )
    dist = profile["cumulative_distance_m"]
    fig2.add_trace(go.Scatter(x=dist, y=optimized_path[:, 2], mode="lines+markers",
                             marker=dict(size=4), line=dict(color="#1f77b4"),
                             name="Altitude"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=profile["segment_times_s"], y=profile["speeds_mps"], mode="lines+markers",
                             marker=dict(size=4), line=dict(color="#1f77b4"),
                             name="Speed"), row=1, col=2)
    fig2.add_hline(y=config.max_speed, line_dash="dash", line_color="#d62728", row=1, col=2,
                   annotation_text=f"Speed limit {config.max_speed} m/s")
    fig2.add_trace(go.Scatter(x=profile["segment_times_s"], y=profile["segment_min_clearance_m"],
                             mode="lines+markers", marker=dict(size=4), line=dict(color="#9467bd"),
                             name="Clearance"), row=2, col=1)
    fig2.add_hline(y=0, line_dash="dash", line_color="#d62728", row=2, col=1)
    fig2.add_trace(go.Scatter(x=profile["segment_times_s"], y=profile["gps_hdop_profile"],
                             mode="lines+markers", marker=dict(size=4), line=dict(color="#2ca02c"),
                             name="GPS HDOP"), row=2, col=2)
    fig2.add_hline(y=1.0, line_dash="dot", line_color="#888", row=2, col=2)

    summary_text = (
        f"<b>Path Length:</b> {metrics['path_length_m']:.1f} m &nbsp; "
        f"<b>Flight Time:</b> {metrics['flight_time_s']:.1f} s &nbsp; "
        f"<b>Battery Used:</b> {metrics['battery_range_used_pct']:.1f}%<br>"
        f"<b>Max Speed:</b> {metrics['max_speed_mps']:.2f}/{config.max_speed:.0f} m/s &nbsp; "
        f"<b>Max Climb:</b> {metrics['max_climb_mps']:.2f} m/s &nbsp; "
        f"<b>Max Turn:</b> {metrics['max_turn_deg']:.1f}°<br>"
        f"<b>Min Clearance:</b> {metrics['min_clearance_m']:.3f} m &nbsp; "
        f"<b>No-Fly Violations:</b> {metrics['no_fly_zone_violations']:.0f} &nbsp; "
        f"<b>Max GPS HDOP:</b> {metrics['max_gps_hdop']:.2f}"
    )
    fig2.update_layout(
        title=dict(text=f"Flight Profile Metrics<br><sup>{summary_text}</sup>", x=0.5, font=dict(size=11)),
        height=500, showlegend=False,
        margin=dict(l=60, r=60, t=120, b=40),
    )
    fig2.update_xaxes(title_text="Distance (m)", row=1, col=1)
    fig2.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig2.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig2.update_yaxes(title_text="Speed (m/s)", row=1, col=2)
    fig2.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig2.update_yaxes(title_text="Clearance (m)", row=2, col=1)
    fig2.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig2.update_yaxes(title_text="HDOP factor", row=2, col=2)

    # Save
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    dashboard_html_path = output_path.with_name(output_path.stem + "_profiles" + output_path.suffix)
    fig2.write_html(dashboard_html_path, include_plotlyjs="cdn", full_html=True)
    print(f"  Interactive HTML (3D): {output_path}")
    print(f"  Interactive HTML (profiles): {dashboard_html_path}")


def save_waypoints(path: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x_m", "y_m", "z_m"])
        for idx, point in enumerate(path):
            writer.writerow([idx, *[f"{value:.6f}" for value in point]])


def serialize_array(array: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in array]


def run_planner(config: PlannerConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    coarse_path = astar_3d(config)
    downsampled = resample_by_arclength(coarse_path, config.n_segments + 1)
    initial_path = bspline_resample(downsampled, config.n_segments + 1)
    initial_path[0] = config.start
    initial_path[-1] = config.goal
    optimized_path, stages = optimize_path(initial_path, config)

    metrics = {
        "coarse_astar": compute_metrics(resample_by_arclength(coarse_path, config.n_segments + 1), config),
        "initial_bspline": compute_metrics(initial_path, config),
        "optimized_sqp": compute_metrics(optimized_path, config),
    }

    figure_path = output_dir / "three_dimensional_path_planning.png"
    dashboard_path = output_dir / "three_dimensional_dashboard.png"
    animation_gif_path = output_dir / "three_dimensional_trajectory.gif"
    animation_mp4_path = output_dir / "three_dimensional_trajectory.mp4"
    waypoint_path = output_dir / "three_dimensional_waypoints.csv"
    report_path = output_dir / "three_dimensional_metrics.json"
    html_3d_path = output_dir / "three_dimensional_3dvisualization.html"

    save_visualization(coarse_path, initial_path, optimized_path, config, figure_path)
    save_summary_dashboard(coarse_path, initial_path, optimized_path, config, dashboard_path)
    animation_artifacts = save_trajectory_animation(
        optimized_path, config, animation_gif_path, animation_mp4_path,
    )
    generate_plotly_html(coarse_path, initial_path, optimized_path, config, html_3d_path)
    save_waypoints(optimized_path, waypoint_path)

    # Practical constraints summary
    practical_constraints = {
        "no_fly_zones": [
            {"center": z.center.tolist(), "radius_m": z.radius_m,
             "min_z_m": z.min_z_m, "max_z_m": z.max_z_m, "label": z.label}
            for z in config.no_fly_zones
        ],
        "gps_coverage_zones": [
            {"center": z.center.tolist(), "radius_m": z.radius_m, "hdop_factor": z.hdop_factor}
            for z in config.gps_coverage_zones
        ],
        "wind_fields": [
            {"direction_deg": wf.direction_hdg_deg, "speed_mps": wf.speed_mps,
             "min_z_m": wf.min_z_m, "max_z_m": wf.max_z_m, "gust_factor": wf.gust_factor}
            for wf in config.wind_fields
        ],
        "battery": {
            "capacity_wh": config.battery.capacity_wh,
            "range_m": config.battery_range_m(),
            "max_discharge_pct": config.battery.max_discharge_pct,
        },
    }

    payload: dict[str, object] = {
        "config": {
            "start": config.start.tolist(),
            "goal": config.goal.tolist(),
            "obstacles": config.obstacles.tolist(),
            "n_segments": config.n_segments,
            "drone_radius_m": config.drone_radius,
            "safety_margin_m": config.safety_margin,
            "dt_s": config.dt_s,
            "total_time_s": config.total_time_s,
        },
        "practical_constraints": practical_constraints,
        "optimization_stages": stages,
        "metrics": metrics,
        "optimized_waypoints": serialize_array(optimized_path),
        "artifacts": {
            "figure": str(figure_path),
            "dashboard": str(dashboard_path),
            "waypoints_csv": str(waypoint_path),
            "html_3d": str(html_3d_path),
            **animation_artifacts,
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced A* + B-spline + SQP 三维无人机路径规划（含实际约束和交互可视化）"
    )
    parser.add_argument("--segments", type=int, default=20, help="路径分段数，默认 20")
    parser.add_argument("--safety-margin", type=float, default=5.0, help="额外安全距离（米），默认 5")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "results" / "three_dimensional", help="输出目录")
    return parser.parse_args()


def print_summary(payload: dict[str, object]) -> None:
    metrics = payload["metrics"]
    stages = payload["optimization_stages"]
    artifacts = payload["artifacts"]
    optimized = metrics["optimized_sqp"]
    initial = metrics["initial_bspline"]

    print("=" * 72)
    print("Enhanced A* + B-spline + SQP 三维路径规划完成")
    print("=" * 72)
    print(f"初始 B 样条路径长度: {initial['path_length_m']:.3f} m")
    print(f"优化后路径长度:     {optimized['path_length_m']:.3f} m")
    print(f"最小安全净距:       {optimized['min_clearance_m']:.3f} m")
    print(f"最大速度:           {optimized['max_speed_mps']:.3f} m/s")
    print(f"最大爬升速度:       {optimized['max_climb_mps']:.3f} m/s")
    print(f"最大下降速度:       {optimized['max_descent_mps']:.3f} m/s")
    print(f"最大转弯角:         {optimized['max_turn_deg']:.3f}°")
    print("-" * 72)
    print("Practical Boundary Conditions:")
    print(f"  No-fly zone violations: {optimized['no_fly_zone_violations']:.0f}")
    print(f"  Max GPS HDOP:           {optimized['max_gps_hdop']:.2f}")
    print(f"  Max wind speed:         {optimized['max_wind_speed_mps']:.2f} m/s")
    print(f"  Battery used:          {optimized['battery_range_used_pct']:.1f}%")
    print(f"  Terrain clearance:      {optimized['terrain_clearance_min_m']:.2f} m")
    print("-" * 72)
    for stage in stages:
        print(
            f"同伦阶段 margin={stage['margin_m']:.1f} m | "
            f"success={stage['success']} | "
            f"min_constraint={stage['min_constraint']:.6f}"
        )
    print("-" * 72)
    print(f"图像输出:      {artifacts['figure']}")
    print(f"综合图输出:    {artifacts['dashboard']}")
    print(f"动画 GIF 输出: {artifacts['animation_gif']}")
    if "animation_mp4" in artifacts:
        print(f"动画 MP4 输出: {artifacts['animation_mp4']}")
    elif artifacts.get("animation_mp4_status") == "ffmpeg_not_found":
        print("动画 MP4 未生成: 当前环境未检测到 ffmpeg，已改为输出 GIF。")
    print(f"交互HTML输出: {artifacts['html_3d']}")
    print(f"航路点输出:    {artifacts['waypoints_csv']}")


def main() -> None:
    args = parse_args()
    if args.segments < 4:
        raise ValueError("segments 至少需要为 4，才能形成有意义的平滑路径。")
    config = make_default_config(n_segments=args.segments, safety_margin=args.safety_margin)
    payload = run_planner(config, args.output_dir)
    print_summary(payload)


if __name__ == "__main__":
    main()