# -*- coding: utf-8 -*-

import numpy as np

from GlobeCanvas import GlobeCanvas


RE = 6378.145
MU = 398600.4418

EARTH_TEXTURE_FILE = "bluemarble-2048.png"
CLOUD_TEXTURE_FILE = "clouds_2048.png"

NUM_PLANES = 6
SATS_PER_PLANE = 20

CONSTELLATION_ALTITUDE = 550.0
CONSTELLATION_INCLINATION = 53.0

ANIMATION_INTERVAL_MS = 30
INITIAL_TIME_SCALE = 60.0
MIN_TIME_SCALE = 0.1
MAX_TIME_SCALE = 5000.0


def rotation_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def circular_orbit(altitude_km, raan_deg, inclination_deg, n_points=1000):
    radius = RE + altitude_km
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    orbit = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))
    rotation = rotation_z(np.deg2rad(raan_deg)) @ rotation_x(np.deg2rad(inclination_deg))
    return orbit @ rotation.T


def elliptical_orbit(perigee_altitude_km, apogee_altitude_km, inclination_deg, raan_deg, n_points=1200):
    rp = RE + perigee_altitude_km
    ra = RE + apogee_altitude_km
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    radius = a * (1.0 - e**2) / (1.0 + e * np.cos(theta))
    orbit = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))
    rotation = rotation_z(np.deg2rad(raan_deg)) @ rotation_x(np.deg2rad(inclination_deg))
    return orbit @ rotation.T


def create_constellation(num_planes=NUM_PLANES, sats_per_plane=SATS_PER_PLANE, altitude_km=CONSTELLATION_ALTITUDE, inclination_deg=CONSTELLATION_INCLINATION):
    trajectories = []
    radius = RE + altitude_km

    for plane in range(num_planes):
        raan_deg = 360.0 * plane / num_planes
        phase_offset = 360.0 * plane / (num_planes * sats_per_plane)
        theta = np.linspace(0.0, 2.0 * np.pi, 400)
        theta_shifted = theta + np.deg2rad(phase_offset)
        orbit = np.column_stack((radius * np.cos(theta_shifted), radius * np.sin(theta_shifted), np.zeros_like(theta_shifted)))
        rotation = rotation_z(np.deg2rad(raan_deg)) @ rotation_x(np.deg2rad(inclination_deg))
        trajectories.append(orbit @ rotation.T)

    return trajectories


def circular_position(t, altitude_km, raan_deg, inclination_deg, phase_deg=0.0):
    radius = RE + altitude_km
    mean_motion = np.sqrt(MU / radius**3)
    phase = np.deg2rad(phase_deg) + mean_motion * t
    position_orbital = np.array([radius * np.cos(phase), radius * np.sin(phase), 0.0])
    rotation = rotation_z(np.deg2rad(raan_deg)) @ rotation_x(np.deg2rad(inclination_deg))
    return rotation @ position_orbital


def constellation_positions(t, num_planes=NUM_PLANES, sats_per_plane=SATS_PER_PLANE, altitude_km=CONSTELLATION_ALTITUDE, inclination_deg=CONSTELLATION_INCLINATION):
    positions = []

    for plane in range(num_planes):
        raan_deg = 360.0 * plane / num_planes

        for satellite in range(sats_per_plane):
            phase_deg = 360.0 * satellite / sats_per_plane
            phase_deg += 360.0 * plane / (num_planes * sats_per_plane)
            positions.append(circular_position(t, altitude_km, raan_deg, inclination_deg, phase_deg))

    return np.asarray(positions, dtype=float)


def create_demo_trajectories():
    sat1 = circular_orbit(500.0, 0.0, 0.0, 1000)
    sat2 = circular_orbit(800.0, 45.0, 30.0, 1000)
    sat3 = circular_orbit(1000.0, 90.0, 60.0, 1000)
    sat4 = elliptical_orbit(500.0, 12000.0, 55.0, 120.0, 1200)
    return [sat1, sat2, sat3, sat4]


def create_constellation_trajectories():
    return create_constellation(NUM_PLANES, SATS_PER_PLANE, CONSTELLATION_ALTITUDE, CONSTELLATION_INCLINATION)


def main():
    globe = GlobeCanvas(title="Earth Satellite Constellation — PyVistaQt", earth_radius=RE, earth_texture=EARTH_TEXTURE_FILE, cloud_texture=CLOUD_TEXTURE_FILE, window_size=(1500, 950), show_clouds=True, show_atmosphere=True, show_equator=True, show_controls=True, show_sim_time=True, show_asset_logo=True)

    demo_trajectories = create_demo_trajectories()

    globe.plot_trajectory(demo_trajectories[0], color="red", width=3.0, label="Satellite 1 — 500 km")
    globe.plot_trajectory(demo_trajectories[1], color="lime", width=3.0, label="Satellite 2 — 800 km / 45°")
    globe.plot_trajectory(demo_trajectories[2], color="deepskyblue", width=3.0, label="Satellite 3 — 1000 km / 90°")
    globe.plot_trajectory(demo_trajectories[3], color="gold", width=3.0, label="Satellite 4 — 500×12000 km / 55°")

    constellation_trajectories = create_constellation_trajectories()
    globe.plot_batched_trajectories(constellation_trajectories, color="white", width=1.5, label="120-Satellite Constellation")

    initial_positions = constellation_positions(0.0, NUM_PLANES, SATS_PER_PLANE, CONSTELLATION_ALTITUDE, CONSTELLATION_INCLINATION)
    globe.plot_points(initial_positions, color="white", size=8.0, label="Constellation Satellites")

    def update_satellites(t):
        positions = constellation_positions(t, NUM_PLANES, SATS_PER_PLANE, CONSTELLATION_ALTITUDE, CONSTELLATION_INCLINATION)
        globe.update_points(positions)

    def reset_satellites():
        globe.update_points(initial_positions)

    globe.add_animation(update_function=update_satellites, interval=ANIMATION_INTERVAL_MS, time_scale=INITIAL_TIME_SCALE, min_time_scale=MIN_TIME_SCALE, max_time_scale=MAX_TIME_SCALE, reset_function=reset_satellites, start=True)

    globe.add_legend()
    globe.frame(padding=1.25)
    globe.show()


if __name__ == "__main__":
    main()