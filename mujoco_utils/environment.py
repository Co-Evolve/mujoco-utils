from __future__ import annotations

import abc
from typing import Callable, List

import dm_control.composer
import gymnasium as gym
import numpy as np
from dm_control import composer
from fprs import frps_random_state

from mujoco_utils.gym_wrapper import DMC2GymWrapper
from mujoco_utils.robot import MJCMorphology


def default_make_mjc_env(
        config: MJCEnvironmentConfig,
        morphology: MJCMorphology
        ) -> composer.Environment:
    task = config.task(config, morphology)
    task.set_timesteps(
            control_timestep=config.control_timestep, physics_timestep=config.physics_timestep
            )
    env = composer.Environment(
            task=task, random_state=frps_random_state, time_limit=config.simulation_time
            )
    return env


def dm_control_to_gym_environment(
        config: MJCEnvironmentConfig,
        environment: composer.Environment
        ) -> gym.Env:
    env = DMC2GymWrapper(
            env=environment, camera_ids=config.camera_ids
            )
    return env


class MJCEnvironmentConfig:
    def __init__(
            self,
            task: Callable[[MJCEnvironmentConfig, MJCMorphology], composer.Task],
            time_scale: float = 1.0,
            control_substeps: int = 1,
            simulation_time: float = 10,
            camera_ids: List[int] | None = None
            ) -> None:
        self.task = task
        self.time_scale = time_scale
        self.control_substeps = control_substeps
        self.simulation_time = simulation_time
        self.camera_ids = camera_ids or [0]

    def environment(
            self,
            morphology: MJCMorphology,
            wrap2gym: bool = True
            ) -> gym.Env | dm_control.composer.Environment:
        env = default_make_mjc_env(config=self, morphology=morphology)
        if wrap2gym:
            env = dm_control_to_gym_environment(config=self, environment=env)
            return env

    @property
    def original_physics_timestep(
            self
            ) -> float:
        return 0.002

    @property
    def control_timestep(
            self
            ) -> float:
        return self.control_substeps * self.physics_timestep

    @property
    def physics_timestep(
            self
            ) -> float:
        return self.original_physics_timestep * self.time_scale

    @property
    def total_num_timesteps(
            self
            ) -> int:
        return int(np.ceil(self.simulation_time / self.control_timestep))
