from __future__ import annotations

from typing import Any, Dict, List

import chex
import numpy as np
from gymnasium.core import RenderFrame

from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.base import BaseEnvState, BaseEnvironment, MuJoCoEnvironmentConfiguration, SpaceType
from mujoco_utils.environment.mjc_env import MJCEnv
from mujoco_utils.environment.mjx_env import MJXEnv
from mujoco_utils.morphology import MJCFMorphology


class DualMuJoCoEnvironment(BaseEnvironment):
    MJC_ENV_CLASS: MJCEnv
    MJX_ENV_CLASS: MJXEnv

    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration,
            backend: str, ) -> None:
        assert backend in ["MJC", "MJX"], f"Backend must either be 'MJC' or 'MJX'. {backend} was given."

        super().__init__(configuration=configuration)
        if backend == "MJC":
            env_class = self.MJC_ENV_CLASS
        else:
            env_class = self.MJX_ENV_CLASS

        self._env = env_class.__init__(
                mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
                )

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFMorphology,
            arena: MJCFArena,
            configuration: MuJoCoEnvironmentConfiguration,
            backend: str, ) -> DualMuJoCoEnvironment:
        arena.attach(other=morphology, free_joint=True)
        mjcf_str, mjcf_assets = arena.get_mjcf_str(), arena.get_mjcf_assets()
        return cls(
                mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration, backend=backend
                )

    @property
    def action_space(
            self
            ) -> SpaceType:
        return self._env.action_space

    @property
    def actuators(
            self
            ) -> List[str]:
        return self._env.actuators

    @property
    def observation_space(
            self
            ) -> SpaceType:
        return self._env.observation_space

    def step(
            self,
            state: BaseEnvState,
            action: chex.Array
            ) -> BaseEnvState:
        return self._env.step(state=state, action=action)

    def reset(
            self,
            rng: np.random.RandomState | chex.PRNGKey
            ) -> BaseEnvState:
        return self._env.reset(rng=rng)

    def render(
            self,
            state: BaseEnvState
            ) -> List[RenderFrame] | None:
        return self._env.render(state=state)

    def close(
            self
            ) -> None:
        return self._env.close()
