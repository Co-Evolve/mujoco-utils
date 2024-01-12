import abc
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, SupportsFloat

import gymnasium
import mujoco
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame

from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.base import BaseMuJoCoEnvironment, BaseObservable, BaseWithArenaAndMorphology, \
    MuJoCoEnvironmentConfiguration
from mujoco_utils.morphology import MJCFMorphology


class MJCObservable(BaseObservable):
    def __init__(
            self,
            name: str,
            low: np.ndarray,
            high: np.ndarray,
            retriever: Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray]
            ) -> None:
        super().__init__(name=name, low=low, high=high, retriever=retriever)

    def __call__(
            self,
            mj_model: mujoco.MjModel,
            mj_data: mujoco.MjData
            ) -> np.ndarray:
        return super().__call__(model=mj_model, data=mj_data)


class BaseMJCEnv(BaseMuJoCoEnvironment, gymnasium.Env, ABC):
    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        BaseMuJoCoEnvironment.__init__(
                self=self, mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
                )
        self._observables: Optional[List[MJCObservable]] = self._get_observables()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def _get_action_space(
            self
            ) -> gymnasium.Space:
        bounds = self.mj_model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    @property
    def observables(
            self
            ) -> List[MJCObservable]:
        return self._observables

    @abc.abstractmethod
    def _get_observables(
            self
            ) -> List[MJCObservable]:
        raise NotImplementedError

    def _get_observation_space(
            self
            ) -> gymnasium.spaces.Dict:
        observation_space = dict()
        for observable in self.observables:
            observation_space[observable.name] = gymnasium.spaces.Box(
                    low=observable.low, high=observable.high, shape=observable.shape
                    )
        return gymnasium.spaces.Dict(observation_space)

    def _get_observations(
            self
            ) -> Dict[str, np.ndarray]:
        observations = dict()
        for observable in self.observables:
            observations[observable.name] = observable(
                    mj_model=self.mj_model, mj_data=self.mj_data
                    )
        return observations

    def _take_n_steps(
            self,
            ctrl: ActType
            ) -> None:
        self.mj_data.ctrl[:] = ctrl
        mujoco.mj_step(
                m=self.mj_model, d=self.mj_data, nstep=self.environment_configuration.num_physics_steps_per_control_step
                )
        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)

    def render(
            self
            ) -> RenderFrame | list[RenderFrame] | None:
        # model updates have happened in self.mj_model, which is used within the renderers
        #   we thus don't need to explicitly update the renderer's model here
        camera_ids = self.environment_configuration.camera_ids or [-1]
        if self.environment_configuration.render_mode == "human":
            return self.get_renderer().render(
                    render_mode="human", camera_id=camera_ids[0]
                    )
        else:
            frames = []
            for camera_id in camera_ids:
                self.get_renderer().update_scene(data=self.mj_data, camera=camera_id)
                frame = self.get_renderer().render()[:, :, ::-1]
                frames.append(frame)
            return frames[0] if len(camera_ids) == 1 else frames

    def close(
            self
            ) -> None:
        super().close()
        del self._observables

    @abc.abstractmethod
    def step(
            self,
            action: ActType
            ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
            ) -> tuple[ObsType, dict[str, Any]]:
        raise NotImplementedError


class MJCEnv(BaseWithArenaAndMorphology, BaseMJCEnv, ABC):
    def __init__(
            self,
            morphology: MJCFMorphology,
            arena: MJCFArena,
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        BaseWithArenaAndMorphology.__init__(self=self, morphology=morphology, arena=arena)
        BaseMJCEnv.__init__(
                self=self, mjcf_str=self._mjcf_str, mjcf_assets=self._mjcf_assets, configuration=configuration
                )
