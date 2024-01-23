from __future__ import annotations

import abc
import dataclasses
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

import gymnasium
import mujoco
import numpy as np
from gymnasium.core import ActType

from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.base import BaseEnvState, BaseMuJoCoEnvironment, BaseObservable, \
    MuJoCoEnvironmentConfiguration
from mujoco_utils.morphology import MJCFMorphology


@dataclasses.dataclass
class MJCEnvState(BaseEnvState):
    model: mujoco.MjModel
    data: mujoco.MjData
    observations: Dict[str, np.ndarray]
    rng: int


class MJCObservable(BaseObservable):
    def __init__(
            self,
            name: str,
            low: np.ndarray,
            high: np.ndarray,
            retriever: Callable[[mujoco.MjModel, mujoco.MjData, Any, Any], np.ndarray]
            ) -> None:
        super().__init__(name=name, low=low, high=high, retriever=retriever)

    def __call__(
            self,
            mj_model: mujoco.MjModel,
            mj_data: mujoco.MjData,
            *args,
            **kwargs
            ) -> np.ndarray:
        return super().__call__(model=mj_model, data=mj_data, *args, **kwargs)


class MJCEnv(BaseMuJoCoEnvironment, ABC):
    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        super().__init__(self, mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration)

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFMorphology,
            arena: MJCFArena,
            configuration: MuJoCoEnvironmentConfiguration
            ) -> MJCEnv:
        return super().from_morphology_and_arena(morphology=morphology, arena=arena, configuration=configuration)

    def _get_mj_models_and_datas_to_render(
            self,
            state: MJCEnvState
            ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        return [state.model], [state.data]

    @property
    def observables(
            self
            ) -> List[MJCObservable]:
        return self._observables

    def _create_observation_space(
            self
            ) -> gymnasium.spaces.Dict:
        observation_space = dict()
        for observable in self.observables:
            observation_space[observable.name] = gymnasium.spaces.Box(
                    low=observable.low, high=observable.high, shape=observable.shape
                    )
        return gymnasium.spaces.Dict(observation_space)

    def _create_action_space(
            self
            ) -> gymnasium.Space:
        bounds = self.mj_model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    def _get_observations(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            *args,
            **kwargs
            ) -> Dict[str, np.ndarray]:
        observations = dict()
        for observable in self.observables:
            observations[observable.name] = observable(
                    mj_model=model, mj_data=data, *args, **kwargs
                    )
        return observations

    def _forward_simulation(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            ctrl: ActType
            ) -> mujoco.MjData:
        data.ctrl[:] = ctrl
        mujoco.mj_step(
                m=model, d=data, nstep=self.environment_configuration.num_physics_steps_per_control_step
                )
        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(model, data)

        return data

    @abc.abstractmethod
    def _create_observables(
            self
            ) -> List[MJCObservable]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
            self,
            state: MJCEnvState,
            action: np.ndarray
            ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
            self,
            rng: int
            ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_reward(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            *args,
            **kwargs
            ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _should_terminate(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            *args,
            **kwargs
            ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _should_truncate(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            *args,
            **kwargs
            ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_info(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            *args,
            **kwargs
            ) -> Dict[str, Any]:
        raise NotImplementedError
