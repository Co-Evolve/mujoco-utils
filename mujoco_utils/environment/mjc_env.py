from __future__ import annotations

import abc
import copy
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

import gymnasium
import mujoco
import numpy as np
from flax import struct
from gymnasium.core import ActType

from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.base import BaseEnvState, BaseMuJoCoEnvironment, BaseObservable, \
    MuJoCoEnvironmentConfiguration
from mujoco_utils.morphology import MJCFMorphology


@struct.dataclass
class MJCEnvState(BaseEnvState):
    observations: Dict[str, np.ndarray]
    rng: np.random.RandomState


class MJCObservable(BaseObservable):
    def __init__(
            self,
            name: str,
            low: np.ndarray,
            high: np.ndarray,
            retriever: Callable[[MJCEnvState], np.ndarray]
            ) -> None:
        super().__init__(name=name, low=low, high=high, retriever=retriever)

    def __call__(
            self,
            state: MJCEnvState
            ) -> np.ndarray:
        return super().__call__(state=state)


class MJCEnv(BaseMuJoCoEnvironment, ABC):
    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        super().__init__(mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration)

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
        return [state.mj_model], [state.mj_data]

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
        bounds = self.frozen_mj_model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    def _update_observations(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        observations = dict()
        for observable in self.observables:
            observations[observable.name] = observable(
                    state=state
                    )
        # noinspection PyUnresolvedReferences
        return state.replace(observations=observations)

    def _update_simulation(
            self,
            state: MJCEnvState,
            ctrl: ActType
            ) -> MJCEnvState:
        mj_data = copy.deepcopy(state.mj_data)
        mujoco.mj_step(
                m=state.mj_model, d=mj_data, nstep=self.environment_configuration.num_physics_steps_per_control_step
                )
        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(state.mj_model, mj_data)

        # noinspection PyUnresolvedReferences
        return state.replace(mj_data=mj_data)

    def step(
            self,
            state: MJCEnvState,
            action: np.ndarray
            ) -> MJCEnvState:
        return super().step(state=state, action=action)

    def _prepare_reset(
            self
            ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        model = copy.deepcopy(self.frozen_mj_model)
        data = copy.deepcopy(self.frozen_mj_data)
        return model, data

    def _finish_reset(
            self,
            models_and_datas: Tuple[mujoco.MjModel, mujoco.MjData],
            rng: np.random.RandomState
            ) -> MJCEnvState:
        mj_model, mj_data = models_and_datas
        mujoco.mj_forward(m=mj_model, d=mj_data)
        state = MJCEnvState(
                mj_model=mj_model,
                mj_data=mj_data,
                observations={},
                reward=0,
                terminated=False,
                truncated=False,
                info={},
                rng=rng
                )
        state = self._update_observations(state=state)
        state = self._update_info(state=state)
        return state

    @abc.abstractmethod
    def _create_observables(
            self
            ) -> List[MJCObservable]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
            self,
            rng: np.random.RandomState
            ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_reward(
            self,
            state: MJCEnvState,
            previous_state: MJCEnvState
            ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_terminated(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_truncated(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_info(
            self,
            state: MJCEnvState
            ) -> MJCEnvState:
        raise NotImplementedError
