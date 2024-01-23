from __future__ import annotations

import abc
import copy
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.base import Base
from flax import struct
from mujoco import mjx

import mujoco_utils.environment.mjx_spaces as mjx_spaces
from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.base import BaseEnvState, BaseMuJoCoEnvironment, BaseObservable, \
    MuJoCoEnvironmentConfiguration
from mujoco_utils.morphology import MJCFMorphology


def mjx_get_model(
        mj_model: mujoco.MjModel,
        mjx_model: mjx.Model,
        n_mj_models: int = 1
        ) -> List[mujoco.MjModel]:
    """
    Transfer mjx.Model to mujoco.MjModel
    """
    mj_models = [copy.deepcopy(mj_model) for _ in range(n_mj_models)]

    offloaded_mjx_model = jax.device_get(mjx_model)
    for key, v in vars(offloaded_mjx_model).items():
        try:
            for i, model in enumerate(mj_models):
                previous_value = getattr(model, key)
                if isinstance(previous_value, np.ndarray):
                    if previous_value.shape != v.shape:
                        actual_value = v[i]
                    else:
                        actual_value = v
                    previous_value[:] = actual_value
                else:
                    setattr(model, key, v)
        except AttributeError:
            pass
        except ValueError:
            pass
    return mj_models


@struct.dataclass
class MJXEnvState(BaseEnvState, Base):
    model: mjx.Model
    data: mjx.Data
    observations: Dict[str, jax.Array]
    rng: chex.PRNGKey


class MJXObservable(BaseObservable):
    def __init__(
            self,
            name: str,
            low: jnp.ndarray,
            high: jnp.ndarray,
            retriever: Callable[[mjx.Model, mjx.Data, Any, Any], jnp.ndarray]
            ) -> None:
        super().__init__(name=name, low=low, high=high, retriever=retriever)

    def __call__(
            self,
            model: mjx.Model,
            data: mjx.Data,
            *args,
            **kwargs
            ) -> jnp.ndarray:
        return super().__call__(
                model=model, data=data, *args, **kwargs
                )


class MJXEnv(BaseMuJoCoEnvironment, ABC):
    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        super().__init__(mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration)
        self._mjx_model, self._mjx_data = self._initialize_mjx_model_and_data()

    @classmethod
    def from_morphology_and_arena(
            cls,
            morphology: MJCFMorphology,
            arena: MJCFArena,
            configuration: MuJoCoEnvironmentConfiguration
            ) -> MJXEnv:
        return super().from_morphology_and_arena(morphology=morphology, arena=arena, configuration=configuration)

    @property
    def mjx_model(
            self
            ) -> mjx.Model:
        return self._mjx_model

    @property
    def mjx_data(
            self
            ) -> mjx.Data:
        return self._mjx_data

    def _initialize_mjx_model_and_data(
            self
            ) -> Tuple[mjx.Model, mjx.Data]:
        return mjx.put_model(m=self.mj_model), mjx.put_data(m=self.mj_model, d=self.mj_data)

    @staticmethod
    def _get_batch_size(
            state: MJXEnvState
            ) -> int:
        try:
            return state.reward.shape[0]
        except IndexError:
            return 1

    def _get_mj_models_and_datas_to_render(
            self,
            state: MJXEnvState
            ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        num_models = 1 if self.environment_configuration.render_mode == "human" else self._get_batch_size(state=state)
        mj_models = mjx_get_model(mj_model=self.mj_model, mjx_model=state.model, n_mj_models=num_models)
        mj_datas = mjx.get_data(m=mj_models[0], d=state.data)
        if not isinstance(mj_datas, list):
            mj_datas = [mj_datas]
        return mj_models, mj_datas

    def _initialize_mjx_data(
            self,
            model: mjx.Model,
            data: mjx.Data,
            qpos: jnp.ndarray,
            qvel: jnp.ndarray
            ) -> mjx.Data:
        mjx_data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        mjx_data = mjx.forward(m=model, d=mjx_data)
        return mjx_data

    @property
    def observables(
            self
            ) -> List[MJXObservable]:
        return self._observables

    def _create_observation_space(
            self
            ) -> mjx_spaces.Dict:
        observation_space = dict()
        for observable in self.observables:
            observation_space[observable.name] = mjx_spaces.Box(
                    low=observable.low, high=observable.high, shape=observable.shape
                    )
        return mjx_spaces.Dict(observation_space)

    def _create_action_space(
            self
            ) -> mjx_spaces.Box:
        bounds = jnp.array(self.mj_model.actuator_ctrlrange.copy().astype(np.float32))
        low, high = bounds.T
        action_space = mjx_spaces.Box(low=low, high=high, shape=low.shape, dtype=jnp.float32)
        return action_space

    def _get_observations(
            self,
            model: mjx.Model,
            data: mjx.Data,
            *args,
            **kwargs
            ) -> Dict[str, jnp.ndarray]:
        observations = jax.tree_util.tree_map(
                lambda
                    observable: (observable.name, observable(
                        model=model, data=data, *args, **kwargs
                        )), self.observables
                )
        return dict(observations)

    def _forward_simulation(
            self,
            model: mjx.Model,
            data: mjx.Data,
            ctrl: jnp.ndarray
            ) -> mjx.Data:
        def _simulation_step(
                _data,
                _
                ) -> Tuple[mjx.Data, None]:
            _data = _data.replace(ctrl=ctrl)
            return mjx.step(model, _data), None

        mjx_data, _ = jax.lax.scan(
                _simulation_step, data, (), self.environment_configuration.num_physics_steps_per_control_step
                )
        return mjx_data

    def close(
            self
            ) -> None:
        super().close()
        del self._mjx_model
        del self._mjx_data

    @abc.abstractmethod
    def _create_observables(
            self
            ) -> List[MJXObservable]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
            self,
            state: MJXEnvState,
            action: jnp.ndarray
            ) -> MJXEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
            self,
            rng: jnp.ndarray
            ) -> MJXEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_reward(
            self,
            model: mjx.Model,
            data: mjx.Data,
            *args,
            **kwargs
            ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _should_terminate(
            self,
            model: mjx.Model,
            data: mjx.Data,
            *args,
            **kwargs
            ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _should_truncate(
            self,
            model: mjx.Model,
            data: mjx.Data,
            *args,
            **kwargs
            ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_info(
            self,
            model: mjx.Model,
            data: mjx.Data,
            *args,
            **kwargs
            ) -> Dict[str, Any]:
        raise NotImplementedError
