import abc
import copy
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.base import Base
from flax import struct
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.vector.utils import batch_space
from mujoco import mjx

from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.base import BaseMuJoCoEnvironment, BaseObservable, BaseWithArenaAndMorphology, \
    MuJoCoEnvironmentConfiguration
from mujoco_utils.morphology import MJCFMorphology


def mjx_get_model(
        mj_model: mujoco.MjModel,
        mjx_model: mjx.Model,
        n_mj_models: int = 1
        ) -> Union[List[mujoco.MjModel], mujoco.MjModel]:
    """
    Transfer mjx.Model to mujoco.MjModel
    :param result:
    :param value:
    :return:
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
class MJXState(Base):
    """Environment state for MJX.

    Args:
      mjx_model: the current Model, mjx.Model
      mjx_data: the physics state, mjx.Data
      observations: environment observations
      reward: environment reward
      terminated: boolean, True if the current episode has terminated (e.g. by reaching a goal)
      truncated: boolean, True if the current episode was truncated (e.g. by time limits)
      info: metrics that get tracked per environment step
    """
    mjx_model: mjx.Model
    mjx_data: mjx.Data
    observations: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: Dict[str, Any] = struct.field(default_factory=dict)


class MJXObservable(BaseObservable):
    def __init__(
            self,
            name: str,
            low: jnp.ndarray,
            high: jnp.ndarray,
            retriever: Callable[[mjx.Model, mjx.Data], jnp.ndarray]
            ) -> None:
        super().__init__(name=name, low=low, high=high, retriever=retriever)

    def __call__(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> np.ndarray:
        return super().__call__(model=mjx_model, data=mjx_data)


class BaseMJXEnv(BaseMuJoCoEnvironment, ABC):
    # todo: Once Gymnasium releases Gymnasium-MJX, inherit from there
    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        BaseMuJoCoEnvironment.__init__(
                self=self, mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
                )
        self._mjx_model, self._mjx_data = self._initialize_mjx_model_and_data()
        self._mjx_data = mjx.put_data(m=self.mj_model, d=self.mj_data)
        self._observables: Optional[List[MJXObservable]] = self._get_observables()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

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

    def render(
            self,
            state: MJXState
            ) -> RenderFrame | list[RenderFrame] | list[list[RenderFrame]] | None:
        # need to update renderer's model and data (we updated stuff in mjx model that should be in the data)
        # If we are in batch mode,
        #   the human render mode will only render the first environment
        #   rgb_array mode will render a frame for every environment
        camera_ids = self.environment_configuration.camera_ids or [-1]

        try:
            batch_size = state.reward.shape[0]
        except IndexError:
            batch_size = 1

        if self.environment_configuration.render_mode == "human":
            mj_models = mjx_get_model(mj_model=self.mj_model, mjx_model=state.mjx_model, n_mj_models=1)
            mj_datas = mjx.get_data(m=mj_models[0], d=state.mjx_data)
            if not isinstance(mj_datas, list):
                mj_datas = [mj_datas]
            else:
                mj_datas = [mj_datas[0]]
        else:
            mj_models = mjx_get_model(mj_model=self.mj_model, mjx_model=state.mjx_model, n_mj_models=batch_size)
            mj_datas = mjx.get_data(m=self.mj_model, d=state.mjx_data)
            if not isinstance(mj_datas, list):
                mj_datas = [mj_datas]

        frames_per_env = []
        for i, (m, d) in enumerate(zip(mj_models, mj_datas)):
            mujoco.mj_forward(m=m, d=d)
            renderer = self.get_renderer(
                    identifier=i, mj_model=m, mj_data=d
                    )

            if self.environment_configuration.render_mode == "human":
                # noinspection PyProtectedMember
                viewer = renderer._get_viewer("human")
                viewer.model = m
                viewer.data = d

                # Break the loop: only render the first environment in human mode
                return renderer.render(
                        render_mode="human", camera_id=camera_ids[0]
                        )
            else:
                renderer._model = m
                frames = []
                for camera_id in camera_ids:
                    renderer.update_scene(data=d, camera=camera_id)
                    frame = renderer.render()[:, :, ::-1]
                    frames.append(frame)
                frames = frames[0] if len(camera_ids) == 1 else frames
                frames_per_env.append(frames)

        return frames_per_env if batch_size > 1 else frames_per_env[0]

    def _initialize_mjx_data(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data,
            qpos: jnp.ndarray,
            qvel: jnp.ndarray
            ) -> mjx.Data:
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        mjx_data = mjx.forward(m=mjx_model, d=mjx_data)

        return mjx_data

    def _get_action_space(
            self
            ) -> gymnasium.Space:
        bounds = self.mj_model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    @property
    def observables(
            self
            ) -> List[MJXObservable]:
        return self._observables

    @abc.abstractmethod
    def _get_observables(
            self
            ) -> List[MJXObservable]:
        raise NotImplementedError

    def _get_observation_space(
            self
            ) -> gymnasium.spaces.Dict:
        observation_space = dict()
        for observable in self.observables:
            observation_space[observable.name] = gymnasium.spaces.Box(
                    low=np.array(observable.low), high=np.array(observable.high), shape=observable.shape
                    )
        return gymnasium.spaces.Dict(observation_space)

    def _get_observations(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data
            ) -> Dict[str, jnp.ndarray]:
        observations = jax.tree_util.tree_map(
                lambda
                    observable: (observable.name, observable(
                        mjx_model=mjx_model, mjx_data=mjx_data
                        )), self.observables
                )
        return dict(observations)

    def _take_n_steps(
            self,
            mjx_model: mjx.Model,
            mjx_data: mjx.Data,
            ctrl: jnp.ndarray
            ) -> mjx.Data:
        def f(
                data,
                _
                ) -> Tuple[mjx.Data, None]:
            data = data.replace(ctrl=ctrl)
            return (mjx.step(mjx_model, data), None,)

        mjx_data, _ = jax.lax.scan(
                f, mjx_data, (), self.environment_configuration.num_physics_steps_per_control_step
                )
        return mjx_data

    def backend(
            self
            ) -> str:
        return "mjx"

    def close(
            self
            ) -> None:
        super().close()
        del self._observables
        del self._mjx_model
        del self._mjx_data

    @abc.abstractmethod
    def step(
            self,
            state: MJXState,
            action: jnp.ndarray
            ) -> MJXState:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
            self,
            rng: jnp.ndarray
            ) -> MJXState:
        raise NotImplementedError


class MJXEnv(BaseWithArenaAndMorphology, BaseMJXEnv, ABC):
    def __init__(
            self,
            morphology: MJCFMorphology,
            arena: MJCFArena,
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        BaseWithArenaAndMorphology.__init__(self=self, morphology=morphology, arena=arena)
        BaseMJXEnv.__init__(
                self=self, mjcf_str=self._mjcf_str, mjcf_assets=self._mjcf_assets, configuration=configuration
                )


StepFnType = Callable[[MJXState, jnp.ndarray], MJXState]
ResetFnType = Callable[[jnp.ndarray], MJXState]
StepReturnType = Tuple[ObsType, float, bool, bool, dict]
StepBatchReturnType = Tuple[ObsType, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]
ResetReturnType = Tuple[ObsType, dict]


class MJXGymEnvWrapper:
    def __init__(
            self,
            env: MJXEnv,
            num_envs: int
            ) -> None:
        self._env = env
        self._num_envs = num_envs

        self.__jit_step: Optional[StepFnType] = None
        self.__jit_reset: Optional[ResetFnType] = None
        self._mjx_state: Optional[MJXState] = None
        self._rng = jax.random.PRNGKey(seed=42)

        self.single_action_space: gymnasium.spaces.Box = env.action_space
        self.single_observation_space: gymnasium.spaces.Dict = env.observation_space
        if self._num_envs > 1:
            self.action_space = batch_space(self.single_action_space, num_envs)
            self.observation_space = batch_space(self.single_observation_space, num_envs)
        else:
            self.action_space = self.single_action_space
            self.observation_space = self.single_observation_space

    @property
    def mjx_environment(
            self
            ) -> MJXEnv:
        return self._env

    def _prepare_env_fn(
            self,
            fn: Callable
            ) -> Callable:
        if self._num_envs > 1:
            fn = jax.vmap(fn)
        return jax.jit(fn)

    @property
    def _jit_step(
            self
            ) -> StepFnType:
        if self.__jit_step is None:
            self.__jit_step = self._prepare_env_fn(fn=self.mjx_environment.step)
        return self.__jit_step

    @property
    def _jit_reset(
            self
            ) -> ResetFnType:
        if self.__jit_reset is None:
            self.__jit_reset = self._prepare_env_fn(fn=self.mjx_environment.reset)
        return self.__jit_reset

    def step(
            self,
            actions: ActType
            ) -> Union[StepReturnType, StepBatchReturnType]:
        """Steps through the environment with action."""
        self._mjx_state = self._jit_step(self._mjx_state, actions)

        return (
        self._mjx_state.observations, self._mjx_state.reward, self._mjx_state.terminated, self._mjx_state.truncated,
        self._mjx_state.info)

    def reset(
            self,
            seed: int | None = None,
            **kwargs
            ) -> ResetReturnType:
        """Resets the environment with kwargs."""
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed=seed)
        self._rng, *sub_rngs = jax.random.split(key=self._rng, num=self._num_envs + 1)
        sub_rngs = jnp.array(sub_rngs)
        if self._num_envs == 1:
            sub_rngs = sub_rngs[0]

        self._mjx_state = self._jit_reset(sub_rngs)

        return self._mjx_state.observations, self._mjx_state.info

    def render(
            self,
            *args,
            **kwargs
            ) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Renders the environment."""
        return self.mjx_environment.render(self._mjx_state, *args, **kwargs)

    def close(
            self
            ):
        """Closes the environment."""
        return self.mjx_environment.close()

    def __str__(
            self
            ):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.mjx_environment}>"

    def __repr__(
            self
            ):
        """Returns the string representation of the wrapper."""
        return str(self)
