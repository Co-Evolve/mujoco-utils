import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy
import mujoco
import numpy as np
from gymnasium import Space
from mujoco import mjx

from mujoco_utils.arena import MJCFArena
from mujoco_utils.environment.renderer import MujocoRenderer
from mujoco_utils.morphology import MJCFMorphology


class MuJoCoEnvironmentConfiguration:
    PHYSICS_TIMESTEP = 0.002

    def __init__(
            self,
            seed: int = 42,
            time_scale: float = 1.0,
            num_physics_steps_per_control_step: int = 1,
            simulation_time: float = 10,
            camera_ids: List[int] | None = None,
            render_size: Tuple[int, int] | None = (240, 320),
            render_mode: Optional[str] = None
            ) -> None:
        self.seed = seed
        self.time_scale = time_scale
        self.num_physics_steps_per_control_step = num_physics_steps_per_control_step
        self.simulation_time = simulation_time
        self.camera_ids = camera_ids or [0]
        self.render_size = render_size
        self.render_mode = render_mode

    @property
    def control_timestep(
            self
            ) -> float:
        return self.num_physics_steps_per_control_step * self.physics_timestep

    @property
    def physics_timestep(
            self
            ) -> float:
        return self.PHYSICS_TIMESTEP * self.time_scale

    @property
    def total_num_control_steps(
            self
            ) -> int:
        return int(np.ceil(self.simulation_time / self.control_timestep))

    @property
    def total_num_physics_steps(
            self
            ) -> int:
        return int(np.ceil(self.simulation_time / self.physics_timestep))


ArrayType = Union[np.ndarray, jax.numpy.ndarray]
ModelType = Union[mujoco.MjModel, mjx.Model]
DataType = Union[mujoco.MjData, mjx.Data]


class BaseObservable:
    def __init__(
            self,
            name: str,
            low: ArrayType,
            high: ArrayType,
            retriever: Callable[[ModelType, DataType], ArrayType]
            ) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.retriever = retriever

    def __call__(
            self,
            model: ModelType,
            data: DataType
            ) -> ArrayType:
        return self.retriever(model, data)

    @property
    def shape(
            self
            ) -> Tuple[int, ...]:
        return self.low.shape


class BaseMuJoCoEnvironment(abc.ABC):
    metadata = {"render_modes": []}

    def __init__(
            self,
            mjcf_str: str,
            mjcf_assets: Dict[str, Any],
            configuration: MuJoCoEnvironmentConfiguration
            ) -> None:
        self._mjcf_str = mjcf_str
        self._mjcf_assets = mjcf_assets

        self._configuration = configuration

        self._renderers: Dict[int, Union[MujocoRenderer, mujoco.Renderer]] = dict()

        self._mj_model, self._mj_data = self._initialize_mj_model_and_data()

        self.action_space: Optional[Space] = None
        self.observation_space: Optional[Space] = None

        assert configuration.render_mode is None or self.environment_configuration.render_mode in self.metadata[
            "render_modes"], (f"Unsupported render mode: '{self.environment_configuration.render_mode}'. Must be one "
                              f"of {self.metadata['render_modes']}")

    @property
    def environment_configuration(
            self
            ) -> MuJoCoEnvironmentConfiguration:
        return self._configuration

    @property
    def mj_model(
            self
            ) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mj_data(
            self
            ) -> mujoco.MjData:
        return self._mj_data

    @property
    def actuators(
            self
            ) -> List[str]:
        return [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]

    @abc.abstractmethod
    def _get_observation_space(
            self
            ) -> Space:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_action_space(
            self
            ) -> Space:
        raise NotImplementedError

    def _initialize_mj_model_and_data(
            self
            ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        mj_model = mujoco.MjModel.from_xml_string(xml=self._mjcf_str, assets=self._mjcf_assets)
        mj_model.vis.global_.offheight = self.environment_configuration.render_size[0]
        mj_model.vis.global_.offwidth = self.environment_configuration.render_size[1]
        mj_data = mujoco.MjData(mj_model)
        return mj_model, mj_data

    def get_renderer(
            self,
            identifier: int = 0,
            mj_model: Optional[mujoco.MjModel] = None,
            mj_data: Optional[mujoco.MjData] = None
            ) -> Union[MujocoRenderer, mujoco.Renderer]:
        if identifier not in self._renderers:
            if self.environment_configuration.render_mode == "human":
                renderer = MujocoRenderer(
                        model=mj_model or self.mj_model,
                        data=mj_data or self.mj_data,
                        default_cam_config=None
                        )
            else:
                renderer = mujoco.Renderer(
                        model=mj_model or self.mj_model,
                        height=self.environment_configuration.render_size[0],
                        width=self.environment_configuration.render_size[1]
                        )
            self._renderers[identifier] = renderer
        return self._renderers[identifier]

    def _close_renderers(
            self
            ) -> None:
        for renderer in self._renderers.values():
            renderer.close()

    def close(
            self
            ) -> None:
        self._close_renderers()


class BaseWithArenaAndMorphology(abc.ABC):
    def __init__(
            self,
            morphology: MJCFMorphology,
            arena: MJCFArena
            ) -> None:
        self._morphology = morphology
        self._arena = arena
        self.arena.attach(other=self.morphology, free_joint=True)
        self._mjcf_str, self._mjcf_assets = arena.get_mjcf_str(), arena.get_mjcf_assets()

    @property
    def morphology(
            self
            ) -> MJCFMorphology:
        return self._morphology

    @property
    def arena(
            self
            ) -> MJCFArena:
        return self._arena
