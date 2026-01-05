import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    """Return (x, y) of center of mass (mass-weighted average of body positions)."""
    num = np.einsum("b,bj->j", model.body_mass, data.xipos)
    denom = model.body_mass.sum()
    return (num / denom)[0:2].copy()


class {{ENV_CLASS}}(MujocoEnv, utils.EzPickle):
    """Gymnasium MuJoCo Humanoid environment (v5-style) with configurable observation components."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "humanoid.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # If True: drop absolute (x, y) torso position from obs (position-invariant bias)
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Toggle optional observation blocks
        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation


        xml_path = Path(xml_file)
        if not xml_path.is_absolute():
            xml_path = ASSET_DIR / xml_path

        MujocoEnv.__init__(
            self,
            str(xml_path),
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Build observation size based on enabled blocks
        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation  # drop x,y from qpos
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Helpful for tooling/wrappers: how the observation is composed
        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        # Penalize large controls (uses applied control in data.ctrl)
        return self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))

    @property
    def contact_cost(self):
        # Penalize large external contact forces
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        return np.clip(contact_cost, min_cost, max_cost)

    @property
    def is_healthy(self):
        # Healthy if torso height (qpos[2]) in range
        min_z, max_z = self._healthy_z_range
        return min_z < self.data.qpos[2] < max_z

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        com_inertia = self.data.cinert[1:].flatten() if self._include_cinert_in_observation else np.array([])
        com_velocity = self.data.cvel[1:].flatten() if self._include_cvel_in_observation else np.array([])

        # Skip the first 6 dofs (free joint) for actuator forces
        actuator_forces = (
            self.data.qfrc_actuator[6:].flatten()
            if self._include_qfrc_actuator_in_observation
            else np.array([])
        )
        external_contact_forces = (
            self.data.cfrc_ext[1:].flatten()
            if self._include_cfrc_ext_in_observation
            else np.array([])
        )

        if self._exclude_current_positions_from_observation:
            position = position[2:]  # drop absolute x,y

        return np.concatenate(
            (position, velocity, com_inertia, com_velocity, actuator_forces, external_contact_forces)
        )

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)

        terminated = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        # Truncation is handled by the TimeLimit wrapper added in gym.make()
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        reward = (forward_reward + healthy_reward) - (ctrl_cost + contact_cost)

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }
        return reward, reward_info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
