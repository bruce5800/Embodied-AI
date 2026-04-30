"""GraspEnv — 最小可跑的抓取-放置 RL 环境。

Action space: 4-dim continuous in [-1, 1]
    [Δx, Δy, Δz, gripper]
    Δxyz: 末端位置增量，缩放到 ±2cm/step
    gripper: -1 = 闭合, +1 = 张开（线性映射到 joint6 角度）

Observation space: 26-dim 状态向量
    arm_qpos(5) + arm_qvel(5) + gripper_qpos(1)
    + ee_pos(3) + obj_pos(3) + target_pos(3)
    + ee_to_obj(3) + obj_to_target(3)

Reward (dense)：
    -‖ee - obj‖                 接近物体
    + 1 if obj_z > 0.06         抬起奖励
    -‖obj_xy - target_xy‖       靠近目标
    + 10 if 放置成功            成功奖励
    - 0.001 ‖action‖²           动作平滑

Episode：
    reset       — 臂回 READY，物体随机化，物理稳定 200 步
    target      — 单物体；__init__ 指定固定，否则每个 episode 随机
    terminated  — 物体落入目标区半径 6cm 内
    truncated   — 200 步超时
"""

from typing import Optional

import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # 退回旧 gym
    import gym
    from gym import spaces

from .scene_constants import (
    SCENE_XML, N_ARM_JOINTS, GRIPPER_IDX,
    GRIPPER_OPEN, GRIPPER_CLOSE, READY_JOINTS,
    OBJECT_NAMES, ZONES, DEFAULT_SORT_RULES,
    EE_BOUNDS_LOW, EE_BOUNDS_HIGH,
    LIFT_THRESHOLD, SUCCESS_RADIUS, PLACED_Z_MAX,
)
from .randomization import randomize_objects


# 动作缩放
EE_DELTA_MAX = 0.02            # 单步末端最大位移（米）
SUBSTEPS_PER_ACTION = 5        # 每个 RL step 跑几次 mj_step

# IK 求解（单次从当前姿态出发，速度优先）
IK_MAX_ITER = 30
IK_TOL = 1e-3
IK_DAMPING = 0.05
IK_MAX_STEP = 0.1


class GraspEnv(gym.Env):
    """单物体 pick-and-place 强化学习环境。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        target_object: Optional[str] = None,
        max_episode_steps: int = 200,
        render_mode: Optional[str] = None,
        camera_name: str = "camera_front",
    ):
        super().__init__()

        if target_object is not None and target_object not in OBJECT_NAMES:
            raise ValueError(
                f"Unknown target_object {target_object!r}; "
                f"choose from {OBJECT_NAMES} or None for random."
            )

        # 仿真
        self._model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self._data = mujoco.MjData(self._model)
        self._site_ee = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

        # 配置
        self._fixed_target = target_object
        self._max_steps = int(max_episode_steps)
        self._render_mode = render_mode
        self._camera_name = camera_name
        self._renderer = None

        # 运行时状态
        self._steps = 0
        self._target_obj: Optional[str] = None
        self._target_zone_pos = np.zeros(3)

        # 缓存的 body id（reset 时刷新）
        self._target_body_id: int = -1

        # gym spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32,
        )

    # ──────────────────────────────────────────────────────────
    #  gym API
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        # 臂回就绪 + 张爪
        self._data.qpos[:N_ARM_JOINTS] = READY_JOINTS
        self._data.qpos[GRIPPER_IDX] = GRIPPER_OPEN
        self._data.ctrl[:N_ARM_JOINTS] = READY_JOINTS
        self._data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN

        # 物体随机化
        randomize_objects(self._model, self._data, np_random=self.np_random)

        # 物理稳定（让物体落到桌面）
        for _ in range(200):
            mujoco.mj_step(self._model, self._data)

        # 选目标
        if self._fixed_target is not None:
            self._target_obj = self._fixed_target
        else:
            idx = int(self.np_random.integers(0, len(OBJECT_NAMES)))
            self._target_obj = OBJECT_NAMES[idx]

        zone_name = DEFAULT_SORT_RULES[self._target_obj]
        self._target_zone_pos = ZONES[zone_name].copy()
        self._target_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, self._target_obj
        )

        self._steps = 0
        return self._build_obs(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # 解码动作
        ee_delta = action[:3] * EE_DELTA_MAX
        gripper_cmd = self._decode_gripper(float(action[3]))

        # 当前 EE → 目标 EE（裁剪到工作空间）
        mujoco.mj_forward(self._model, self._data)
        cur_ee = self._data.site_xpos[self._site_ee].copy()
        target_ee = np.clip(cur_ee + ee_delta, EE_BOUNDS_LOW, EE_BOUNDS_HIGH)

        # IK 求解关节命令
        target_q = self._solve_ik(target_ee)

        # 写入 ctrl
        self._data.ctrl[:N_ARM_JOINTS] = target_q
        self._data.ctrl[GRIPPER_IDX] = gripper_cmd

        # 推进物理
        for _ in range(SUBSTEPS_PER_ACTION):
            mujoco.mj_step(self._model, self._data)

        self._steps += 1

        obs = self._build_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self._steps >= self._max_steps
        return obs, reward, terminated, truncated, self._info()

    def render(self):
        if self._render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
        self._renderer.update_scene(self._data, camera=self._camera_name)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────
    #  内部
    # ──────────────────────────────────────────────────────────

    def _decode_gripper(self, a: float) -> float:
        """[-1, 1] → [GRIPPER_CLOSE, GRIPPER_OPEN]，-1=闭，+1=开。"""
        return GRIPPER_CLOSE + (a + 1.0) * 0.5 * (GRIPPER_OPEN - GRIPPER_CLOSE)

    def _solve_ik(self, target_pos):
        """单次 DLS IK，从当前姿态出发。比 04_/ik_solver.py 的 8 次随机重启快得多，
        作为 RL 低层控制器使用。"""
        ik_data = mujoco.MjData(self._model)
        ik_data.qpos[:] = self._data.qpos[:]
        ik_data.qvel[:] = 0.0

        eye3 = np.eye(3)
        for _ in range(IK_MAX_ITER):
            mujoco.mj_forward(self._model, ik_data)
            err = target_pos - ik_data.site_xpos[self._site_ee]
            if np.linalg.norm(err) < IK_TOL:
                break

            jacp = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, ik_data, jacp, None, self._site_ee)
            J = jacp[:, :N_ARM_JOINTS]

            try:
                dq = J.T @ np.linalg.solve(
                    J @ J.T + (IK_DAMPING ** 2) * eye3, err
                )
            except np.linalg.LinAlgError:
                break

            scale = float(np.max(np.abs(dq)))
            if scale > IK_MAX_STEP:
                dq *= IK_MAX_STEP / scale

            ik_data.qpos[:N_ARM_JOINTS] += dq
            for j in range(N_ARM_JOINTS):
                lo, hi = self._model.jnt_range[j]
                ik_data.qpos[j] = np.clip(ik_data.qpos[j], lo, hi)

        return ik_data.qpos[:N_ARM_JOINTS].copy()

    def _build_obs(self):
        mujoco.mj_forward(self._model, self._data)

        arm_qpos = self._data.qpos[:N_ARM_JOINTS].copy()
        arm_qvel = self._data.qvel[:N_ARM_JOINTS].copy()
        gripper_q = self._data.qpos[GRIPPER_IDX:GRIPPER_IDX + 1].copy()

        ee_pos = self._data.site_xpos[self._site_ee].copy()
        obj_pos = self._data.xpos[self._target_body_id].copy()
        target_pos = self._target_zone_pos.copy()

        ee_to_obj = obj_pos - ee_pos
        obj_to_target = target_pos - obj_pos

        return np.concatenate([
            arm_qpos, arm_qvel, gripper_q,
            ee_pos, obj_pos, target_pos,
            ee_to_obj, obj_to_target,
        ]).astype(np.float32)

    def _compute_reward(self, action):
        ee_pos = self._data.site_xpos[self._site_ee]
        obj_pos = self._data.xpos[self._target_body_id]
        target_pos = self._target_zone_pos

        d_ee_obj = float(np.linalg.norm(obj_pos - ee_pos))
        d_obj_tgt_xy = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        lifted = obj_pos[2] > LIFT_THRESHOLD
        placed = (d_obj_tgt_xy < SUCCESS_RADIUS) and (obj_pos[2] < PLACED_Z_MAX)

        reward = (
            -d_ee_obj
            + (1.0 if lifted else 0.0)
            - d_obj_tgt_xy
            + (10.0 if placed else 0.0)
            - 0.001 * float(np.sum(action ** 2))
        )
        return float(reward), bool(placed)

    def _info(self):
        obj_pos = self._data.xpos[self._target_body_id].copy()
        return {
            "target_object": self._target_obj,
            "target_zone": self._target_zone_pos.copy(),
            "object_pos": obj_pos,
            "object_lifted": bool(obj_pos[2] > LIFT_THRESHOLD),
            "ee_pos": self._data.site_xpos[self._site_ee].copy(),
            "steps": self._steps,
        }
