"""GraspEnv — 抓取-放置 RL 环境（M1/M2 通用）。

Action space: 4-dim continuous in [-1, 1]
    [Δx, Δy, Δz, gripper]
    Δxyz: 末端位置增量，缩放到 ±EE_DELTA_MAX 米/step
    gripper: -1 = 闭合, +1 = 张开（线性映射到 joint6 角度）

Observation space: 28-dim 状态向量
    arm_qpos(5) + arm_qvel(5) + gripper_qpos(1)
    + ee_pos(3) + obj_pos(3) + target_pos(3)
    + ee_to_obj(3) + obj_to_target(3)
    + touch(1) + t_norm(1)

Reward: phase-gated dense（见 reward_shaping.py）

Episode：
    reset       — 臂回 READY+扰动，物体随机化，物理稳定 200 步
    target      — __init__ 指定固定，否则每个 episode 随机
    terminated  — placed = True
    truncated   — steps ≥ MAX_EPISODE_STEPS 或 物体被推出 OOB
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
    GRIPPER_OPEN, GRIPPER_CLOSE, READY_JOINTS, READY_PERTURB_STD,
    TOUCH_SENSOR_NAME,
    OBJECT_NAMES, ZONES, DEFAULT_SORT_RULES,
    EE_BOUNDS_LOW, EE_BOUNDS_HIGH, EE_DELTA_MAX,
    MAX_EPISODE_STEPS, SUBSTEPS_PER_ACTION,
    LIFT_THRESHOLD,
)
from .randomization import randomize_objects
from .reward_shaping import compute_reward


# IK 求解（单次从当前姿态出发，速度优先）
IK_MAX_ITER = 30
IK_TOL = 1e-3
IK_DAMPING = 0.05
IK_MAX_STEP = 0.1
IK_FALLBACK_RESIDUAL = 0.05    # 残差 > 此值视为不收敛，回退用上一步 target_q


class GraspEnv(gym.Env):
    """单物体 pick-and-place 强化学习环境。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        target_object: Optional[str] = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        render_mode: Optional[str] = None,
        camera_name: str = "camera_front",
    ):
        super().__init__()

        if target_object is not None and target_object not in OBJECT_NAMES:
            raise ValueError(
                f"Unknown target_object {target_object!r}; "
                f"choose from {OBJECT_NAMES} or None for random."
            )

        # ── 仿真 ──
        self._model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self._data = mujoco.MjData(self._model)

        self._site_ee = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

        # 触觉传感器 sensordata 索引
        sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, TOUCH_SENSOR_NAME
        )
        if sensor_id < 0:
            raise RuntimeError(
                f"Sensor {TOUCH_SENSOR_NAME!r} not found in scene; "
                "check test4.xml <sensor> section."
            )
        self._touch_adr = int(self._model.sensor_adr[sensor_id])

        # 缓存 body id（每 episode 刷新）
        self._target_body_id: int = -1

        # ── 配置 ──
        self._fixed_target = target_object
        self._max_steps = int(max_episode_steps)
        self._render_mode = render_mode
        self._camera_name = camera_name
        self._renderer = None

        # ── 运行时状态 ──
        self._steps = 0
        self._target_obj: Optional[str] = None
        self._target_zone_pos = np.zeros(3)
        self._last_target_q = READY_JOINTS.copy()   # IK 兜底用

        # 复用同一个 ik MjData 减少分配开销
        self._ik_data = mujoco.MjData(self._model)
        self._eye3 = np.eye(3)

        # ── gym spaces ──
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32,
        )

    # ──────────────────────────────────────────────────────────
    #  gym API
    # ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        # 臂回就绪（带高斯扰动）+ 张爪
        perturb = self.np_random.normal(0.0, READY_PERTURB_STD, size=N_ARM_JOINTS)
        ready = READY_JOINTS + perturb
        # 关节限幅
        for j in range(N_ARM_JOINTS):
            lo, hi = self._model.jnt_range[j]
            ready[j] = np.clip(ready[j], lo, hi)

        self._data.qpos[:N_ARM_JOINTS] = ready
        self._data.qpos[GRIPPER_IDX] = GRIPPER_OPEN
        self._data.ctrl[:N_ARM_JOINTS] = ready
        self._data.ctrl[GRIPPER_IDX] = GRIPPER_OPEN
        self._last_target_q = ready.copy()

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
        return self._build_obs(), self._build_info_reset()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # ── 解码动作 ──
        ee_delta = action[:3] * EE_DELTA_MAX
        gripper_cmd = self._decode_gripper(float(action[3]))

        # ── 当前 EE → 目标 EE（裁剪到工作空间） ──
        mujoco.mj_forward(self._model, self._data)
        cur_ee = self._data.site_xpos[self._site_ee].copy()
        target_ee = np.clip(cur_ee + ee_delta, EE_BOUNDS_LOW, EE_BOUNDS_HIGH)

        # ── IK 求解（不收敛则回退上一步） ──
        target_q, ik_residual, ik_ok = self._solve_ik(target_ee)
        if not ik_ok:
            target_q = self._last_target_q
        else:
            self._last_target_q = target_q

        # ── 写入 ctrl 并推进物理 ──
        self._data.ctrl[:N_ARM_JOINTS] = target_q
        self._data.ctrl[GRIPPER_IDX] = gripper_cmd

        for _ in range(SUBSTEPS_PER_ACTION):
            mujoco.mj_step(self._model, self._data)

        self._steps += 1

        # ── obs / reward / 终止 ──
        mujoco.mj_forward(self._model, self._data)

        reward, terminated, truncated_oob, rew_info = compute_reward(
            self._data,
            site_ee_id=self._site_ee,
            target_body_id=self._target_body_id,
            target_zone_pos=self._target_zone_pos,
            touch_sensor_adr=self._touch_adr,
            action=action,
        )

        truncated_timeout = self._steps >= self._max_steps
        truncated = bool(truncated_oob or truncated_timeout)

        obs = self._build_obs()
        info = self._build_info_step(rew_info, ik_residual, ik_ok,
                                     truncated_oob, truncated_timeout)
        return obs, reward, terminated, truncated, info

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
        """单次 DLS IK，从当前姿态出发。

        Returns:
            target_q (5,)        关节命令
            residual (float)     最终末端误差
            ok (bool)            残差 < IK_FALLBACK_RESIDUAL
        """
        ik_data = self._ik_data
        ik_data.qpos[:] = self._data.qpos[:]
        ik_data.qvel[:] = 0.0

        residual = float("inf")
        for _ in range(IK_MAX_ITER):
            mujoco.mj_forward(self._model, ik_data)
            err = target_pos - ik_data.site_xpos[self._site_ee]
            residual = float(np.linalg.norm(err))
            if residual < IK_TOL:
                break

            jacp = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, ik_data, jacp, None, self._site_ee)
            J = jacp[:, :N_ARM_JOINTS]

            try:
                dq = J.T @ np.linalg.solve(
                    J @ J.T + (IK_DAMPING ** 2) * self._eye3, err
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

        target_q = ik_data.qpos[:N_ARM_JOINTS].copy()
        ok = residual < IK_FALLBACK_RESIDUAL
        return target_q, residual, ok

    def _build_obs(self):
        # 调用方应已经 mj_forward 过；这里再保险一次
        mujoco.mj_forward(self._model, self._data)

        arm_qpos = self._data.qpos[:N_ARM_JOINTS].copy()
        arm_qvel = self._data.qvel[:N_ARM_JOINTS].copy()
        gripper_q = self._data.qpos[GRIPPER_IDX:GRIPPER_IDX + 1].copy()

        ee_pos = self._data.site_xpos[self._site_ee].copy()
        obj_pos = self._data.xpos[self._target_body_id].copy()
        target_pos = self._target_zone_pos.copy()

        ee_to_obj = obj_pos - ee_pos
        obj_to_target = target_pos - obj_pos

        touch = np.array([self._data.sensordata[self._touch_adr]])
        t_norm = np.array([self._steps / max(self._max_steps, 1)])

        return np.concatenate([
            arm_qpos, arm_qvel, gripper_q,
            ee_pos, obj_pos, target_pos,
            ee_to_obj, obj_to_target,
            touch, t_norm,
        ]).astype(np.float32)

    def _build_info_reset(self):
        obj_pos = self._data.xpos[self._target_body_id].copy()
        return {
            "target_object": self._target_obj,
            "target_zone": self._target_zone_pos.copy(),
            "object_pos": obj_pos,
            "object_lifted": bool(obj_pos[2] > LIFT_THRESHOLD),
            "ee_pos": self._data.site_xpos[self._site_ee].copy(),
            "steps": self._steps,
        }

    def _build_info_step(self, rew_info, ik_residual, ik_ok,
                         oob, timeout):
        info = {
            "target_object": self._target_obj,
            "target_zone": self._target_zone_pos.copy(),
            "object_pos": self._data.xpos[self._target_body_id].copy(),
            "object_lifted": rew_info["lifted"],
            "ee_pos": self._data.site_xpos[self._site_ee].copy(),
            "steps": self._steps,
            "ik_residual": ik_residual,
            "ik_ok": ik_ok,
            "truncated_oob": oob,
            "truncated_timeout": timeout,
        }
        info.update(rew_info)
        return info
