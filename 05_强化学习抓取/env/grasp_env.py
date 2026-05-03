"""GraspEnv — 抓取-放置 RL 环境（M1/M2 通用）。

Action space: 6-dim continuous in [-1, 1]
    [Δq1, Δq2, Δq3, Δq4, Δq5, gripper]
    Δq1-5: 关节角增量，缩放到 ±JOINT_DELTA_MAX 弧度/step
    gripper: -1 = 闭合, +1 = 张开（线性映射到 joint6 角度）

为什么不用 EE-space (Δxyz) 动作 + IK？早期单次 IK 单步可解，多步累积后会陷入
局部解（在某些 pose 下 IK 残差从 0.001 涨到 0.02）。多重启 IK 太慢。
关节空间动作直接、可靠，且 SO-100 只有 5 DOF。

Observation space: 28-dim 状态向量
    arm_qpos(5) + arm_qvel(5) + gripper_qpos(1)
    + ee_pos(3) + obj_pos(3) + target_pos(3)
    + ee_to_obj(3) + obj_to_target(3)
    + contact_flag(1) + t_norm(1)

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
    OBJECT_NAMES, ZONES, DEFAULT_SORT_RULES,
    MAX_EPISODE_STEPS, SUBSTEPS_PER_ACTION, JOINT_DELTA_MAX,
    LIFT_THRESHOLD, KP_BOOST,
)
from .randomization import randomize_objects
from .reward_shaping import compute_reward


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
        # 04 stage XML 的 actuator kp 太软，joint1-5 的 gainprm/biasprm 同步乘 KP_BOOST
        for i in range(N_ARM_JOINTS):
            self._model.actuator_gainprm[i, 0] *= KP_BOOST
            self._model.actuator_biasprm[i, 1] *= KP_BOOST
        # ── 碰撞分组（这是 0% lift rate 的真因之一）──
        # 默认所有 geom contype/conaffinity=1，arm 的 link5/gripper 会撞地板，下不去。
        # 三组分组：
        #   floor:   ct=1, ca=1  (只与 obj 碰)
        #   obj:     ct=5, ca=3  (与 floor 和 arm 都碰)
        #   arm:     ct=2, ca=4  (只与 obj 碰，不撞地板，也不自碰)
        self._setup_collision_groups()

        self._data = mujoco.MjData(self._model)

        self._site_ee = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

        # 夹爪几何（用于 mj_contact 接触检测，比 site touch sensor 可靠）
        self._grip_geom_ids = set()
        for name in ("link5_geom", "gripper_geom"):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._grip_geom_ids.add(gid)
        if not self._grip_geom_ids:
            raise RuntimeError("No gripper geoms found; check XML names.")

        # 缓存 body / geom id（每 episode 刷新）
        self._target_body_id: int = -1
        self._target_geom_id: int = -1

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
        self._first_lift_pending = True  # episode 内是否还没发过 first_lift bonus
        self._first_near_pending = True  # 是否还没发过 first_near_target bonus
        self._ever_lifted = False        # episode 内是否曾经 lift 过（堵推杆 hack）

        # ── gym spaces ──
        # 6 dim：5 关节增量 + 1 夹爪
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32,
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
        self._target_geom_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{self._target_obj}_geom"
        )

        self._steps = 0
        self._first_lift_pending = True
        self._first_near_pending = True
        self._ever_lifted = False
        return self._build_obs(), self._build_info_reset()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # ── 解码动作 ──
        joint_delta = action[:N_ARM_JOINTS] * JOINT_DELTA_MAX
        gripper_cmd = self._decode_gripper(float(action[5]))

        # ── 关节增量控制：ctrl[i] += delta，clip 到 joint range ──
        new_ctrl = self._data.ctrl[:N_ARM_JOINTS] + joint_delta
        for j in range(N_ARM_JOINTS):
            lo, hi = self._model.jnt_range[j]
            new_ctrl[j] = np.clip(new_ctrl[j], lo, hi)
        self._data.ctrl[:N_ARM_JOINTS] = new_ctrl
        self._data.ctrl[GRIPPER_IDX] = gripper_cmd

        # ── 推进物理 ──
        for _ in range(SUBSTEPS_PER_ACTION):
            mujoco.mj_step(self._model, self._data)

        self._steps += 1

        # ── obs / reward / 终止 ──
        mujoco.mj_forward(self._model, self._data)
        has_contact = self._check_grip_contact()

        reward, terminated, truncated_oob, rew_info = compute_reward(
            self._data,
            site_ee_id=self._site_ee,
            target_body_id=self._target_body_id,
            target_zone_pos=self._target_zone_pos,
            has_contact=has_contact,
            action=action,
            first_lift_pending=self._first_lift_pending,
            first_near_target_pending=self._first_near_pending,
            ever_lifted=self._ever_lifted,
        )
        # episode 状态更新：lift 过的标志一次置 True 不再翻回
        if rew_info.get("lifted", False):
            self._ever_lifted = True
        # 一次性 milestone bonus 已发，关闭 flag
        if rew_info.get("first_lift_consumed", False):
            self._first_lift_pending = False
        if rew_info.get("first_near_consumed", False):
            self._first_near_pending = False

        truncated_timeout = self._steps >= self._max_steps
        truncated = bool(truncated_oob or truncated_timeout)

        obs = self._build_obs()
        info = self._build_info_step(rew_info, truncated_oob, truncated_timeout)
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

    def _setup_collision_groups(self):
        """三组碰撞分组：floor 只与 obj，arm 只与 obj，obj 与 floor+arm。

        位掩码规则（MuJoCo）：geom1.contype & geom2.conaffinity 或反之非零则碰撞。
        floor:  ct=1, ca=1
        obj:    ct=5, ca=3   (5=1|4 与 floor 的 ca=1 相 AND, 与 arm 的 ca=4 相 AND)
        arm:    ct=2, ca=4   (与 floor 的 ct=1 相 AND=0 ⇒ 不碰；与 obj 的 ca=3 相 AND=2 ⇒ 碰)
        """
        m = self._model

        floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            m.geom_contype[floor_id] = 1
            m.geom_conaffinity[floor_id] = 1

        # 物体 geom
        for obj_name in OBJECT_NAMES:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_name}_geom")
            if gid >= 0:
                m.geom_contype[gid] = 5
                m.geom_conaffinity[gid] = 3

        # 所有 arm geom
        for name in ("base_geom", "link1_geom", "link2_geom", "link3_geom",
                     "link4_geom", "link5_geom", "gripper_geom"):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                m.geom_contype[gid] = 2
                m.geom_conaffinity[gid] = 4

    def _check_grip_contact(self) -> bool:
        """gripper 几何对 target 物体几何，是否存在 mj_contact。"""
        if self._target_geom_id < 0:
            return False
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = c.geom1, c.geom2
            hit_obj = g1 == self._target_geom_id or g2 == self._target_geom_id
            hit_grip = g1 in self._grip_geom_ids or g2 in self._grip_geom_ids
            if hit_obj and hit_grip:
                return True
        return False

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

        contact_flag = np.array([1.0 if self._check_grip_contact() else 0.0])
        t_norm = np.array([self._steps / max(self._max_steps, 1)])

        return np.concatenate([
            arm_qpos, arm_qvel, gripper_q,
            ee_pos, obj_pos, target_pos,
            ee_to_obj, obj_to_target,
            contact_flag, t_norm,
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

    def _build_info_step(self, rew_info, oob, timeout):
        info = {
            "target_object": self._target_obj,
            "target_zone": self._target_zone_pos.copy(),
            "object_pos": self._data.xpos[self._target_body_id].copy(),
            "object_lifted": rew_info["lifted"],
            "ee_pos": self._data.site_xpos[self._site_ee].copy(),
            "steps": self._steps,
            "truncated_oob": oob,
            "truncated_timeout": timeout,
        }
        info.update(rew_info)
        return info
