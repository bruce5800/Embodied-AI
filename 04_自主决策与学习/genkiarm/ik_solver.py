"""
IK 求解器 — 基于 MuJoCo Jacobian + 阻尼最小二乘（多次随机重启）
"""

import mujoco
import numpy as np

from config import N_ARM_JOINTS, READY_JOINTS


def _ik_once(model, target, site_id, initial_qpos, max_iter=500, tol=1e-3):
    """
    单次 IK 求解（内部函数）。
    从 initial_qpos 出发，用 Jacobian + 阻尼最小二乘迭代。
    """
    data_ik = mujoco.MjData(model)
    data_ik.qpos[:] = initial_qpos[:]

    for _ in range(max_iter):
        mujoco.mj_forward(model, data_ik)

        err = target - data_ik.site_xpos[site_id]
        dist = np.linalg.norm(err)

        if dist < tol:
            return data_ik.qpos[:N_ARM_JOINTS].copy(), dist

        # Jacobian: 3×nv 矩阵
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_ik, jacp, None, site_id)
        J = jacp[:, :N_ARM_JOINTS]

        # 自适应阻尼：远离目标时大步快走，靠近时小步精调
        lam = max(0.01, min(0.1, dist * 0.5))
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)

        # 限制单步幅度
        max_step = 0.2
        scale = np.max(np.abs(dq))
        if scale > max_step:
            dq *= max_step / scale

        data_ik.qpos[:N_ARM_JOINTS] += dq

        # 关节限幅
        for j in range(N_ARM_JOINTS):
            lo, hi = model.jnt_range[j]
            data_ik.qpos[j] = np.clip(data_ik.qpos[j], lo, hi)

    mujoco.mj_forward(model, data_ik)
    final_err = np.linalg.norm(target - data_ik.site_xpos[site_id])
    return data_ik.qpos[:N_ARM_JOINTS].copy(), final_err


def solve_ik(model, target_pos, current_qpos=None, max_iter=500, tol=1e-3,
             num_attempts=8):
    """
    使用 MuJoCo 内置 mj_jacSite 求解逆运动学（多次随机重启）。
    仅控制 joint1-5（臂体），不动 joint6（夹爪）。

    原理:
      1. 计算末端当前位置与目标的误差 e
      2. 通过 mj_jacSite 计算末端对关节的 Jacobian J
      3. 阻尼最小二乘: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
      4. 更新关节角，限幅，重复直到收敛
      5. 多次随机初始猜测，取最优解（避免局部最小值）

    Returns:
        (joint_angles, success)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target = np.array(target_pos, dtype=np.float64)

    best_angles = None
    best_err = float('inf')

    # 构建初始猜测列表
    base_qpos = np.zeros(model.nq)
    if current_qpos is not None:
        base_qpos[:] = current_qpos[:]

    for attempt in range(num_attempts):
        init_qpos = base_qpos.copy()

        if attempt == 0:
            # 第 1 次：从当前姿态出发
            pass
        elif attempt == 1:
            # 第 2 次：从就绪姿态出发
            init_qpos[:N_ARM_JOINTS] = READY_JOINTS
        else:
            # 其余：随机采样关节角
            for j in range(N_ARM_JOINTS):
                lo, hi = model.jnt_range[j]
                init_qpos[j] = np.random.uniform(lo, hi)

        angles, err = _ik_once(model, target, site_id, init_qpos,
                               max_iter=max_iter, tol=tol)

        if err < best_err:
            best_err = err
            best_angles = angles

        # 足够精确就提前退出
        if best_err < tol:
            break

    success = best_err < tol * 5
    return best_angles, success
