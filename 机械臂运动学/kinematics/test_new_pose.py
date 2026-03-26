#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from forword_kinematics_01 import GenkiArmForwardKinematics
from inverse_kinematics_02 import GenkiArmInverseKinematics

def test_new_initial_pose():
    """测试新的初始姿态"""
    print("=== 测试新的初始姿态 ===")
    
    # 创建运动学对象
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 新的初始姿态
    initial_angles = [0.0, -0.3, 0.6, 0.0, -0.6, 0.0]
    
    print(f"新初始关节角度: {[f'{angle:.3f}' for angle in initial_angles]}")
    
    # 计算当前位置
    current_pos = fk.forward_kinematics(initial_angles)
    print(f"当前末端位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    
    # 计算雅可比矩阵
    jacobian = ik.compute_jacobian(initial_angles)
    print(f"\n雅可比矩阵形状: {jacobian.shape}")
    print("雅可比矩阵:")
    for i, row in enumerate(jacobian):
        print(f"  {['X', 'Y', 'Z'][i]}: [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}, {row[3]:8.5f}, {row[4]:8.5f}, {row[5]:8.5f}]")
    
    # 分析雅可比矩阵的性质
    JJT = np.dot(jacobian, jacobian.T)
    print(f"\nJ*J^T 矩阵:")
    for i, row in enumerate(JJT):
        print(f"  [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")
    
    # 计算条件数
    cond_num = np.linalg.cond(JJT)
    print(f"\nJ*J^T 条件数: {cond_num:.2e}")
    
    # 计算特征值
    eigenvals = np.linalg.eigvals(JJT)
    print(f"J*J^T 特征值: {[f'{val:.2e}' for val in eigenvals]}")
    
    # 检查每个轴的可操作性
    print("\n各轴可操作性分析:")
    for axis in range(3):
        axis_jacobian = jacobian[axis, :]
        axis_norm = np.linalg.norm(axis_jacobian)
        print(f"  {['X', 'Y', 'Z'][axis]}轴: 雅可比行向量模长 = {axis_norm:.6f}")
        if axis_norm < 1e-6:
            print(f"    警告: {['X', 'Y', 'Z'][axis]}轴几乎不可操作!")
    
    return jacobian, current_pos

def test_movement_from_new_pose():
    """测试从新姿态开始的移动"""
    print("\n=== 测试从新姿态开始的移动 ===")
    
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 新的初始姿态
    initial_angles = [0.0, -0.3, 0.6, 0.0, -0.6, 0.0]
    current_pos = fk.forward_kinematics(initial_angles)
    
    # 测试多个方向的移动
    movements = [
        ("向上10mm", [0, 0, 0.01]),
        ("向下10mm", [0, 0, -0.01]),
        ("向前10mm", [0.01, 0, 0]),
        ("向后10mm", [-0.01, 0, 0]),
        ("向左10mm", [0, 0.01, 0]),
        ("向右10mm", [0, -0.01, 0])
    ]
    
    for direction, delta in movements:
        print(f"\n--- {direction} ---")
        target_pos = np.array(current_pos) + np.array(delta)
        
        print(f"当前位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
        print(f"目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        
        # 尝试逆运动学求解
        result_angles, success, info = ik.inverse_kinematics(target_pos, initial_angles, verbose=False)
        
        if success:
            # 验证结果
            final_pos = fk.forward_kinematics(result_angles)
            error = np.linalg.norm(np.array(final_pos) - target_pos)
            print(f"求解成功! 最终误差: {error:.6f}m")
            print(f"迭代次数: {info['iterations']}")
        else:
            print(f"求解失败! 最终误差: {info['final_error']:.6f}m")

if __name__ == "__main__":
    jacobian, pos = test_new_initial_pose()
    test_movement_from_new_pose()