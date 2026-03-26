#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from forword_kinematics_01 import GenkiArmForwardKinematics
from inverse_kinematics_02 import GenkiArmInverseKinematics

def debug_jacobian_at_initial_pose():
    """调试初始姿态下的雅可比矩阵"""
    print("=== 调试初始姿态下的雅可比矩阵 ===")
    
    # 创建运动学对象
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 初始姿态（避免奇异点的弯曲姿态）
    initial_angles = [0.0, -0.5, 0.8, 0.0, -0.3, 0.0]
    
    print(f"初始关节角度: {[f'{angle:.3f}' for angle in initial_angles]}")
    
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

def test_movement_jacobian():
    """测试移动过程中的雅可比矩阵变化"""
    print("\n=== 测试移动过程中的雅可比矩阵变化 ===")
    
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 初始姿态
    initial_angles = [0.0, -0.5, 0.8, 0.0, -0.3, 0.0]
    current_pos = fk.forward_kinematics(initial_angles)
    
    # 目标位置（向上移动10mm）
    target_pos = np.array([current_pos[0], current_pos[1], current_pos[2] + 0.01])
    
    print(f"当前位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    print(f"目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
    
    # 计算位置误差
    position_error = target_pos - np.array(current_pos)
    print(f"位置误差: [{position_error[0]:.6f}, {position_error[1]:.6f}, {position_error[2]:.6f}]")
    print(f"误差模长: {np.linalg.norm(position_error):.6f}")
    
    # 计算雅可比矩阵
    jacobian = ik.compute_jacobian(initial_angles)
    
    # 尝试计算关节角度变化
    JJT = np.dot(jacobian, jacobian.T)
    
    # 检查是否可逆
    try:
        damped_inverse = np.linalg.inv(JJT + 0.01 * np.eye(3))
        delta_angles = 0.1 * np.dot(jacobian.T, np.dot(damped_inverse, position_error))
        
        print(f"\n计算得到的关节角度变化:")
        for i, delta in enumerate(delta_angles):
            print(f"  关节{i+1}: {delta:.6f} rad ({np.degrees(delta):.3f}°)")
        
        # 预测新位置
        new_angles = [initial_angles[i] + delta_angles[i] for i in range(6)]
        predicted_pos = fk.forward_kinematics(new_angles)
        
        print(f"\n预测新位置: [{predicted_pos[0]:.4f}, {predicted_pos[1]:.4f}, {predicted_pos[2]:.4f}]")
        
        # 计算实际移动
        actual_movement = np.array(predicted_pos) - np.array(current_pos)
        print(f"实际移动: [{actual_movement[0]:.6f}, {actual_movement[1]:.6f}, {actual_movement[2]:.6f}]")
        print(f"期望移动: [{position_error[0]:.6f}, {position_error[1]:.6f}, {position_error[2]:.6f}]")
        
        # 计算移动效率
        movement_ratio = actual_movement / (position_error + 1e-10)
        print(f"移动效率: [{movement_ratio[0]:.3f}, {movement_ratio[1]:.3f}, {movement_ratio[2]:.3f}]")
        
    except np.linalg.LinAlgError as e:
        print(f"矩阵求逆失败: {e}")

if __name__ == "__main__":
    jacobian, pos = debug_jacobian_at_initial_pose()
    test_movement_jacobian()