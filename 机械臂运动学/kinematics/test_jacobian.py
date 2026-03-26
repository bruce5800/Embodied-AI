#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试雅可比矩阵计算
"""

import numpy as np
import math
from forword_kinematics_01 import GenkiArmForwardKinematics
from inverse_kinematics_02 import GenkiArmInverseKinematics

def test_jacobian():
    """测试雅可比矩阵"""
    print("=" * 50)
    print("测试雅可比矩阵计算")
    print("=" * 50)
    
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 测试零位的雅可比矩阵
    zero_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"零位关节角度: {zero_angles}")
    
    try:
        jacobian = ik.compute_jacobian(zero_angles)
        print(f"雅可比矩阵形状: {jacobian.shape}")
        print("雅可比矩阵:")
        for i in range(3):
            row_str = " ".join([f"{jacobian[i,j]:8.4f}" for j in range(6)])
            print(f"  [{row_str}]")
        
        # 检查雅可比矩阵的条件数
        JJT = np.dot(jacobian, jacobian.T)
        print(f"\nJ*J^T 矩阵:")
        for i in range(3):
            row_str = " ".join([f"{JJT[i,j]:8.4f}" for j in range(3)])
            print(f"  [{row_str}]")
        
        # 计算条件数
        try:
            cond_num = np.linalg.cond(JJT)
            print(f"J*J^T 条件数: {cond_num:.2e}")
            
            # 尝试求逆
            JJT_inv = np.linalg.inv(JJT + 0.01 * np.eye(3))
            print("J*J^T + λI 可以求逆")
            
        except np.linalg.LinAlgError as e:
            print(f"矩阵求逆失败: {e}")
            
    except Exception as e:
        print(f"雅可比矩阵计算失败: {e}")

def test_small_movement():
    """测试小幅移动的雅可比矩阵"""
    print("\n" + "=" * 50)
    print("测试小幅移动")
    print("=" * 50)
    
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 从零位开始
    current_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current_pos = np.array(fk.forward_kinematics(current_angles))
    
    print(f"当前位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
    
    # 目标：向上移动1cm
    target_pos = current_pos.copy()
    target_pos[2] += 0.01  # 向上1cm
    
    print(f"目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
    
    # 计算位置误差
    position_error = target_pos - current_pos
    print(f"位置误差: [{position_error[0]:.4f}, {position_error[1]:.4f}, {position_error[2]:.4f}]")
    
    # 计算雅可比矩阵
    jacobian = ik.compute_jacobian(current_angles)
    print("雅可比矩阵:")
    for i in range(3):
        row_str = " ".join([f"{jacobian[i,j]:8.4f}" for j in range(6)])
        print(f"  [{row_str}]")
    
    # 使用阻尼最小二乘法计算关节角度变化
    try:
        JJT = np.dot(jacobian, jacobian.T)
        damping = 0.01
        damped_inverse = np.linalg.inv(JJT + damping * np.eye(3))
        delta_angles = 0.5 * np.dot(jacobian.T, np.dot(damped_inverse, position_error))
        
        print(f"计算得到的关节角度变化 (弧度): {[f'{a:.4f}' for a in delta_angles]}")
        print(f"计算得到的关节角度变化 (度): {[f'{math.degrees(a):.2f}°' for a in delta_angles]}")
        
        # 应用变化
        new_angles = [current_angles[i] + delta_angles[i] for i in range(6)]
        new_pos = np.array(fk.forward_kinematics(new_angles))
        
        print(f"预测新位置: [{new_pos[0]:.4f}, {new_pos[1]:.4f}, {new_pos[2]:.4f}]")
        
        # 计算实际移动
        actual_movement = new_pos - current_pos
        print(f"实际移动: [{actual_movement[0]:.4f}, {actual_movement[1]:.4f}, {actual_movement[2]:.4f}]")
        
        # 计算误差
        remaining_error = target_pos - new_pos
        error_magnitude = np.linalg.norm(remaining_error)
        print(f"剩余误差: {error_magnitude:.6f}m")
        
    except Exception as e:
        print(f"阻尼最小二乘法计算失败: {e}")

def main():
    """主函数"""
    print("雅可比矩阵测试程序")
    
    # 测试雅可比矩阵
    test_jacobian()
    
    # 测试小幅移动
    test_small_movement()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()