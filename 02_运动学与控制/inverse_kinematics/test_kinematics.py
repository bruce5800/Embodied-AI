#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试正运动学和逆运动学的计算
"""

import numpy as np
import math
from forword_kinematics_01 import GenkiArmForwardKinematics
from inverse_kinematics_02 import GenkiArmInverseKinematics

def test_forward_kinematics():
    """测试正运动学"""
    print("=" * 50)
    print("测试正运动学")
    print("=" * 50)
    
    fk = GenkiArmForwardKinematics()
    
    # 测试零位
    zero_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"零位关节角度: {zero_angles}")
    
    try:
        position = fk.forward_kinematics(zero_angles)
        print(f"零位末端位置: {position}")
        print(f"零位末端位置 (mm): [{position[0]*1000:.1f}, {position[1]*1000:.1f}, {position[2]*1000:.1f}]")
    except Exception as e:
        print(f"正运动学计算失败: {e}")
        return None
    
    # 测试一些典型位置
    test_angles = [
        [0.0, 0.5, -0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    ]
    
    for i, angles in enumerate(test_angles):
        try:
            position = fk.forward_kinematics(angles)
            print(f"测试{i+1} - 关节角度: {[f'{a:.3f}' for a in angles]}")
            print(f"测试{i+1} - 末端位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        except Exception as e:
            print(f"测试{i+1} 正运动学计算失败: {e}")
    
    return position

def test_inverse_kinematics():
    """测试逆运动学"""
    print("\n" + "=" * 50)
    print("测试逆运动学")
    print("=" * 50)
    
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 首先计算零位的末端位置
    zero_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        zero_position = fk.forward_kinematics(zero_angles)
        print(f"零位末端位置: [{zero_position[0]:.4f}, {zero_position[1]:.4f}, {zero_position[2]:.4f}]")
    except Exception as e:
        print(f"无法计算零位: {e}")
        return
    
    # 测试逆运动学求解零位
    print("\n测试逆运动学求解零位...")
    try:
        joint_angles, success, info = ik.inverse_kinematics(zero_position, verbose=True)
        print(f"逆运动学求解结果:")
        print(f"  成功: {success}")
        print(f"  关节角度: {[f'{a:.4f}' for a in joint_angles]}")
        print(f"  迭代次数: {info.get('iterations', 'N/A')}")
        print(f"  最终误差: {info.get('final_error', 'N/A')}")
        
        # 验证解的正确性
        if success:
            verify_position = fk.forward_kinematics(joint_angles)
            error = np.linalg.norm(np.array(verify_position) - np.array(zero_position))
            print(f"  验证位置: [{verify_position[0]:.4f}, {verify_position[1]:.4f}, {verify_position[2]:.4f}]")
            print(f"  验证误差: {error:.6f}m")
            
    except Exception as e:
        print(f"逆运动学求解失败: {e}")
    
    # 测试一些简单的目标位置
    test_targets = [
        [0.0, 0.0, 0.35],  # 向上移动5cm
        [0.05, 0.0, 0.30], # 向右移动5cm
        [0.0, 0.05, 0.30], # 向前移动5cm
    ]
    
    for i, target in enumerate(test_targets):
        print(f"\n测试目标位置 {i+1}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
        try:
            joint_angles, success, info = ik.inverse_kinematics(target, verbose=False)
            print(f"  求解成功: {success}")
            print(f"  最终误差: {info.get('final_error', 'N/A'):.6f}m")
            
            if success:
                # 验证解
                verify_position = fk.forward_kinematics(joint_angles)
                error = np.linalg.norm(np.array(verify_position) - np.array(target))
                print(f"  验证误差: {error:.6f}m")
                
        except Exception as e:
            print(f"  求解失败: {e}")

def test_step_movement():
    """测试步进移动"""
    print("\n" + "=" * 50)
    print("测试步进移动")
    print("=" * 50)
    
    fk = GenkiArmForwardKinematics()
    ik = GenkiArmInverseKinematics()
    
    # 从零位开始
    current_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current_position = fk.forward_kinematics(current_angles)
    
    print(f"起始位置: [{current_position[0]:.4f}, {current_position[1]:.4f}, {current_position[2]:.4f}]")
    
    # 测试向上移动10mm
    step = 0.01  # 10mm
    target_position = current_position.copy()
    target_position[2] += step
    
    print(f"目标位置 (向上10mm): [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]")
    
    try:
        joint_angles, success, info = ik.inverse_kinematics(target_position, current_angles, verbose=True)
        print(f"求解结果: 成功={success}, 误差={info.get('final_error', 'N/A')}")
        
        if success:
            # 验证
            verify_position = fk.forward_kinematics(joint_angles)
            error = np.linalg.norm(np.array(verify_position) - np.array(target_position))
            print(f"验证位置: [{verify_position[0]:.4f}, {verify_position[1]:.4f}, {verify_position[2]:.4f}]")
            print(f"验证误差: {error:.6f}m")
        
    except Exception as e:
        print(f"步进移动测试失败: {e}")

def main():
    """主函数"""
    print("机械臂运动学测试程序")
    
    # 测试正运动学
    test_forward_kinematics()
    
    # 测试逆运动学
    test_inverse_kinematics()
    
    # 测试步进移动
    test_step_movement()
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()