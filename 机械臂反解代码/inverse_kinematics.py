#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6关节机械臂运动学逆解计算
基于正解代码，使用迭代逼近方法计算关节角度
采用牛顿-拉夫逊迭代法求解逆运动学问题

关节说明：
1. Rotation (腰部旋转) - 绕X轴旋转
2. Rotation2 (大臂控制) - 绕Y轴旋转  
3. Rotation3 (小臂控制) - 绕Y轴旋转
4. Rotation4 (腕部控制) - 绕Y轴旋转
5. Rotation5 (腕部旋转) - 绕Z轴旋转
6. Rotation6 (夹爪控制) - 绕X轴旋转
"""

import numpy as np
import math
from forward_kinematics import RobotArmKinematics

class RobotArmInverseKinematics:
    def __init__(self):
        """
        初始化机械臂逆运动学求解器
        """
        # 使用正解类进行计算
        self.forward_kinematics = RobotArmKinematics()
        
        # 关节角度限制 (弧度)
        self.joint_limits = {
            'joint1': [-1.57, 1.57],   # ±90度
            'joint2': [-1.57, 1.57],   # ±90度
            'joint3': [-1.57, 1.57],   # ±90度
            'joint4': [-1.57, 1.57],   # ±90度
            'joint5': [-1.57, 1.57],   # ±90度
            'joint6': [0, 1.57]        # 0-90度
        }
        
        # 迭代参数
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.step_size = 0.1
        
    def compute_jacobian(self, joint_angles):
        """
        计算雅可比矩阵 (数值微分方法)
        
        Args:
            joint_angles: 当前关节角度 [θ1, θ2, θ3, θ4, θ5, θ6]
        
        Returns:
            3x6雅可比矩阵 (位置部分)
        """
        jacobian = np.zeros((3, 6))
        delta = 1e-6  # 微小增量
        
        # 计算当前位置
        current_pos = self.forward_kinematics.forward_kinematics(joint_angles)
        
        # 对每个关节计算偏导数
        for i in range(6):
            # 创建增量角度
            delta_angles = joint_angles.copy()
            delta_angles[i] += delta
            
            # 计算增量后的位置
            new_pos = self.forward_kinematics.forward_kinematics(delta_angles)
            
            # 计算偏导数 (数值微分)
            jacobian[:, i] = (new_pos - current_pos) / delta
        
        return jacobian
    
    def clamp_joint_angles(self, joint_angles):
        """
        将关节角度限制在允许范围内
        
        Args:
            joint_angles: 关节角度列表
        
        Returns:
            限制后的关节角度列表
        """
        clamped_angles = joint_angles.copy()
        
        for i, (joint_name, limits) in enumerate(self.joint_limits.items()):
            if i < len(clamped_angles):
                clamped_angles[i] = np.clip(clamped_angles[i], limits[0], limits[1])
        
        return clamped_angles
    
    def inverse_kinematics(self, target_position, initial_guess=None):
        """
        使用迭代逼近方法计算逆运动学
        
        Args:
            target_position: 目标位置 [x, y, z]
            initial_guess: 初始关节角度猜测值，如果为None则使用零位
        
        Returns:
            tuple: (joint_angles, success, iterations, final_error)
                - joint_angles: 计算得到的关节角度
                - success: 是否成功收敛
                - iterations: 迭代次数
                - final_error: 最终误差
        """
        # 初始化关节角度
        if initial_guess is None:
            joint_angles = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            joint_angles = np.array(initial_guess)
        
        target_pos = np.array(target_position)
        
        for iteration in range(self.max_iterations):
            # 计算当前末端位置
            current_pos = self.forward_kinematics.forward_kinematics(joint_angles)
            
            # 计算位置误差
            error = target_pos - current_pos
            error_magnitude = np.linalg.norm(error)
            
            # 检查收敛条件
            if error_magnitude < self.tolerance:
                return joint_angles, True, iteration + 1, error_magnitude
            
            # 计算雅可比矩阵
            jacobian = self.compute_jacobian(joint_angles)
            
            # 检查雅可比矩阵是否奇异
            try:
                # 使用伪逆求解
                jacobian_pinv = np.linalg.pinv(jacobian)
                
                # 计算关节角度增量
                delta_angles = self.step_size * jacobian_pinv @ error
                
                # 更新关节角度
                joint_angles += delta_angles
                
                # 限制关节角度在允许范围内
                joint_angles = self.clamp_joint_angles(joint_angles)
                
            except np.linalg.LinAlgError:
                # 雅可比矩阵奇异，尝试添加阻尼
                damping = 1e-6
                jacobian_damped = jacobian.T @ jacobian + damping * np.eye(6)
                try:
                    delta_angles = self.step_size * np.linalg.solve(jacobian_damped, jacobian.T @ error)
                    joint_angles += delta_angles
                    joint_angles = self.clamp_joint_angles(joint_angles)
                except np.linalg.LinAlgError:
                    # 如果仍然失败，返回当前结果
                    break
        
        # 计算最终误差
        final_pos = self.forward_kinematics.forward_kinematics(joint_angles)
        final_error = np.linalg.norm(target_pos - final_pos)
        
        return joint_angles, False, self.max_iterations, final_error
    
    def inverse_kinematics_with_multiple_attempts(self, target_position, num_attempts=5):
        """
        使用多个不同的初始猜测值尝试逆解
        
        Args:
            target_position: 目标位置 [x, y, z]
            num_attempts: 尝试次数
        
        Returns:
            最佳解的结果 (joint_angles, success, iterations, final_error)
        """
        best_result = None
        best_error = float('inf')
        
        for attempt in range(num_attempts):
            # 生成随机初始猜测值
            if attempt == 0:
                # 第一次尝试使用零位
                initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                # 其他尝试使用随机初始值
                initial_guess = []
                for joint_name, limits in self.joint_limits.items():
                    random_angle = np.random.uniform(limits[0], limits[1])
                    initial_guess.append(random_angle)
            
            # 尝试求解
            result = self.inverse_kinematics(target_position, initial_guess)
            joint_angles, success, iterations, final_error = result
            
            # 如果成功收敛，直接返回
            if success:
                return result
            
            # 记录最佳结果
            if final_error < best_error:
                best_error = final_error
                best_result = result
        
        return best_result if best_result else ([0]*6, False, 0, float('inf'))


def main():
    """
    测试逆运动学求解
    """
    # 创建逆运动学求解器
    inverse_solver = RobotArmInverseKinematics()
    
    print("=== 机械臂逆运动学测试 ===")
    
    # 测试用例1: 使用正解的已知结果进行验证
    print("\n=== 测试用例1: 验证逆解正确性 ===")
    
    # 先用正解计算一个位置
    test_angles = [0.5, 0.3, -0.2, 0.4, 0.6, 0.1]
    forward_solver = RobotArmKinematics()
    target_pos = forward_solver.forward_kinematics(test_angles)
    
    print(f"原始关节角度: {[f'{angle:.4f}' for angle in test_angles]}")
    print(f"目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
    
    # 使用逆解求解
    result_angles, success, iterations, error = inverse_solver.inverse_kinematics_with_multiple_attempts(target_pos)
    
    print(f"逆解结果: {[f'{angle:.4f}' for angle in result_angles]}")
    print(f"收敛状态: {'成功' if success else '失败'}")
    print(f"迭代次数: {iterations}")
    print(f"最终误差: {error:.6f}")
    
    # 验证逆解结果
    verify_pos = forward_solver.forward_kinematics(result_angles)
    print(f"验证位置: [{verify_pos[0]:.4f}, {verify_pos[1]:.4f}, {verify_pos[2]:.4f}]")
    position_error = np.linalg.norm(target_pos - verify_pos)
    print(f"位置误差: {position_error:.6f}")
    
    # 测试用例2: 指定目标位置
    print("\n=== 测试用例2: 指定目标位置 ===")
    target_positions = [
        [0.2, 0.1, 0.3],
        [0.15, 0.0, 0.25],
        [0.1, -0.1, 0.2],
        [-0.0129, -0.0132, 0.4280],
    ]
    
    for i, target_pos in enumerate(target_positions):
        print(f"\n--- 目标位置 {i+1}: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] ---")
        
        result_angles, success, iterations, error = inverse_solver.inverse_kinematics_with_multiple_attempts(target_pos)
        
        print(f"求解结果: {[f'{angle:.4f}' for angle in result_angles]}")
        print(f"收敛状态: {'成功' if success else '失败'}")
        print(f"迭代次数: {iterations}")
        print(f"最终误差: {error:.6f}")
        
        # 验证结果
        verify_pos = forward_solver.forward_kinematics(result_angles)
        position_error = np.linalg.norm(np.array(target_pos) - verify_pos)
        print(f"位置验证误差: {position_error:.6f}")


if __name__ == "__main__":
    main()