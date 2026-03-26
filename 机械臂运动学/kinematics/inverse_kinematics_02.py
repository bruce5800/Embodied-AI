#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂逆运动学控制代码
基于雅可比矩阵的迭代法，通过逐步逼近求解关节角度
输入：末端夹爪的目标位置 (x, y, z)
输出：6个电机的角度
"""

import numpy as np
import math
from typing import Tuple, List, Optional
import sys
import os

# 导入正运动学类
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from forword_kinematics_01 import GenkiArmForwardKinematics
except ImportError:
    # 如果导入失败，直接在当前文件中重新定义正运动学类
    import numpy as np
    import math
    
    class GenkiArmForwardKinematics:
        def __init__(self):
            """
            初始化机械臂正运动学计算器
            根据URDF文件中的关节参数设置DH参数
            """
            # 根据URDF文件重新分析的关节变换参数
            self.joint_transforms = [
                # 关节1: Base -> yao (腰部旋转，绕X轴)
                {"translation": [-0.013, 0, 0.0265], "rotation": [0, -1.57, 0], "axis": [1, 0, 0]},
                
                # 关节2: yao -> jian1 (大臂，绕Y轴)
                {"translation": [0.081, 0, 0.0], "rotation": [0, 1.57, 0], "axis": [0, 1, 0]},
                
                # 关节3: jian1 -> jian2 (小臂，绕Y轴)
                {"translation": [0, 0, 0.118], "rotation": [0, 0, 0], "axis": [0, 1, 0]},
                
                # 关节4: jian2 -> wan (腕部，绕Y轴)
                {"translation": [0, 0, 0.118], "rotation": [0, 0, 0], "axis": [0, 1, 0]},
                
                # 关节5: wan -> wan2 (腕部旋转，绕Z轴)
                {"translation": [0, 0, 0.0635], "rotation": [0, 0, 0], "axis": [0, 0, 1]},
                
                # 关节6: wan2 -> zhua (夹爪，绕X轴)
                {"translation": [0, -0.0132, 0.021], "rotation": [0, 0, 0], "axis": [1, 0, 0]}
            ]
            
            # 关节限制 [lower, upper] (弧度)
            self.joint_limits = [
                [-1.57, 1.57],  # 关节1
                [-1.57, 1.57],  # 关节2
                [-1.57, 1.57],  # 关节3
                [-1.57, 1.57],  # 关节4
                [-1.57, 1.57],  # 关节5
                [0, 1.57]       # 关节6 (夹爪)
            ]
        
        def rotation_matrix_x(self, angle):
            """绕X轴旋转矩阵"""
            c = math.cos(angle)
            s = math.sin(angle)
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        
        def rotation_matrix_y(self, angle):
            """绕Y轴旋转矩阵"""
            c = math.cos(angle)
            s = math.sin(angle)
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        
        def rotation_matrix_z(self, angle):
            """绕Z轴旋转矩阵"""
            c = math.cos(angle)
            s = math.sin(angle)
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        
        def create_transform_matrix(self, translation, rotation):
            """根据平移和旋转创建4x4变换矩阵"""
            # 创建旋转矩阵
            R_x = self.rotation_matrix_x(rotation[0])  # roll
            R_y = self.rotation_matrix_y(rotation[1])  # pitch  
            R_z = self.rotation_matrix_z(rotation[2])  # yaw
            
            # 按照RPY顺序相乘：R = R_z * R_y * R_x
            R = np.dot(R_z, np.dot(R_y, R_x))
            
            # 创建4x4变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = translation
            
            return T
        
        def create_joint_rotation_matrix(self, axis, angle):
            """根据旋转轴和角度创建旋转矩阵"""
            if axis[0] == 1:  # 绕X轴旋转
                return self.rotation_matrix_x(angle)
            elif axis[1] == 1:  # 绕Y轴旋转
                return self.rotation_matrix_y(angle)
            elif axis[2] == 1:  # 绕Z轴旋转
                return self.rotation_matrix_z(angle)
            else:
                return np.eye(3)  # 无旋转
        
        def forward_kinematics(self, joint_angles):
            """正运动学计算"""
            if len(joint_angles) != 6:
                raise ValueError("需要输入6个关节角度")
            
            # 初始化变换矩阵为单位矩阵
            T = np.eye(4)
            
            # 逐个计算每个关节的变换矩阵并累乘
            for i in range(6):
                joint_transform = self.joint_transforms[i]
                translation = joint_transform["translation"]
                rotation = joint_transform["rotation"]
                axis = joint_transform["axis"]
                
                # 创建固定的平移和旋转变换矩阵（来自URDF的origin）
                T_fixed = self.create_transform_matrix(translation, rotation)
                
                # 创建关节旋转变换矩阵
                joint_rotation = self.create_joint_rotation_matrix(axis, joint_angles[i])
                T_joint = np.eye(4)
                T_joint[:3, :3] = joint_rotation
                
                # 组合变换：先应用固定变换，再应用关节旋转
                T_combined = np.dot(T_fixed, T_joint)
                
                # 累乘变换矩阵
                T = np.dot(T, T_combined)
            
            # 提取末端位置
            end_effector_position = T[:3, 3]
            
            return end_effector_position


class GenkiArmInverseKinematics:
    def __init__(self):
        """
        初始化机械臂逆运动学求解器
        """
        # 使用正运动学类进行前向计算
        self.fk = GenkiArmForwardKinematics()
        
        # 迭代参数
        self.max_iterations = 200       # 减少最大迭代次数
        self.position_tolerance = 0.005 # 放宽位置容差到5mm
        self.step_size = 0.5           # 增加步长
        self.damping_factor = 0.01     # 减小阻尼因子
        
        # 关节角度变化限制 (每次迭代的最大变化量，弧度)
        self.max_joint_change = 0.2    # 增加每次迭代的最大关节变化量
        
        # 工作空间限制 (米)
        self.workspace_limits = {
            'x': [-0.3, 0.3],
            'y': [-0.3, 0.3], 
            'z': [0.1, 0.6]
        }
    
    def compute_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        计算雅可比矩阵
        使用数值微分方法计算每个关节对末端位置的影响
        
        Args:
            joint_angles: 当前关节角度列表 (弧度)
            
        Returns:
            jacobian: 3x6 雅可比矩阵 (位置部分)
        """
        jacobian = np.zeros((3, 6))
        delta = 1e-4  # 增大微小变化量，提高数值稳定性
        
        # 计算当前末端位置
        current_pos = np.array(self.fk.forward_kinematics(joint_angles))
        
        # 对每个关节计算偏导数
        for i in range(6):
            # 创建扰动后的关节角度
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += delta
            
            # 确保关节角度在限制范围内
            perturbed_angles = self.clamp_joint_angles(perturbed_angles)
            
            # 计算扰动后的末端位置
            perturbed_pos = np.array(self.fk.forward_kinematics(perturbed_angles))
            
            # 计算偏导数 (数值微分)
            jacobian[:, i] = (perturbed_pos - current_pos) / delta
        
        return jacobian
    
    def check_workspace_limits(self, target_position: List[float]) -> bool:
        """
        检查目标位置是否在工作空间内
        
        Args:
            target_position: 目标位置 [x, y, z]
            
        Returns:
            bool: 是否在工作空间内
        """
        x, y, z = target_position
        
        if (self.workspace_limits['x'][0] <= x <= self.workspace_limits['x'][1] and
            self.workspace_limits['y'][0] <= y <= self.workspace_limits['y'][1] and
            self.workspace_limits['z'][0] <= z <= self.workspace_limits['z'][1]):
            return True
        else:
            return False
    
    def clamp_joint_angles(self, joint_angles: List[float]) -> List[float]:
        """
        将关节角度限制在允许范围内
        
        Args:
            joint_angles: 关节角度列表
            
        Returns:
            clamped_angles: 限制后的关节角度列表
        """
        clamped_angles = []
        for i, angle in enumerate(joint_angles):
            lower, upper = self.fk.joint_limits[i]
            clamped_angle = max(lower, min(upper, angle))
            clamped_angles.append(clamped_angle)
        
        return clamped_angles
    
    def inverse_kinematics(self, target_position: List[float], 
                          initial_guess: Optional[List[float]] = None,
                          verbose: bool = False) -> Tuple[List[float], bool, dict]:
        """
        逆运动学求解主函数
        使用阻尼最小二乘法 (Damped Least Squares) 迭代求解
        
        Args:
            target_position: 目标位置 [x, y, z] (米)
            initial_guess: 初始关节角度猜测 (弧度)，如果为None则使用零位
            verbose: 是否打印详细信息
            
        Returns:
            joint_angles: 求解得到的关节角度 (弧度)
            success: 是否成功求解
            info: 求解信息字典
        """
        # 检查目标位置是否在工作空间内
        if not self.check_workspace_limits(target_position):
            if verbose:
                print(f"警告: 目标位置 {target_position} 可能超出工作空间范围")
        
        # 初始化关节角度
        if initial_guess is None:
            current_angles = [0.0] * 6  # 从零位开始
        else:
            current_angles = initial_guess.copy()
        
        # 确保初始角度在限制范围内
        current_angles = self.clamp_joint_angles(current_angles)
        
        target_pos = np.array(target_position)
        
        # 迭代求解
        for iteration in range(self.max_iterations):
            # 计算当前末端位置
            current_pos = self.fk.forward_kinematics(current_angles)
            current_pos = np.array(current_pos)
            
            # 计算位置误差
            position_error = target_pos - current_pos
            error_magnitude = np.linalg.norm(position_error)
            
            if verbose and iteration % 20 == 0:
                print(f"迭代 {iteration}: 位置误差 = {error_magnitude:.6f}m")
                print(f"  当前位置: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
                print(f"  目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            
            # 检查收敛条件
            if error_magnitude < self.position_tolerance:
                if verbose:
                    print(f"收敛成功! 迭代次数: {iteration}, 最终误差: {error_magnitude:.6f}m")
                
                return (current_angles, True, {
                    'iterations': iteration,
                    'final_error': error_magnitude,
                    'final_position': current_pos.tolist()
                })
            
            # 计算雅可比矩阵
            jacobian = self.compute_jacobian(current_angles)
            
            # 检查雅可比矩阵的有效性
            jacobian_norm = np.linalg.norm(jacobian)
            if jacobian_norm < 1e-8:
                if verbose:
                    print(f"警告: 雅可比矩阵接近零，可能处于奇异点")
                break
            
            # 检查奇异点 (雅可比矩阵的条件数)
            try:
                # 使用阻尼最小二乘法求解
                # Δθ = J^T * (J*J^T + λ*I)^(-1) * Δx
                JJT = np.dot(jacobian, jacobian.T)
                
                # 自适应阻尼因子
                cond_num = np.linalg.cond(JJT)
                adaptive_damping = self.damping_factor
                if cond_num > 1e6:  # 条件数过大时增加阻尼
                    adaptive_damping = self.damping_factor * (cond_num / 1e6)
                
                damped_inverse = np.linalg.inv(JJT + adaptive_damping * np.eye(3))
                delta_angles = self.step_size * np.dot(jacobian.T, np.dot(damped_inverse, position_error))
                
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"警告: 在迭代 {iteration} 遇到奇异点，增加阻尼因子")
                # 增加阻尼因子重试
                JJT = np.dot(jacobian, jacobian.T)
                damped_inverse = np.linalg.inv(JJT + (self.damping_factor * 100) * np.eye(3))
                delta_angles = self.step_size * np.dot(jacobian.T, np.dot(damped_inverse, position_error))
            
            # 限制每次迭代的关节角度变化量
            delta_angles = np.clip(delta_angles, -self.max_joint_change, self.max_joint_change)
            
            # 更新关节角度
            new_angles = [current_angles[i] + delta_angles[i] for i in range(6)]
            
            # 应用关节限制
            new_angles = self.clamp_joint_angles(new_angles)
            
            current_angles = new_angles
        
        # 达到最大迭代次数仍未收敛
        final_pos = self.fk.forward_kinematics(current_angles)
        final_error = np.linalg.norm(target_pos - np.array(final_pos))
        
        if verbose:
            print(f"未收敛: 达到最大迭代次数 {self.max_iterations}")
            print(f"最终误差: {final_error:.6f}m")
        
        return (current_angles, False, {
            'iterations': self.max_iterations,
            'final_error': final_error,
            'final_position': final_pos
        })
    
    def inverse_kinematics_multiple_attempts(self, target_position: List[float],
                                           num_attempts: int = 5,
                                           verbose: bool = False) -> Tuple[List[float], bool, dict]:
        """
        多次尝试逆运动学求解，使用不同的初始猜测
        
        Args:
            target_position: 目标位置 [x, y, z]
            num_attempts: 尝试次数
            verbose: 是否打印详细信息
            
        Returns:
            最佳求解结果
        """
        best_result = None
        best_error = float('inf')
        
        for attempt in range(num_attempts):
            if verbose:
                print(f"\n=== 尝试 {attempt + 1}/{num_attempts} ===")
            
            # 生成随机初始猜测
            if attempt == 0:
                # 第一次尝试使用零位
                initial_guess = [0.0] * 6
            else:
                # 后续尝试使用随机初始位置
                initial_guess = []
                for i in range(6):
                    lower, upper = self.fk.joint_limits[i]
                    random_angle = np.random.uniform(lower, upper)
                    initial_guess.append(random_angle)
            
            # 求解
            angles, success, info = self.inverse_kinematics(
                target_position, initial_guess, verbose=verbose
            )
            
            # 记录最佳结果
            if info['final_error'] < best_error:
                best_error = info['final_error']
                best_result = (angles, success, info)
                
                if success:
                    if verbose:
                        print(f"找到成功解! 误差: {best_error:.6f}m")
                    break
        
        return best_result
    
    def degrees_to_radians(self, angles_deg: List[float]) -> List[float]:
        """角度转弧度"""
        return [math.radians(angle) for angle in angles_deg]
    
    def radians_to_degrees(self, angles_rad: List[float]) -> List[float]:
        """弧度转角度"""
        return [math.degrees(angle) for angle in angles_rad]
    
    def verify_solution(self, joint_angles: List[float], target_position: List[float]) -> dict:
        """
        验证求解结果的准确性
        
        Args:
            joint_angles: 求解得到的关节角度
            target_position: 目标位置
            
        Returns:
            验证结果字典
        """
        # 使用正运动学计算实际末端位置
        actual_position = self.fk.forward_kinematics(joint_angles)
        
        # 计算位置误差
        position_error = np.array(target_position) - np.array(actual_position)
        error_magnitude = np.linalg.norm(position_error)
        
        return {
            'target_position': target_position,
            'actual_position': actual_position.tolist(),
            'position_error': position_error.tolist(),
            'error_magnitude': error_magnitude,
            'joint_angles_deg': self.radians_to_degrees(joint_angles),
            'joint_angles_rad': joint_angles
        }


def main():
    """
    主函数 - 演示逆运动学求解
    """
    # 创建逆运动学求解器
    ik = GenkiArmInverseKinematics()
    
    print("机械臂逆运动学求解器")
    print("=" * 50)
    
    # 测试用例1: 简单位置
    print("测试用例1: 目标位置 [0.2, 0.1, 0.3]")
    target_pos = [0.2, 0.1, 0.3]
    
    joint_angles, success, info = ik.inverse_kinematics_multiple_attempts(
        target_pos, num_attempts=3, verbose=True
    )
    
    if success:
        print(f"\n✓ 求解成功!")
        verification = ik.verify_solution(joint_angles, target_pos)
        print(f"关节角度 (度): {[f'{angle:.2f}' for angle in verification['joint_angles_deg']]}")
        print(f"实际位置: [{verification['actual_position'][0]:.4f}, {verification['actual_position'][1]:.4f}, {verification['actual_position'][2]:.4f}]")
        print(f"位置误差: {verification['error_magnitude']:.6f}m")
    else:
        print(f"\n✗ 求解失败，最终误差: {info['final_error']:.6f}m")
    
    print("\n" + "=" * 50)
    
    # 测试用例2: 更复杂的位置
    print("测试用例2: 目标位置 [0.15, 0.15, 0.25]")
    target_pos = [0.15, 0.15, 0.25]
    
    joint_angles, success, info = ik.inverse_kinematics_multiple_attempts(
        target_pos, num_attempts=3, verbose=False
    )
    
    if success:
        print(f"✓ 求解成功!")
        verification = ik.verify_solution(joint_angles, target_pos)
        print(f"关节角度 (度): {[f'{angle:.2f}' for angle in verification['joint_angles_deg']]}")
        print(f"实际位置: [{verification['actual_position'][0]:.4f}, {verification['actual_position'][1]:.4f}, {verification['actual_position'][2]:.4f}]")
        print(f"位置误差: {verification['error_magnitude']:.6f}m")
    else:
        print(f"✗ 求解失败，最终误差: {info['final_error']:.6f}m")
    
    # 交互式输入
    print("\n" + "=" * 50)
    print("交互式逆运动学求解")
    print("请输入目标位置 (x, y, z)，单位：米")
    print("建议范围: x[-0.3, 0.3], y[-0.3, 0.3], z[0.1, 0.5]")
    
    try:
        user_input = input("目标位置 (x y z): ")
        target_coords = list(map(float, user_input.split()))
        
        if len(target_coords) != 3:
            print("错误: 需要输入3个坐标值")
            return
        
        print(f"\n正在求解目标位置: [{target_coords[0]:.3f}, {target_coords[1]:.3f}, {target_coords[2]:.3f}]")
        
        joint_angles, success, info = ik.inverse_kinematics_multiple_attempts(
            target_coords, num_attempts=5, verbose=True
        )
        
        print(f"\n{'='*50}")
        print("求解结果:")
        
        if success:
            print(f"✓ 求解成功! (迭代次数: {info['iterations']})")
            verification = ik.verify_solution(joint_angles, target_coords)
            
            print(f"\n关节角度:")
            joint_names = ["腰部旋转", "大臂", "小臂", "腕部", "腕部旋转", "夹爪"]
            for i, (name, angle_deg) in enumerate(zip(joint_names, verification['joint_angles_deg'])):
                print(f"  关节{i+1} ({name}): {angle_deg:.2f}°")
            
            print(f"\n位置验证:")
            print(f"  目标位置: [{verification['target_position'][0]:.4f}, {verification['target_position'][1]:.4f}, {verification['target_position'][2]:.4f}]")
            print(f"  实际位置: [{verification['actual_position'][0]:.4f}, {verification['actual_position'][1]:.4f}, {verification['actual_position'][2]:.4f}]")
            print(f"  位置误差: {verification['error_magnitude']:.6f}m")
            
        else:
            print(f"✗ 求解失败")
            print(f"  最终误差: {info['final_error']:.6f}m")
            print(f"  可能原因: 目标位置超出工作空间或接近奇异点")
        
    except ValueError as e:
        print(f"输入错误: {e}")
    except KeyboardInterrupt:
        print("\n程序退出")


if __name__ == "__main__":
    main()