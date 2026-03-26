#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂正运动学控制代码
基于URDF文件中的机械臂结构，使用矩阵运算计算末端夹爪位置
输入：6个电机的角度
输出：末端夹爪的x, y, z坐标
"""

import numpy as np
import math

class GenkiArmForwardKinematics:
    def __init__(self):
        """
        初始化机械臂正运动学计算器
        根据URDF文件中的关节参数设置DH参数
        """
        # 根据URDF文件提取的关节参数
        # 关节1 (腰部旋转): Base -> yao
        # 关节2 (大臂): yao -> jian1  
        # 关节3 (小臂): jian1 -> jian2
        # 关节4 (腕部): jian2 -> wan
        # 关节5 (腕部旋转): wan -> wan2
        # 关节6 (夹爪): wan2 -> zhua
        
        # 根据URDF文件重新分析的关节变换参数
        # 每个关节的变换包括：平移 + 旋转
        # 存储格式：[translation_xyz, rotation_rpy, axis_xyz]
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
        """
        绕X轴旋转矩阵
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def rotation_matrix_y(self, angle):
        """
        绕Y轴旋转矩阵
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def rotation_matrix_z(self, angle):
        """
        绕Z轴旋转矩阵
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def create_transform_matrix(self, translation, rotation):
        """
        根据平移和旋转创建4x4变换矩阵
        translation: [x, y, z] 平移向量
        rotation: [roll, pitch, yaw] 旋转角度（弧度）
        """
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
        """
        根据旋转轴和角度创建旋转矩阵
        axis: [x, y, z] 旋转轴向量
        angle: 旋转角度（弧度）
        """
        if axis[0] == 1:  # 绕X轴旋转
            return self.rotation_matrix_x(angle)
        elif axis[1] == 1:  # 绕Y轴旋转
            return self.rotation_matrix_y(angle)
        elif axis[2] == 1:  # 绕Z轴旋转
            return self.rotation_matrix_z(angle)
        else:
            return np.eye(3)  # 无旋转
    
    def check_joint_limits(self, joint_angles):
        """
        检查关节角度是否在限制范围内
        """
        if len(joint_angles) != 6:
            raise ValueError("需要输入6个关节角度")
        
        for i, angle in enumerate(joint_angles):
            lower, upper = self.joint_limits[i]
            if angle < lower or angle > upper:
                print(f"警告: 关节{i+1}角度 {angle:.3f} 超出限制范围 [{lower:.3f}, {upper:.3f}]")
    
    def forward_kinematics(self, joint_angles):
        """
        正运动学计算
        输入: joint_angles - 6个关节角度的列表 (弧度)
        输出: 末端夹爪的位置 [x, y, z]
        """
        # 检查输入
        if len(joint_angles) != 6:
            raise ValueError("需要输入6个关节角度")
        
        # 检查关节限制
        self.check_joint_limits(joint_angles)
        
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
    
    def forward_kinematics_with_orientation(self, joint_angles):
        """
        正运动学计算（包含姿态）
        输入: joint_angles - 6个关节角度的列表 (弧度)
        输出: 末端夹爪的位置和姿态矩阵
        """
        # 检查输入
        if len(joint_angles) != 6:
            raise ValueError("需要输入6个关节角度")
        
        # 检查关节限制
        self.check_joint_limits(joint_angles)
        
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
        
        # 提取末端位置和姿态
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        
        return position, rotation_matrix, T
    
    def degrees_to_radians(self, angles_deg):
        """
        角度转弧度
        """
        return [math.radians(angle) for angle in angles_deg]
    
    def print_joint_info(self):
        """
        打印关节信息
        """
        joint_names = ["腰部旋转", "大臂", "小臂", "腕部", "腕部旋转", "夹爪"]
        print("机械臂关节信息:")
        for i, name in enumerate(joint_names):
            lower, upper = self.joint_limits[i]
            print(f"关节{i+1} ({name}): 限制范围 [{math.degrees(lower):.1f}°, {math.degrees(upper):.1f}°]")


def main():
    """
    主函数 - 演示正运动学计算
    """
    # 创建正运动学计算器
    fk = GenkiArmForwardKinematics()
    
    # 打印关节信息
    fk.print_joint_info()
    print()
    
    # 测试用例1: 所有关节角度为0
    print("测试用例1: 所有关节角度为0度")
    joint_angles_deg = [0, 0, 0, 0, 0, 0]
    joint_angles_rad = fk.degrees_to_radians(joint_angles_deg)
    
    position = fk.forward_kinematics(joint_angles_rad)
    print(f"输入关节角度 (度): {joint_angles_deg}")
    print(f"末端夹爪位置: x={position[0]:.4f}m, y={position[1]:.4f}m, z={position[2]:.4f}m")
    print()
    
    # 测试用例2: 部分关节有角度
    print("测试用例2: 腰部旋转45度，大臂抬起30度")
    joint_angles_deg = [45, 30, 0, 0, 0, 0]
    joint_angles_rad = fk.degrees_to_radians(joint_angles_deg)
    
    position = fk.forward_kinematics(joint_angles_rad)
    print(f"输入关节角度 (度): {joint_angles_deg}")
    print(f"末端夹爪位置: x={position[0]:.4f}m, y={position[1]:.4f}m, z={position[2]:.4f}m")
    print()
    
    # 测试用例3: 复杂姿态
    print("测试用例3: 复杂姿态")
    joint_angles_deg = [30, 45, -30, 60, 90, 45]
    joint_angles_rad = fk.degrees_to_radians(joint_angles_deg)
    
    position, rotation, transform = fk.forward_kinematics_with_orientation(joint_angles_rad)
    print(f"输入关节角度 (度): {joint_angles_deg}")
    print(f"末端夹爪位置: x={position[0]:.4f}m, y={position[1]:.4f}m, z={position[2]:.4f}m")
    print(f"末端姿态矩阵:")
    print(rotation)
    print()
    
    # 交互式输入
    print("=" * 50)
    print("交互式正运动学计算")
    print("请输入6个关节角度 (度)，用空格分隔:")
    print("关节顺序: 腰部旋转 大臂 小臂 腕部 腕部旋转 夹爪")
    
    try:
        user_input = input("关节角度: ")
        angles_deg = list(map(float, user_input.split()))
        
        if len(angles_deg) != 6:
            print("错误: 需要输入6个角度值")
            return
        
        angles_rad = fk.degrees_to_radians(angles_deg)
        position = fk.forward_kinematics(angles_rad)
        
        print(f"\n计算结果:")
        print(f"输入关节角度 (度): {angles_deg}")
        print(f"末端夹爪位置: x={position[0]:.4f}m, y={position[1]:.4f}m, z={position[2]:.4f}m")
        
    except ValueError as e:
        print(f"输入错误: {e}")
    except KeyboardInterrupt:
        print("\n程序退出")


if __name__ == "__main__":
    main()