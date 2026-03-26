#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6关节机械臂运动学正解计算
基于URDF文件中的关节参数，使用齐次变换矩阵计算末端执行器位置

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

class RobotArmKinematics:
    def __init__(self):
        """
        初始化机械臂运动学参数
        根据URDF文件中的关节参数设置
        """
        # 从URDF文件提取的关节变换参数
        # 格式: [x, y, z, roll, pitch, yaw]
        self.joint_transforms = {
            'base_to_yao': [-0.013, 0, 0.0265, 0, -1.57, 0],
            'yao_to_jian1': [0.081, 0, 0.0, 0, 1.57, 0],
            'jian1_to_jian2': [0, 0, 0.118, 0, 0, 0],
            'jian2_to_wan': [0, 0, 0.118, 0, 0, 0],
            'wan_to_wan2': [0, 0, 0.0635, 0, 0, 0],
            'wan2_to_zhua': [0, -0.0132, 0.021, 0, 0, 0]
        }
        
        # 关节旋转轴定义
        self.joint_axes = {
            'joint1': [1, 0, 0],  # X轴
            'joint2': [0, 1, 0],  # Y轴
            'joint3': [0, 1, 0],  # Y轴
            'joint4': [0, 1, 0],  # Y轴
            'joint5': [0, 0, 1],  # Z轴
            'joint6': [1, 0, 0]   # X轴
        }
    
    def create_rotation_matrix(self, axis, angle):
        """
        创建绕指定轴旋转的旋转矩阵
        
        Args:
            axis: 旋转轴 [x, y, z]
            angle: 旋转角度 (弧度)
        
        Returns:
            3x3旋转矩阵
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        if axis == [1, 0, 0]:  # 绕X轴旋转
            return np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        elif axis == [0, 1, 0]:  # 绕Y轴旋转
            return np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif axis == [0, 0, 1]:  # 绕Z轴旋转
            return np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError("不支持的旋转轴")
    
    def create_transform_matrix(self, translation, rotation_rpy, joint_angle=0, joint_axis=None):
        """
        创建4x4齐次变换矩阵
        
        Args:
            translation: 平移向量 [x, y, z]
            rotation_rpy: 固定旋转 [roll, pitch, yaw]
            joint_angle: 关节角度 (弧度)
            joint_axis: 关节旋转轴 [x, y, z]
        
        Returns:
            4x4齐次变换矩阵
        """
        # 创建平移矩阵
        T = np.eye(4)
        T[0:3, 3] = translation
        
        # 创建固定旋转矩阵 (RPY)
        R_fixed = np.eye(3)
        if rotation_rpy[0] != 0:  # Roll (绕X轴)
            R_x = self.create_rotation_matrix([1, 0, 0], rotation_rpy[0])
            R_fixed = R_fixed @ R_x
        if rotation_rpy[1] != 0:  # Pitch (绕Y轴)
            R_y = self.create_rotation_matrix([0, 1, 0], rotation_rpy[1])
            R_fixed = R_fixed @ R_y
        if rotation_rpy[2] != 0:  # Yaw (绕Z轴)
            R_z = self.create_rotation_matrix([0, 0, 1], rotation_rpy[2])
            R_fixed = R_fixed @ R_z
        
        # 添加关节旋转
        if joint_axis is not None and joint_angle != 0:
            R_joint = self.create_rotation_matrix(joint_axis, joint_angle)
            R_total = R_fixed @ R_joint
        else:
            R_total = R_fixed
        
        T[0:3, 0:3] = R_total
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        计算机械臂运动学正解
        
        Args:
            joint_angles: 6个关节角度列表 [θ1, θ2, θ3, θ4, θ5, θ6] (弧度)
        
        Returns:
            末端执行器位置 [x, y, z]
        """
        if len(joint_angles) != 6:
            raise ValueError("需要输入6个关节角度")
        
        # 初始化为单位矩阵
        T_total = np.eye(4)
        
        # 1. Base到yao的变换 (关节1: 腰部旋转)
        trans = self.joint_transforms['base_to_yao']
        T1 = self.create_transform_matrix(
            translation=trans[0:3],
            rotation_rpy=trans[3:6],
            joint_angle=joint_angles[0],
            joint_axis=self.joint_axes['joint1']
        )
        T_total = T_total @ T1
        
        # 2. yao到jian1的变换 (关节2: 大臂控制)
        trans = self.joint_transforms['yao_to_jian1']
        T2 = self.create_transform_matrix(
            translation=trans[0:3],
            rotation_rpy=trans[3:6],
            joint_angle=joint_angles[1],
            joint_axis=self.joint_axes['joint2']
        )
        T_total = T_total @ T2
        
        # 3. jian1到jian2的变换 (关节3: 小臂控制)
        trans = self.joint_transforms['jian1_to_jian2']
        T3 = self.create_transform_matrix(
            translation=trans[0:3],
            rotation_rpy=trans[3:6],
            joint_angle=joint_angles[2],
            joint_axis=self.joint_axes['joint3']
        )
        T_total = T_total @ T3
        
        # 4. jian2到wan的变换 (关节4: 腕部控制)
        trans = self.joint_transforms['jian2_to_wan']
        T4 = self.create_transform_matrix(
            translation=trans[0:3],
            rotation_rpy=trans[3:6],
            joint_angle=joint_angles[3],
            joint_axis=self.joint_axes['joint4']
        )
        T_total = T_total @ T4
        
        # 5. wan到wan2的变换 (关节5: 腕部旋转)
        trans = self.joint_transforms['wan_to_wan2']
        T5 = self.create_transform_matrix(
            translation=trans[0:3],
            rotation_rpy=trans[3:6],
            joint_angle=joint_angles[4],
            joint_axis=self.joint_axes['joint5']
        )
        T_total = T_total @ T5
        
        # 6. wan2到zhua的变换 (关节6: 夹爪控制)
        trans = self.joint_transforms['wan2_to_zhua']
        T6 = self.create_transform_matrix(
            translation=trans[0:3],
            rotation_rpy=trans[3:6],
            joint_angle=joint_angles[5],
            joint_axis=self.joint_axes['joint6']
        )
        T_total = T_total @ T6
        
        # 提取末端执行器位置
        end_effector_position = T_total[0:3, 3]
        
        return end_effector_position
    
    def get_transform_matrix(self, joint_angles):
        """
        获取完整的变换矩阵 (用于调试)
        
        Args:
            joint_angles: 6个关节角度列表
        
        Returns:
            4x4变换矩阵
        """
        if len(joint_angles) != 6:
            raise ValueError("需要输入6个关节角度")
        
        T_total = np.eye(4)
        
        # 计算所有变换矩阵的乘积
        transforms = [
            ('base_to_yao', 'joint1'),
            ('yao_to_jian1', 'joint2'),
            ('jian1_to_jian2', 'joint3'),
            ('jian2_to_wan', 'joint4'),
            ('wan_to_wan2', 'joint5'),
            ('wan2_to_zhua', 'joint6')
        ]
        
        for i, (trans_key, joint_key) in enumerate(transforms):
            trans = self.joint_transforms[trans_key]
            T = self.create_transform_matrix(
                translation=trans[0:3],
                rotation_rpy=trans[3:6],
                joint_angle=joint_angles[i],
                joint_axis=self.joint_axes[joint_key]
            )
            T_total = T_total @ T
        
        return T_total


def main():
    """
    测试运动学正解计算
    """
    # 创建机械臂运动学对象
    robot = RobotArmKinematics()
    
    # 测试用例1: 所有关节角度为0
    print("=== 测试用例1: 所有关节角度为0 ===")
    joint_angles_1 = [0, 0, 0, 0, 0, 0]
    position_1 = robot.forward_kinematics(joint_angles_1)
    print(f"关节角度: {joint_angles_1}")
    print(f"末端位置: [{position_1[0]:.4f}, {position_1[1]:.4f}, {position_1[2]:.4f}]")
    
    # 测试用例2: 部分关节有角度
    print("\n=== 测试用例2: 部分关节有角度 ===")
    joint_angles_2 = [math.pi/4, math.pi/6, -math.pi/6, math.pi/4, math.pi/2, 0]
    position_2 = robot.forward_kinematics(joint_angles_2)
    print(f"关节角度: {[f'{angle:.4f}' for angle in joint_angles_2]}")
    print(f"末端位置: [{position_2[0]:.4f}, {position_2[1]:.4f}, {position_2[2]:.4f}]")
    
    # 测试用例3: 最大关节角度
    print("\n=== 测试用例3: 接近最大关节角度 ===")
    joint_angles_3 = [1.5, 1.5, -1.5, 1.5, 1.5, 1.5]
    position_3 = robot.forward_kinematics(joint_angles_3)
    print(f"关节角度: {[f'{angle:.4f}' for angle in joint_angles_3]}")
    print(f"末端位置: [{position_3[0]:.4f}, {position_3[1]:.4f}, {position_3[2]:.4f}]")
    
    # 显示变换矩阵 (调试用)
    print("\n=== 完整变换矩阵 (测试用例1) ===")
    T_matrix = robot.get_transform_matrix(joint_angles_1)
    print(T_matrix)


if __name__ == "__main__":
    main()