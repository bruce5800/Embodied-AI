/**
 * 机械臂末端位置控制模块
 * 提供前后、左右、上下移动控制功能
 */

import { inverseKinematics } from './inverseKinematics.js';
import { forwardKinematics } from './forwardKinematics.js';

export class PositionController {
    constructor(robot, inverseKinematicsInstance) {
        this.robot = robot;
        this.inverseKinematics = inverseKinematicsInstance;
        this.forwardKinematics = forwardKinematics;
        
        // 当前末端位置
        this.currentPosition = { x: 0, y: 0, z: 0 };
        
        // 移动步长 (米) - 增大步长以便更容易求解
        this.stepSize = 0.05; // 默认5cm，更适合机械臂的移动范围
        
        // 移动方向定义
        this.directions = {
            forward: { x: 1, y: 0, z: 0 },   // 前 (X+)
            backward: { x: -1, y: 0, z: 0 }, // 后 (X-)
            left: { x: 0, y: 1, z: 0 },      // 左 (Y+)
            right: { x: 0, y: -1, z: 0 },    // 右 (Y-)
            up: { x: 0, y: 0, z: 1 },        // 上 (Z+)
            down: { x: 0, y: 0, z: -1 }      // 下 (Z-)
        };
    }
    
    /**
     * 初始化位置控制器
     */
    init() {
        // 初始化UI事件
        this.initializeUI();
        
        // 更新当前位置
        this.updateCurrentPosition();
    }
    
    /**
     * 初始化UI事件监听
     */
    initializeUI() {
        // 位置控制按钮事件
        document.getElementById('moveForward')?.addEventListener('click', () => this.moveInDirection('forward'));
        document.getElementById('moveBackward')?.addEventListener('click', () => this.moveInDirection('backward'));
        document.getElementById('moveLeft')?.addEventListener('click', () => this.moveInDirection('left'));
        document.getElementById('moveRight')?.addEventListener('click', () => this.moveInDirection('right'));
        document.getElementById('moveUp')?.addEventListener('click', () => this.moveInDirection('up'));
        document.getElementById('moveDown')?.addEventListener('click', () => this.moveInDirection('down'));
        
        // 步长控制滑块事件
        const stepSizeControl = document.getElementById('stepSizeControl');
        const stepSizeValue = document.getElementById('stepSizeValue');
        
        if (stepSizeControl && stepSizeValue) {
            stepSizeControl.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.stepSize = value / 100; // 转换为米
                stepSizeValue.textContent = value.toFixed(1);
            });
        }
        
        // 键盘快捷键支持
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey) { // 按住Ctrl键时启用位置控制
                switch(e.key.toLowerCase()) {
                    case 'arrowup':
                        e.preventDefault();
                        this.moveInDirection('forward');
                        break;
                    case 'arrowdown':
                        e.preventDefault();
                        this.moveInDirection('backward');
                        break;
                    case 'arrowleft':
                        e.preventDefault();
                        this.moveInDirection('left');
                        break;
                    case 'arrowright':
                        e.preventDefault();
                        this.moveInDirection('right');
                        break;
                    case 'pageup':
                        e.preventDefault();
                        this.moveInDirection('up');
                        break;
                    case 'pagedown':
                        e.preventDefault();
                        this.moveInDirection('down');
                        break;
                }
            }
        });
    }
    
    /**
     * 更新当前末端位置
     */
    updateCurrentPosition() {
        try {
            if (!this.robot || !this.robot.joints) {
                console.warn('机器人对象未准备好');
                return;
            }
            
            // 获取当前关节角度
            const currentAngles = [];
            const jointNames = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'];
            
            for (const jointName of jointNames) {
                const joint = this.robot.joints[jointName];
                if (joint) {
                    currentAngles.push(joint.jointValue[0] || 0);
                } else {
                    currentAngles.push(0);
                }
            }
            
            // 计算当前末端位置
            const position = this.forwardKinematics.forwardKinematics(currentAngles);
            
            this.currentPosition = {
                x: position[0] || position.x || 0,  // 兼容数组和对象格式
                y: position[1] || position.y || 0,
                z: position[2] || position.z || 0
            };
            
            console.log('当前位置更新:', this.currentPosition, '关节角度:', currentAngles);
            
            // 更新UI显示
            this.updatePositionDisplay();
            
        } catch (error) {
            console.warn('更新当前位置失败:', error);
        }
    }
    
    /**
     * 更新位置显示
     */
    updatePositionDisplay() {
        const xElement = document.getElementById('endEffectorX');
        const yElement = document.getElementById('endEffectorY');
        const zElement = document.getElementById('endEffectorZ');
        
        if (xElement) xElement.textContent = (this.currentPosition.x * 1000).toFixed(1); // 转换为mm
        if (yElement) yElement.textContent = (this.currentPosition.y * 1000).toFixed(1);
        if (zElement) zElement.textContent = (this.currentPosition.z * 1000).toFixed(1);
    }
    
    /**
     * 向指定方向移动
     * @param {string} direction - 移动方向
     */
    async moveInDirection(direction) {
        try {
            // 获取方向向量
            const dirVector = this.directions[direction];
            if (!dirVector) {
                console.error('未知的移动方向:', direction);
                return;
            }
            
            // 计算目标位置
            const targetPosition = [
                this.currentPosition.x + dirVector.x * this.stepSize,
                this.currentPosition.y + dirVector.y * this.stepSize,
                this.currentPosition.z + dirVector.z * this.stepSize
            ];
            
            // 显示移动提示
            this.showMovementFeedback(direction);
            
            // 使用逆运动学求解关节角度 (现在会自动调整到工作空间内)
            const result = this.inverseKinematics.inverseKinematicsMultipleAttempts(
                targetPosition, 
                3, // 尝试3次
                false // 不显示详细信息
            );
            
            // 检查结果是否有效
            if (!result) {
                this.showAlert('逆运动学求解失败：无法找到有效解', 'error');
                console.error('逆运动学求解返回null');
                return;
            }
            
            if (result.success) {
                // 移动机械臂到目标位置
                await this.moveToJointAngles(result.jointAngles);
                
                // 更新当前位置 (使用实际到达的位置)
                const actualPosition = result.adjustedPosition || targetPosition;
                this.currentPosition = {
                    x: actualPosition[0],
                    y: actualPosition[1],
                    z: actualPosition[2]
                };
                
                this.updatePositionDisplay();
                
                // 如果位置被调整，显示提示
                if (result.adjustedPosition) {
                    this.showAlert(`已移动到最接近的可达位置: X=${actualPosition[0].toFixed(3)}, Y=${actualPosition[1].toFixed(3)}, Z=${actualPosition[2].toFixed(3)}`, 'info');
                    console.log(`位置已调整到工作空间内: X=${actualPosition[0].toFixed(3)}, Y=${actualPosition[1].toFixed(3)}, Z=${actualPosition[2].toFixed(3)}`);
                } else {
                    console.log(`成功移动到位置: X=${targetPosition[0].toFixed(3)}, Y=${targetPosition[1].toFixed(3)}, Z=${targetPosition[2].toFixed(3)}`);
                }
                
            } else {
                console.warn('逆运动学求解失败:', result.error);
                this.showAlert('无法移动到目标位置: ' + result.error, 'error');
            }
            
        } catch (error) {
            console.error('移动过程中发生错误:', error);
            this.showAlert('移动失败: ' + error.message, 'error');
        }
    }
    
    /**
     * 移动机械臂到指定关节角度
     * @param {Array} jointAngles - 目标关节角度 (弧度)
     */
    async moveToJointAngles(jointAngles) {
        try {
            if (!this.robot || !this.robot.joints) {
                console.warn('机器人对象未准备好');
                return false;
            }
            
            const jointNames = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'];
            
            for (let i = 0; i < jointNames.length && i < jointAngles.length; i++) {
                const joint = this.robot.joints[jointNames[i]];
                if (joint) {
                    joint.jointValue[0] = jointAngles[i];
                }
            }
            
            // 更新当前位置
            this.updateCurrentPosition();
            
            return true;
        } catch (error) {
            console.error('设置关节角度失败:', error);
            throw error;
        }
    }
    
    /**
     * 显示移动反馈
     * @param {string} direction - 移动方向
     */
    showMovementFeedback(direction) {
        const directionNames = {
            forward: '前',
            backward: '后',
            left: '左',
            right: '右',
            up: '上',
            down: '下'
        };
        
        const button = document.getElementById(`move${direction.charAt(0).toUpperCase() + direction.slice(1)}`);
        if (button) {
            button.style.transform = 'scale(0.95)';
            setTimeout(() => {
                button.style.transform = '';
            }, 150);
        }
        
        console.log(`向${directionNames[direction]}移动 ${(this.stepSize * 100).toFixed(1)}cm`);
    }
    
    /**
     * 显示警告信息
     * @param {string} message - 警告信息
     * @param {string} type - 警告类型 ('warning', 'error', 'info')
     */
    showAlert(message, type = 'info') {
        // 创建或获取警告元素
        let alertElement = document.getElementById('positionControlAlert');
        
        if (!alertElement) {
            alertElement = document.createElement('div');
            alertElement.id = 'positionControlAlert';
            alertElement.style.cssText = `
                position: fixed;
                top: 80px;
                left: 50%;
                transform: translateX(-50%);
                padding: 12px 20px;
                border-radius: 8px;
                z-index: 2000;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                text-align: center;
                max-width: 400px;
                font-weight: bold;
                animation: fadeInOut 0.3s ease;
                pointer-events: none;
                display: none;
            `;
            document.body.appendChild(alertElement);
        }
        
        // 设置样式和内容
        const colors = {
            warning: 'rgba(255, 152, 0, 0.9)',
            error: 'rgba(244, 67, 54, 0.9)',
            info: 'rgba(33, 150, 243, 0.9)'
        };
        
        alertElement.style.background = colors[type] || colors.info;
        alertElement.style.color = 'white';
        alertElement.textContent = message;
        alertElement.style.display = 'block';
        
        // 自动隐藏
        setTimeout(() => {
            alertElement.style.display = 'none';
        }, 3000);
    }
    
    /**
     * 延迟函数
     * @param {number} ms - 延迟毫秒数
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * 重置到初始位置
     */
    async resetToHome() {
        try {
            const homeAngles = [0, 0, 0, 0, 0, 0]; // 所有关节归零
            await this.moveToJointAngles(homeAngles);
            
            // 更新当前位置
            setTimeout(() => {
                this.updateCurrentPosition();
            }, 500);
            
            this.showAlert('已重置到初始位置', 'info');
            
        } catch (error) {
            console.error('重置失败:', error);
            this.showAlert('重置失败: ' + error.message, 'error');
        }
    }
    
    /**
     * 移动到指定位置
     * @param {number} x - X坐标 (米)
     * @param {number} y - Y坐标 (米)
     * @param {number} z - Z坐标 (米)
     */
    async moveToPosition(x, y, z) {
        try {
            const targetPosition = [x, y, z];
            
            if (!this.inverseKinematics.checkWorkspaceLimits(targetPosition)) {
                this.showAlert('目标位置超出工作空间范围', 'warning');
                return false;
            }
            
            const result = this.inverseKinematics.inverseKinematicsMultipleAttempts(targetPosition, 5);
            
            if (result.success) {
                await this.moveToJointAngles(result.jointAngles);
                
                this.currentPosition = { x, y, z };
                this.updatePositionDisplay();
                
                return true;
            } else {
                this.showAlert('无法移动到目标位置', 'error');
                return false;
            }
            
        } catch (error) {
            console.error('移动到指定位置失败:', error);
            this.showAlert('移动失败: ' + error.message, 'error');
            return false;
        }
    }
}

// 导出位置控制器类
export default PositionController;