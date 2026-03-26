/**
 * 机械臂逆运动学计算模块
 * 基于雅可比矩阵的迭代法，通过逐步逼近求解关节角度
 * 输入：末端夹爪的目标位置 (x, y, z)
 * 输出：6个电机的角度
 */

import { forwardKinematics } from './forwardKinematics.js';

export class GenkiArmInverseKinematics {
    constructor() {
        this.forwardKinematics = forwardKinematics;
        
        // 算法参数 - 调整为更宽松的参数以提高求解成功率
        this.config = {
            maxIterations: 1000,    // 增加最大迭代次数
            tolerance: 0.01,        // 放宽容忍度到1cm
            dampingFactor: 0.01,    // 减小阻尼因子
            stepSize: 0.1,
            maxStepSize: 0.2,
            minStepSize: 0.01
        };
        
        // 工作空间限制 (米) - 参考Python版本并调整为更合理的范围
        this.workspaceLimits = {
            x: [-0.5, 0.5],
            y: [-0.5, 0.5], 
            z: [0.0, 0.6]
        };
    }
    
    /**
     * 计算雅可比矩阵
     * @param {Array} jointAngles - 当前关节角度 (弧度)
     * @returns {Array} 3x6 雅可比矩阵
     */
    computeJacobian(jointAngles) {
        const jacobian = [];
        const delta = 0.0001; // 数值微分步长
        
        // 计算当前位置
        const currentPos = this.forwardKinematics.forwardKinematics(jointAngles);
        
        // 对每个关节计算偏导数
        for (let j = 0; j < 6; j++) {
            const perturbedAngles = [...jointAngles];
            perturbedAngles[j] += delta;
            
            const perturbedPos = this.forwardKinematics.forwardKinematics(perturbedAngles);
            
            // 计算位置变化率
            const dx = (perturbedPos.x - currentPos.x) / delta;
            const dy = (perturbedPos.y - currentPos.y) / delta;
            const dz = (perturbedPos.z - currentPos.z) / delta;
            
            jacobian.push([dx, dy, dz]);
        }
        
        // 转置矩阵 (6x3 -> 3x6)
        const jacobianT = [];
        for (let i = 0; i < 3; i++) {
            jacobianT[i] = [];
            for (let j = 0; j < 6; j++) {
                jacobianT[i][j] = jacobian[j][i];
            }
        }
        
        return jacobianT;
    }
    
    /**
     * 将目标位置限制到工作空间内
     * @param {Array} targetPosition - 目标位置 [x, y, z]
     * @returns {Array} 限制后的位置
     */
    clampToWorkspace(targetPosition) {
        const [x, y, z] = targetPosition;
        
        const clampedX = Math.max(this.workspaceLimits.x[0], Math.min(this.workspaceLimits.x[1], x));
        const clampedY = Math.max(this.workspaceLimits.y[0], Math.min(this.workspaceLimits.y[1], y));
        const clampedZ = Math.max(this.workspaceLimits.z[0], Math.min(this.workspaceLimits.z[1], z));
        
        return [clampedX, clampedY, clampedZ];
    }

    /**
     * 检查目标位置是否在工作空间内
     * @param {Array} targetPosition - 目标位置 [x, y, z]
     * @returns {boolean} 是否在工作空间内
     */
    checkWorkspaceLimits(targetPosition) {
        const [x, y, z] = targetPosition;
        
        return (
            x >= this.workspaceLimits.x[0] && x <= this.workspaceLimits.x[1] &&
            y >= this.workspaceLimits.y[0] && y <= this.workspaceLimits.y[1] &&
            z >= this.workspaceLimits.z[0] && z <= this.workspaceLimits.z[1]
        );
    }
    
    /**
     * 限制关节角度在允许范围内
     * @param {Array} jointAngles - 关节角度数组
     * @returns {Array} 限制后的关节角度
     */
    clampJointAngles(jointAngles) {
        const limits = this.forwardKinematics.jointLimits;
        return jointAngles.map((angle, i) => {
            const [lower, upper] = limits[i];
            return Math.max(lower, Math.min(upper, angle));
        });
    }
    
    /**
     * 矩阵乘法
     * @param {Array} A - 矩阵A
     * @param {Array} B - 矩阵B
     * @returns {Array} 结果矩阵
     */
    matrixMultiply(A, B) {
        const rows = A.length;
        const cols = B[0].length;
        const common = B.length;
        const result = [];
        
        for (let i = 0; i < rows; i++) {
            result[i] = [];
            for (let j = 0; j < cols; j++) {
                let sum = 0;
                for (let k = 0; k < common; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        return result;
    }
    
    /**
     * 计算矩阵的伪逆 (使用SVD分解的简化版本)
     * @param {Array} matrix - 输入矩阵
     * @returns {Array} 伪逆矩阵
     */
    pseudoInverse(matrix) {
        // 简化的伪逆计算：使用转置和阻尼最小二乘法
        const rows = matrix.length;
        const cols = matrix[0].length;
        
        // 计算 J^T
        const JT = [];
        for (let i = 0; i < cols; i++) {
            JT[i] = [];
            for (let j = 0; j < rows; j++) {
                JT[i][j] = matrix[j][i];
            }
        }
        
        // 计算 J * J^T
        const JJT = this.matrixMultiply(matrix, JT);
        
        // 添加阻尼项 (J * J^T + λI)
        const lambda = this.config.dampingFactor;
        for (let i = 0; i < JJT.length; i++) {
            JJT[i][i] += lambda;
        }
        
        // 计算逆矩阵 (简化版本，仅适用于3x3矩阵)
        const inv = this.inverse3x3(JJT);
        if (!inv) {
            return null;
        }
        
        // 返回 J^T * (J * J^T + λI)^(-1)
        return this.matrixMultiply(JT, inv);
    }
    
    /**
     * 计算3x3矩阵的逆矩阵
     * @param {Array} matrix - 3x3矩阵
     * @returns {Array|null} 逆矩阵或null（如果不可逆）
     */
    inverse3x3(matrix) {
        const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
        
        const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        
        if (Math.abs(det) < 1e-10) {
            return null; // 矩阵不可逆
        }
        
        const invDet = 1 / det;
        
        return [
            [(e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet],
            [(f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet],
            [(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet]
        ];
    }
    
    /**
     * 逆运动学求解
     * @param {Array} targetPosition - 目标位置 [x, y, z]
     * @param {Array} initialGuess - 初始关节角度猜测
     * @param {boolean} verbose - 是否输出详细信息
     * @returns {Object} 求解结果
     */
    inverseKinematics(targetPosition, initialGuess = null, verbose = false) {
        // 将目标位置限制到工作空间内
        const clampedPosition = this.clampToWorkspace(targetPosition);
        const wasOutOfBounds = !this.checkWorkspaceLimits(targetPosition);
        
        if (wasOutOfBounds && verbose) {
            console.log(`目标位置超出工作空间，已调整为最接近位置: [${clampedPosition.map(x => x.toFixed(3)).join(', ')}]`);
        }
        
        // 初始化关节角度
        let jointAngles = initialGuess || [0, 0, 0, 0, 0, 0];
        jointAngles = this.clampJointAngles(jointAngles);
        
        let bestAngles = [...jointAngles];
        let bestError = Infinity;
        
        for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
            // 计算当前末端位置
            const currentPos = this.forwardKinematics.forwardKinematics(jointAngles);
            
            // 计算位置误差 (使用限制后的目标位置)
            const error = [
                clampedPosition[0] - currentPos.x,
                clampedPosition[1] - currentPos.y,
                clampedPosition[2] - currentPos.z
            ];
            
            const errorMagnitude = Math.sqrt(error[0]**2 + error[1]**2 + error[2]**2);
            
            if (verbose && iteration % 10 === 0) {
                console.log(`迭代 ${iteration}: 误差 = ${errorMagnitude.toFixed(6)}`);
            }
            
            // 记录最佳结果
            if (errorMagnitude < bestError) {
                bestError = errorMagnitude;
                bestAngles = [...jointAngles];
            }
            
            // 检查收敛
            if (errorMagnitude < this.config.tolerance) {
                return {
                    success: true,
                    jointAngles: jointAngles,
                    error: null,
                    iterations: iteration + 1,
                    finalError: errorMagnitude,
                    adjustedPosition: wasOutOfBounds ? clampedPosition : null
                };
            }
            
            // 计算雅可比矩阵
            const jacobian = this.computeJacobian(jointAngles);
            
            // 计算雅可比矩阵的伪逆
            const jacobianPinv = this.pseudoInverse(jacobian);
            
            if (!jacobianPinv) {
                if (verbose) {
                    console.log("雅可比矩阵奇异，无法继续迭代");
                }
                break;
            }
            
            // 计算关节角度增量
            const deltaAngles = [];
            for (let i = 0; i < 6; i++) {
                let sum = 0;
                for (let j = 0; j < 3; j++) {
                    sum += jacobianPinv[i][j] * error[j];
                }
                deltaAngles[i] = sum * this.config.stepSize;
            }
            
            // 限制步长
            const maxDelta = Math.max(...deltaAngles.map(Math.abs));
            if (maxDelta > this.config.maxStepSize) {
                const scale = this.config.maxStepSize / maxDelta;
                for (let i = 0; i < 6; i++) {
                    deltaAngles[i] *= scale;
                }
            }
            
            // 更新关节角度
            for (let i = 0; i < 6; i++) {
                jointAngles[i] += deltaAngles[i];
            }
            
            // 限制关节角度
            jointAngles = this.clampJointAngles(jointAngles);
        }
        
        return {
            success: false,
            jointAngles: bestAngles,
            error: `未能在${this.config.maxIterations}次迭代内收敛`,
            iterations: this.config.maxIterations,
            finalError: bestError,
            adjustedPosition: wasOutOfBounds ? clampedPosition : null
        };
    }
    
    /**
     * 多次尝试逆运动学求解
     * @param {Array} targetPosition - 目标位置
     * @param {number} numAttempts - 尝试次数
     * @param {boolean} verbose - 是否输出详细信息
     * @returns {Object} 最佳求解结果
     */
    inverseKinematicsMultipleAttempts(targetPosition, numAttempts = 5, verbose = false) {
        let bestResult = null;
        let bestError = Infinity;
        
        for (let attempt = 0; attempt < numAttempts; attempt++) {
            // 生成随机初始猜测
            const initialGuess = [];
            for (let i = 0; i < 6; i++) {
                const [lower, upper] = this.forwardKinematics.jointLimits[i];
                initialGuess[i] = lower + Math.random() * (upper - lower);
            }
            
            const result = this.inverseKinematics(targetPosition, initialGuess, verbose);
            
            if (result && result.success) {
                return result;
            }
            
            if (result && result.finalError < bestError) {
                bestError = result.finalError;
                bestResult = result;
            }
        }
        
        // 确保返回一个有效的结果对象
        if (!bestResult) {
            return {
                success: false,
                jointAngles: [0, 0, 0, 0, 0, 0],
                error: '所有尝试都失败了',
                iterations: 0,
                finalError: Infinity,
                adjustedPosition: null
            };
        }
        
        return bestResult;
    }
    
    /**
     * 角度转换：度 -> 弧度
     */
    degreesToRadians(anglesDeg) {
        return anglesDeg.map(angle => angle * Math.PI / 180);
    }
    
    /**
     * 角度转换：弧度 -> 度
     */
    radiansToDegrees(anglesRad) {
        return anglesRad.map(angle => angle * 180 / Math.PI);
    }
    
    /**
     * 验证求解结果
     * @param {Array} jointAngles - 关节角度
     * @param {Array} targetPosition - 目标位置
     * @returns {Object} 验证结果
     */
    verifySolution(jointAngles, targetPosition) {
        const actualPos = this.forwardKinematics.forwardKinematics(jointAngles);
        
        const error = [
            targetPosition[0] - actualPos.x,
            targetPosition[1] - actualPos.y,
            targetPosition[2] - actualPos.z
        ];
        
        const errorMagnitude = Math.sqrt(error[0]**2 + error[1]**2 + error[2]**2);
        
        return {
            targetPosition: targetPosition,
            actualPosition: [actualPos.x, actualPos.y, actualPos.z],
            error: error,
            errorMagnitude: errorMagnitude,
            withinTolerance: errorMagnitude < this.config.tolerance
        };
    }
}

// 创建全局实例
export const inverseKinematics = new GenkiArmInverseKinematics();