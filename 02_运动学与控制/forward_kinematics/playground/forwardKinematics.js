/**
 * 机械臂正运动学计算模块
 * 基于URDF文件中的机械臂结构，使用矩阵运算计算末端夹爪位置
 * 输入：6个电机的角度
 * 输出：末端夹爪的x, y, z坐标
 */

export class GenkiArmForwardKinematics {
    constructor() {
        /**
         * 根据URDF文件提取的关节参数
         * 关节1 (腰部旋转): Base -> yao
         * 关节2 (大臂): yao -> jian1  
         * 关节3 (小臂): jian1 -> jian2
         * 关节4 (腕部): jian2 -> wan
         * 关节5 (腕部旋转): wan -> wan2
         * 关节6 (夹爪): wan2 -> zhua
         */
        this.jointTransforms = [
            // 关节1: Base -> yao (腰部旋转，绕X轴)
            {translation: [-0.013, 0, 0.0265], rotation: [0, -1.57, 0], axis: [1, 0, 0]},
            
            // 关节2: yao -> jian1 (大臂，绕Y轴)
            {translation: [0.081, 0, 0.0], rotation: [0, 1.57, 0], axis: [0, 1, 0]},
            
            // 关节3: jian1 -> jian2 (小臂，绕Y轴)
            {translation: [0, 0, 0.118], rotation: [0, 0, 0], axis: [0, 1, 0]},
            
            // 关节4: jian2 -> wan (腕部，绕Y轴)
            {translation: [0, 0, 0.118], rotation: [0, 0, 0], axis: [0, 1, 0]},
            
            // 关节5: wan -> wan2 (腕部旋转，绕Z轴)
            {translation: [0, 0, 0.0635], rotation: [0, 0, 0], axis: [0, 0, 1]},
            
            // 关节6: wan2 -> zhua (夹爪，绕X轴)
            {translation: [0, -0.0132, 0.021], rotation: [0, 0, 0], axis: [1, 0, 0]}
        ];
        
        // 关节限制 [lower, upper] (弧度)
        this.jointLimits = [
            [-1.57, 1.57],  // 关节1
            [-1.57, 1.57],  // 关节2
            [-1.57, 1.57],  // 关节3
            [-1.57, 1.57],  // 关节4
            [-1.57, 1.57],  // 关节5
            [0, 1.57]       // 关节6 (夹爪)
        ];
    }
    
    /**
     * 绕X轴旋转矩阵
     */
    rotationMatrixX(angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ];
    }
    
    /**
     * 绕Y轴旋转矩阵
     */
    rotationMatrixY(angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ];
    }
    
    /**
     * 绕Z轴旋转矩阵
     */
    rotationMatrixZ(angle) {
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        return [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ];
    }
    
    /**
     * 矩阵乘法
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
     * 创建4x4单位矩阵
     */
    createIdentityMatrix() {
        return [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ];
    }
    
    /**
     * 根据平移和旋转创建4x4变换矩阵
     * translation: [x, y, z] 平移向量
     * rotation: [roll, pitch, yaw] 旋转角度（弧度）
     */
    createTransformMatrix(translation, rotation) {
        // 创建旋转矩阵
        const Rx = this.rotationMatrixX(rotation[0]);  // roll
        const Ry = this.rotationMatrixY(rotation[1]);  // pitch  
        const Rz = this.rotationMatrixZ(rotation[2]);  // yaw
        
        // 按照RPY顺序相乘：R = Rz * Ry * Rx
        const RyRx = this.matrixMultiply(Ry, Rx);
        const R = this.matrixMultiply(Rz, RyRx);
        
        // 创建4x4变换矩阵
        const T = this.createIdentityMatrix();
        
        // 设置旋转部分
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                T[i][j] = R[i][j];
            }
        }
        
        // 设置平移部分
        T[0][3] = translation[0];
        T[1][3] = translation[1];
        T[2][3] = translation[2];
        
        return T;
    }
    
    /**
     * 根据旋转轴和角度创建旋转矩阵
     * axis: [x, y, z] 旋转轴向量
     * angle: 旋转角度（弧度）
     */
    createJointRotationMatrix(axis, angle) {
        if (axis[0] === 1) {  // 绕X轴旋转
            return this.rotationMatrixX(angle);
        } else if (axis[1] === 1) {  // 绕Y轴旋转
            return this.rotationMatrixY(angle);
        } else if (axis[2] === 1) {  // 绕Z轴旋转
            return this.rotationMatrixZ(angle);
        } else {
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];  // 无旋转
        }
    }
    
    /**
     * 检查关节角度是否在限制范围内
     */
    checkJointLimits(jointAngles) {
        if (jointAngles.length !== 6) {
            throw new Error("需要输入6个关节角度");
        }
        
        for (let i = 0; i < jointAngles.length; i++) {
            const angle = jointAngles[i];
            const [lower, upper] = this.jointLimits[i];
            if (angle < lower || angle > upper) {
                console.warn(`警告: 关节${i+1}角度 ${angle.toFixed(3)} 超出限制范围 [${lower.toFixed(3)}, ${upper.toFixed(3)}]`);
            }
        }
    }
    
    /**
     * 正运动学计算
     * 输入: jointAngles - 6个关节角度的数组 (弧度)
     * 输出: 末端夹爪的位置 [x, y, z]
     */
    forwardKinematics(jointAngles) {
        // 检查输入
        if (jointAngles.length !== 6) {
            throw new Error("需要输入6个关节角度");
        }
        
        // 检查关节限制
        this.checkJointLimits(jointAngles);
        
        // 初始化变换矩阵为单位矩阵
        let T = this.createIdentityMatrix();
        
        // 逐个计算每个关节的变换矩阵并累乘
        for (let i = 0; i < 6; i++) {
            const jointTransform = this.jointTransforms[i];
            const translation = jointTransform.translation;
            const rotation = jointTransform.rotation;
            const axis = jointTransform.axis;
            
            // 创建固定的平移和旋转变换矩阵（来自URDF的origin）
            const TFixed = this.createTransformMatrix(translation, rotation);
            
            // 创建关节旋转变换矩阵
            const jointRotation = this.createJointRotationMatrix(axis, jointAngles[i]);
            const TJoint = this.createIdentityMatrix();
            
            // 设置关节旋转部分
            for (let row = 0; row < 3; row++) {
                for (let col = 0; col < 3; col++) {
                    TJoint[row][col] = jointRotation[row][col];
                }
            }
            
            // 组合变换：先应用固定变换，再应用关节旋转
            const TCombined = this.matrixMultiply(TFixed, TJoint);
            
            // 累乘变换矩阵
            T = this.matrixMultiply(T, TCombined);
        }
        
        // 提取末端位置
        const endEffectorPosition = [T[0][3], T[1][3], T[2][3]];
        
        return endEffectorPosition;
    }
    
    /**
     * 正运动学计算（包含姿态）
     * 输入: jointAngles - 6个关节角度的数组 (弧度)
     * 输出: {position: [x, y, z], rotationMatrix: 3x3矩阵, transformMatrix: 4x4矩阵}
     */
    forwardKinematicsWithOrientation(jointAngles) {
        // 检查输入
        if (jointAngles.length !== 6) {
            throw new Error("需要输入6个关节角度");
        }
        
        // 检查关节限制
        this.checkJointLimits(jointAngles);
        
        // 初始化变换矩阵为单位矩阵
        let T = this.createIdentityMatrix();
        
        // 逐个计算每个关节的变换矩阵并累乘
        for (let i = 0; i < 6; i++) {
            const jointTransform = this.jointTransforms[i];
            const translation = jointTransform.translation;
            const rotation = jointTransform.rotation;
            const axis = jointTransform.axis;
            
            // 创建固定的平移和旋转变换矩阵（来自URDF的origin）
            const TFixed = this.createTransformMatrix(translation, rotation);
            
            // 创建关节旋转变换矩阵
            const jointRotation = this.createJointRotationMatrix(axis, jointAngles[i]);
            const TJoint = this.createIdentityMatrix();
            
            // 设置关节旋转部分
            for (let row = 0; row < 3; row++) {
                for (let col = 0; col < 3; col++) {
                    TJoint[row][col] = jointRotation[row][col];
                }
            }
            
            // 组合变换：先应用固定变换，再应用关节旋转
            const TCombined = this.matrixMultiply(TFixed, TJoint);
            
            // 累乘变换矩阵
            T = this.matrixMultiply(T, TCombined);
        }
        
        // 提取末端位置
        const position = [T[0][3], T[1][3], T[2][3]];
        
        // 提取旋转矩阵
        const rotationMatrix = [
            [T[0][0], T[0][1], T[0][2]],
            [T[1][0], T[1][1], T[1][2]],
            [T[2][0], T[2][1], T[2][2]]
        ];
        
        return {
            position: position,
            rotationMatrix: rotationMatrix,
            transformMatrix: T
        };
    }
    
    /**
     * 角度转弧度
     */
    degreesToRadians(anglesDeg) {
        return anglesDeg.map(angle => angle * Math.PI / 180);
    }
    
    /**
     * 弧度转角度
     */
    radiansToDegrees(anglesRad) {
        return anglesRad.map(angle => angle * 180 / Math.PI);
    }
}

// 创建全局实例
export const forwardKinematics = new GenkiArmForwardKinematics();