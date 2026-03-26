import numpy as np
import matplotlib.pyplot as plt

# 同样，我们用 tensorflow.keras.datasets 来方便地加载数据
import tensorflow as tf
from joblib import dump, load
print("正在加载 MNIST 数据集...")
data = np.load("mnist.npz")
# 提取数据
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
print("数据集加载完成！")

# 打印数据形状
print(f"训练数据 x_train 形状: {x_train.shape}") # (60000, 28, 28)
print(f"训练标签 y_train 形状: {y_train.shape}") # (60000,)

# --- 数据预处理 ---

# 1. 展平 (Flatten) 图像数据
# Scikit-learn 的神经网络也需要一维向量作为输入
# (60000, 28, 28) -> (60000, 784)
num_pixels = x_train.shape[1] * x_train.shape[2] # 28 * 28 = 784
x_train_flat = x_train.reshape(x_train.shape[0], num_pixels)
x_test_flat = x_test.reshape(x_test.shape[0], num_pixels)

print(f"展平后的 x_train 形状: {x_train_flat.shape}")

# 2. 归一化 (Normalize)
# 将像素值从 0-255 缩放到 0-1
x_train_norm = x_train_flat / 255.
x_test_norm = x_test_flat / 255.

print("\n数据预处理完成！")
# 注意：使用 Scikit-learn 时，标签 y 不需要进行独热编码 (One-Hot Encoding)，
# 它可以直接处理像 0, 1, 2... 这样的整数标签，非常方便！

from sklearn.neural_network import MLPClassifier

print("开始创建和训练神经网络...")

# 创建一个 MLPClassifier 实例
# 这就是我们的三层神经网络（输入层是自动确定的）
mlp = MLPClassifier(
    hidden_layer_sizes=(128,),  # 一个包含128个神经元的隐藏层。可以写 (128, 64) 来创建两个隐藏层
    max_iter=20,                 # 训练迭代次数 (epochs)
    activation='relu',           # 隐藏层的激活函数
    solver='adam',               # 优化器，adam 是一个高效的默认选项
    random_state=1,              # 保证每次运行结果一致
    verbose=True                 # 打印每一轮的损失值，方便我们观察训练过程
)

# 训练模型！
# .fit() 方法会自动完成所有的训练工作
mlp.fit(x_train_norm, y_train)


# 保存模型
dump(mlp, 'mnist_mlp_model.joblib') 
print("模型已保存为 mnist_mlp_model.joblib")
print("\n模型训练完成！")

from sklearn.metrics import accuracy_score, classification_report

print("\n开始在测试集上评估模型...")

# 使用 .predict() 方法进行预测
y_pred = mlp.predict(x_test_norm)

# 1. 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy * 100:.2f}%")

# 2. 显示更详细的分类报告
# 它会告诉你每个数字（0-9）的精确率、召回率和 F1-score
print("\n分类报告:")
print(classification_report(y_test, y_pred))


# 随机选择一些测试图片进行展示
num_examples = 10
indices = np.random.choice(range(len(x_test_norm)), num_examples)

plt.figure(figsize=(12, 6))

for i, idx in enumerate(indices):
    # 获取原始图像 (28x28) 用于显示
    image = x_test[idx]
    
    # 获取展平并归一化的数据用于预测
    # 注意：.predict() 需要一个二维数组，所以我们用 [ ... ] 或 .reshape(1, -1)
    data_point = x_test_norm[idx]
    
    # 进行预测
    prediction = mlp.predict([data_point])
    predicted_label = prediction[0]
    
    # 获取真实标签
    true_label = y_test[idx]
    
    # 绘图
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"预测: {predicted_label}\n真实: {true_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()




# 加载模型（在其他地方使用时）
# mlp_loaded = load('mnist_mlp_model.joblib')
