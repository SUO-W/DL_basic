# RNN网络
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
 
# 1. 数据预处理
# 读取数据
data = pd.read_excel(r'E:\study\code\AI\data\realibility\gear_data_lstm.csv')  # 替换为你的数据文件路径  
 
# 提取特征和标签
labels = data['Steering_Angle'].values
features = data[['real1',  'real2',  'real3',  'real4',  'real5',  'real6',  'real7']].values  # 使用 NumPy 数组
 
# 添加历史方向盘转角作为特征 (假设历史窗口长度为5)
window_size = 7
history_features = []
for i in range(window_size, len(data)):
    past_angles = labels[i - window_size:i]
    history_features.append(list(past_angles))
features = features[window_size:]
labels = labels[window_size:]
 
# 合并特征
features = np.hstack((features, history_features))
 
# 归一化
scaler_x = StandardScaler()
scaler_y = StandardScaler()
features = scaler_x.fit_transform(features)
labels = scaler_y.fit_transform(labels.reshape(-1, 1))
 
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
 
# 将特征转换为三维张量，形状为 [样本数, 时间序列长度, 特征数]
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1, window_size + 2)  # [batch_size, seq_len, input_size]
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1, window_size + 2)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
 
# 2. 创建RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # 使用RNN
        self.fc = nn.Linear(hidden_size, 1)  # 输出层
 
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
 
        # 前向传播
        out, _ = self.rnn(x, h0)  # RNN输出形状为 (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出
        return out
 
 
# 实例化模型
input_size = window_size + 2  # 输入特征维度
hidden_size = 64  # 隐藏层大小
num_layers = 2  # RNN层数
model = RNNModel(input_size, hidden_size, num_layers)
 
 
# 3. 设置损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
 
# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # 前向传播
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
 
    # 后向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
 
 
# 5. 预测
model.eval()
with torch.no_grad():
    y_pred_tensor = model(x_test_tensor)
 
y_pred = scaler_y.inverse_transform(y_pred_tensor.numpy())  # 将预测值逆归一化
y_test = scaler_y.inverse_transform(y_test_tensor.numpy())  # 逆归一化真实值
 
# 评估指标
r2 = r2_score(y_test, y_pred)
mae_score = mae(y_test, y_pred)
print(f"R^2 score: {r2:.4f}")
print(f"MAE: {mae_score:.4f}")
 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
 
# 绘制实际值和预测值的对比图
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='实际值', color='blue')
plt.plot(range(len(y_pred)), y_pred, label='预测值', color='red')
plt.xlabel('样本索引')
plt.ylabel('Steering Angle')
plt.title('实际值与预测值对比图')
plt.legend()
plt.grid(True)
plt.show()