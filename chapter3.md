# 第3章：时空采样与轨迹对齐

视频自回归模型面临的核心挑战之一是如何有效处理时空连续性。本章深入探讨时域采样策略、变帧率处理、位置编码设计，以及如何融合GPS轨迹信息实现更准确的时空建模。这些技术对于构建能够理解和预测复杂驾驶场景的模型至关重要。

## 3.1 视频时域采样策略

### 采样率与信息密度权衡

**信息论视角**：
视频帧间信息冗余度：
$$I(F_t; F_{t+\Delta t}) = H(F_t) - H(F_t|F_{t+\Delta t})$$

随着$\Delta t$增加，互信息减少，独立信息增加。

**采样策略对比**：
```
策略              采样率        优势                劣势
──────────────────────────────────────────────────────────
均匀采样          固定fps      简单、可预测        忽略内容变化
自适应采样        动态fps      信息效率高          复杂度高
关键帧采样        事件驱动    捕获重要变化        可能丢失过渡
分层采样          多尺度      多粒度建模          存储开销大
```

### 动态采样率设计

**基于运动幅度的采样**：
$$\text{fps}_t = \text{fps}_{base} \cdot \left(1 + \alpha \cdot \frac{\|\mathbf{v}_t\|}{\|\mathbf{v}_{max}\|}\right)$$

其中$\mathbf{v}_t$为光流速度场。

**基于预测误差的采样**：
```python
def adaptive_sampling(video, model):
    frames = []
    t = 0
    while t < len(video):
        frames.append(video[t])
        # 预测下一帧
        pred = model.predict(frames[-1])
        # 计算跳帧数
        skip = 1
        while skip < max_skip:
            error = ||video[t+skip] - pred||
            if error > threshold:
                break
            skip += 1
        t += skip
    return frames
```

**场景复杂度感知采样**：
$$\text{Complexity}_t = \sum_{i} w_i \cdot f_i(S_t)$$

其中$f_i$包括：
- 物体数量：$f_1 = \text{count}(\text{objects})$
- 交互强度：$f_2 = \text{interaction\_score}$
- 速度分布：$f_3 = \text{std}(\text{velocities})$

### 时序金字塔采样

**多尺度时序表示**：
```
Level 0: [t₀, t₁, t₂, t₃, t₄, t₅, t₆, t₇]  # 30fps
Level 1: [t₀,     t₂,     t₄,     t₆    ]  # 15fps
Level 2: [t₀,           t₄              ]  # 7.5fps
Level 3: [t₀                            ]  # 3.75fps
```

**分层特征融合**：
$$F_{merged} = \sum_{l=0}^L \alpha_l \cdot \text{Upsample}(F_l)$$

其中$\alpha_l$为可学习的层级权重。

### 因果采样与未来帧处理

**因果mask设计**：
$$M_{ij} = \begin{cases}
1 & \text{if } t_i \leq t_j \\
0 & \text{if } t_i > t_j
\end{cases}$$

**双向采样用于训练**：
```
Past Context: [t-3, t-2, t-1, t]
Future Hint:  [t+1, t+3, t+5]    # 稀疏未来帧
Target:       [t+1, t+2, ..., t+n]
```

训练时使用未来信息，推理时仅用过去信息。

**Rule of Thumb**：
- 基础采样率：10-30fps
- 最大跳帧：5-10帧
- 历史窗口：1-3秒
- 预测范围：3-5秒

## 3.2 变帧率处理与帧率感知PE

### 变帧率的挑战与机遇

**真实场景的帧率变化**：
```
场景类型          典型帧率      变化原因
────────────────────────────────────────
高速公路          10-15fps     场景简单，压缩传输
城市路口          25-30fps     复杂交互，需要细节
停车场            5-10fps      低速运动
紧急制动          30-60fps     关键安全事件
```

**时间间隔编码**：
不均匀采样下的时间戳：
$$\Delta t_i = t_i - t_{i-1}$$

需要模型理解变化的时间间隔。

### 帧率感知位置编码设计

**连续时间位置编码**：
$$PE(t) = \left[\sin\left(\frac{t}{10000^{0/d}}\right), \cos\left(\frac{t}{10000^{0/d}}\right), ..., \sin\left(\frac{t}{10000^{(d-1)/d}}\right), \cos\left(\frac{t}{10000^{(d-1)/d}}\right)\right]$$

使用实际时间戳而非帧索引。

**相对时间编码**：
$$PE_{rel}(i,j) = f(\Delta t_{ij}) = \text{MLP}([\sin(\omega_k \Delta t_{ij}), \cos(\omega_k \Delta t_{ij})]_{k=1}^K)$$

其中$\omega_k = 1/10000^{2k/d}$为不同频率。

**混合编码策略**：
$$PE_{hybrid} = PE_{absolute}(t) + \alpha \cdot PE_{relative}(\Delta t) + \beta \cdot PE_{fps}(\text{fps}_t)$$

### 变帧率下的注意力机制

**时间加权注意力**：
$$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}} - \lambda \cdot |\Delta t_{ij}|\right)$$

时间距离越远，注意力权重衰减。

**帧率条件层归一化**：
$$\text{FPS-LN}(x) = \gamma(\text{fps}) \odot \frac{x - \mu}{\sigma} + \beta(\text{fps})$$

其中$\gamma, \beta$通过fps条件生成。

### 插值与外推策略

**时序插值网络**：
生成中间帧：
$$F_{t+\alpha} = (1-\alpha) \cdot F_t + \alpha \cdot F_{t+1} + \text{ResNet}(F_t, F_{t+1}, \alpha)$$

**自适应跳帧预测**：
```python
def adaptive_skip_prediction(model, context, target_time):
    # 根据目标时间选择预测步长
    dt = target_time - context[-1].time
    if dt < 0.1:  # 100ms内
        return direct_predict(model, context)
    else:  # 长跳预测
        intermediate = predict_intermediate_points(model, context, dt)
        return refine_prediction(intermediate, target_time)
```

**Rule of Thumb**：
- 时间编码维度：64-128
- 频率范围：1-10000
- 最大时间跨度：10秒
- 插值分辨率：10ms

## 3.3 NTK位置编码与长序列外插

### RoPE与NTK理论基础

**旋转位置编码（RoPE）**：
$$f_{\{q,k\}}(x_m, m) = W_{\{q,k\}} x_m e^{im\theta}$$

其中$\theta = 10000^{-2k/d}$，$m$为位置。

**NTK缩放**：
通过调整基频实现长度外插：
$$\theta_{NTK} = \alpha \cdot 10000^{-2k/d}$$

缩放因子：
$$\alpha = \left(\frac{L_{target}}{L_{train}}\right)^{d/(d-2)}$$

### 长序列外插策略

**动态缩放方案**：
```python
def dynamic_ntk_scale(current_len, max_trained_len, dim):
    if current_len <= max_trained_len:
        return 1.0
    else:
        # 超出训练长度时动态调整
        alpha = (current_len / max_trained_len) ** (dim / (dim - 2))
        return alpha
```

**分段线性插值**：
$$\theta(L) = \begin{cases}
\theta_0 & L \leq L_{train} \\
\theta_0 \cdot (1 + \lambda(L - L_{train})) & L_{train} < L \leq 2L_{train} \\
\theta_0 \cdot (1 + \lambda L_{train}) \cdot \sqrt{L/2L_{train}} & L > 2L_{train}
\end{cases}$$

### 位置编码的优化技巧

**ALiBi（Attention with Linear Biases）**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + m \cdot [-(j-i)]_{i,j}\right)V$$

其中$m$为可学习的斜率。

**相对位置桶化**：
$$b(i,j) = \begin{cases}
|i-j| & |i-j| \leq 8 \\
8 + \log_2(|i-j|/8) \cdot 4 & |i-j| > 8
\end{cases}$$

将相对位置映射到有限数量的桶。

**Flash Attention兼容性**：
```
Standard Attention: O(n²) memory
Flash Attention:    O(n) memory
NTK Compatible:     Yes, with modified kernel
```

### 多尺度位置编码

**层级位置编码**：
不同层使用不同的位置编码精度：
$$PE_l = PE_{fine} \cdot (1 - \alpha_l) + PE_{coarse} \cdot \alpha_l$$

底层关注局部，高层关注全局。

**2D时空位置编码**：
$$PE_{2D}(t, s) = [PE_t(t) || PE_s(s) || PE_{ts}(t \times s)]$$

时间、空间及其交互的联合编码。

**Rule of Thumb**：
- NTK缩放因子：1.0-4.0
- 训练长度：2048-4096 tokens
- 外插能力：2-4倍训练长度
- ALiBi斜率：8个注意力头用[1/2^i for i in range(8)]

## 3.4 GPS轨迹融合与Loop Closure

### GPS信号处理与噪声过滤

**卡尔曼滤波平滑**：
状态方程：
$$\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k + \mathbf{w}_k$$

观测方程：
$$\mathbf{z}_k = \mathbf{H}\mathbf{x}_k + \mathbf{v}_k$$

其中$\mathbf{x} = [x, y, v_x, v_y]^T$为位置和速度。

**GPS置信度评估**：
$$\text{Confidence} = \exp\left(-\frac{\text{HDOP}^2}{2\sigma_{hdop}^2} - \frac{(n_{sat} - n_{min})^2}{2\sigma_{sat}^2}\right)$$

HDOP：水平精度因子，$n_{sat}$：卫星数量。

### 轨迹表示与编码

**分段三次样条**：
$$\mathbf{p}(t) = \sum_{i=0}^3 \mathbf{c}_i \cdot t^i, \quad t \in [t_k, t_{k+1}]$$

保证$C^2$连续性。

**Frenet坐标系**：
```
Cartesian (x, y) ──> Frenet (s, l, ψ)
         │                    │
         ↓                    ↓
    全局坐标            道路相对坐标
                      s: 沿道路距离
                      l: 横向偏移
                      ψ: 航向角差
```

**轨迹特征提取**：
$$\mathbf{f}_{traj} = \text{LSTM}([\mathbf{p}_t, \mathbf{v}_t, \mathbf{a}_t, \kappa_t]_{t=1}^T)$$

其中$\kappa$为曲率。

### Loop Closure检测

**位置相似性**：
$$S_{pos}(i,j) = \exp\left(-\frac{\|\mathbf{p}_i - \mathbf{p}_j\|^2}{2\sigma_{pos}^2}\right)$$

**视觉相似性**：
$$S_{vis}(i,j) = \frac{\mathbf{f}_i^T \mathbf{f}_j}{\|\mathbf{f}_i\| \|\mathbf{f}_j\|}$$

**联合Loop检测**：
$$\text{Loop}(i,j) = \mathbb{1}[S_{pos}(i,j) > \tau_{pos} \land S_{vis}(i,j) > \tau_{vis} \land |i-j| > \tau_{time}]$$

### 轨迹引导的视频预测

**条件生成**：
$$\mathbf{F}_{t+1} = f(\mathbf{F}_{\leq t}, \mathbf{T}_{t+1})$$

其中$\mathbf{T}_{t+1}$为未来轨迹。

**轨迹-视觉对齐**：
$$\mathcal{L}_{align} = \sum_t \|\text{PoseNet}(\mathbf{F}_t) - \mathbf{p}_t^{GPS}\|^2$$

**多模态融合**：
```python
def trajectory_guided_fusion(visual_features, trajectory_features):
    # Cross attention
    visual_guided = cross_attention(visual_features, trajectory_features)
    trajectory_guided = cross_attention(trajectory_features, visual_features)

    # Gated fusion
    gate = sigmoid(W_g @ concat([visual_guided, trajectory_guided]))
    fused = gate * visual_guided + (1 - gate) * trajectory_guided
    return fused
```

**Rule of Thumb**：
- GPS采样率：1-10Hz
- 轨迹平滑窗口：5-10个点
- Loop检测距离阈值：5-10米
- 视觉相似度阈值：0.8-0.9

## 3.5 时空一致性约束

### 物理约束建模

**运动学一致性**：
$$\begin{align}
\mathbf{p}_{t+1} &= \mathbf{p}_t + \mathbf{v}_t \Delta t + \frac{1}{2}\mathbf{a}_t \Delta t^2 \\
\mathbf{v}_{t+1} &= \mathbf{v}_t + \mathbf{a}_t \Delta t
\end{align}$$

**动力学约束**：
$$\begin{align}
|a_{lat}| &\leq \mu g \\
|a_{lon}| &\leq \mu g \\
|\dot{\psi}| &\leq v/r_{min}
\end{align}$$

其中$\mu$为摩擦系数，$r_{min}$为最小转弯半径。

### 时序一致性损失

**光流一致性**：
$$\mathcal{L}_{flow} = \sum_{t} \|\mathbf{F}_{t+1} - \text{Warp}(\mathbf{F}_t, \mathbf{O}_{t \to t+1})\|_1$$

**循环一致性**：
$$\mathcal{L}_{cycle} = \|\mathbf{F}_t - \text{Warp}(\text{Warp}(\mathbf{F}_t, \mathbf{O}_{t \to t+1}), \mathbf{O}_{t+1 \to t})\|_1$$

**长程依赖**：
$$\mathcal{L}_{long} = \sum_{k=1}^K \lambda_k \|\mathbf{F}_{t+k} - \hat{\mathbf{F}}_{t+k|t}\|^2$$

随距离衰减的多步预测损失。

### 空间一致性约束

**多视图几何一致性**：
$$\mathbf{p}_2 = \mathbf{R} \mathbf{p}_1 + \mathbf{t}$$

极线约束：
$$\mathbf{p}_2^T \mathbf{F} \mathbf{p}_1 = 0$$

**深度一致性**：
$$\mathcal{L}_{depth} = \sum_{i,j \in \mathcal{N}} \|\mathbf{D}_i - \text{Project}(\mathbf{D}_j, \mathbf{T}_{j \to i})\|$$

**语义一致性**：
$$\mathcal{L}_{semantic} = -\sum_{c} \mathbb{1}[S_i = c] \log p(S_j = c | \text{neighbor})$$

### 对抗训练增强一致性

**时序判别器**：
```python
def temporal_discriminator(sequence):
    # 判断序列是否时序一致
    features = []
    for t in range(len(sequence)-1):
        feat = concat([
            sequence[t],
            sequence[t+1],
            sequence[t+1] - sequence[t]  # 差分特征
        ])
        features.append(feat)

    score = MLP(aggregate(features))
    return score  # 真实序列为1，生成序列为0
```

**空间判别器**：
判断多视图是否一致：
$$D_{spatial}(\{V_i\}_{i=1}^N) \to [0, 1]$$

**联合优化**：
$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_t \mathcal{L}_{temp} + \lambda_s \mathcal{L}_{spatial} - \lambda_d (\mathcal{L}_{D_t} + \mathcal{L}_{D_s})$$

**Rule of Thumb**：
- 光流损失权重：0.1-0.5
- 循环一致性权重：0.05-0.2
- 判别器更新比例：1:5（D:G）
- 物理约束松弛因子：1.2-1.5

## 高级话题：4D占用网格与时空连续性建模

### 4D占用表示

**体素化时空**：
$$\mathcal{V}_{4D} = \{v_{x,y,z,t}\} \in \{0, 1\}^{X \times Y \times Z \times T}$$

每个体素表示该时空位置是否被占用。

**概率占用场**：
$$p(v_{x,y,z,t} = 1) = \sigma(\mathbf{W}^T \mathbf{f}_{x,y,z,t} + b)$$

使用神经网络预测占用概率。

### 神经隐式4D表示

**4D NeRF扩展**：
$$F: (x, y, z, t) \to (\mathbf{c}, \sigma)$$

**时空解耦**：
$$F(x,y,z,t) = F_{static}(x,y,z) + F_{dynamic}(x,y,z,t)$$

分离静态背景和动态物体。

**速度场建模**：
$$\mathbf{v}(x,y,z,t) = \nabla_{\mathbf{p}} F_{dynamic}(\mathbf{p}, t)$$

从4D场导出速度场。

### 连续时间卷积

**时空卷积核**：
$$K(x,y,t) = \exp\left(-\frac{x^2 + y^2}{2\sigma_s^2} - \frac{t^2}{2\sigma_t^2}\right)$$

**可变形4D卷积**：
$$y(\mathbf{p}, t) = \sum_{k} w_k \cdot x(\mathbf{p} + \Delta \mathbf{p}_k, t + \Delta t_k)$$

其中偏移量$\Delta \mathbf{p}_k, \Delta t_k$通过网络学习。

### 4D Transformer

**4D位置编码**：
$$PE_{4D}(x,y,z,t) = \sum_{i,j,k,l} \sin\left(\frac{2\pi}{\lambda_{ijkl}}(ix + jy + kz + lt)\right)$$

**稀疏4D注意力**：
```python
def sparse_4d_attention(features_4d, mask_4d):
    # 只在占用体素间计算注意力
    occupied_indices = torch.where(mask_4d > 0)
    sparse_features = features_4d[occupied_indices]

    # 计算稀疏注意力
    attention = compute_attention(sparse_features)

    # 映射回密集表示
    output_4d = scatter(attention, occupied_indices, features_4d.shape)
    return output_4d
```

**时空池化策略**：
- 空间：max/average pooling
- 时间：递归聚合或注意力池化

### 4D数据增强

**时空扭曲**：
$$\mathbf{p}'(t) = \mathbf{p}(t) + \mathbf{A}(t) \cdot \mathbf{n}(t)$$

其中$\mathbf{A}(t)$为时变扰动幅度。

**4D混合**：
```python
def mixup_4d(sample1, sample2, alpha):
    # 时空维度的mixup
    lambda_t = np.random.beta(alpha, alpha)

    # 时间维度对齐
    t_split = int(lambda_t * T)
    mixed = concat([
        sample1[:, :, :, :t_split],
        sample2[:, :, :, t_split:]
    ])

    return mixed
```

## 本章小结

本章系统介绍了视频自回归模型的时空采样与建模技术：

**关键概念**：
1. **时域采样策略**：动态采样、分层采样、因果采样
2. **变帧率处理**：帧率感知PE、时间加权注意力
3. **NTK位置编码**：长序列外插、动态缩放
4. **GPS轨迹融合**：轨迹编码、Loop closure、多模态对齐
5. **时空一致性**：物理约束、循环一致性、对抗训练

**核心公式**：
- 连续时间PE：$PE(t) = [\sin(t/10000^{k/d}), \cos(t/10000^{k/d})]$
- NTK缩放：$\alpha = (L_{target}/L_{train})^{d/(d-2)}$
- 轨迹样条：$\mathbf{p}(t) = \sum_{i=0}^3 \mathbf{c}_i \cdot t^i$
- 4D占用场：$F: (x,y,z,t) \to (\mathbf{c}, \sigma)$

**核心论文**：
1. [RoPE] **RoFormer: Enhanced Transformer with Rotary Position Embedding**, 2021
2. [NTK] **NTK-Aware Scaled RoPE**, 2023
3. [Flash Attention] **FlashAttention: Fast and Memory-Efficient Exact Attention**, NeurIPS 2022
4. [4D NeRF] **D-NeRF: Neural Radiance Fields for Dynamic Scenes**, CVPR 2021
5. [TimeSformer] **Is Space-Time Attention All You Need for Video Understanding?**, ICML 2021

## 常见陷阱与错误 (Gotchas)

### 1. 采样率与模型容量不匹配
**问题**：高采样率导致序列过长，模型无法处理
**症状**：OOM错误，推理速度慢
**解决**：
- 使用分层采样策略
- 采用稀疏注意力机制
- 动态调整采样率

### 2. 时间编码溢出
**问题**：长视频导致位置编码数值溢出
**症状**：NaN loss，梯度爆炸
**解决**：
- 使用相对位置编码
- 归一化时间戳
- 周期性重置位置计数

### 3. GPS信号丢失处理不当
**问题**：隧道/高楼导致GPS中断
**症状**：轨迹跳变，预测错误
**解决**：
- 惯性导航补充
- 视觉里程计融合
- 插值和外推策略

### 4. Loop closure误检
**问题**：相似场景误判为同一地点
**症状**：地图扭曲，定位错误
**解决**：
- 多模态验证（GPS+视觉）
- 时间约束（最小时间间隔）
- 语义一致性检查

### 5. 物理约束过强
**问题**：严格物理约束限制模型表达力
**症状**：无法建模异常行为
**解决**：
- 软约束（损失函数项）
- 自适应约束强度
- 异常检测旁路

### 6. 4D表示内存爆炸
**问题**：密集4D网格内存需求巨大
**症状**：内存溢出，无法扩展
**解决**：
- 稀疏表示（octree）
- 隐式神经表示
- 分块处理策略

### 7. 时序不一致累积
**问题**：长时预测误差累积
**症状**：预测漂移，物体变形
**解决**：
- 定期重置（关键帧）
- 误差修正机制
- 短程预测组合

### 8. 多尺度特征对齐错误
**问题**：不同时间尺度特征不对齐
**症状**：模糊预测，细节丢失
**解决**：
- 显式对齐模块
- 特征金字塔网络
- 注意力引导对齐