# 第5章：高效长序列注意力机制

处理自动驾驶场景的长视频序列是视频自回归模型面临的核心计算挑战。标准注意力机制的O(n²)复杂度在处理数千帧的视频时变得不可行。本章深入探讨各种高效注意力机制，包括线性注意力、稀疏模式、以及O(n log n)和O(n√n)复杂度的算法实现，为大规模视频建模提供实用的解决方案。

## 5.1 长序列建模的计算挑战

### 复杂度分析与瓶颈

**标准自注意力复杂度**：
```
操作              计算复杂度    内存复杂度    瓶颈类型
───────────────────────────────────────────────────
QK^T计算          O(n²d)       O(n²)        内存带宽
Softmax          O(n²)        O(n²)        内存访问
Attention×V      O(n²d)       O(n²)        计算密集
总体             O(n²d)       O(n²+nd)     内存为主
```

其中n为序列长度，d为特征维度。

**实际场景挑战**：
- 30fps视频，1分钟=1800帧
- 每帧256个tokens
- 总序列长度：460,800 tokens
- 内存需求：~800GB（fp32）

### 计算-内存权衡

**Roofline模型分析**：
$$\text{Performance} = \min\left(\text{Peak\_FLOPS}, \text{Peak\_Bandwidth} \times \text{Arithmetic\_Intensity}\right)$$

算术强度：
$$AI = \frac{\text{FLOPs}}{\text{Memory\_Access}}$$

对于注意力：$AI \approx \frac{2n^2d}{4nd + 4n^2} \approx \frac{d}{2}$

**IO复杂度**：
```python
def attention_io_complexity(n, d, block_size):
    # 标准注意力
    standard_io = O(n * d + n * n)  # 读Q,K,V + 读写attention matrix

    # 分块注意力
    num_blocks = n // block_size
    block_io = O(num_blocks * block_size * d)  # 减少重复读取

    return standard_io, block_io
```

### 长序列特有问题

**梯度消失/爆炸**：
$$\frac{\partial \mathcal{L}}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}}$$

长序列导致梯度累积不稳定。

**注意力退化**：
```
Attention entropy随序列长度增加：
n=512:  H=6.2
n=1024: H=7.8
n=2048: H=8.9
n=4096: H=9.6 (接近均匀分布)
```

**位置编码失效**：
超出训练长度后性能急剧下降。

**Rule of Thumb**：
- 序列长度<1k：标准注意力
- 1k-8k：稀疏/局部注意力
- 8k-32k：线性注意力
- >32k：层级/压缩方法

## 5.2 线性注意力与稀疏模式

### 线性注意力机制

**核技巧（Kernel Trick）**：
将注意力重写为：
$$\text{Attention}(Q,K,V) = \phi(Q)(\phi(K)^T V)$$

其中$\phi$为特征映射。

**Performer实现**：
使用随机特征近似：
$$\phi(x) = \frac{1}{\sqrt{m}}\left[\exp\left(\omega_1^T x - \|x\|^2/2\right), ..., \exp\left(\omega_m^T x - \|x\|^2/2\right)\right]$$

复杂度：$O(nmd)$，其中$m \ll n$。

**Linear Transformer**：
$$Y = \frac{\phi(Q) \cdot (\phi(K)^T \cdot V)}{\phi(Q) \cdot \sum_i \phi(K_i)}$$

分母为归一化项。

### 稀疏注意力模式

**局部窗口注意力**：
```python
def local_attention(q, k, v, window_size):
    n = q.shape[0]
    attention_scores = []

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)

        q_i = q[i:i+1]
        k_local = k[start:end]
        v_local = v[start:end]

        score = softmax(q_i @ k_local.T) @ v_local
        attention_scores.append(score)

    return torch.cat(attention_scores)
```

复杂度：$O(n \cdot w \cdot d)$

**跨步注意力（Strided Attention）**：
$$M_{ij} = \begin{cases}
1 & \text{if } |i-j| \bmod s = 0 \\
0 & \text{otherwise}
\end{cases}$$

每隔s个位置计算注意力。

**Longformer模式**：
组合局部+全局注意力：
```
Pattern = Local(w=512) + Global(tokens=[CLS, SEP]) + Dilated(r=2)
```

### 混合注意力策略

**Big Bird架构**：
```python
def bigbird_attention(q, k, v):
    # 三种注意力模式组合
    random_attn = random_attention(q, k, v, num_blocks=r)
    window_attn = sliding_window_attention(q, k, v, window_size=w)
    global_attn = global_attention(q, k, v, global_tokens=g)

    # 加权组合
    return alpha * random_attn + beta * window_attn + gamma * global_attn
```

**Reformer (LSH注意力)**：
使用局部敏感哈希分桶：
$$h(x) = \arg\max([xR_1, -xR_1, ..., xR_k, -xR_k])$$

相似的向量大概率分到同一桶。

**稀疏性自适应**：
```python
def adaptive_sparse_attention(q, k, v, threshold):
    # 计算注意力分数
    scores = q @ k.T

    # 保留top-k
    topk_scores, indices = torch.topk(scores, k=int(threshold * n))

    # 稀疏注意力
    sparse_attn = torch.zeros_like(scores)
    sparse_attn.scatter_(1, indices, softmax(topk_scores))

    return sparse_attn @ v
```

**Rule of Thumb**：
- 窗口大小：256-512
- 全局token比例：<1%
- 随机块数：3-5
- LSH桶数：n/64

## 5.3 O(n log n)复杂度算法实现

### Fast Attention via Orthogonal Random Features

**正交随机特征**：
使用正交矩阵保证更好的近似：
$$\Omega = \text{orth}(G) \cdot \sqrt{d}$$

其中G为高斯随机矩阵。

**FAVOR+算法**：
```python
def favor_plus_attention(q, k, v, m):
    # 生成正交随机特征
    omega = generate_orthogonal_features(d, m)

    # 特征映射
    q_prime = feature_map(q, omega)  # [n, m]
    k_prime = feature_map(k, omega)  # [n, m]

    # 线性注意力
    kv = k_prime.T @ v  # [m, d]
    normalizer = k_prime.sum(0, keepdim=True)  # [1, m]

    output = q_prime @ (kv / normalizer)
    return output
```

复杂度：$O(nmd)$，选择$m = O(\log n)$得到$O(n \log n)$。

### 分层注意力机制

**二叉树分解**：
```
Level 0: [1][2][3][4][5][6][7][8]  # n个叶节点
Level 1:   [1,2] [3,4] [5,6] [7,8]  # n/2个节点
Level 2:     [1-4]     [5-8]        # n/4个节点
Level 3:         [1-8]               # 根节点
```

每层计算复杂度：$O(n \cdot d)$
总复杂度：$O(n \log n \cdot d)$

**实现细节**：
```python
def hierarchical_attention(x, num_levels):
    outputs = []
    current = x

    for level in range(num_levels):
        # 当前层注意力
        attn = local_attention(current, window_size=2**level)
        outputs.append(attn)

        # 池化到下一层
        current = pool(attn, factor=2)

    # 上采样并组合
    combined = sum(upsample(out, factor=2**i)
                   for i, out in enumerate(outputs))
    return combined
```

### FFT加速的注意力

**循环卷积视角**：
注意力可视为循环卷积：
$$y_i = \sum_{j} a_{(i-j) \bmod n} \cdot v_j$$

**FFT加速**：
$$Y = \text{IFFT}(\text{FFT}(A) \odot \text{FFT}(V))$$

复杂度：$O(n \log n \cdot d)$

**Toeplitz矩阵近似**：
```python
def fft_attention(q, k, v):
    n = q.shape[0]

    # 构造Toeplitz矩阵的第一行
    first_row = compute_attention_kernel(q[0], k)

    # FFT加速
    fft_kernel = fft(first_row)
    fft_values = fft(v, axis=0)

    # 逐点乘法
    fft_output = fft_kernel[:, None] * fft_values

    # 逆FFT
    output = ifft(fft_output, axis=0).real
    return output
```

**Rule of Thumb**：
- 随机特征数：64-256
- 层级数：log₂(n/512)
- FFT块大小：2的幂次

## 5.4 O(n√n)块状注意力设计

### 块分解策略

**基本块注意力**：
将序列分成$\sqrt{n}$个块，每块$\sqrt{n}$个元素：
```
Sequence: [Block₁][Block₂]...[Block_{√n}]
           ↓      ↓          ↓
        √n tokens each
```

**块内+块间注意力**：
$$\text{BlockAttn} = \text{IntraBlock} + \text{InterBlock}$$

复杂度：
- 块内：$\sqrt{n}$块×$O((\sqrt{n})^2 d) = O(n\sqrt{n}d)$
- 块间：$O((\sqrt{n})^2 d) = O(nd)$
- 总计：$O(n\sqrt{n}d)$

### Routing Transformer

**路由机制**：
```python
def routing_attention(x, num_clusters):
    n, d = x.shape
    k = int(np.sqrt(n))  # 聚类数

    # K-means聚类
    centroids, assignments = kmeans(x, k)

    # 块内注意力
    clustered_outputs = []
    for i in range(k):
        cluster_idx = (assignments == i)
        cluster_x = x[cluster_idx]

        # 计算注意力
        output = standard_attention(cluster_x)
        clustered_outputs.append((cluster_idx, output))

    # 重组输出
    final_output = torch.zeros_like(x)
    for idx, output in clustered_outputs:
        final_output[idx] = output

    return final_output
```

**动态路由**：
$$p_{ij} = \frac{\exp(x_i^T c_j / \tau)}{\sum_k \exp(x_i^T c_k / \tau)}$$

token i分配到簇j的概率。

### 分块矩阵乘法优化

**Blocked Matrix Multiplication**：
```python
def blocked_matmul(A, B, block_size):
    m, k = A.shape
    k2, n = B.shape
    assert k == k2

    C = torch.zeros(m, n)

    for i in range(0, m, block_size):
        for j in range(0, n, block_size):
            for k in range(0, k, block_size):
                # 块乘法
                A_block = A[i:i+block_size, k:k+block_size]
                B_block = B[k:k+block_size, j:j+block_size]
                C[i:i+block_size, j:j+block_size] += A_block @ B_block

    return C
```

**内存访问优化**：
- 块大小选择：适配L2 cache
- 数据布局：行优先vs列优先
- 预取策略：提前加载下一块

### Linformer降维

**低秩投影**：
$$\text{Attention}(Q,K,V) = QW_q(EK)^T(FV)$$

其中$E, F \in \mathbb{R}^{k \times n}$，$k = O(\sqrt{n})$

**实现**：
```python
class Linformer(nn.Module):
    def __init__(self, n, d, k):
        super().__init__()
        self.E = nn.Parameter(torch.randn(k, n))
        self.F = nn.Parameter(torch.randn(k, n))

    def forward(self, q, k, v):
        # 降维
        k_reduced = self.E @ k  # [k, d]
        v_reduced = self.F @ v  # [k, d]

        # 标准注意力（低维）
        scores = q @ k_reduced.T  # [n, k]
        attn = softmax(scores)
        output = attn @ v_reduced  # [n, d]

        return output
```

**Rule of Thumb**：
- 块大小：$\sqrt{n}$或512
- 聚类数：$\sqrt{n}$到$n/64$
- 投影维度：256-512
- Cache块：32-64KB

## 5.5 混合注意力架构

### 层级混合策略

**不同层使用不同注意力**：
```python
class HybridTransformer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i < num_layers // 3:
                # 底层：局部注意力
                layer = LocalAttentionLayer(window_size=256)
            elif i < 2 * num_layers // 3:
                # 中层：稀疏注意力
                layer = SparseAttentionLayer(sparsity=0.1)
            else:
                # 高层：全局注意力（降采样后）
                layer = GlobalAttentionLayer(downsample=4)

            self.layers.append(layer)
```

**动态选择**：
```python
def dynamic_attention_selection(x, length):
    if length < 1024:
        return standard_attention(x)
    elif length < 4096:
        return sparse_attention(x)
    elif length < 16384:
        return linear_attention(x)
    else:
        return hierarchical_attention(x)
```

### Sandwich Transformer

**三明治结构**：
```
[Local] → [Local] → [Global] → [Local] → [Local]
```

全局层捕获长程依赖，局部层细化。

**实现**：
```python
class SandwichLayer(nn.Module):
    def __init__(self, d_model, is_global=False):
        super().__init__()
        if is_global:
            self.attn = GlobalAttention(d_model)
        else:
            self.attn = LocalAttention(d_model, window_size=256)

    def forward(self, x):
        return self.attn(x) + x
```

### 门控混合机制

**门控选择**：
$$y = g \cdot y_{local} + (1-g) \cdot y_{global}$$

其中$g = \sigma(W_g x + b_g)$

**多头混合**：
不同head使用不同模式：
```python
class MixedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = nn.ModuleList()

        for i in range(num_heads):
            if i % 4 == 0:
                head = GlobalHead(d_model // num_heads)
            elif i % 4 == 1:
                head = LocalHead(d_model // num_heads)
            elif i % 4 == 2:
                head = StridedHead(d_model // num_heads)
            else:
                head = RandomHead(d_model // num_heads)

            self.heads.append(head)
```

### 自适应计算

**长度感知路由**：
```python
def length_aware_attention(x):
    n = x.shape[0]

    if n < 512:
        return full_attention(x)

    # 分段处理
    segments = []
    for i in range(0, n, 512):
        segment = x[i:i+512]

        # 段内精细注意力
        local_out = full_attention(segment)

        # 段间粗糙注意力
        if i > 0:
            context = x[max(0, i-512):i:64]  # 下采样
            cross_out = cross_attention(segment, context)
            local_out = local_out + 0.1 * cross_out

        segments.append(local_out)

    return torch.cat(segments)
```

**计算预算分配**：
```python
def budget_aware_attention(x, compute_budget):
    importance = estimate_importance(x)

    # 根据重要性分配计算
    attention_type = []
    remaining_budget = compute_budget

    for i, imp in enumerate(importance):
        if imp > 0.8 and remaining_budget > FULL_COST:
            attention_type.append('full')
            remaining_budget -= FULL_COST
        elif imp > 0.5 and remaining_budget > SPARSE_COST:
            attention_type.append('sparse')
            remaining_budget -= SPARSE_COST
        else:
            attention_type.append('linear')
            remaining_budget -= LINEAR_COST

    return apply_mixed_attention(x, attention_type)
```

**Rule of Thumb**：
- 全局层比例：1/4到1/3
- 门控阈值：0.5
- 混合head比例：2:1:1（local:strided:global）
- 计算预算：2x-4x标准注意力

## 高级话题：State Space Models (Mamba)与自回归视频生成

### State Space Models基础

**连续时间状态空间**：
$$\begin{align}
\frac{dx(t)}{dt} &= Ax(t) + Bu(t) \\
y(t) &= Cx(t) + Du(t)
\end{align}$$

**离散化（ZOH）**：
$$\begin{align}
x_k &= \bar{A}x_{k-1} + \bar{B}u_k \\
y_k &= Cx_k
\end{align}$$

其中$\bar{A} = e^{A\Delta}, \bar{B} = A^{-1}(e^{A\Delta} - I)B$

### Mamba架构

**选择性SSM**：
```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state

        # 投影
        self.in_proj = nn.Linear(d_model, d_model * 2)

        # SSM参数
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.randn(1))

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 选择性扫描
        z, r = self.in_proj(x).chunk(2, dim=-1)
        z = torch.sigmoid(z)

        # SSM
        h = torch.zeros(x.shape[0], self.d_state)
        outputs = []

        for t in range(x.shape[1]):
            h = torch.tanh(self.A @ h + self.B(x[:, t]))
            y = self.C(x[:, t]) @ h + self.D * x[:, t]
            outputs.append(y * z[:, t])

        output = torch.stack(outputs, dim=1)
        return self.out_proj(output)
```

**并行扫描算法**：
利用关联性质并行化：
$$y_i = C\left(\prod_{j=1}^i \bar{A}\right) \bar{B}u_0 + ... + C\bar{B}u_i$$

### 视频生成中的应用

**时空SSM**：
```python
class SpatioTemporalSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.temporal_ssm = MambaBlock(d_model)
        self.spatial_ssm = MambaBlock(d_model)

    def forward(self, video_tokens):
        # video_tokens: [B, T, H, W, D]
        B, T, H, W, D = video_tokens.shape

        # 时间维度SSM
        temporal_out = self.temporal_ssm(
            video_tokens.reshape(B*H*W, T, D)
        ).reshape(B, T, H, W, D)

        # 空间维度SSM
        spatial_out = self.spatial_ssm(
            temporal_out.reshape(B*T, H*W, D)
        ).reshape(B, T, H, W, D)

        return spatial_out
```

### 长程建模优势

**线性复杂度**：
- Transformer: $O(n^2d)$
- Mamba: $O(nd^2)$

**无限上下文**：
状态向量隐式编码历史信息。

**因果性保证**：
自然满足自回归要求。

**训练效率**：
```python
def efficient_mamba_training(sequences, model):
    # 并行处理多个序列
    states = initialize_states(batch_size)

    all_outputs = []
    for chunk in chunk_sequences(sequences):
        # 并行扫描
        outputs, states = parallel_scan(chunk, states, model)
        all_outputs.append(outputs)

        # 梯度检查点
        if should_checkpoint():
            checkpoint(outputs)

    return concat(all_outputs)
```

## 本章小结

本章系统介绍了高效长序列注意力机制的设计与实现：

**关键概念**：
1. **计算挑战**：O(n²)复杂度、内存瓶颈、梯度问题
2. **线性注意力**：核技巧、随机特征、FAVOR+
3. **稀疏模式**：局部窗口、LSH、Longformer
4. **高效算法**：O(n log n)层级、O(n√n)块状
5. **混合架构**：Sandwich、门控、自适应

**核心公式**：
- 线性注意力：$\text{Attn} = \phi(Q)(\phi(K)^TV)$
- 块复杂度：$O(n\sqrt{n}d)$
- SSM递归：$x_k = \bar{A}x_{k-1} + \bar{B}u_k$
- FFT加速：$Y = \text{IFFT}(\text{FFT}(A) \odot \text{FFT}(V))$

**核心论文**：
1. [Performer] **Rethinking Attention with Performers**, ICLR 2021
2. [Linformer] **Linformer: Self-Attention with Linear Complexity**, 2020
3. [Reformer] **Reformer: The Efficient Transformer**, ICLR 2020
4. [Mamba] **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, 2023
5. [Flash Attention] **FlashAttention-2: Faster Attention with Better Parallelism**, 2023

## 常见陷阱与错误 (Gotchas)

### 1. 近似误差累积
**问题**：线性注意力近似误差随序列增长
**症状**：长序列性能退化
**解决**：
- 增加随机特征数
- 使用正交特征
- 定期重新校准

### 2. 稀疏模式选择不当
**问题**：固定模式不适合所有数据
**症状**：关键信息丢失
**解决**：
- 学习稀疏模式
- 动态调整稀疏度
- 保留全局token

### 3. 块大小与硬件不匹配
**问题**：块大小不适配cache
**症状**：性能远低于理论值
**解决**：
- Profile确定最优块大小
- 考虑GPU warp大小
- 内存对齐

### 4. 梯度流断裂
**问题**：稀疏连接导致梯度消失
**症状**：深层不更新
**解决**：
- 残差连接
- 梯度重路由
- 混合密集层

### 5. 位置信息丢失
**问题**：高效注意力破坏位置编码
**症状**：顺序敏感任务性能差
**解决**：
- 显式位置编码
- 相对位置偏置
- 因果mask保持

### 6. 训推不一致
**问题**：训练用全注意力，推理用稀疏
**症状**：推理性能下降
**解决**：
- 训练时模拟稀疏
- 渐进式稀疏化
- 知识蒸馏

### 7. 数值稳定性
**问题**：长序列数值溢出
**症状**：NaN或Inf
**解决**：
- Log-space计算
- 定期归一化
- 混合精度训练

### 8. 内存碎片
**问题**：动态稀疏导致碎片
**症状**：OOM despite有剩余内存
**解决**：
- 预分配缓冲区
- 内存池管理
- 定期整理