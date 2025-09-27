# 第4章：视觉编码器与Tokenization

视觉tokenization是连接原始像素空间和高层语义空间的关键桥梁。本章深入探讨如何设计高效的视觉编码器，将连续的视频流转换为离散或连续的token序列，为自回归建模奠定基础。我们将重点讨论连续与离散表示的权衡、压缩策略、语义对齐，以及tokenizer的损失函数设计。

## 4.1 连续vs离散：表示学习的权衡

### 离散化的动机与挑战

**离散表示的优势**：
```
优势类型          具体表现                   应用场景
───────────────────────────────────────────────────
计算效率          查表替代计算               推理加速
压缩能力          有限码本表示               带宽受限
组合泛化          离散符号组合               新场景生成
可解释性          语义单元明确               调试分析
```

**连续表示的优势**：
- 信息保真度高
- 梯度流畅通
- 细粒度控制
- 无量化噪声

### Vector Quantization (VQ) 架构

**VQ-VAE基础**：
编码过程：
$$\mathbf{z}_e = \text{Encoder}(\mathbf{x})$$

量化过程：
$$\mathbf{z}_q = \arg\min_{\mathbf{e}_k \in \mathcal{C}} \|\mathbf{z}_e - \mathbf{e}_k\|_2$$

其中$\mathcal{C} = \{\mathbf{e}_1, ..., \mathbf{e}_K\}$为码本。

**直通估计器（Straight-Through Estimator）**：
前向传播：使用量化值
$$\mathbf{z} = \mathbf{z}_q$$

反向传播：跳过量化
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_e} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_q}$$

### 连续表示的正则化

**KL正则化**：
$$\mathcal{L}_{KL} = D_{KL}(q(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

其中$p(\mathbf{z}) = \mathcal{N}(0, I)$为先验分布。

**信息瓶颈**：
$$\mathcal{L}_{IB} = \mathcal{L}_{recon} + \beta \cdot I(\mathbf{x}; \mathbf{z})$$

控制表示的信息量。

### 混合表示策略

**层级混合**：
```python
def hierarchical_representation(x):
    # 底层：连续细节
    continuous_low = encoder_low(x)

    # 中层：半离散
    semi_discrete = soft_quantize(encoder_mid(continuous_low))

    # 高层：完全离散
    discrete_high = hard_quantize(encoder_high(semi_discrete))

    return {
        'fine': continuous_low,
        'medium': semi_discrete,
        'coarse': discrete_high
    }
```

**软量化（Gumbel-Softmax）**：
$$\mathbf{z} = \sum_{k=1}^K \frac{\exp((l_k + g_k)/\tau)}{\sum_{j=1}^K \exp((l_j + g_j)/\tau)} \mathbf{e}_k$$

其中$g_k \sim \text{Gumbel}(0,1)$，$\tau$为温度参数。

**Rule of Thumb**：
- 码本大小：512-8192
- 向量维度：256-512
- 量化层级：2-3层
- 温度退火：从5.0到0.5

## 4.2 视觉Tokenizer架构设计

### 空间Token化策略

**Patch分割**：
```
Image (H×W×3) ──> Patches (N×P²×3)
       │                │
       ↓                ↓
   原始图像      N = HW/P²个patches
```

标准patch大小：8×8, 16×16, 32×32

**重叠Patch**：
$$\text{Patch}_{i,j} = \text{Image}[i \cdot s : i \cdot s + p, j \cdot s : j \cdot s + p]$$

其中$s < p$为步长，实现重叠。

**自适应分割**：
```python
def adaptive_patching(image, saliency_map):
    # 高显著性区域细粒度分割
    patches = []
    for region in get_regions(saliency_map):
        if region.saliency > threshold:
            patch_size = 8
        else:
            patch_size = 32
        patches.extend(extract_patches(image, region, patch_size))
    return patches
```

### 时序Token化设计

**3D卷积Tokenizer**：
$$\mathbf{z} = \text{Conv3D}(\mathbf{X}) \in \mathbb{R}^{T' \times H' \times W' \times D}$$

空间和时间同时下采样。

**分离式Tokenizer**：
```python
def factorized_tokenizer(video):
    # 空间编码
    spatial_tokens = []
    for frame in video:
        spatial_tokens.append(spatial_encoder(frame))

    # 时序聚合
    temporal_tokens = temporal_encoder(stack(spatial_tokens))
    return temporal_tokens
```

**Tube Token**：
将时空管作为基本单元：
$$\text{Tube}_{i,j,t} = \text{Video}[t:t+\tau, i \cdot s : i \cdot s + h, j \cdot s : j \cdot s + w]$$

### 多尺度Token设计

**金字塔Token**：
```
Level 0: Fine tokens   (64×64)
Level 1: Medium tokens (32×32)
Level 2: Coarse tokens (16×16)
Level 3: Global token  (1×1)
```

**跨尺度连接**：
$$\mathbf{z}_l = f_l(\mathbf{z}_{l-1}) + g_l(\text{Pool}(\mathbf{z}_{l-1}))$$

**动态Token分配**：
根据内容复杂度分配token：
$$N_{tokens}(region) = N_{base} \cdot \exp(\alpha \cdot \text{Complexity}(region))$$

### Vision Transformer (ViT) 编码器

**标准ViT架构**：
$$\begin{align}
\mathbf{z}_0 &= [\mathbf{x}_{cls}; \mathbf{x}_p^1\mathbf{E}; ...; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos} \\
\mathbf{z}_l &= \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1} \\
\mathbf{z}_l &= \text{MLP}(\text{LN}(\mathbf{z}_l)) + \mathbf{z}_l
\end{align}$$

**Window Attention优化**：
```python
def window_attention(x, window_size):
    # 分割成窗口
    windows = partition(x, window_size)

    # 窗口内注意力
    attended = []
    for window in windows:
        attended.append(self_attention(window))

    # 合并窗口
    return merge_windows(attended)
```

**Rule of Thumb**：
- Patch大小：16×16 (ViT-B/16)
- 嵌入维度：768-1024
- 注意力头数：12-16
- MLP比例：4倍隐藏层

## 4.3 BPE与动态压缩策略

### 视觉BPE（Byte Pair Encoding）

**Token合并算法**：
```python
def visual_bpe(tokens, vocab_size):
    # 初始化：每个patch为一个token
    sequences = initialize_sequences(tokens)

    while len(unique_tokens(sequences)) < vocab_size:
        # 找最频繁的token对
        pair = find_most_frequent_pair(sequences)

        # 合并为新token
        new_token = merge_pair(pair)

        # 更新序列
        sequences = replace_pair(sequences, pair, new_token)

    return sequences, vocabulary
```

**层级BPE**：
$$\text{BPE}_l = \text{Merge}(\text{BPE}_{l-1}, \text{threshold}_l)$$

不同层级使用不同合并阈值。

### 内容感知压缩

**重要性评分**：
$$\text{Importance}(\mathbf{t}_i) = \alpha \cdot \text{Attention}(\mathbf{t}_i) + \beta \cdot \text{Gradient}(\mathbf{t}_i) + \gamma \cdot \text{Variance}(\mathbf{t}_i)$$

**Token剪枝**：
```python
def token_pruning(tokens, keep_ratio):
    # 计算重要性分数
    scores = compute_importance(tokens)

    # 保留top-k
    k = int(len(tokens) * keep_ratio)
    indices = torch.topk(scores, k).indices

    # 返回剪枝后的tokens
    return tokens[indices], indices
```

**动态池化**：
$$\mathbf{t}_{merged} = \sum_{i \in \mathcal{N}} w_i \cdot \mathbf{t}_i$$

其中权重$w_i$基于相似度动态计算。

### 学习型压缩

**端到端压缩网络**：
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda \cdot R$$

其中$R$为码率（bits per pixel）。

**熵编码**：
```python
def entropy_encode(tokens):
    # 估计概率分布
    probs = estimate_distribution(tokens)

    # 算术编码
    bitstream = arithmetic_encode(tokens, probs)

    # 理论熵
    entropy = -sum(p * log2(p) for p in probs)

    return bitstream, entropy
```

**可变长度编码**：
频繁token使用短码：
$$L(t) = \lceil -\log_2 P(t) \rceil$$

### 时序冗余压缩

**运动补偿压缩**：
$$\mathbf{t}_t = \mathbf{t}_{ref} + \Delta \mathbf{t}$$

只编码残差$\Delta \mathbf{t}$。

**关键帧策略**：
```python
def keyframe_compression(video_tokens):
    compressed = []
    for i, tokens in enumerate(video_tokens):
        if is_keyframe(i):
            # 完整编码
            compressed.append(full_encode(tokens))
        else:
            # 差分编码
            ref = find_reference(i)
            diff = tokens - video_tokens[ref]
            compressed.append(differential_encode(diff))
    return compressed
```

**Rule of Thumb**：
- 压缩率目标：10-50倍
- 关键帧间隔：30-60帧
- Token保留率：20-50%
- 熵编码开销：<10% extra

## 4.4 Grounding与语义对齐

### 视觉-语言对齐

**CLIP对齐损失**：
$$\mathcal{L}_{CLIP} = -\log\frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j)/\tau)}$$

其中$\mathbf{v}_i, \mathbf{t}_i$为配对的视觉和文本特征。

**细粒度对齐**：
```python
def fine_grained_alignment(visual_tokens, text_tokens):
    # 计算相似度矩阵
    similarity = visual_tokens @ text_tokens.T

    # 最优传输对齐
    alignment = optimal_transport(similarity)

    # 对齐损失
    loss = -torch.sum(alignment * similarity)
    return loss, alignment
```

### 物体级Grounding

**区域-词对齐**：
$$\mathcal{A}_{ij} = \frac{\exp(f(\mathbf{r}_i, \mathbf{w}_j))}{\sum_k \exp(f(\mathbf{r}_i, \mathbf{w}_k))}$$

其中$\mathbf{r}_i$为区域特征，$\mathbf{w}_j$为词嵌入。

**Grounding Transformer**：
```python
class GroundingTransformer(nn.Module):
    def forward(self, visual_tokens, text_tokens):
        # Cross-modal attention
        visual_grounded = cross_attention(
            query=visual_tokens,
            key=text_tokens,
            value=text_tokens
        )

        # Reverse grounding
        text_grounded = cross_attention(
            query=text_tokens,
            key=visual_tokens,
            value=visual_tokens
        )

        return visual_grounded, text_grounded
```

### 语义一致性约束

**跨模态循环一致性**：
$$\mathcal{L}_{cycle} = \|\mathbf{v} - D_v(E_t(D_t(E_v(\mathbf{v}))))\|^2$$

视觉→文本→视觉的循环。

**语义保持损失**：
$$\mathcal{L}_{semantic} = \sum_{(i,j) \in \mathcal{P}} \|s(\mathbf{t}_i, \mathbf{t}_j) - s(\mathbf{v}_i, \mathbf{v}_j)\|^2$$

保持语义相似性结构。

### 层级语义对齐

**多粒度对齐**：
```
像素级 ──> 对象级 ──> 场景级 ──> 事件级
  │          │          │          │
  ↓          ↓          ↓          ↓
颜色纹理  物体类别   场景理解   行为理解
```

**层级对齐损失**：
$$\mathcal{L}_{hierarchical} = \sum_{l=1}^L \lambda_l \mathcal{L}_{align}^l$$

不同层级使用不同权重。

**Rule of Thumb**：
- 对齐温度：0.07-0.1
- 负样本数：1024-4096
- 语义维度：512-768
- 对齐层数：最后3-4层

## 4.5 Tokenizer Loss设计与优化

### 重构损失设计

**多尺度重构**：
$$\mathcal{L}_{recon} = \sum_{s \in \{1, 1/2, 1/4\}} \lambda_s \|\mathbf{x}_s - \hat{\mathbf{x}}_s\|_p$$

不同尺度使用不同范数（$p \in \{1, 2\}$）。

**感知损失**：
$$\mathcal{L}_{perceptual} = \sum_{l} \lambda_l \|\phi_l(\mathbf{x}) - \phi_l(\hat{\mathbf{x}})\|_2$$

其中$\phi_l$为预训练网络的第$l$层特征。

**对抗损失**：
$$\mathcal{L}_{GAN} = \mathbb{E}[\log D(\mathbf{x})] + \mathbb{E}[\log(1 - D(\hat{\mathbf{x}}))]$$

### 正则化设计

**码本正则化**：
```python
def codebook_regularization(codebook, tokens_freq):
    # 使用率正则化
    usage_loss = -entropy(tokens_freq)

    # 正交性正则化
    gram = codebook @ codebook.T
    ortho_loss = ||gram - I||_F

    # 多样性正则化
    diversity_loss = -log_det(gram + εI)

    return usage_loss + ortho_loss + diversity_loss
```

**Commitment Loss**：
$$\mathcal{L}_{commit} = \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|_2^2 + \beta \|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2$$

其中sg为stop gradient操作。

### 任务特定损失

**下游任务对齐**：
$$\mathcal{L}_{task} = \mathcal{L}_{cls}(\text{Classifier}(\mathbf{z}), y)$$

确保token包含任务相关信息。

**时序连续性损失**：
$$\mathcal{L}_{temporal} = \sum_{t} \|\mathbf{z}_t - \mathbf{z}_{t-1}\|_2 \cdot \exp(-\|\mathbf{v}_t - \mathbf{v}_{t-1}\|_2)$$

运动小时token应该相似。

### 优化策略

**两阶段训练**：
```python
# Stage 1: 训练tokenizer
for epoch in range(stage1_epochs):
    loss = recon_loss + reg_loss
    optimize_tokenizer(loss)

# Stage 2: 固定tokenizer，训练下游
freeze(tokenizer)
for epoch in range(stage2_epochs):
    tokens = tokenizer.encode(x)
    loss = downstream_task_loss(tokens)
    optimize_downstream(loss)
```

**渐进式训练**：
逐步增加复杂度：
1. 先训练空间tokenizer
2. 加入时序建模
3. 加入语义对齐
4. 微调整体

**EMA更新码本**：
$$\mathbf{e}_k^{(t+1)} = \gamma \mathbf{e}_k^{(t)} + (1-\gamma) \mathbf{z}_{avg}^{(k)}$$

**Rule of Thumb**：
- 重构损失权重：1.0
- 感知损失权重：0.1-0.5
- Commitment损失：0.25
- EMA动量：0.99-0.999

## 高级话题：Vector Quantization与Diffusion超分辨率融合

### VQ-Diffusion架构

**两阶段生成**：
```
Stage 1: Discrete tokens ──> VQ Decoder ──> Coarse image
Stage 2: Coarse image ──> Diffusion ──> High-res image
```

**条件Diffusion细化**：
$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{c}) = \mathcal{N}(\mu_\theta(\mathbf{x}_t, t, \mathbf{c}), \sigma_t^2\mathbf{I})$$

其中$\mathbf{c}$为VQ解码的粗糙图像。

### 混合离散-连续建模

**Residual VQ**：
$$\mathbf{x} = \sum_{l=1}^L \mathbf{d}_l$$

其中$\mathbf{d}_l$为第$l$层的离散码本解码。

**连续细节叠加**：
$$\mathbf{x}_{final} = \mathbf{x}_{VQ} + \alpha \cdot \mathbf{x}_{continuous}$$

离散捕获结构，连续捕获细节。

### 级联超分辨率

**多级级联**：
```python
def cascaded_super_resolution(tokens):
    # Level 0: 64x64
    x_64 = vq_decode(tokens)

    # Level 1: 128x128
    x_128 = diffusion_sr_128(x_64)

    # Level 2: 256x256
    x_256 = diffusion_sr_256(x_128)

    # Level 3: 512x512
    x_512 = diffusion_sr_512(x_256)

    return x_512
```

**Token引导的超分**：
$$\epsilon_\theta(\mathbf{x}_t, t, \mathbf{z}) = \epsilon_\theta^{uncond}(\mathbf{x}_t, t) + s \cdot (\epsilon_\theta^{cond}(\mathbf{x}_t, t, \mathbf{z}) - \epsilon_\theta^{uncond}(\mathbf{x}_t, t))$$

### 自适应质量控制

**质量感知路由**：
```python
def quality_aware_generation(tokens, quality_target):
    if quality_target == 'fast':
        return vq_decode(tokens)
    elif quality_target == 'balanced':
        coarse = vq_decode(tokens)
        return light_diffusion(coarse, steps=10)
    else:  # high quality
        coarse = vq_decode(tokens)
        fine = heavy_diffusion(coarse, steps=50)
        return super_resolution(fine)
```

**动态步数调整**：
根据内容复杂度调整diffusion步数：
$$T_{steps} = T_{base} \cdot (1 + \alpha \cdot \text{Complexity}(\mathbf{z}))$$

## 本章小结

本章系统介绍了视觉编码器与tokenization的核心技术：

**关键概念**：
1. **连续vs离散权衡**：VQ-VAE、软量化、混合表示
2. **Tokenizer架构**：Patch分割、ViT编码器、多尺度设计
3. **压缩策略**：视觉BPE、内容感知压缩、熵编码
4. **语义对齐**：CLIP对齐、Grounding、层级语义
5. **损失设计**：重构损失、正则化、任务特定损失

**核心公式**：
- VQ量化：$\mathbf{z}_q = \arg\min_{\mathbf{e}_k \in \mathcal{C}} \|\mathbf{z}_e - \mathbf{e}_k\|_2$
- CLIP对齐：$\mathcal{L}_{CLIP} = -\log\frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j)/\tau)}$
- Gumbel-Softmax：$\mathbf{z} = \sum_k \frac{\exp((l_k + g_k)/\tau)}{\sum_j \exp((l_j + g_j)/\tau)} \mathbf{e}_k$
- 感知损失：$\mathcal{L}_{perceptual} = \sum_l \|\phi_l(\mathbf{x}) - \phi_l(\hat{\mathbf{x}})\|_2$

**核心论文**：
1. [VQ-VAE] **Neural Discrete Representation Learning**, NeurIPS 2017
2. [ViT] **An Image is Worth 16x16 Words**, ICLR 2021
3. [CLIP] **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021
4. [BEiT] **BEiT: BERT Pre-Training of Image Transformers**, ICLR 2022
5. [MAGVIT] **Masked Generative Video Transformer**, CVPR 2023

## 常见陷阱与错误 (Gotchas)

### 1. 码本崩塌
**问题**：大部分码本向量未被使用
**症状**：重构质量差，多样性低
**解决**：
- 码本重置机制
- 使用率正则化
- EMA更新替代梯度更新

### 2. 训练不稳定
**问题**：VQ训练早期振荡
**症状**：Loss剧烈波动，码本频繁变化
**解决**：
- 预热学习率
- Commitment loss权重调整
- 分阶段训练

### 3. Token长度爆炸
**问题**：高分辨率导致token序列过长
**症状**：内存溢出，推理慢
**解决**：
- 层级tokenization
- 动态token合并
- 稀疏注意力

### 4. 语义对齐失败
**问题**：视觉和文本特征不对齐
**症状**：检索性能差，grounding不准
**解决**：
- 温度参数调优
- 困难负样本挖掘
- 多级对齐损失

### 5. 压缩artifacts
**问题**：过度压缩导致块效应
**症状**：重构图像有明显边界
**解决**：
- 重叠patch
- 后处理平滑
- 自适应压缩率

### 6. 时序不连续
**问题**：相邻帧token变化剧烈
**症状**：视频闪烁，不自然
**解决**：
- 时序平滑损失
- 运动感知tokenization
- 帧间共享码本

### 7. 梯度消失
**问题**：深层encoder梯度消失
**症状**：深层不更新，性能饱和
**解决**：
- 残差连接
- 层归一化
- 梯度裁剪

### 8. 分辨率泛化差
**问题**：只在固定分辨率训练
**症状**：其他分辨率效果差
**解决**：
- 多分辨率训练
- 位置编码插值
- 尺度不变设计