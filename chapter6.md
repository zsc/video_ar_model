# 第6章：MoE架构与规模化

Mixture of Experts (MoE)是实现模型规模化的关键技术，能够在不成比例增加计算成本的情况下扩展模型容量。本章深入探讨MoE在视频自回归模型中的设计与优化，包括expert粒度选择、router机制、负载均衡，以及训练稳定性等关键问题。

## 6.1 MoE设计动机与原理

### 条件计算的核心思想

**稀疏激活原理**：
```
Dense Model:  All parameters active for all inputs
              Compute = O(N × d²)

MoE Model:    k<<N experts active per input
              Compute = O(k × d²)
              Capacity = O(N × d²)
```

容量与计算解耦，实现高效扩展。

**生物学启发**：
大脑神经元稀疏激活（2-4%），不同区域处理不同信息。

### MoE数学框架

**基本形式**：
$$y = \sum_{i=1}^N g_i(x) \cdot E_i(x)$$

其中：
- $E_i$：第i个expert网络
- $g_i(x)$：router对expert i的权重
- $\sum_i g_i(x) = 1$（归一化）

**Top-k选择**：
$$y = \sum_{i \in \text{TopK}(g(x))} \frac{g_i(x)}{\sum_{j \in \text{TopK}} g_j(x)} \cdot E_i(x)$$

只激活得分最高的k个experts。

### 自动驾驶场景的MoE动机

**场景多样性**：
```
场景类型        所需能力              适合的Expert
────────────────────────────────────────────────
高速公路        车道保持、跟车        运动预测专家
城市路口        多目标交互           交互建模专家
停车场          精确定位             几何推理专家
恶劣天气        鲁棒感知             去噪增强专家
```

**计算效率需求**：
- 实时推理：<100ms延迟
- 边缘部署：<10W功耗
- 大模型能力：20B+参数

**Rule of Thumb**：
- Expert数量：8-64个
- 激活专家数：1-2个
- 专家容量：总容量的1/8到1/4

## 6.2 Expert粒度与数量权衡

### 粗粒度Expert设计

**任务级Expert**：
```python
class TaskExpert(nn.Module):
    def __init__(self, d_model, task_type):
        super().__init__()
        self.task_type = task_type

        if task_type == 'motion':
            # 运动预测专家
            self.layers = MotionPredictionLayers(d_model)
        elif task_type == 'interaction':
            # 交互建模专家
            self.layers = InteractionModelingLayers(d_model)
        elif task_type == 'perception':
            # 感知增强专家
            self.layers = PerceptionEnhancementLayers(d_model)
```

**模态级Expert**：
```python
def modality_experts(features, modality_type):
    experts = {
        'visual': VisualExpert(),
        'trajectory': TrajectoryExpert(),
        'semantic': SemanticExpert(),
        'temporal': TemporalExpert()
    }

    return experts[modality_type](features)
```

### 细粒度Expert设计

**Token级Expert**：
每个token独立选择expert：
```python
class TokenLevelMoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(d_model) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        router_logits = self.router(x)  # [batch, seq_len, num_experts]

        # Top-k selection per token
        topk_logits, topk_indices = router_logits.topk(self.top_k, dim=-1)
        topk_gates = F.softmax(topk_logits, dim=-1)

        # Dispatch to experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]
            gate = topk_gates[..., i:i+1]

            # Gather tokens for each expert
            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += gate[mask] * expert_output

        return output
```

### 层级Expert混合

**浅层细粒度+深层粗粒度**：
```python
class HierarchicalMoE(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i < num_layers // 3:
                # 浅层：细粒度，多expert
                layer = TokenLevelMoE(d_model, num_experts=64, top_k=2)
            elif i < 2 * num_layers // 3:
                # 中层：中等粒度
                layer = SequenceLevelMoE(d_model, num_experts=16, top_k=4)
            else:
                # 深层：粗粒度，少expert
                layer = TaskLevelMoE(d_model, num_experts=4, top_k=1)

            self.layers.append(layer)
```

### Expert容量设计

**固定容量**：
$$\text{Capacity} = \frac{n \cdot k}{N} \cdot (1 + \epsilon)$$

其中n为token数，k为top-k，N为expert数，ε为buffer因子。

**动态容量**：
```python
def dynamic_capacity(router_probs, min_capacity, max_capacity):
    # 根据路由概率动态分配
    expected_tokens = router_probs.sum(dim=0)  # 每个expert期望token数

    capacities = []
    for exp_tokens in expected_tokens:
        capacity = torch.clamp(
            exp_tokens * 1.25,  # 25% buffer
            min=min_capacity,
            max=max_capacity
        )
        capacities.append(int(capacity))

    return capacities
```

**Rule of Thumb**：
- Token级：64-128 experts
- 序列级：8-16 experts
- 任务级：2-4 experts
- 容量buffer：25-50%

## 6.3 Router设计与负载均衡

### Router架构

**线性Router**：
```python
class LinearRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        return self.w_gate(x)
```

**注意力Router**：
```python
class AttentionRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, d_model)
        )

    def forward(self, x):
        # 计算与每个expert的相似度
        scores = torch.einsum('bsd,ed->bse', x, self.expert_embeddings)
        return scores / np.sqrt(d_model)
```

**层级Router**：
```python
class HierarchicalRouter(nn.Module):
    def __init__(self, d_model, num_groups, experts_per_group):
        super().__init__()
        self.group_router = nn.Linear(d_model, num_groups)
        self.expert_routers = nn.ModuleList([
            nn.Linear(d_model, experts_per_group)
            for _ in range(num_groups)
        ])

    def forward(self, x):
        # 先选组
        group_logits = self.group_router(x)
        group_probs = F.softmax(group_logits, dim=-1)

        # 组内选expert
        expert_logits = []
        for i, router in enumerate(self.expert_routers):
            expert_logits.append(router(x) * group_probs[:, :, i:i+1])

        return torch.stack(expert_logits, dim=-1).flatten(-2)
```

### 负载均衡机制

**负载均衡损失**：
$$\mathcal{L}_{balance} = N \cdot \sum_{i=1}^N f_i \cdot P_i$$

其中$f_i$为expert i的token比例，$P_i$为router概率均值。

**实现**：
```python
def load_balancing_loss(router_probs, num_experts):
    # router_probs: [batch, seq_len, num_experts]

    # Token分布
    tokens_per_expert = router_probs.sum(dim=[0, 1])
    f = tokens_per_expert / tokens_per_expert.sum()

    # Router概率分布
    P = router_probs.mean(dim=[0, 1])

    # 负载均衡损失
    loss = num_experts * (f * P).sum()
    return loss
```

### 专家选择vs Token选择

**Expert-Choice Routing**：
Expert选择要处理的token：
```python
def expert_choice_routing(x, router_logits, expert_capacity):
    batch, seq_len, num_experts = router_logits.shape

    # 每个expert选择top-k tokens
    expert_gates = []
    expert_indices = []

    for e in range(num_experts):
        expert_scores = router_logits[:, :, e].flatten()

        # Expert选择得分最高的tokens
        k = min(expert_capacity, len(expert_scores))
        topk_scores, topk_idx = expert_scores.topk(k)

        expert_gates.append(F.softmax(topk_scores, dim=0))
        expert_indices.append(topk_idx)

    return expert_gates, expert_indices
```

**Token-Choice Routing**：
Token选择要去的expert（传统方式）。

### 确定性vs随机路由

**确定性路由**：
```python
def deterministic_routing(x, router):
    logits = router(x)
    # Hard top-k
    _, indices = logits.topk(k, dim=-1)
    return indices
```

**随机路由（训练时）**：
```python
def stochastic_routing(x, router, temperature=1.0):
    logits = router(x) / temperature

    # Gumbel-Softmax采样
    gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
    logits_with_noise = logits + gumbel

    # Soft top-k
    return F.softmax(logits_with_noise, dim=-1)
```

**Rule of Thumb**：
- 负载均衡损失权重：0.01-0.1
- Router温度：1.0-5.0
- Expert容量：1.25×平均负载
- 随机路由比例：10-20%

## 6.4 Common Expert与知识共享

### Common Expert设计

**共享+专用架构**：
```python
class MoEWithCommonExpert(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        # 共享专家（always active）
        self.common_expert = FeedForward(d_model, expand_factor=2)

        # 专用专家
        self.specialized_experts = nn.ModuleList([
            FeedForward(d_model, expand_factor=4)
            for _ in range(num_experts)
        ])

        self.router = Router(d_model, num_experts)

    def forward(self, x):
        # 共享路径
        common_output = self.common_expert(x)

        # 专用路径
        specialized_output = dispatch_to_experts(
            x, self.specialized_experts, self.router
        )

        # 组合
        return common_output + specialized_output
```

### 知识蒸馏与共享

**Expert间知识蒸馏**：
```python
def expert_distillation_loss(expert_outputs):
    # expert_outputs: list of [batch, seq_len, d_model]

    loss = 0
    num_pairs = 0

    for i in range(len(expert_outputs)):
        for j in range(i+1, len(expert_outputs)):
            # KL散度
            loss += F.kl_div(
                F.log_softmax(expert_outputs[i], dim=-1),
                F.softmax(expert_outputs[j], dim=-1),
                reduction='batchmean'
            )
            num_pairs += 1

    return loss / num_pairs
```

### Mixture of Depths

**深度选择机制**：
```python
class MixtureOfDepths(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(num_layers)
        ])
        self.depth_router = nn.Linear(d_model, num_layers)

    def forward(self, x):
        # 决定每个token的处理深度
        depth_logits = self.depth_router(x)  # [batch, seq_len, num_layers]
        depth_probs = F.sigmoid(depth_logits)

        # 渐进处理
        hidden = x
        for i, layer in enumerate(self.layers):
            # 概率性跳过
            mask = depth_probs[:, :, i:i+1]
            hidden = mask * layer(hidden) + (1 - mask) * hidden

        return hidden
```

### 参数共享策略

**层间参数共享**：
```python
class SharedExperts(nn.Module):
    def __init__(self, d_model, num_experts, share_ratio=0.5):
        super().__init__()
        shared_dim = int(d_model * share_ratio)
        private_dim = d_model - shared_dim

        # 共享参数
        self.shared_params = nn.Linear(d_model, shared_dim)

        # 专用参数
        self.private_params = nn.ModuleList([
            nn.Linear(d_model, private_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x, expert_idx):
        shared_feat = self.shared_params(x)
        private_feat = self.private_params[expert_idx](x)
        return torch.cat([shared_feat, private_feat], dim=-1)
```

**Rule of Thumb**：
- Common expert大小：25-50%总容量
- 参数共享比例：30-50%
- 蒸馏损失权重：0.1-0.5
- 深度选择：平均激活50-70%层

## 6.5 辅助Loss与训练稳定性

### 辅助损失设计

**Router z-loss**：
减少router logits幅度，防止过拟合：
$$\mathcal{L}_z = \frac{1}{B \cdot S} \sum_{i,j} \log^2(1 + e^{x_{ij}})$$

**实现**：
```python
def router_z_loss(router_logits):
    # 防止router过度自信
    z_loss = torch.logsumexp(router_logits, dim=-1) ** 2
    return z_loss.mean()
```

**专家多样性损失**：
```python
def expert_diversity_loss(expert_outputs):
    # 鼓励不同expert产生不同输出
    num_experts = len(expert_outputs)
    similarity_matrix = torch.zeros(num_experts, num_experts)

    for i in range(num_experts):
        for j in range(i+1, num_experts):
            # 余弦相似度
            sim = F.cosine_similarity(
                expert_outputs[i].flatten(),
                expert_outputs[j].flatten(),
                dim=0
            )
            similarity_matrix[i, j] = sim

    # 最小化相似度
    return similarity_matrix.mean()
```

### 训练稳定性技巧

**梯度裁剪策略**：
```python
def adaptive_gradient_clipping(model, max_norm=1.0):
    # 每个expert独立裁剪
    for name, param in model.named_parameters():
        if 'expert' in name:
            # Expert参数
            expert_idx = int(name.split('expert')[1].split('.')[0])
            torch.nn.utils.clip_grad_norm_(
                param, max_norm * (1 + 0.1 * expert_idx)
            )
        else:
            # 共享参数
            torch.nn.utils.clip_grad_norm_(param, max_norm)
```

**专家Dropout**：
```python
class ExpertDropout(nn.Module):
    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, expert_outputs, training=True):
        if not training:
            return expert_outputs

        # 随机丢弃部分expert
        keep_prob = 1 - self.drop_rate
        mask = torch.bernoulli(
            torch.full((len(expert_outputs),), keep_prob)
        )

        # 重新归一化
        scaled_outputs = []
        for i, output in enumerate(expert_outputs):
            if mask[i]:
                scaled_outputs.append(output / keep_prob)
            else:
                scaled_outputs.append(torch.zeros_like(output))

        return scaled_outputs
```

### 初始化策略

**Router初始化**：
```python
def init_router(router, num_experts):
    # 均匀初始化，避免初期不平衡
    nn.init.zeros_(router.weight)
    nn.init.normal_(router.weight, std=0.01)

    # 添加噪声打破对称性
    with torch.no_grad():
        router.weight += torch.randn_like(router.weight) * 0.001
```

**Expert初始化差异化**：
```python
def init_experts_diverse(experts):
    for i, expert in enumerate(experts):
        # 不同的初始化策略
        if i % 3 == 0:
            nn.init.xavier_uniform_(expert.weight)
        elif i % 3 == 1:
            nn.init.kaiming_uniform_(expert.weight)
        else:
            nn.init.orthogonal_(expert.weight)

        # 不同的初始scale
        expert.weight.data *= (1 + 0.1 * i)
```

### 训练调度

**渐进激活**：
```python
class ProgressiveActivation:
    def __init__(self, num_experts, warmup_steps):
        self.num_experts = num_experts
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_active_experts(self):
        if self.current_step >= self.warmup_steps:
            return self.num_experts

        # 线性增加
        progress = self.current_step / self.warmup_steps
        active = int(1 + progress * (self.num_experts - 1))
        return max(1, active)

    def step(self):
        self.current_step += 1
```

**Rule of Thumb**：
- z-loss权重：0.001
- 多样性损失权重：0.01
- 梯度裁剪：1.0-10.0
- Expert dropout：0.1-0.2
- 预热步数：1000-5000

## 高级话题：Soft MoE与连续路由机制

### Soft MoE原理

**软分配机制**：
不是选择top-k，而是软组合所有experts：
$$y = \sum_{i=1}^N \text{Softmax}(\phi(x, E_i)) \cdot E_i(x)$$

**Slot-based Soft MoE**：
```python
class SoftMoE(nn.Module):
    def __init__(self, d_model, num_experts, num_slots):
        super().__init__()
        self.num_slots = num_slots

        # Slot embeddings
        self.slot_embeds = nn.Parameter(torch.randn(num_slots, d_model))

        # Experts
        self.experts = nn.ModuleList([
            FeedForward(d_model) for _ in range(num_experts)
        ])

        # Dispatch and combine weights
        self.dispatch = nn.Linear(d_model, num_slots)
        self.combine = nn.Linear(d_model, num_slots)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Dispatch: tokens -> slots
        dispatch_weights = F.softmax(self.dispatch(x), dim=1)  # [B, S, num_slots]
        slots = torch.einsum('bsd,bsn->bnd', x, dispatch_weights)

        # Process through experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(slots))

        # Weighted combination
        combined = sum(expert_outputs) / len(self.experts)

        # Combine: slots -> tokens
        combine_weights = F.softmax(self.combine(x), dim=2)  # [B, S, num_slots]
        output = torch.einsum('bnd,bsn->bsd', combined, combine_weights)

        return output
```

### 连续路由空间

**可微分路由**：
```python
class DifferentiableRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )

    def forward(self, x):
        logits = self.router(x)

        # Soft routing with learnable temperature
        routing_weights = F.softmax(logits / self.temperature, dim=-1)

        # Entropy regularization for exploration
        entropy = -(routing_weights * routing_weights.log()).sum(-1).mean()

        return routing_weights, entropy
```

### 动态Expert生成

**Meta-learning Expert**：
```python
class MetaExpert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 超网络生成expert参数
        self.hyper_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * d_model)
        )

    def forward(self, x, context):
        # 根据context生成expert参数
        batch_size = x.shape[0]

        # 生成权重
        weights = self.hyper_net(context)
        weights = weights.view(batch_size, d_model, d_model)

        # 应用生成的expert
        output = torch.bmm(x.unsqueeze(1), weights).squeeze(1)
        return output
```

### Conditional Computation

**条件计算图**：
```python
class ConditionalMoE(nn.Module):
    def __init__(self, d_model, condition_dim):
        super().__init__()
        # 条件编码器
        self.condition_encoder = nn.Linear(condition_dim, d_model)

        # 条件化experts
        self.experts = nn.ModuleList([
            ConditionalExpert(d_model) for _ in range(8)
        ])

    def forward(self, x, condition):
        # 编码条件
        cond_embedding = self.condition_encoder(condition)

        # 条件化路由
        router_logits = torch.einsum('bsd,d->bs', x, cond_embedding)

        # Soft routing
        routing_weights = F.softmax(router_logits, dim=-1)

        # 加权组合
        output = 0
        for i, expert in enumerate(self.experts):
            weight = routing_weights[:, i:i+1, None]
            output += weight * expert(x, cond_embedding)

        return output
```

## 本章小结

本章系统介绍了MoE架构在视频自回归模型中的设计与优化：

**关键概念**：
1. **MoE原理**：稀疏激活、条件计算、容量vs计算解耦
2. **Expert设计**：粒度选择、数量权衡、容量分配
3. **Router机制**：负载均衡、选择策略、确定性vs随机
4. **知识共享**：Common expert、参数共享、蒸馏
5. **训练稳定性**：辅助损失、初始化、渐进训练

**核心公式**：
- MoE输出：$y = \sum_{i \in \text{TopK}} g_i(x) \cdot E_i(x)$
- 负载均衡：$\mathcal{L}_{balance} = N \cdot \sum_i f_i \cdot P_i$
- Router z-loss：$\mathcal{L}_z = \sum_{ij} \log^2(1 + e^{x_{ij}})$
- Soft MoE：$y = \sum_i \text{Softmax}(\phi(x, E_i)) \cdot E_i(x)$

**核心论文**：
1. [GShard] **GShard: Scaling Giant Models with Conditional Computation**, ICLR 2021
2. [Switch Transformer] **Switch Transformers: Scaling to Trillion Parameter Models**, JMLR 2022
3. [Expert Choice] **Mixture-of-Experts with Expert Choice Routing**, NeurIPS 2022
4. [Soft MoE] **From Sparse to Soft Mixtures of Experts**, ICLR 2024
5. [MoE Scale] **Unified Scaling Laws for Routed Language Models**, ICML 2022

## 常见陷阱与错误 (Gotchas)

### 1. 负载不均衡
**问题**：少数experts处理大部分tokens
**症状**：计算效率低，部分experts未充分训练
**解决**：
- 增加负载均衡损失权重
- 使用expert-choice routing
- 添加噪声打破对称性

### 2. Router坍缩
**问题**：Router退化为固定模式
**症状**：所有输入路由到同样experts
**解决**：
- Router正则化（z-loss）
- Dropout和噪声注入
- 温度退火策略

### 3. 专家同质化
**问题**：不同experts学到相似功能
**症状**：模型容量未充分利用
**解决**：
- 多样性损失
- 差异化初始化
- 专家特化预训练

### 4. 训练不稳定
**问题**：Loss震荡，难以收敛
**症状**：训练曲线剧烈波动
**解决**：
- 降低学习率
- 渐进激活experts
- 每层独立梯度裁剪

### 5. 推理延迟增加
**问题**：动态路由导致延迟不可预测
**症状**：P99延迟远高于平均值
**解决**：
- 推理时固定路由
- 缓存路由决策
- 批处理优化

### 6. 内存碎片化
**问题**：不规则dispatch导致内存碎片
**症状**：OOM但显存未满
**解决**：
- 固定容量缓冲区
- 内存池预分配
- Padding到固定大小

### 7. 通信开销
**问题**：分布式训练all-to-all通信昂贵
**症状**：多卡扩展效率低
**解决**：
- Expert并行优化
- 层级路由减少通信
- 本地Expert优先

### 8. 量化困难
**问题**：不同experts量化敏感度不同
**症状**：量化后性能严重下降
**解决**：
- Per-expert量化
- 混合精度策略
- 量化感知训练