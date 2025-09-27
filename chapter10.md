# 第10章：监督微调策略（Supervised Fine-tuning Strategies）

监督微调（SFT）是将预训练的视频自回归模型适配到特定自动驾驶任务的关键步骤。与传统的视觉-语言模型不同，自动驾驶场景下的SFT需要同时考虑多模态输入、时序一致性、安全性约束和实时性要求。本章将深入探讨如何设计高效的监督微调策略，包括高质量标注数据收集、多阶段微调流程、灾难性遗忘缓解、少样本泛化和领域适应技术。

## 10.1 高质量标注数据收集与筛选

### 10.1.1 人工标注与数据质量控制

自动驾驶场景的标注数据需要极高的准确性和一致性。标注流程设计需要考虑：

**多层次标注体系**：
```
标注层级结构：
L0: 基础感知标注（物体框、车道线、红绿灯）
L1: 语义理解标注（意图识别、场景分类）
L2: 行为预测标注（轨迹、交互关系）
L3: 决策推理标注（驾驶决策链、风险评估）
```

**标注质量评估指标**：
- **一致性评分**：$S_{consistency} = \frac{1}{N}\sum_{i=1}^{N} IoU(A_i, \bar{A})$，其中$A_i$是第$i$个标注员的结果，$\bar{A}$是共识结果
- **完整性评分**：$S_{completeness} = \frac{|D_{annotated}|}{|D_{required}|}$，覆盖所有必需的标注维度
- **时序连续性**：$S_{temporal} = 1 - \frac{1}{T-1}\sum_{t=1}^{T-1}||f_t - f_{t+1}||_2$，相邻帧标注的平滑度

**主动学习采样策略**：
```
不确定性采样：
P(x) ∝ H(y|x) = -∑_y p(y|x) log p(y|x)

多样性采样：
P(x) ∝ min_{x' ∈ S} d(x, x')  # 与已标注集合S的最小距离

混合策略：
P(x) = α·P_uncertainty(x) + β·P_diversity(x) + γ·P_difficulty(x)
```

### 10.1.2 自动标注与伪标签生成

利用教师模型生成高质量伪标签可以大幅降低标注成本：

**教师模型集成**：
```python
class TeacherEnsemble:
    def __init__(self, teachers):
        self.teachers = teachers
        self.confidence_threshold = 0.9

    def generate_pseudo_labels(self, x):
        predictions = []
        confidences = []

        for teacher in self.teachers:
            pred, conf = teacher.predict_with_confidence(x)
            predictions.append(pred)
            confidences.append(conf)

        # 加权投票
        weights = softmax(confidences / temperature)
        consensus = weighted_vote(predictions, weights)

        # 置信度过滤
        final_confidence = compute_agreement_score(predictions)
        if final_confidence > self.confidence_threshold:
            return consensus, final_confidence
        return None, 0
```

**时序一致性约束**：
```
L_temporal = ∑_t ||f(x_t) - T_{t→t+1}(f(x_{t+1}))||_2
```
其中$T_{t→t+1}$是基于光流或运动模型的变换。

### 10.1.3 困难样本挖掘

**困难样本识别**：
- **预测不一致性**：模型在多次前向传播中预测结果差异大
- **梯度异常**：反向传播时梯度范数异常大或小
- **分布外检测**：特征空间中远离训练分布的样本

```python
class HardSampleMiner:
    def __init__(self, model, memory_bank):
        self.model = model
        self.memory_bank = memory_bank  # 存储典型样本特征

    def compute_hardness_score(self, x):
        # 预测不确定性
        uncertainty = self.compute_mc_dropout_uncertainty(x)

        # OOD分数
        features = self.model.encode(x)
        ood_score = self.compute_mahalanobis_distance(
            features, self.memory_bank
        )

        # 梯度信息
        grad_norm = self.compute_gradient_norm(x)

        return α * uncertainty + β * ood_score + γ * grad_norm
```

## 10.2 多阶段微调策略设计

### 10.2.1 渐进式解冻策略

逐步解冻模型层可以更好地保留预训练知识：

```python
class ProgressiveUnfreezing:
    def __init__(self, model, num_stages=4):
        self.model = model
        self.num_stages = num_stages
        self.layers = self.group_layers()

    def group_layers(self):
        # 将模型分为多个组
        total_layers = len(self.model.layers)
        group_size = total_layers // self.num_stages
        return [
            self.model.layers[i:i+group_size]
            for i in range(0, total_layers, group_size)
        ]

    def unfreeze_stage(self, stage):
        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻当前阶段及之后的层
        for i in range(stage, self.num_stages):
            for layer in self.layers[i]:
                for param in layer.parameters():
                    param.requires_grad = True
```

**学习率调度策略**：
```
不同层使用不同学习率：
lr_layer_i = lr_base * decay^(L-i)

其中L是总层数，decay通常取0.9-0.95
```

### 10.2.2 任务递进式微调

从简单任务逐步过渡到复杂任务：

```
Stage 1: 基础感知任务
- 物体检测
- 车道线检测
- 深度估计

Stage 2: 场景理解任务
- 语义分割
- 场景图生成
- 3D重建

Stage 3: 预测任务
- 轨迹预测
- 意图识别
- 风险评估

Stage 4: 决策任务
- 行为规划
- 轨迹规划
- 交互决策
```

**任务相关性矩阵**：
```python
def compute_task_affinity(task_i, task_j, shared_data):
    # 在共享数据上计算任务梯度
    grad_i = compute_gradient(task_i, shared_data)
    grad_j = compute_gradient(task_j, shared_data)

    # 梯度余弦相似度
    affinity = cosine_similarity(grad_i, grad_j)
    return affinity

# 构建任务依赖图
task_graph = build_dependency_graph(task_affinities)
training_order = topological_sort(task_graph)
```

### 10.2.3 混合精度与梯度累积

**混合精度训练**：
```python
class MixedPrecisionTrainer:
    def __init__(self, model):
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()

    def training_step(self, batch):
        with torch.cuda.amp.autocast():
            # 前向传播使用FP16
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, batch)

        # 反向传播
        self.scaler.scale(loss).backward()

        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        # 参数更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

**梯度累积策略**：
```python
accumulation_steps = 8  # 等效增大8倍batch size

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = compute_loss(outputs, batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 10.3 灾难性遗忘与正则化技术

### 10.3.1 弹性权重巩固（EWC）

通过Fisher信息矩阵保护重要参数：

```python
class EWC:
    def __init__(self, model, dataset, importance_weight=1000):
        self.model = model
        self.importance_weight = importance_weight
        self.fisher_matrix = self.compute_fisher_matrix(dataset)
        self.optimal_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }

    def compute_fisher_matrix(self, dataset):
        fisher = {}
        model.eval()

        for n, p in model.named_parameters():
            fisher[n] = torch.zeros_like(p)

        for batch in dataset:
            model.zero_grad()
            output = model(batch)
            loss = F.cross_entropy(output, batch['labels'])
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= len(dataset)

        return fisher

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher_matrix:
                loss += (self.fisher_matrix[n] *
                        (p - self.optimal_params[n]) ** 2).sum()
        return self.importance_weight * loss
```

### 10.3.2 记忆回放与经验池

**优先经验回放**：
```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = []
        self.priorities = []

    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # 替换优先级最低的样本
            min_idx = np.argmin(self.priorities)
            if priority > self.priorities[min_idx]:
                self.buffer[min_idx] = experience
                self.priorities[min_idx] = priority

    def sample(self, batch_size, beta=0.4):
        # 按优先级采样
        probs = np.array(self.priorities) ** beta
        probs /= probs.sum()

        indices = np.random.choice(
            len(self.buffer), batch_size, p=probs
        )

        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化重要性权重

        return experiences, weights, indices
```

### 10.3.3 知识蒸馏与教师-学生框架

**特征级蒸馏**：
```python
class FeatureDistillation:
    def __init__(self, teacher, student, temperature=3.0):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature

    def distillation_loss(self, x):
        # 获取中间层特征
        teacher_features = self.teacher.get_intermediate_features(x)
        student_features = self.student.get_intermediate_features(x)

        loss = 0
        for t_feat, s_feat in zip(teacher_features, student_features):
            # 特征对齐
            if t_feat.shape != s_feat.shape:
                s_feat = self.adapter(s_feat)  # 维度适配

            # MSE损失
            loss += F.mse_loss(s_feat, t_feat.detach())

            # 注意力图蒸馏
            t_att = self.compute_attention_map(t_feat)
            s_att = self.compute_attention_map(s_feat)
            loss += F.kl_div(
                F.log_softmax(s_att / self.temperature, dim=-1),
                F.softmax(t_att / self.temperature, dim=-1)
            )

        return loss
```

## 10.4 少样本学习与零样本泛化

### 10.4.1 元学习方法（MAML）

Model-Agnostic Meta-Learning适配到视频理解：

```python
class VideoMAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

    def inner_loop(self, support_set, num_steps=5):
        # 复制模型参数
        adapted_params = {
            n: p.clone()
            for n, p in self.model.named_parameters()
        }

        for _ in range(num_steps):
            # 在支持集上计算梯度
            loss = self.compute_loss(support_set, adapted_params)
            grads = torch.autograd.grad(loss, adapted_params.values())

            # 梯度下降更新
            adapted_params = {
                n: p - self.inner_lr * g
                for (n, p), g in zip(adapted_params.items(), grads)
            }

        return adapted_params

    def outer_loop(self, tasks):
        outer_loss = 0

        for task in tasks:
            # 内循环适应
            adapted_params = self.inner_loop(task['support'])

            # 在查询集上评估
            query_loss = self.compute_loss(
                task['query'], adapted_params
            )
            outer_loss += query_loss

        # 外循环更新
        outer_loss.backward()
        self.optimizer.step()
```

### 10.4.2 提示学习（Prompt Learning）

**视觉提示设计**：
```python
class VisualPrompt:
    def __init__(self, model, prompt_length=10):
        self.model = model
        self.prompt_length = prompt_length

        # 可学习的提示向量
        self.prompts = nn.Parameter(
            torch.randn(prompt_length, model.hidden_size)
        )

    def forward(self, x):
        # 将提示添加到输入序列
        batch_size = x.shape[0]
        prompts = self.prompts.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # 拼接提示和原始输入
        x_prompted = torch.cat([prompts, x], dim=1)

        # 前向传播
        output = self.model(x_prompted)

        # 移除提示部分的输出
        return output[:, self.prompt_length:]
```

**上下文提示优化**：
```
任务特定提示：
P_task = argmin_P L(f(x; P), y)

动态提示生成：
P = g(x; θ)  # 基于输入生成提示

层级提示：
每层使用不同的提示向量P_l
```

### 10.4.3 对比学习与原型网络

**原型网络实现**：
```python
class PrototypicalNetwork:
    def __init__(self, encoder):
        self.encoder = encoder

    def compute_prototypes(self, support_set):
        prototypes = {}
        for class_id, samples in support_set.items():
            # 计算类别原型（均值）
            embeddings = self.encoder(samples)
            prototypes[class_id] = embeddings.mean(dim=0)
        return prototypes

    def classify(self, query, prototypes):
        query_embedding = self.encoder(query)
        distances = {}

        for class_id, prototype in prototypes.items():
            # 欧氏距离
            distances[class_id] = torch.norm(
                query_embedding - prototype, p=2
            )

        # 转换为概率
        logits = -torch.stack(list(distances.values()))
        return F.softmax(logits, dim=0)
```

## 10.5 领域适应与迁移学习

### 10.5.1 域对抗训练（DANN）

**域判别器设计**：
```python
class DomainAdversarialNetwork:
    def __init__(self, feature_extractor, task_classifier,
                 domain_discriminator):
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.domain_discriminator = domain_discriminator

    def forward(self, x, alpha=1.0):
        # 特征提取
        features = self.feature_extractor(x)

        # 梯度反转层
        reversed_features = GradientReversal.apply(features, alpha)

        # 任务分类
        task_output = self.task_classifier(features)

        # 域判别
        domain_output = self.domain_discriminator(reversed_features)

        return task_output, domain_output

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

### 10.5.2 自适应批归一化

**域特定BN统计量**：
```python
class AdaptiveBatchNorm(nn.Module):
    def __init__(self, num_features, num_domains):
        super().__init__()
        self.num_domains = num_domains

        # 每个域的BN参数
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features)
            for _ in range(num_domains)
        ])

    def forward(self, x, domain_id):
        if self.training:
            # 训练时使用域特定的BN
            return self.bn_layers[domain_id](x)
        else:
            # 测试时混合所有域的统计量
            outputs = []
            for bn in self.bn_layers:
                outputs.append(bn(x))
            return torch.stack(outputs).mean(dim=0)
```

### 10.5.3 自监督域适应

**旋转预测辅助任务**：
```python
class RotationPrediction:
    def __init__(self, model):
        self.model = model
        self.rotations = [0, 90, 180, 270]

    def create_rotation_task(self, x):
        batch_size = x.shape[0]
        rotated_images = []
        labels = []

        for img in x:
            rot_id = random.choice(range(4))
            angle = self.rotations[rot_id]

            # 旋转图像
            rotated = torch.rot90(img, k=rot_id, dims=[-2, -1])
            rotated_images.append(rotated)
            labels.append(rot_id)

        return torch.stack(rotated_images), torch.tensor(labels)

    def auxiliary_loss(self, x):
        rotated_x, rot_labels = self.create_rotation_task(x)
        rot_predictions = self.model.rotation_head(
            self.model.encoder(rotated_x)
        )
        return F.cross_entropy(rot_predictions, rot_labels)
```

## 进阶专题：LoRA与参数高效微调在视频模型中的应用

### LoRA架构设计

Low-Rank Adaptation在大规模视频模型中的实现：

```python
class VideoLoRA(nn.Module):
    def __init__(self, base_model, rank=16, alpha=32):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.scaling = alpha / rank

        # 为每个注意力层添加LoRA适配器
        self.lora_layers = nn.ModuleDict()
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features

                # 低秩分解 A 和 B
                self.lora_layers[name + '_A'] = nn.Linear(
                    in_features, rank, bias=False
                )
                self.lora_layers[name + '_B'] = nn.Linear(
                    rank, out_features, bias=False
                )

                # 初始化
                nn.init.kaiming_uniform_(
                    self.lora_layers[name + '_A'].weight
                )
                nn.init.zeros_(self.lora_layers[name + '_B'].weight)

    def forward(self, x):
        # 基础模型前向传播
        base_output = self.base_model(x)

        # 添加LoRA增量
        for name, module in self.base_model.named_modules():
            if name + '_A' in self.lora_layers:
                lora_A = self.lora_layers[name + '_A']
                lora_B = self.lora_layers[name + '_B']

                # 计算低秩增量
                delta = lora_B(lora_A(x)) * self.scaling
                base_output = base_output + delta

        return base_output
```

### 动态秩选择

根据任务复杂度自适应调整LoRA秩：

```python
class AdaptiveRankLoRA:
    def __init__(self, base_model, initial_rank=8):
        self.base_model = base_model
        self.current_rank = initial_rank
        self.importance_scores = {}

    def compute_importance(self, gradients):
        # 基于梯度的奇异值分解
        U, S, V = torch.svd(gradients)

        # 计算有效秩
        normalized_singular = S / S.sum()
        cumsum = torch.cumsum(normalized_singular, dim=0)

        # 找到解释90%方差的秩
        effective_rank = (cumsum < 0.9).sum().item() + 1
        return min(effective_rank, self.max_rank)

    def update_rank(self, layer_name, new_rank):
        # 动态调整LoRA层的秩
        old_A = self.lora_layers[layer_name + '_A']
        old_B = self.lora_layers[layer_name + '_B']

        # 保留重要的奇异向量
        U, S, V = torch.svd(old_B.weight @ old_A.weight)

        # 创建新的LoRA层
        new_A = nn.Linear(old_A.in_features, new_rank, bias=False)
        new_B = nn.Linear(new_rank, old_B.out_features, bias=False)

        # 初始化为截断的SVD
        new_A.weight.data = V[:, :new_rank].T
        new_B.weight.data = U[:, :new_rank] @ torch.diag(S[:new_rank])

        self.lora_layers[layer_name + '_A'] = new_A
        self.lora_layers[layer_name + '_B'] = new_B
```

### 任务特定LoRA组合

```python
class MultiTaskLoRA:
    def __init__(self, base_model, tasks):
        self.base_model = base_model
        self.task_loras = {}

        for task_name in tasks:
            self.task_loras[task_name] = VideoLoRA(
                base_model, rank=tasks[task_name]['rank']
            )

    def forward(self, x, task_name, task_weights=None):
        if task_weights is None:
            # 单任务推理
            return self.task_loras[task_name](x)
        else:
            # 多任务加权组合
            output = self.base_model(x)
            for task, weight in task_weights.items():
                task_delta = self.task_loras[task](x) - self.base_model(x)
                output += weight * task_delta
            return output
```

## 本章小结

监督微调是将大规模预训练模型适配到具体任务的关键环节。本章介绍了从数据收集到模型适配的完整流程，重点讨论了以下核心技术：

1. **数据工程**：高质量标注、主动学习、困难样本挖掘
2. **微调策略**：渐进解冻、多阶段训练、混合精度优化
3. **知识保持**：EWC、记忆回放、知识蒸馏
4. **少样本学习**：元学习、提示学习、原型网络
5. **领域适应**：域对抗、自适应归一化、自监督适应
6. **参数高效**：LoRA及其变体在视频模型中的应用

### 关键概念与公式

- **Fisher信息矩阵**：$F = \mathbb{E}[\nabla_\theta \log p(x|\theta) \nabla_\theta \log p(x|\theta)^T]$
- **EWC正则化**：$L_{EWC} = L_{new} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$
- **知识蒸馏**：$L_{KD} = \alpha L_{task} + (1-\alpha)KL(p_s||p_t)$
- **MAML更新**：$\theta' = \theta - \alpha \nabla_\theta L_{task}(f_\theta)$
- **LoRA分解**：$W = W_0 + BA$，其中$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$

### 核心论文

1. **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)
2. **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation" (ICML 2017)
3. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
4. **Prompt Learning**: Jia et al., "Visual Prompt Tuning" (ECCV 2022)
5. **DANN**: Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016)

## 常见陷阱与错误（Gotchas）

### 1. 过度微调导致遗忘
- **错误**：对所有层使用相同的大学习率
- **解决**：使用层级学习率衰减，底层使用更小的学习率

### 2. 标注数据分布偏差
- **错误**：只标注简单或典型场景
- **解决**：主动采样困难和边缘案例，保持数据多样性

### 3. 任务间负迁移
- **错误**：盲目多任务训练
- **解决**：计算任务相似度矩阵，分组训练相关任务

### 4. LoRA秩选择不当
- **错误**：所有层使用相同的秩
- **解决**：根据层的重要性和梯度分析动态调整秩

### 5. 验证集泄露
- **错误**：使用验证集进行超参数搜索
- **解决**：严格的三分数据集：训练、验证、测试

### 6. 灾难性遗忘检测滞后
- **错误**：训练后才发现性能退化
- **解决**：持续监控在保留验证集上的性能

### 7. 伪标签噪声累积
- **错误**：无条件接受所有伪标签
- **解决**：设置置信度阈值，使用一致性检查

### 8. 域适应过拟合
- **错误**：在目标域上过度调优
- **解决**：保持源域性能监控，使用正则化约束

这些微调技术的合理组合使用，能够在保持模型泛化能力的同时，实现对特定任务的高效适配。在实际应用中，需要根据具体场景和资源约束选择合适的策略组合。