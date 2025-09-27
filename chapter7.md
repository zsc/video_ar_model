# 第7章：多任务学习框架

视频自回归模型的强大能力不仅体现在视频预测本身，更在于其能够同时完成多个相关任务。本章深入探讨如何设计有效的多任务学习框架，包括任务选择、权重平衡、梯度协调，以及如何利用LiDAR等额外模态进行伪监督，构建一个全面理解驾驶场景的统一模型。

## 7.1 主任务：视频预测

### 视频预测的多层次目标

**像素级预测**：
$$\mathcal{L}_{pixel} = \mathbb{E}_{t}\left[\|I_{t+1} - \hat{I}_{t+1}\|_p\right]$$

其中$p \in \{1, 2\}$选择L1或L2范数。

**特征级预测**：
$$\mathcal{L}_{feature} = \sum_{l} \lambda_l \|\phi_l(I_{t+1}) - \phi_l(\hat{I}_{t+1})\|_2$$

使用预训练网络的中间层特征。

**结构级预测**：
```python
def structural_prediction_loss(pred, target):
    # SSIM损失
    ssim_loss = 1 - ssim(pred, target)

    # 边缘损失
    edge_pred = sobel_filter(pred)
    edge_target = sobel_filter(target)
    edge_loss = F.l1_loss(edge_pred, edge_target)

    # 光流一致性
    flow = compute_optical_flow(target[:-1], target[1:])
    warped = warp_frames(pred[:-1], flow)
    flow_loss = F.l1_loss(warped, pred[1:])

    return ssim_loss + 0.1 * edge_loss + 0.05 * flow_loss
```

### 条件视频生成

**轨迹条件预测**：
$$\hat{V}_{t:t+H} = f_\theta(V_{t-C:t}, T_{t:t+H})$$

其中$T$为未来轨迹条件。

**语义地图条件**：
```python
class SemanticConditionedPredictor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.video_encoder = VideoEncoder(d_model)
        self.semantic_encoder = SemanticMapEncoder(d_model)
        self.decoder = VideoDecoder(d_model)

    def forward(self, past_video, future_semantic_map):
        # 编码历史视频
        video_features = self.video_encoder(past_video)

        # 编码未来语义地图
        semantic_features = self.semantic_encoder(future_semantic_map)

        # Cross-attention融合
        fused = cross_attention(video_features, semantic_features)

        # 解码未来视频
        future_video = self.decoder(fused)
        return future_video
```

### 不确定性建模

**概率视频预测**：
使用VAE建模不确定性：
$$p(V_{t+1}|V_{\leq t}) = \int p(V_{t+1}|z, V_{\leq t})p(z|V_{\leq t})dz$$

**多模态预测**：
```python
class MultiModalVideoPredictor(nn.Module):
    def __init__(self, d_model, num_modes=5):
        super().__init__()
        self.num_modes = num_modes
        self.encoder = VideoEncoder(d_model)

        # 多个预测头
        self.prediction_heads = nn.ModuleList([
            PredictionHead(d_model) for _ in range(num_modes)
        ])

        # 模态选择网络
        self.mode_selector = nn.Linear(d_model, num_modes)

    def forward(self, past_video):
        features = self.encoder(past_video)

        # 预测每个模态的概率
        mode_logits = self.mode_selector(features.mean(dim=1))
        mode_probs = F.softmax(mode_logits, dim=-1)

        # 生成多个预测
        predictions = []
        for head in self.prediction_heads:
            pred = head(features)
            predictions.append(pred)

        return predictions, mode_probs
```

**时序一致性约束**：
$$\mathcal{L}_{temporal} = \sum_{t} \|\mathcal{F}(V_t, V_{t+1}) - \mathcal{F}(\hat{V}_t, \hat{V}_{t+1})\|$$

其中$\mathcal{F}$计算帧间转换（光流、仿射变换等）。

**Rule of Thumb**：
- 预测范围：3-5秒
- 历史窗口：1-2秒
- 预测FPS：5-10 Hz
- 不确定性模态数：3-5个

## 7.2 目标检测与语义分割辅助任务

### 统一的多任务架构

**共享编码器设计**：
```python
class MultiTaskBackbone(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 共享的视觉编码器
        self.shared_encoder = ViTEncoder(d_model)

        # 任务特定的适配器
        self.task_adapters = nn.ModuleDict({
            'video': VideoAdapter(d_model),
            'detection': DetectionAdapter(d_model),
            'segmentation': SegmentationAdapter(d_model),
            'depth': DepthAdapter(d_model)
        })

    def forward(self, x, tasks):
        # 共享特征提取
        shared_features = self.shared_encoder(x)

        # 任务特定处理
        outputs = {}
        for task in tasks:
            if task in self.task_adapters:
                adapter = self.task_adapters[task]
                task_features = adapter(shared_features)
                outputs[task] = task_features

        return outputs
```

### 3D目标检测集成

**BEV空间检测**：
$$\mathcal{L}_{det} = \mathcal{L}_{cls} + \lambda_{loc} \mathcal{L}_{loc} + \lambda_{dim} \mathcal{L}_{dim} + \lambda_{rot} \mathcal{L}_{rot}$$

各项分别为：
- 分类损失
- 位置回归损失
- 尺寸回归损失
- 朝向回归损失

**时序检测关联**：
```python
def temporal_detection_loss(detections_sequence):
    """跨帧检测一致性损失"""
    loss = 0
    for t in range(len(detections_sequence) - 1):
        curr_dets = detections_sequence[t]
        next_dets = detections_sequence[t + 1]

        # 匹配跨帧检测
        matches = hungarian_matching(curr_dets, next_dets)

        for i, j in matches:
            # 运动一致性
            expected_pos = curr_dets[i].position + curr_dets[i].velocity * dt
            actual_pos = next_dets[j].position
            loss += torch.norm(expected_pos - actual_pos)

            # 外观一致性
            loss += 1 - cosine_similarity(curr_dets[i].features,
                                         next_dets[j].features)

    return loss / (len(detections_sequence) - 1)
```

### 全景分割任务

**实例与语义联合**：
```python
class PanopticHead(nn.Module):
    def __init__(self, d_model, num_classes, max_instances=100):
        super().__init__()
        # 语义分割分支
        self.semantic_head = nn.Conv2d(d_model, num_classes, 1)

        # 实例分割分支
        self.instance_embed = nn.Conv2d(d_model, 64, 1)
        self.instance_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(64, 8),
            num_layers=3
        )

        # 实例查询
        self.instance_queries = nn.Parameter(
            torch.randn(max_instances, 64)
        )

    def forward(self, features):
        # 语义分割
        semantic_logits = self.semantic_head(features)

        # 实例嵌入
        instance_embeds = self.instance_embed(features)

        # Transformer解码实例
        instance_masks = self.instance_decoder(
            self.instance_queries.unsqueeze(0),
            instance_embeds.flatten(2).permute(2, 0, 1)
        )

        return {
            'semantic': semantic_logits,
            'instances': instance_masks
        }
```

### 任务间知识传递

**检测引导的视频预测**：
```python
def detection_guided_prediction(video_features, detections):
    """使用检测结果改善视频预测"""

    # 构建物体中心的注意力mask
    attention_mask = torch.zeros(video_features.shape[:-1])
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        attention_mask[..., y1:y2, x1:x2] = 1.0

    # 物体感知的特征增强
    object_features = video_features * attention_mask.unsqueeze(-1)
    background_features = video_features * (1 - attention_mask).unsqueeze(-1)

    # 分别处理前景和背景
    object_pred = object_predictor(object_features)
    bg_pred = background_predictor(background_features)

    # 组合预测
    return object_pred + bg_pred
```

**Rule of Thumb**：
- 检测类别数：10-20（车、人、骑行者等）
- 分割类别数：20-30
- 检测NMS阈值：0.5-0.7
- 实例最大数量：50-100

## 7.3 深度估计与3D重建

### 单目深度估计

**尺度不变深度损失**：
$$\mathcal{L}_{depth} = \sqrt{\frac{1}{n}\sum_i d_i^2 - \frac{1}{n^2}(\sum_i d_i)^2}$$

其中$d_i = \log \hat{D}_i - \log D_i$。

**多尺度深度预测**：
```python
class MultiScaleDepth(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.decoders = nn.ModuleList([
            DepthDecoder(d_model, scale)
            for scale in [1, 2, 4, 8]
        ])

    def forward(self, features_pyramid):
        predictions = []
        for decoder, features in zip(self.decoders, features_pyramid):
            depth = decoder(features)
            predictions.append(depth)

        # 上采样到原始分辨率
        depths_upsampled = []
        for i, depth in enumerate(predictions):
            scale = 2 ** i
            if scale > 1:
                depth = F.interpolate(depth, scale_factor=scale)
            depths_upsampled.append(depth)

        # 加权融合
        weights = F.softmax(self.scale_weights, dim=0)
        final_depth = sum(w * d for w, d in zip(weights, depths_upsampled))

        return final_depth, predictions
```

### 3D占用网格预测

**体素化表示**：
$$O_{xyz} = \sigma(f_\theta(F_{BEV}, z))$$

其中$O \in [0,1]^{X \times Y \times Z}$表示占用概率。

**稀疏3D卷积**：
```python
class Sparse3DReconstruction(nn.Module):
    def __init__(self, voxel_size=0.2):
        super().__init__()
        self.voxel_size = voxel_size

        # 2D到3D提升
        self.lift_net = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 128 * 16, 1)  # 16个高度层
        )

        # 稀疏3D处理
        self.sparse_conv = spconv.Sequential(
            spconv.SubMConv3d(128, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SubMConv3d(128, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64, 1, 1)  # 占用概率
        )

    def forward(self, bev_features):
        batch, C, H, W = bev_features.shape

        # 提升到3D
        lifted = self.lift_net(bev_features)
        lifted = lifted.view(batch, 128, 16, H, W)
        lifted = lifted.permute(0, 2, 3, 4, 1)  # [B, Z, H, W, C]

        # 转换为稀疏表示
        indices = torch.nonzero(lifted.abs().sum(-1) > 0.1)
        features = lifted[indices[:, 0], indices[:, 1],
                         indices[:, 2], indices[:, 3]]

        sparse_tensor = spconv.SparseConvTensor(
            features, indices,
            spatial_shape=[16, H, W],
            batch_size=batch
        )

        # 稀疏卷积处理
        occupancy = self.sparse_conv(sparse_tensor)

        return occupancy
```

### 表面法向量估计

**法向量一致性**：
$$\mathcal{L}_{normal} = \frac{1}{|\Omega|}\sum_{p \in \Omega} (1 - \mathbf{n}_p^T \hat{\mathbf{n}}_p)$$

**几何约束**：
```python
def geometric_consistency_loss(depth, normal):
    """深度与法向量的几何一致性"""

    # 从深度计算法向量
    grad_x = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    grad_y = depth[:, :, 1:, :] - depth[:, :, :-1, :]

    # 叉积得到法向量
    normal_from_depth = torch.cross(
        torch.stack([grad_x, torch.zeros_like(grad_x),
                     torch.ones_like(grad_x)], dim=-1),
        torch.stack([torch.zeros_like(grad_y), grad_y,
                     torch.ones_like(grad_y)], dim=-1),
        dim=-1
    )
    normal_from_depth = F.normalize(normal_from_depth, dim=-1)

    # 一致性损失
    consistency = 1 - (normal * normal_from_depth).sum(dim=-1)
    return consistency.mean()
```

**Rule of Thumb**：
- 深度范围：0.5-100m
- 体素大小：0.1-0.5m
- 占用网格分辨率：200×200×16
- 法向量损失权重：0.1-0.2

## 7.4 LiDAR点云伪监督

### LiDAR-Camera投影对齐

**点云投影**：
$$\mathbf{p}_{img} = \mathbf{K} \cdot \mathbf{T}_{cam}^{lidar} \cdot \mathbf{P}_{3D}$$

**深度补全网络**：
```python
class DepthCompletion(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_encoder = SparseConvNet()
        self.rgb_encoder = ResNet()
        self.fusion = FusionModule()
        self.decoder = DepthDecoder()

    def forward(self, sparse_depth, rgb_image):
        # 编码稀疏深度
        sparse_features = self.sparse_encoder(sparse_depth)

        # 编码RGB
        rgb_features = self.rgb_encoder(rgb_image)

        # 融合
        fused = self.fusion(sparse_features, rgb_features)

        # 解码密集深度
        dense_depth = self.decoder(fused)

        # 保持LiDAR点的精确值
        mask = (sparse_depth > 0).float()
        final_depth = mask * sparse_depth + (1 - mask) * dense_depth

        return final_depth
```

### 点云序列运动估计

**场景流估计**：
$$\mathcal{L}_{flow} = \sum_{i} \|\mathbf{p}_i^{t+1} - (\mathbf{p}_i^t + \mathbf{f}_i)\|_2 + \lambda \|\mathbf{f}_i\|_2$$

**动静点分离**：
```python
def dynamic_static_segmentation(point_cloud_seq):
    """分离动态和静态点"""

    # 计算点云间的对应关系
    correspondences = []
    for t in range(len(point_cloud_seq) - 1):
        corr = nearest_neighbor_matching(
            point_cloud_seq[t],
            point_cloud_seq[t+1]
        )
        correspondences.append(corr)

    # RANSAC拟合自车运动
    ego_motion = ransac_rigid_transform(correspondences)

    # 补偿自车运动后的残差
    residuals = []
    for t, corr in enumerate(correspondences):
        compensated = apply_transform(point_cloud_seq[t], ego_motion[t])
        residual = torch.norm(compensated - point_cloud_seq[t+1], dim=-1)
        residuals.append(residual)

    # 阈值分割
    dynamic_mask = torch.stack(residuals).mean(0) > 0.2  # 20cm阈值

    return dynamic_mask, ego_motion
```

### 多模态一致性学习

**跨模态蒸馏**：
```python
class CrossModalDistillation(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_teacher = LiDARNet()
        self.camera_student = CameraNet()
        self.align_proj = nn.Linear(256, 256)

    def forward(self, lidar_data, camera_data):
        # Teacher预测
        with torch.no_grad():
            teacher_features = self.lidar_teacher(lidar_data)
            teacher_pred = teacher_features['occupancy']

        # Student预测
        student_features = self.camera_student(camera_data)
        student_pred = student_features['occupancy']

        # 特征对齐
        student_aligned = self.align_proj(student_features['features'])

        # 蒸馏损失
        distill_loss = F.kl_div(
            F.log_softmax(student_pred / self.temperature, dim=1),
            F.softmax(teacher_pred / self.temperature, dim=1),
            reduction='batchmean'
        )

        # 特征匹配损失
        feature_loss = F.mse_loss(student_aligned, teacher_features['features'])

        return distill_loss + 0.5 * feature_loss
```

### 点云增强视频预测

**几何引导的生成**：
```python
def lidar_guided_video_generation(past_frames, future_lidar):
    """使用未来LiDAR指导视频生成"""

    # 编码历史帧
    past_encoding = encode_video(past_frames)

    # 编码未来点云序列
    lidar_encoding = encode_point_cloud_sequence(future_lidar)

    # 投影点云到图像平面
    projected_depth = project_lidar_to_image_plane(future_lidar)

    # 条件生成
    generated_frames = []
    hidden = past_encoding

    for t, depth_t in enumerate(projected_depth):
        # Cross-attention with LiDAR
        hidden = cross_attention(hidden, lidar_encoding[t])

        # 深度条件解码
        frame = decode_with_depth(hidden, depth_t)
        generated_frames.append(frame)

        # 更新隐状态
        hidden = update_hidden(hidden, frame)

    return torch.stack(generated_frames)
```

**Rule of Thumb**：
- LiDAR点数：100k-200k/frame
- 投影深度稀疏度：5-10%
- 场景流阈值：0.2m
- 蒸馏温度：3-5

## 7.5 多任务权重平衡策略

### 动态权重调整

**不确定性加权**：
$$\mathcal{L}_{total} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$$

其中$\sigma_i$为任务i的可学习不确定性。

**梯度归一化**：
```python
class GradNorm:
    def __init__(self, num_tasks, alpha=1.5):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks))

    def compute_grad_norm(self, losses, shared_params):
        """计算各任务梯度范数"""
        grads = []
        for loss in losses:
            grad = torch.autograd.grad(loss, shared_params,
                                       retain_graph=True)
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grad]))
            grads.append(grad_norm)
        return torch.stack(grads)

    def balance_gradients(self, losses, shared_params, loss_ratios):
        """平衡各任务梯度"""
        grad_norms = self.compute_grad_norm(losses, shared_params)
        mean_norm = grad_norms.mean()

        # 计算目标梯度范数
        target_grads = mean_norm * (loss_ratios ** self.alpha)

        # 更新权重
        for i in range(self.num_tasks):
            self.weights[i] *= (target_grads[i] / grad_norms[i]).detach()

        # 归一化权重
        self.weights.data = self.weights.data / self.weights.data.mean()

        return self.weights
```

### 任务优先级调度

**课程式任务学习**：
```python
class CurriculumMultiTask:
    def __init__(self, tasks, difficulties):
        self.tasks = tasks
        self.difficulties = difficulties
        self.progress = 0

    def get_active_tasks(self, epoch):
        """根据训练进度激活任务"""
        active = []

        for task, difficulty in zip(self.tasks, self.difficulties):
            # 简单任务先激活
            if epoch >= difficulty * 10:
                active.append(task)

        return active

    def get_task_weights(self, epoch):
        """动态任务权重"""
        weights = {}

        for task, diff in zip(self.tasks, self.difficulties):
            if epoch < diff * 10:
                weights[task] = 0
            elif epoch < diff * 20:
                # 渐进增加权重
                progress = (epoch - diff * 10) / (diff * 10)
                weights[task] = progress
            else:
                weights[task] = 1.0

        return weights
```

### 任务间冲突解决

**梯度手术（Gradient Surgery）**：
```python
def gradient_surgery(grads):
    """修改冲突梯度使其不互相干扰"""
    num_tasks = len(grads)

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i != j:
                # 计算梯度内积
                dot_product = (grads[i] * grads[j]).sum()

                if dot_product < 0:  # 梯度冲突
                    # 投影去除冲突分量
                    proj = dot_product / (grads[j].norm() ** 2)
                    grads[i] = grads[i] - proj * grads[j]

    return grads
```

### 多任务评估指标

**相对改进度量**：
$$\Delta_i = \frac{M_i^{MTL} - M_i^{STL}}{M_i^{STL}}$$

其中$M_i^{MTL}$和$M_i^{STL}$分别为多任务和单任务性能。

**任务间相关性分析**：
```python
def task_affinity_matrix(model, tasks, validation_data):
    """计算任务相关性矩阵"""
    n_tasks = len(tasks)
    affinity = torch.zeros(n_tasks, n_tasks)

    for i, task_i in enumerate(tasks):
        # 训练task_i
        model_i = train_single_task(model, task_i, validation_data)

        for j, task_j in enumerate(tasks):
            if i != j:
                # 评估在task_j上的性能
                perf = evaluate(model_i, task_j, validation_data)
                affinity[i, j] = perf

    # 对称化
    affinity = (affinity + affinity.T) / 2

    return affinity
```

**Rule of Thumb**：
- 任务数量：3-5个主要任务
- 权重更新频率：每100个batch
- GradNorm α：1.5
- 梯度裁剪：1.0-5.0

## 高级话题：任务间梯度冲突与多目标优化

### 帕累托最优解

**多目标优化框架**：
$$\min_\theta \mathbf{L}(\theta) = [L_1(\theta), L_2(\theta), ..., L_K(\theta)]^T$$

寻找帕累托前沿。

**Multiple Gradient Descent Algorithm (MGDA)**：
```python
def mgda_solver(gradients):
    """寻找帕累托稳定点的最小范数梯度"""
    from scipy.optimize import minimize

    num_tasks = len(gradients)

    # 展平梯度
    grads_matrix = torch.stack([g.flatten() for g in gradients])

    def objective(weights):
        # 加权梯度的范数
        weighted_grad = (weights[:, None] * grads_matrix).sum(0)
        return weighted_grad.norm().item()

    # 约束：权重和为1，非负
    constraints = [
        {'type': 'eq', 'fun': lambda w: sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: w}
    ]

    # 初始化
    w0 = np.ones(num_tasks) / num_tasks

    # 优化
    result = minimize(objective, w0, constraints=constraints)

    return torch.tensor(result.x, dtype=torch.float32)
```

### 元学习任务权重

**MAML for Multi-task**：
```python
class MetaMultiTask(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def meta_update(self, support_tasks, query_tasks):
        # 内循环：在support上更新
        adapted_params = []
        for task_data in support_tasks:
            task_params = self.base_model.parameters()
            task_loss = compute_task_loss(task_params, task_data)

            # 梯度下降一步
            grads = torch.autograd.grad(task_loss, task_params)
            adapted = []
            for param, grad in zip(task_params, grads):
                adapted.append(param - self.inner_lr * grad)
            adapted_params.append(adapted)

        # 外循环：在query上评估
        meta_loss = 0
        for params, task_data in zip(adapted_params, query_tasks):
            with torch.no_grad():
                loss = compute_task_loss(params, task_data)
            meta_loss += self.task_weights[task_idx] * loss

        # 更新元参数
        meta_grads = torch.autograd.grad(meta_loss, self.parameters())
        return meta_grads
```

### 条件任务生成

**任务条件网络**：
```python
class TaskConditionedNetwork(nn.Module):
    def __init__(self, d_model, num_tasks):
        super().__init__()
        self.task_embeddings = nn.Embedding(num_tasks, d_model)
        self.film_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )

    def forward(self, x, task_id):
        # 获取任务嵌入
        task_emb = self.task_embeddings(task_id)

        # 生成FiLM参数
        film_params = self.film_generator(task_emb)
        gamma, beta = film_params.chunk(2, dim=-1)

        # 应用FiLM
        x = gamma * x + beta

        return x
```

### 动态任务图

**任务依赖图构建**：
```python
class TaskDependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_task(self, task, dependencies=[]):
        self.graph.add_node(task)
        for dep in dependencies:
            self.graph.add_edge(dep, task)

    def get_execution_order(self):
        """拓扑排序获取执行顺序"""
        return list(nx.topological_sort(self.graph))

    def get_parallel_groups(self):
        """获取可并行的任务组"""
        groups = []
        remaining = set(self.graph.nodes())

        while remaining:
            # 找出入度为0的节点
            group = [n for n in remaining
                    if self.graph.in_degree(n) == 0]
            groups.append(group)

            # 移除这些节点
            for node in group:
                self.graph.remove_node(node)
                remaining.remove(node)

        return groups
```

## 本章小结

本章系统介绍了多任务学习框架的设计与优化：

**关键概念**：
1. **视频预测主任务**：多层次目标、条件生成、不确定性建模
2. **辅助任务设计**：检测、分割、深度估计、3D重建
3. **LiDAR伪监督**：投影对齐、场景流、跨模态蒸馏
4. **权重平衡**：动态调整、GradNorm、梯度手术
5. **多目标优化**：帕累托最优、MGDA、元学习

**核心公式**：
- 多任务损失：$\mathcal{L} = \sum_i w_i \mathcal{L}_i$
- 不确定性加权：$\mathcal{L} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$
- GradNorm：$w_i \propto (L_i^{ratio})^\alpha / \|\nabla L_i\|$
- 梯度手术：$g_i' = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j$

**核心论文**：
1. [GradNorm] **GradNorm: Gradient Normalization for Adaptive Loss Balancing**, ICML 2018
2. [PCGrad] **Gradient Surgery for Multi-Task Learning**, NeurIPS 2020
3. [MTL Survey] **Multi-Task Learning for Dense Prediction Tasks**, CVPR 2021
4. [Uncertainty] **Multi-Task Learning Using Uncertainty**, CVPR 2018
5. [MGDA] **Multiple-gradient Descent Algorithm**, NeurIPS 2018

## 常见陷阱与错误 (Gotchas)

### 1. 任务不平衡
**问题**：某个任务主导训练
**症状**：其他任务性能差
**解决**：
- 调整损失权重
- 使用GradNorm
- 分阶段训练

### 2. 负迁移
**问题**：多任务性能低于单任务
**症状**：MTL < STL
**解决**：
- 任务选择优化
- 增加任务特定参数
- 使用任务条件网络

### 3. 梯度冲突
**问题**：任务梯度方向相反
**症状**：训练振荡
**解决**：
- PCGrad梯度手术
- MGDA优化
- 任务交替训练

### 4. 评估偏差
**问题**：过度优化某个指标
**症状**：其他指标退化
**解决**：
- 多指标综合评估
- 帕累托前沿分析
- 早停策略

### 5. 计算开销
**问题**：多任务推理慢
**症状**：延迟增加
**解决**：
- 任务级联优化
- 共享计算复用
- 动态任务选择

### 6. 标注不一致
**问题**：不同任务标注质量差异
**症状**：噪声传播
**解决**：
- 标注质量加权
- 鲁棒损失函数
- 自动标注校验

### 7. 任务耦合
**问题**：任务间依赖复杂
**症状**：级联错误
**解决**：
- 显式建模依赖
- 错误隔离机制
- 独立验证

### 8. 超参数爆炸
**问题**：每个任务都有超参数
**症状**：调参困难
**解决**：
- 自动超参搜索
- 元学习超参数
- 经验参数迁移