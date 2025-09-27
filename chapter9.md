# 第9章：预训练监控与可解释性

大规模模型的预训练是一个复杂且耗时的过程，需要精细的监控和深入的理解。本章探讨如何监控训练过程、诊断问题、理解模型行为，以及设计有效的评估基准。我们将深入介绍Loss曲线分析、梯度流诊断、注意力可视化、可解释性工具，以及评估体系的构建。

## 9.1 Loss曲线诊断与异常检测

### 多尺度Loss监控

**层级Loss追踪**：
```python
class MultiScaleLossMonitor:
    def __init__(self, window_sizes=[100, 1000, 10000]):
        self.window_sizes = window_sizes
        self.loss_history = []
        self.stats = {w: {} for w in window_sizes}

    def update(self, loss_dict, step):
        """更新损失统计"""
        self.loss_history.append({
            'step': step,
            **loss_dict
        })

        # 计算不同窗口的统计
        for window in self.window_sizes:
            if len(self.loss_history) >= window:
                recent = self.loss_history[-window:]

                for key in loss_dict.keys():
                    values = [d[key] for d in recent]

                    self.stats[window][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'trend': self.compute_trend(values)
                    }

    def compute_trend(self, values):
        """计算趋势（线性回归斜率）"""
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        return z[0]  # 斜率

    def detect_anomalies(self):
        """检测异常"""
        anomalies = []

        for window, stats in self.stats.items():
            for loss_name, stat in stats.items():
                # 检测突增
                if stat['max'] > stat['mean'] + 3 * stat['std']:
                    anomalies.append({
                        'type': 'spike',
                        'loss': loss_name,
                        'window': window,
                        'severity': (stat['max'] - stat['mean']) / stat['std']
                    })

                # 检测平台期
                if stat['std'] < 0.001 * stat['mean']:
                    anomalies.append({
                        'type': 'plateau',
                        'loss': loss_name,
                        'window': window
                    })

                # 检测发散
                if stat['trend'] > 0 and window >= 1000:
                    anomalies.append({
                        'type': 'divergence',
                        'loss': loss_name,
                        'window': window,
                        'trend': stat['trend']
                    })

        return anomalies
```

### Loss分解分析

**组件级Loss追踪**：
```python
class LossDecomposition:
    def __init__(self):
        self.components = {
            'reconstruction': [],
            'regularization': [],
            'auxiliary': [],
            'adversarial': []
        }

    def compute_losses(self, model, batch):
        """计算并分解损失"""
        losses = {}

        # 重构损失
        pred = model(batch['input'])
        losses['recon_pixel'] = F.mse_loss(pred, batch['target'])
        losses['recon_perceptual'] = self.perceptual_loss(pred, batch['target'])

        # 正则化损失
        losses['reg_kl'] = self.kl_divergence(model.get_latents())
        losses['reg_l2'] = sum(p.pow(2).sum() for p in model.parameters()) * 1e-5

        # 辅助任务损失
        if 'depth' in batch:
            losses['aux_depth'] = F.l1_loss(model.predict_depth(), batch['depth'])
        if 'segmentation' in batch:
            losses['aux_seg'] = F.cross_entropy(model.predict_seg(), batch['segmentation'])

        # 对抗损失
        if self.use_adversarial:
            losses['adv_gen'] = -torch.log(self.discriminator(pred)).mean()
            losses['adv_disc'] = self.discriminator_loss(pred.detach(), batch['target'])

        # 记录分解
        for key, value in losses.items():
            category = key.split('_')[0]
            if category in self.components:
                self.components[category].append(value.item())

        return losses

    def analyze_balance(self):
        """分析损失平衡"""
        analysis = {}

        for category, values in self.components.items():
            if values:
                analysis[category] = {
                    'magnitude': np.mean(values[-100:]),
                    'variance': np.var(values[-100:]),
                    'contribution': np.mean(values[-100:]) / sum(
                        np.mean(v[-100:]) for v in self.components.values() if v
                    )
                }

        return analysis
```

### 收敛性诊断

**收敛指标计算**：
```python
class ConvergenceAnalyzer:
    def __init__(self, patience=1000, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.steps_without_improvement = 0

    def check_convergence(self, current_loss):
        """检查是否收敛"""
        # 相对改进
        relative_improvement = (self.best_loss - current_loss) / abs(self.best_loss)

        if relative_improvement > self.threshold:
            self.best_loss = current_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        # 收敛判断
        convergence_status = {
            'converged': self.steps_without_improvement > self.patience,
            'improving': relative_improvement > self.threshold,
            'plateau': self.steps_without_improvement > self.patience // 2,
            'relative_improvement': relative_improvement,
            'steps_without_improvement': self.steps_without_improvement
        }

        return convergence_status

    def estimate_convergence_time(self, loss_history):
        """估计收敛时间"""
        if len(loss_history) < 100:
            return None

        # 拟合指数衰减
        x = np.arange(len(loss_history))
        y = np.array(loss_history)

        # L(t) = a * exp(-b * t) + c
        from scipy.optimize import curve_fit

        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c

        try:
            params, _ = curve_fit(exp_decay, x, y, p0=[y[0], 0.001, y[-1]])
            a, b, c = params

            # 估计到达目标的时间
            target_loss = c * 1.01  # 接近渐近线的1%
            if a > 0 and b > 0:
                t_convergence = -np.log((target_loss - c) / a) / b
                return int(t_convergence)
        except:
            return None
```

### 异常模式识别

**Loss异常模式库**：
```python
class AnomalyPatternDetector:
    def __init__(self):
        self.patterns = {
            'gradient_explosion': self.detect_gradient_explosion,
            'mode_collapse': self.detect_mode_collapse,
            'overfitting': self.detect_overfitting,
            'underfitting': self.detect_underfitting,
            'oscillation': self.detect_oscillation
        }

    def detect_gradient_explosion(self, metrics):
        """检测梯度爆炸"""
        if 'grad_norm' in metrics:
            recent_grads = metrics['grad_norm'][-100:]
            if any(g > 100 for g in recent_grads):
                return True, {
                    'max_grad': max(recent_grads),
                    'frequency': sum(g > 100 for g in recent_grads) / len(recent_grads)
                }
        return False, {}

    def detect_mode_collapse(self, metrics):
        """检测模式崩塌"""
        if 'output_diversity' in metrics:
            diversity = metrics['output_diversity'][-100:]
            if np.mean(diversity) < 0.1:  # 多样性过低
                return True, {
                    'diversity': np.mean(diversity),
                    'trend': np.polyfit(range(len(diversity)), diversity, 1)[0]
                }
        return False, {}

    def detect_overfitting(self, metrics):
        """检测过拟合"""
        if 'train_loss' in metrics and 'val_loss' in metrics:
            train_loss = np.mean(metrics['train_loss'][-100:])
            val_loss = np.mean(metrics['val_loss'][-100:])
            gap = val_loss - train_loss

            if gap > 0.2 * train_loss:  # 验证损失高20%以上
                return True, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'gap': gap
                }
        return False, {}

    def detect_oscillation(self, metrics):
        """检测振荡"""
        if 'loss' in metrics:
            recent_loss = metrics['loss'][-100:]
            # 计算自相关
            autocorr = np.correlate(recent_loss, recent_loss, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # 寻找周期性
            peaks = self.find_peaks(autocorr)
            if peaks and autocorr[peaks[0]] > 0.5:
                return True, {
                    'period': peaks[0],
                    'strength': autocorr[peaks[0]]
                }
        return False, {}

    def analyze_all(self, metrics):
        """运行所有检测器"""
        results = {}
        for name, detector in self.patterns.items():
            detected, details = detector(metrics)
            if detected:
                results[name] = details
        return results
```

**Rule of Thumb**：
- 监控窗口：100、1k、10k步
- 异常阈值：3σ原则
- 收敛判断：1000步无改进
- 平滑系数：0.95-0.99

## 9.2 梯度流分析与死神经元

### 梯度流监控

**层级梯度统计**：
```python
class GradientFlowMonitor:
    def __init__(self, model):
        self.model = model
        self.gradient_stats = {}
        self.register_hooks()

    def register_hooks(self):
        """注册梯度钩子"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                module.register_backward_hook(
                    lambda m, grad_in, grad_out, n=name:
                    self.save_gradient(n, grad_in, grad_out)
                )

    def save_gradient(self, name, grad_in, grad_out):
        """保存梯度统计"""
        if grad_out[0] is not None:
            grad = grad_out[0].detach()

            self.gradient_stats[name] = {
                'mean': grad.abs().mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'min': grad.abs().min().item(),
                'norm': grad.norm().item(),
                'zero_ratio': (grad == 0).float().mean().item()
            }

    def analyze_flow(self):
        """分析梯度流"""
        analysis = {}

        # 检测梯度消失
        vanishing_layers = []
        for name, stats in self.gradient_stats.items():
            if stats['mean'] < 1e-7:
                vanishing_layers.append(name)

        # 检测梯度爆炸
        exploding_layers = []
        for name, stats in self.gradient_stats.items():
            if stats['max'] > 1e3:
                exploding_layers.append(name)

        # 计算梯度流健康度
        gradient_norms = [stats['norm'] for stats in self.gradient_stats.values()]
        flow_health = np.std(np.log10(gradient_norms + 1e-8))

        analysis['vanishing_layers'] = vanishing_layers
        analysis['exploding_layers'] = exploding_layers
        analysis['flow_health'] = flow_health
        analysis['dead_neurons'] = self.detect_dead_neurons()

        return analysis

    def detect_dead_neurons(self):
        """检测死神经元"""
        dead_info = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 检查梯度为0的参数
                zero_grad = (param.grad == 0).float().mean().item()
                if zero_grad > 0.5:  # 超过50%梯度为0
                    dead_info[name] = {
                        'zero_ratio': zero_grad,
                        'param_norm': param.norm().item(),
                        'grad_norm': param.grad.norm().item()
                    }

        return dead_info
```

### 激活值分析

**激活分布监控**：
```python
class ActivationMonitor:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.register_forward_hooks()

    def register_forward_hooks(self):
        """注册前向钩子"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                module.register_forward_hook(
                    lambda m, inp, out, n=name:
                    self.save_activation(n, out)
                )

    def save_activation(self, name, activation):
        """保存激活统计"""
        act = activation.detach()

        self.activations[name] = {
            'mean': act.mean().item(),
            'std': act.std().item(),
            'sparsity': (act == 0).float().mean().item(),
            'saturation': (act > 0.99).float().mean().item(),
            'distribution': self.compute_distribution(act)
        }

    def compute_distribution(self, tensor):
        """计算分布统计"""
        flat = tensor.flatten()
        return {
            'q25': torch.quantile(flat, 0.25).item(),
            'q50': torch.quantile(flat, 0.50).item(),
            'q75': torch.quantile(flat, 0.75).item(),
            'skew': self.skewness(flat),
            'kurtosis': self.kurtosis(flat)
        }

    def skewness(self, x):
        """计算偏度"""
        mean = x.mean()
        std = x.std()
        return ((x - mean) ** 3).mean() / (std ** 3)

    def kurtosis(self, x):
        """计算峰度"""
        mean = x.mean()
        std = x.std()
        return ((x - mean) ** 4).mean() / (std ** 4) - 3

    def diagnose_issues(self):
        """诊断激活问题"""
        issues = []

        for name, stats in self.activations.items():
            # 死ReLU问题
            if stats['sparsity'] > 0.8:
                issues.append({
                    'layer': name,
                    'issue': 'dead_relu',
                    'severity': stats['sparsity']
                })

            # 饱和问题
            if stats['saturation'] > 0.1:
                issues.append({
                    'layer': name,
                    'issue': 'saturation',
                    'severity': stats['saturation']
                })

            # 分布偏移
            if abs(stats['distribution']['skew']) > 2:
                issues.append({
                    'layer': name,
                    'issue': 'distribution_shift',
                    'skewness': stats['distribution']['skew']
                })

        return issues
```

### 参数更新监控

**更新比率分析**：
```python
class ParameterUpdateMonitor:
    def __init__(self, model):
        self.model = model
        self.param_history = {}
        self.save_initial_params()

    def save_initial_params(self):
        """保存初始参数"""
        for name, param in self.model.named_parameters():
            self.param_history[name] = {
                'initial': param.data.clone(),
                'previous': param.data.clone(),
                'updates': []
            }

    def compute_update_metrics(self):
        """计算更新指标"""
        metrics = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                history = self.param_history[name]

                # 更新幅度
                update = param.data - history['previous']
                update_norm = update.norm().item()

                # 相对更新
                relative_update = update_norm / (param.norm().item() + 1e-8)

                # 累计变化
                total_change = (param.data - history['initial']).norm().item()

                # 更新方向一致性
                if len(history['updates']) > 0:
                    direction_consistency = F.cosine_similarity(
                        update.flatten(),
                        history['updates'][-1].flatten(),
                        dim=0
                    ).item()
                else:
                    direction_consistency = 0

                metrics[name] = {
                    'update_norm': update_norm,
                    'relative_update': relative_update,
                    'total_change': total_change,
                    'direction_consistency': direction_consistency,
                    'grad_to_param_ratio': param.grad.norm().item() / (param.norm().item() + 1e-8)
                }

                # 保存历史
                history['previous'] = param.data.clone()
                history['updates'].append(update.clone())
                if len(history['updates']) > 10:
                    history['updates'].pop(0)

        return metrics

    def detect_frozen_params(self, threshold=1e-6):
        """检测冻结的参数"""
        frozen = []

        for name, metrics in self.compute_update_metrics().items():
            if metrics['relative_update'] < threshold:
                frozen.append({
                    'name': name,
                    'update_norm': metrics['update_norm'],
                    'param_norm': self.model.get_parameter(name).norm().item()
                })

        return frozen
```

### 梯度修复策略

**自适应梯度裁剪**：
```python
class AdaptiveGradientClipper:
    def __init__(self, percentile=90, history_size=100):
        self.percentile = percentile
        self.history_size = history_size
        self.grad_norm_history = []

    def clip(self, model):
        """自适应裁剪梯度"""
        # 计算当前梯度范数
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm() ** 2
        total_norm = total_norm ** 0.5

        # 更新历史
        self.grad_norm_history.append(total_norm.item())
        if len(self.grad_norm_history) > self.history_size:
            self.grad_norm_history.pop(0)

        # 计算自适应阈值
        if len(self.grad_norm_history) >= 10:
            threshold = np.percentile(self.grad_norm_history, self.percentile)
        else:
            threshold = 10.0  # 默认值

        # 裁剪
        if total_norm > threshold:
            clip_coef = threshold / total_norm
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(clip_coef)

        return {
            'grad_norm': total_norm.item(),
            'threshold': threshold,
            'clipped': total_norm.item() > threshold
        }
```

**Rule of Thumb**：
- 梯度消失阈值：<1e-7
- 梯度爆炸阈值：>1e3
- 死神经元比例：>50%
- 裁剪百分位：90-95%

## 9.3 注意力模式可视化

### 注意力矩阵可视化

**多头注意力可视化**：
```python
class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.register_attention_hooks()

    def register_attention_hooks(self):
        """注册注意力钩子"""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                module.register_forward_hook(
                    lambda m, inp, out, n=name:
                    self.save_attention(n, m, out)
                )

    def save_attention(self, name, module, output):
        """保存注意力权重"""
        if hasattr(module, 'attention_weights'):
            self.attention_maps[name] = module.attention_weights.detach()

    def visualize_head_patterns(self, layer_name):
        """可视化各头的注意力模式"""
        attention = self.attention_maps[layer_name]  # [B, H, S, S]

        num_heads = attention.shape[1]
        fig, axes = plt.subplots(2, num_heads // 2, figsize=(20, 8))

        for head_idx in range(num_heads):
            ax = axes[head_idx // (num_heads // 2), head_idx % (num_heads // 2)]

            # 取第一个样本的注意力
            head_attn = attention[0, head_idx].cpu().numpy()

            # 可视化
            im = ax.imshow(head_attn, cmap='hot', interpolation='nearest')
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Keys')
            ax.set_ylabel('Queries')

            # 添加颜色条
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        return fig

    def analyze_attention_patterns(self, attention):
        """分析注意力模式"""
        patterns = {}

        # 对角线模式（局部注意力）
        diagonal_strength = self.compute_diagonal_strength(attention)
        patterns['diagonal'] = diagonal_strength

        # 垂直模式（特定token被广泛关注）
        vertical_strength = attention.std(dim=-2).mean()
        patterns['vertical'] = vertical_strength.item()

        # 水平模式（某些token广泛关注其他）
        horizontal_strength = attention.std(dim=-1).mean()
        patterns['horizontal'] = horizontal_strength.item()

        # 块状模式
        block_strength = self.compute_block_pattern(attention)
        patterns['block'] = block_strength

        # 稀疏性
        sparsity = (attention < 0.01).float().mean()
        patterns['sparsity'] = sparsity.item()

        return patterns

    def compute_diagonal_strength(self, attention):
        """计算对角线模式强度"""
        seq_len = attention.shape[-1]
        diagonal = torch.diagonal(attention, dim1=-2, dim2=-1)
        return diagonal.mean().item()

    def compute_block_pattern(self, attention, block_size=16):
        """计算块状模式"""
        # 将注意力矩阵分块
        blocks = attention.unfold(-2, block_size, block_size).unfold(-1, block_size, block_size)
        # 计算块内平均注意力
        block_means = blocks.mean(dim=(-2, -1))
        # 计算块间方差
        block_variance = block_means.var()
        return block_variance.item()
```

### 注意力流追踪

**Token重要性传播**：
```python
class AttentionFlowTracer:
    def __init__(self, model):
        self.model = model
        self.attention_flows = []

    def trace_token_importance(self, input_tokens, target_position):
        """追踪特定位置的注意力流"""
        # 初始化重要性（目标位置为1，其他为0）
        importance = torch.zeros(len(input_tokens))
        importance[target_position] = 1.0

        layer_importances = [importance]

        # 逐层传播
        for layer_idx, layer in enumerate(self.model.layers):
            # 获取该层的注意力权重
            attention = self.get_layer_attention(layer, input_tokens)

            # 反向传播重要性
            # importance_new[i] = sum_j (importance[j] * attention[j, i])
            importance = torch.matmul(importance, attention)

            layer_importances.append(importance)

        return layer_importances

    def compute_attention_entropy(self, attention):
        """计算注意力熵"""
        # attention: [batch, heads, seq, seq]
        entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=-1)
        return entropy.mean(dim=(0, 1))  # 平均over batch和heads

    def identify_information_bottlenecks(self):
        """识别信息瓶颈"""
        bottlenecks = []

        for layer_idx, attention in enumerate(self.attention_flows):
            entropy = self.compute_attention_entropy(attention)

            # 低熵表示信息瓶颈
            if entropy.mean() < 2.0:  # 阈值
                bottlenecks.append({
                    'layer': layer_idx,
                    'entropy': entropy.mean().item(),
                    'concentrated_positions': torch.where(entropy < 1.0)[0].tolist()
                })

        return bottlenecks
```

### 跨层注意力分析

**层间相似性**：
```python
class CrossLayerAttentionAnalyzer:
    def __init__(self):
        self.layer_attentions = {}

    def compute_layer_similarity(self, attn1, attn2):
        """计算两层注意力的相似性"""
        # 展平注意力矩阵
        flat1 = attn1.flatten(start_dim=2)
        flat2 = attn2.flatten(start_dim=2)

        # 计算余弦相似度
        similarity = F.cosine_similarity(flat1, flat2, dim=-1)

        return similarity.mean()

    def analyze_redundancy(self, model):
        """分析层间冗余"""
        similarity_matrix = torch.zeros(len(self.layer_attentions),
                                      len(self.layer_attentions))

        layers = list(self.layer_attentions.keys())
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                if i != j:
                    sim = self.compute_layer_similarity(
                        self.layer_attentions[layer1],
                        self.layer_attentions[layer2]
                    )
                    similarity_matrix[i, j] = sim

        # 识别高度相似的层对
        redundant_pairs = []
        threshold = 0.9
        for i in range(len(layers)):
            for j in range(i+1, len(layers)):
                if similarity_matrix[i, j] > threshold:
                    redundant_pairs.append((layers[i], layers[j],
                                          similarity_matrix[i, j].item()))

        return redundant_pairs

    def compute_attention_flow_efficiency(self):
        """计算注意力流效率"""
        efficiencies = []

        layers = list(self.layer_attentions.keys())
        for i in range(len(layers) - 1):
            curr_attn = self.layer_attentions[layers[i]]
            next_attn = self.layer_attentions[layers[i+1]]

            # 计算信息传递效率
            # 使用矩阵秩作为信息量的代理
            curr_rank = torch.matrix_rank(curr_attn.mean(dim=1))
            next_rank = torch.matrix_rank(next_attn.mean(dim=1))

            efficiency = next_rank.float() / (curr_rank.float() + 1e-8)
            efficiencies.append(efficiency.item())

        return efficiencies
```

**Rule of Thumb**：
- 注意力熵阈值：<2.0为瓶颈
- 相似度阈值：>0.9为冗余
- 稀疏度目标：50-70%
- 块大小：16-32

## 9.4 Transformer-lens应用

### 机械可解释性框架

**TransformerLens集成**：
```python
class MechanisticInterpretability:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.cache = {}

    def setup_hooks(self):
        """设置钩子收集中间激活"""
        hook_points = [
            'embed', 'pos_embed',
            'attn.q', 'attn.k', 'attn.v',
            'attn.pattern', 'attn.result',
            'mlp.pre', 'mlp.mid', 'mlp.post',
            'resid_pre', 'resid_mid', 'resid_post'
        ]

        for layer_idx in range(self.model.num_layers):
            for hook_point in hook_points:
                hook_name = f'blocks.{layer_idx}.{hook_point}'
                self.add_hook(hook_name)

    def add_hook(self, name):
        """添加激活钩子"""
        def hook_fn(activation, hook):
            self.cache[name] = activation.detach()

        self.hooks[name] = self.model.add_hook(name, hook_fn)

    def run_with_cache(self, input_data):
        """运行模型并缓存激活"""
        self.cache = {}
        output = self.model(input_data)
        return output, self.cache

    def decompose_residual_stream(self, layer_idx):
        """分解残差流"""
        components = {
            'embed': self.cache.get('embed'),
            'pos_embed': self.cache.get('pos_embed'),
            'attn_outputs': [],
            'mlp_outputs': []
        }

        for l in range(layer_idx + 1):
            attn_out = self.cache.get(f'blocks.{l}.attn.result')
            mlp_out = self.cache.get(f'blocks.{l}.mlp.post')

            if attn_out is not None:
                components['attn_outputs'].append(attn_out)
            if mlp_out is not None:
                components['mlp_outputs'].append(mlp_out)

        # 计算总和
        residual = components['embed'] + components['pos_embed']
        for attn in components['attn_outputs']:
            residual += attn
        for mlp in components['mlp_outputs']:
            residual += mlp

        return components, residual

    def analyze_attention_heads(self, layer_idx):
        """分析注意力头的功能"""
        attn_pattern = self.cache[f'blocks.{layer_idx}.attn.pattern']

        head_analysis = {}
        for head_idx in range(attn_pattern.shape[1]):
            head_pattern = attn_pattern[:, head_idx]

            # 分析模式
            analysis = {
                'copying': self.detect_copying_head(head_pattern),
                'induction': self.detect_induction_head(head_pattern),
                'positional': self.detect_positional_head(head_pattern),
                'global': self.detect_global_head(head_pattern)
            }

            head_analysis[f'head_{head_idx}'] = analysis

        return head_analysis

    def detect_copying_head(self, pattern):
        """检测复制头"""
        # 检查是否主要关注相同的token
        diagonal = torch.diagonal(pattern, dim1=-2, dim2=-1)
        return diagonal.mean() > 0.5

    def detect_induction_head(self, pattern):
        """检测归纳头"""
        # 检查是否关注重复模式
        # 简化实现：检查是否关注固定偏移
        shifted = torch.roll(pattern, shifts=1, dims=-1)
        similarity = F.cosine_similarity(pattern.flatten(), shifted.flatten(), dim=0)
        return similarity > 0.7
```

### 电路发现

**自动电路识别**：
```python
class CircuitDiscovery:
    def __init__(self, model):
        self.model = model
        self.circuits = []

    def find_circuits(self, task_examples):
        """发现任务相关的电路"""
        # 运行示例收集激活
        activations = []
        for example in task_examples:
            _, cache = self.model.run_with_cache(example)
            activations.append(cache)

        # 识别重要连接
        important_edges = self.identify_important_edges(activations)

        # 构建电路图
        circuit = self.build_circuit_graph(important_edges)

        return circuit

    def identify_important_edges(self, activations, threshold=0.5):
        """识别重要的连接"""
        edges = []

        for layer_idx in range(self.model.num_layers - 1):
            for head_i in range(self.model.num_heads):
                for head_j in range(self.model.num_heads):
                    # 计算连接强度
                    strength = self.compute_edge_strength(
                        activations, layer_idx, head_i, head_j
                    )

                    if strength > threshold:
                        edges.append({
                            'from': (layer_idx, head_i),
                            'to': (layer_idx + 1, head_j),
                            'strength': strength
                        })

        return edges

    def compute_edge_strength(self, activations, layer_idx, head_i, head_j):
        """计算边的强度"""
        strengths = []

        for act in activations:
            # 获取注意力输出
            attn_out_i = act[f'blocks.{layer_idx}.attn.result'][:, head_i]
            attn_in_j = act[f'blocks.{layer_idx+1}.attn.q'][:, head_j]

            # 计算相关性
            corr = torch.corrcoef(torch.stack([
                attn_out_i.flatten(),
                attn_in_j.flatten()
            ]))[0, 1]

            strengths.append(abs(corr.item()))

        return np.mean(strengths)

    def ablate_circuit(self, circuit, input_data):
        """消融电路以验证其功能"""
        # 保存原始输出
        original_output = self.model(input_data)

        # 逐个消融边
        ablation_effects = []
        for edge in circuit['edges']:
            # 临时移除连接
            with self.zero_ablate_edge(edge):
                ablated_output = self.model(input_data)

                # 计算影响
                effect = (original_output - ablated_output).norm()
                ablation_effects.append({
                    'edge': edge,
                    'effect': effect.item()
                })

        return ablation_effects
```

### 任务向量分析

**任务特定方向发现**：
```python
class TaskVectorAnalyzer:
    def __init__(self, model):
        self.model = model
        self.task_vectors = {}

    def extract_task_vector(self, task_name, positive_examples, negative_examples):
        """提取任务向量"""
        # 收集正例激活
        positive_acts = []
        for example in positive_examples:
            _, cache = self.model.run_with_cache(example)
            positive_acts.append(cache)

        # 收集负例激活
        negative_acts = []
        for example in negative_examples:
            _, cache = self.model.run_with_cache(example)
            negative_acts.append(cache)

        # 计算差异向量
        task_vector = {}
        for key in positive_acts[0].keys():
            pos_mean = torch.stack([a[key] for a in positive_acts]).mean(dim=0)
            neg_mean = torch.stack([a[key] for a in negative_acts]).mean(dim=0)
            task_vector[key] = pos_mean - neg_mean

        self.task_vectors[task_name] = task_vector
        return task_vector

    def project_onto_task(self, activation, task_name):
        """将激活投影到任务向量"""
        if task_name not in self.task_vectors:
            raise ValueError(f"Task {task_name} not found")

        task_vec = self.task_vectors[task_name]
        projections = {}

        for key, act in activation.items():
            if key in task_vec:
                # 计算投影
                vec_norm = task_vec[key].norm()
                if vec_norm > 0:
                    proj = (act * task_vec[key]).sum() / (vec_norm ** 2)
                    projections[key] = proj.item()

        return projections

    def steer_model_behavior(self, input_data, task_name, strength=1.0):
        """通过添加任务向量来引导模型行为"""
        task_vec = self.task_vectors[task_name]

        def steering_hook(activation, hook):
            # 添加任务向量
            return activation + strength * task_vec[hook.name]

        # 添加钩子
        hooks = []
        for key in task_vec.keys():
            hook = self.model.add_hook(key, steering_hook)
            hooks.append(hook)

        # 运行模型
        output = self.model(input_data)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return output
```

**Rule of Thumb**：
- 电路阈值：相关性>0.5
- 任务向量强度：0.5-2.0
- 消融步长：逐个或逐层
- 投影维度：降到128-512

## 9.5 评估Benchmark设计

### 多维度评估体系

**评估维度设计**：
```python
class ComprehensiveBenchmark:
    def __init__(self):
        self.metrics = {
            'perception': PerceptionMetrics(),
            'prediction': PredictionMetrics(),
            'planning': PlanningMetrics(),
            'safety': SafetyMetrics(),
            'generalization': GeneralizationMetrics()
        }

    def evaluate(self, model, test_data):
        """全面评估模型"""
        results = {}

        for category, metric_class in self.metrics.items():
            results[category] = metric_class.evaluate(model, test_data)

        # 计算综合得分
        results['overall'] = self.compute_overall_score(results)

        return results

    def compute_overall_score(self, results):
        """计算综合得分"""
        weights = {
            'perception': 0.2,
            'prediction': 0.3,
            'planning': 0.2,
            'safety': 0.2,
            'generalization': 0.1
        }

        overall = 0
        for category, weight in weights.items():
            if category in results:
                overall += weight * results[category]['score']

        return overall
```

**感知评估**：
```python
class PerceptionMetrics:
    def evaluate(self, model, test_data):
        """评估感知能力"""
        metrics = {}

        # 目标检测
        metrics['detection'] = self.evaluate_detection(model, test_data)

        # 语义分割
        metrics['segmentation'] = self.evaluate_segmentation(model, test_data)

        # 深度估计
        metrics['depth'] = self.evaluate_depth(model, test_data)

        # 追踪
        metrics['tracking'] = self.evaluate_tracking(model, test_data)

        return metrics

    def evaluate_detection(self, model, data):
        """评估检测性能"""
        predictions = []
        ground_truths = []

        for batch in data:
            pred = model.detect(batch['image'])
            predictions.append(pred)
            ground_truths.append(batch['boxes'])

        # 计算mAP
        mAP = compute_mAP(predictions, ground_truths)

        # 分类别性能
        per_class_ap = {}
        for class_name in ['car', 'pedestrian', 'cyclist']:
            per_class_ap[class_name] = compute_ap_for_class(
                predictions, ground_truths, class_name
            )

        return {
            'mAP': mAP,
            'per_class': per_class_ap,
            'score': mAP
        }
```

### 场景覆盖度评估

**场景分类与覆盖**：
```python
class ScenarioCoverageEvaluator:
    def __init__(self):
        self.scenario_taxonomy = {
            'weather': ['clear', 'rain', 'snow', 'fog'],
            'lighting': ['day', 'night', 'dawn', 'dusk'],
            'traffic': ['free', 'moderate', 'congested'],
            'road_type': ['highway', 'urban', 'rural', 'parking'],
            'maneuver': ['straight', 'turn', 'lane_change', 'merge']
        }

    def evaluate_coverage(self, model, scenario_test_sets):
        """评估场景覆盖度"""
        coverage_results = {}

        for dim, categories in self.scenario_taxonomy.items():
            dim_results = {}

            for category in categories:
                if category in scenario_test_sets[dim]:
                    test_data = scenario_test_sets[dim][category]
                    performance = self.evaluate_scenario(model, test_data)
                    dim_results[category] = performance

            coverage_results[dim] = {
                'results': dim_results,
                'coverage': len(dim_results) / len(categories),
                'mean_performance': np.mean(list(dim_results.values()))
            }

        return coverage_results

    def evaluate_scenario(self, model, test_data):
        """评估特定场景"""
        metrics = []

        for sample in test_data:
            pred = model(sample['input'])
            metric = self.compute_metric(pred, sample['target'])
            metrics.append(metric)

        return np.mean(metrics)

    def identify_weak_scenarios(self, coverage_results, threshold=0.7):
        """识别性能较差的场景"""
        weak_scenarios = []

        for dim, results in coverage_results.items():
            for category, performance in results['results'].items():
                if performance < threshold:
                    weak_scenarios.append({
                        'dimension': dim,
                        'category': category,
                        'performance': performance
                    })

        return sorted(weak_scenarios, key=lambda x: x['performance'])
```

### 长尾性能评估

**稀有事件测试**：
```python
class LongTailEvaluator:
    def __init__(self):
        self.rare_events = {
            'emergency_brake': 0.001,
            'pedestrian_jaywalking': 0.01,
            'vehicle_breakdown': 0.005,
            'construction_zone': 0.02,
            'emergency_vehicle': 0.01
        }

    def evaluate_rare_events(self, model, rare_event_data):
        """评估稀有事件处理"""
        results = {}

        for event_type, frequency in self.rare_events.items():
            if event_type in rare_event_data:
                # 评估性能
                performance = self.evaluate_event_handling(
                    model, rare_event_data[event_type]
                )

                # 加权by频率
                weighted_score = performance * (1 / frequency) ** 0.1

                results[event_type] = {
                    'performance': performance,
                    'frequency': frequency,
                    'weighted_score': weighted_score
                }

        return results

    def evaluate_event_handling(self, model, event_data):
        """评估事件处理能力"""
        correct_responses = 0
        total = len(event_data)

        for sample in event_data:
            response = model(sample['scenario'])

            # 检查响应是否适当
            if self.is_appropriate_response(response, sample['expected']):
                correct_responses += 1

        return correct_responses / total

    def compute_tail_robustness(self, results):
        """计算长尾鲁棒性分数"""
        scores = []
        weights = []

        for event, metrics in results.items():
            scores.append(metrics['weighted_score'])
            weights.append(1 / metrics['frequency'])

        # 加权平均
        weighted_mean = np.average(scores, weights=weights)

        # 最差情况
        worst_case = min(scores)

        return {
            'weighted_mean': weighted_mean,
            'worst_case': worst_case,
            'robustness_score': 0.7 * weighted_mean + 0.3 * worst_case
        }
```

### 实时性能评估

**延迟与吞吐量测试**：
```python
class LatencyBenchmark:
    def __init__(self, target_fps=10):
        self.target_fps = target_fps
        self.target_latency = 1000 / target_fps  # ms

    def benchmark(self, model, test_data, num_runs=100):
        """基准测试"""
        latencies = []
        throughputs = []

        # 预热
        for _ in range(10):
            _ = model(test_data[0])

        # 测试延迟
        for i in range(num_runs):
            sample = test_data[i % len(test_data)]

            start = time.perf_counter()
            _ = model(sample)
            end = time.perf_counter()

            latency = (end - start) * 1000  # ms
            latencies.append(latency)

        # 测试吞吐量
        batch_sizes = [1, 2, 4, 8, 16]
        for batch_size in batch_sizes:
            if batch_size <= len(test_data):
                batch = test_data[:batch_size]

                start = time.perf_counter()
                _ = model(batch)
                end = time.perf_counter()

                throughput = batch_size / (end - start)
                throughputs.append({
                    'batch_size': batch_size,
                    'throughput': throughput
                })

        results = {
            'mean_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'meets_target': np.percentile(latencies, 95) < self.target_latency,
            'throughputs': throughputs
        }

        return results
```

**Rule of Thumb**：
- mAP目标：>0.7
- 场景覆盖：>80%
- P95延迟：<100ms
- 稀有事件准确率：>90%

## 高级话题：机械可解释性与电路发现

### 因果追踪

**激活修补实验**：
```python
class CausalTracing:
    def __init__(self, model):
        self.model = model

    def trace_causal_path(self, input_data, target_output, num_steps=10):
        """追踪因果路径"""
        # 获取干净运行的激活
        clean_output, clean_cache = self.model.run_with_cache(input_data)

        # 创建损坏的输入
        corrupted_input = self.corrupt_input(input_data)
        corrupted_output, corrupted_cache = self.model.run_with_cache(corrupted_input)

        # 逐步修补
        causal_effects = []
        for layer_idx in range(self.model.num_layers):
            for position in range(input_data.shape[1]):
                # 修补特定位置的激活
                effect = self.patch_activation(
                    corrupted_cache, clean_cache,
                    layer_idx, position
                )

                causal_effects.append({
                    'layer': layer_idx,
                    'position': position,
                    'effect': effect
                })

        return self.analyze_causal_effects(causal_effects)

    def patch_activation(self, corrupted_cache, clean_cache, layer_idx, position):
        """修补特定激活"""
        # 复制损坏的缓存
        patched_cache = corrupted_cache.copy()

        # 替换特定位置的激活
        layer_key = f'blocks.{layer_idx}.resid_post'
        patched_cache[layer_key][:, position] = clean_cache[layer_key][:, position]

        # 从该层继续前向传播
        output = self.model.forward_from_layer(patched_cache, layer_idx)

        # 计算恢复程度
        clean_output = self.model(input_data)
        corrupted_output = self.model(self.corrupt_input(input_data))

        recovery = (output - corrupted_output).norm() / (clean_output - corrupted_output).norm()

        return recovery.item()
```

### 功能定位

**神经元功能分析**：
```python
class NeuronAnalyzer:
    def __init__(self, model):
        self.model = model
        self.neuron_activations = {}

    def analyze_neuron_selectivity(self, dataset):
        """分析神经元选择性"""
        # 收集激活
        for sample in dataset:
            _, cache = self.model.run_with_cache(sample['input'])
            self.collect_neuron_activations(cache, sample['label'])

        # 分析每个神经元
        selective_neurons = {}
        for layer_name, activations in self.neuron_activations.items():
            selective = self.find_selective_neurons(activations)
            selective_neurons[layer_name] = selective

        return selective_neurons

    def find_selective_neurons(self, activations_by_label):
        """找到选择性神经元"""
        selective = []

        num_neurons = activations_by_label[0][0].shape[-1]
        for neuron_idx in range(num_neurons):
            # 计算每个类别的平均激活
            mean_activations = {}
            for label, acts in activations_by_label.items():
                neuron_acts = [a[:, neuron_idx] for a in acts]
                mean_activations[label] = np.mean(neuron_acts)

            # 计算选择性（最大差异）
            values = list(mean_activations.values())
            selectivity = max(values) - min(values)

            if selectivity > 0.5:  # 阈值
                selective.append({
                    'neuron': neuron_idx,
                    'selectivity': selectivity,
                    'preferred_label': max(mean_activations, key=mean_activations.get)
                })

        return selective
```

### 表示工程

**概念向量提取**：
```python
class ConceptVectorExtractor:
    def __init__(self, model):
        self.model = model
        self.concept_vectors = {}

    def extract_concept(self, concept_name, positive_examples, negative_examples):
        """提取概念向量"""
        # 获取正例和负例的激活差异
        pos_acts = self.get_activations(positive_examples)
        neg_acts = self.get_activations(negative_examples)

        concept_vector = {}
        for layer_name in pos_acts.keys():
            # 计算差异向量
            pos_mean = torch.stack(pos_acts[layer_name]).mean(dim=0)
            neg_mean = torch.stack(neg_acts[layer_name]).mean(dim=0)

            diff = pos_mean - neg_mean

            # 归一化
            concept_vector[layer_name] = F.normalize(diff, dim=-1)

        self.concept_vectors[concept_name] = concept_vector
        return concept_vector

    def measure_concept_presence(self, input_data, concept_name):
        """测量概念存在程度"""
        if concept_name not in self.concept_vectors:
            raise ValueError(f"Concept {concept_name} not found")

        _, cache = self.model.run_with_cache(input_data)
        concept_vec = self.concept_vectors[concept_name]

        scores = {}
        for layer_name, vec in concept_vec.items():
            if layer_name in cache:
                activation = cache[layer_name]
                # 计算投影
                score = F.cosine_similarity(activation, vec.unsqueeze(0), dim=-1)
                scores[layer_name] = score.mean().item()

        return scores

    def edit_concept(self, input_data, concept_name, strength=1.0):
        """编辑概念强度"""
        concept_vec = self.concept_vectors[concept_name]

        # 添加钩子修改激活
        hooks = []
        for layer_name, vec in concept_vec.items():
            def edit_hook(activation, hook, v=vec, s=strength):
                # 添加概念向量
                return activation + s * v

            hook = self.model.add_hook(layer_name, edit_hook)
            hooks.append(hook)

        # 运行模型
        output = self.model(input_data)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return output
```

## 本章小结

本章系统介绍了预训练监控与可解释性的核心技术：

**关键概念**：
1. **Loss监控**：多尺度分析、异常检测、收敛诊断
2. **梯度分析**：梯度流、死神经元、激活分布
3. **注意力可视化**：模式分析、信息流、跨层相似性
4. **机械可解释性**：电路发现、因果追踪、概念向量
5. **评估体系**：多维度基准、场景覆盖、长尾评估

**核心公式**：
- 收敛判断：$\frac{L_{best} - L_{current}}{|L_{best}|} < \epsilon$
- 梯度流健康度：$\sigma(\log_{10}(||\nabla||))$
- 注意力熵：$H = -\sum_i p_i \log p_i$
- 概念投影：$score = \cos(activation, concept\_vector)$

**核心论文**：
1. [Transformer Circuits] **A Mathematical Framework for Transformer Circuits**, 2021
2. [Mechanistic Interpretability] **Toy Models of Superposition**, 2022
3. [BIG-Bench] **Beyond the Imitation Game Benchmark**, 2023
4. [Attention Analysis] **What Does BERT Look At?**, ACL 2019
5. [Causal Tracing] **Locating and Editing Factual Associations**, NeurIPS 2022

## 常见陷阱与错误 (Gotchas)

### 1. 过度解释
**问题**：将随机模式解释为有意义
**症状**：不可重复的发现
**解决**：
- 多次运行验证
- 统计显著性检验
- 对照实验

### 2. 监控开销过大
**问题**：钩子和日志拖慢训练
**症状**：训练速度慢50%+
**解决**：
- 采样监控
- 异步日志
- 分级详细度

### 3. 梯度消失误诊
**问题**：正常的梯度衰减被误判
**症状**：频繁假警报
**解决**：
- 考虑层深度
- 相对阈值
- 历史基线

### 4. 注意力模式误导
**问题**：注意力≠重要性
**症状**：错误归因
**解决**：
- 结合梯度信息
- 消融验证
- 多种方法交叉验证

### 5. 评估过拟合
**问题**：针对benchmark优化
**症状**：实际性能差
**解决**：
- 隐藏测试集
- 动态评估
- 真实场景验证

### 6. 长尾忽视
**问题**：平均指标掩盖问题
**症状**：稀有情况失败
**解决**：
- 分层评估
- 最差情况分析
- 加权指标

### 7. 因果混淆
**问题**：相关性≠因果性
**症状**：错误的因果判断
**解决**：
- 干预实验
- 反事实测试
- 多重验证

### 8. 概念漂移
**问题**：学到的概念不稳定
**症状**：解释不一致
**解决**：
- 多数据集验证
- 概念锚定
- 增量更新