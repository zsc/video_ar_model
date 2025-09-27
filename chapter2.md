# 第2章：数据工程与质量控制

大规模视频自回归模型的成功依赖于高质量、多样化的训练数据。本章深入探讨自动驾驶场景下的数据工程挑战，从海量数据的去重、多样性优化，到主动学习和数据飞轮的构建。我们将系统性地建立一套数据质量控制体系，确保模型能够从有限的计算资源中获得最大的学习收益。

## 2.1 大规模数据去重算法

### 去重的必要性与挑战

自动驾驶数据采集面临严重的冗余问题：
- **时间冗余**：30fps视频中相邻帧相似度极高（>95%）
- **空间冗余**：高速公路等场景高度重复
- **车队冗余**：多辆车经过同一路段产生相似数据

数据规模挑战：
```
数据源         日增量        累计规模       去重需求
────────────────────────────────────────────────────
测试车队       10TB/天       PB级别        90%去重率
众包车辆       100TB/天      10PB级别      95%去重率
仿真数据       1TB/天        100TB级别     70%去重率
```

### 多级去重架构

**Level 1: 帧级去重**

感知哈希（Perceptual Hash）快速筛选：
```
Image ──> Resize(64x64) ──> DCT ──> Quantize ──> Hash(64bit)
                │                        │
                ↓                        ↓
         减少计算量              保留主要特征
```

汉明距离判定相似度：
$$d_{hamming}(h_1, h_2) = \sum_{i=1}^{64} h_1[i] \oplus h_2[i]$$

当$d_{hamming} < \tau$（典型值5-10）时认为重复。

**Level 2: 片段级去重**

视频片段特征提取：
$$\mathbf{v}_{clip} = \text{Pool}([f_1, f_2, ..., f_T])$$

其中Pool可以是：
- Mean pooling：捕获平均特征
- Max pooling：捕获显著特征
- Temporal attention：自适应加权

MinHash实现高效相似度估计：
$$P(h_{min}(A) = h_{min}(B)) = \frac{|A \cap B|}{|A \cup B|} = J(A, B)$$

使用k个独立哈希函数，Jaccard相似度估计误差为$O(1/\sqrt{k})$。

**Level 3: 轨迹级去重**

GPS轨迹相似度度量：
$$d_{traj} = \text{DTW}(T_1, T_2) + \lambda \cdot d_{Fréchet}(T_1, T_2)$$

其中DTW处理时间对齐，Fréchet距离保持空间形状。

### 分布式去重实现

**LSH森林索引**：
```
新数据 ──> 多个LSH函数 ──> 哈希桶
             │                │
             ↓                ↓
        降维投影         候选集合
             │                │
             └────> 精确匹配 <─┘
```

LSH函数设计（for cosine similarity）：
$$h(\mathbf{v}) = \text{sign}(\mathbf{r}^T \mathbf{v})$$

其中$\mathbf{r}$从高斯分布采样。

**分布式处理流程**：

1. **分片策略**：按时空网格划分
   ```
   Grid(lat, lon, time) ──> Shard_ID
   ```

2. **并行去重**：
   ```
   Map Phase:  Data ──> Feature ──> (Hash, Data_ID)
   Reduce Phase: Group by Hash ──> Deduplicate
   ```

3. **增量更新**：
   维护Bloom Filter快速预过滤：
   $$P_{false\_positive} = (1 - e^{-kn/m})^k$$

   其中k为哈希函数数，n为元素数，m为位数组大小。

**Rule of Thumb**：
- 感知哈希：64-128位足够
- MinHash签名：128-256个置换
- LSH森林：32-64棵树
- Bloom Filter：10倍预期容量，k=7

### 时序一致性去重

**滑动窗口去重**：
滑动窗口去重维护一个时间窗口[t-W, t]的历史缓冲区，对每个新帧t+1进行相似性检查。算法将窗口内的历史帧作为参考集，新帧作为候选项进行比对。如果新帧与窗口内任何帧的相似度超过阈值，则认为是重复帧并丢弃。窗口随时间滑动前进，既保证了时序局部性，又限制了比对的计算量。这种方法特别适合处理视频流中的静止场景和慢速运动导致的帧间冗余。

**关键帧提取**：
基于信息增益选择关键帧：
$$I(f_t) = H(f_t|f_{<t}) = -\sum_i p_i \log p_i$$

当$I(f_t) > \tau_{info}$时保留为关键帧。

**场景变化检测**：
使用LSTM预测下一帧，预测误差大表示场景变化：
$$\Delta_t = \|f_t - \hat{f}_t\|_2 > \tau_{change}$$

## 2.2 数据多样性度量与优化

### 多样性的多维度定义

数据多样性不是单一指标，需要从多个维度综合评估：

**维度分解**：
```
多样性维度        描述                     度量方法
─────────────────────────────────────────────────────
场景多样性        天气/光照/地点           熵、覆盖率
对象多样性        车辆/行人/物体类别       类别分布、稀有度
行为多样性        驾驶行为/交互模式        轨迹聚类、动作空间
时序多样性        速度/加速度分布          统计矩、分布距离
语义多样性        道路类型/交通状况        场景图多样性
```

### 特征空间多样性度量

**嵌入空间覆盖度**：
使用预训练编码器将数据映射到特征空间：
$$\mathbf{z} = \text{Encoder}(\mathbf{x})$$

计算特征空间的覆盖度：
$$\text{Coverage} = \frac{|\text{VoronoiCell}(\mathcal{Z})|}{|\mathcal{Z}_{total}|}$$

**核密度估计**：
$$\hat{p}(\mathbf{z}) = \frac{1}{n h^d} \sum_{i=1}^n K\left(\frac{\mathbf{z} - \mathbf{z}_i}{h}\right)$$

多样性度量为熵：
$$H = -\int \hat{p}(\mathbf{z}) \log \hat{p}(\mathbf{z}) d\mathbf{z}$$

**最远点采样（FPS）**：
迭代选择距离已选集合最远的点：
$$\mathbf{z}_{next} = \arg\max_{\mathbf{z} \in \mathcal{Z}} \min_{\mathbf{z}_i \in \mathcal{S}} d(\mathbf{z}, \mathbf{z}_i)$$

FPS保证了空间的均匀覆盖。

### 场景级多样性优化

**场景图表示**：
```
Scene Graph: G = (V, E)
- V: {road, vehicle, pedestrian, traffic_sign, ...}
- E: {near, on, crossing, following, ...}
```

场景图编辑距离：
$$d_{GED}(G_1, G_2) = \min_{P \in \mathcal{P}} \sum_{op \in P} c(op)$$

其中$\mathcal{P}$为所有可能的编辑路径。

**场景聚类与平衡采样**：

1. K-means++初始化保证聚类中心分散
2. 每个聚类采样数量反比于聚类大小：
   $$n_i = N \cdot \frac{1/|C_i|^{\alpha}}{\sum_j 1/|C_j|^{\alpha}}$$

   其中$\alpha \in [0.5, 1]$控制平衡程度。

**稀有场景加权**：
$$w(\mathbf{x}) = \frac{1}{\hat{p}(\mathbf{x})^\beta}$$

其中$\beta$控制对稀有样本的重视程度。

### 时序多样性优化

**动作空间覆盖**：
将连续动作空间离散化为网格：
```
Steering: [-1, -0.5, 0, 0.5, 1]
Throttle: [0, 0.3, 0.6, 1.0]
Brake:    [0, 0.5, 1.0]
```

统计动作组合覆盖率：
$$\text{ActionCoverage} = \frac{|\text{VisitedCells}|}{|\text{TotalCells}|}$$

**轨迹多样性**：
使用DTW聚类轨迹：
$$\text{DTW}(T_1, T_2) = \min_{\pi} \sum_{(i,j) \in \pi} d(t_1^i, t_2^j)$$

选择每个聚类的代表性轨迹。

**速度分布匹配**：
使用Wasserstein距离度量分布差异：
$$W_2(P, Q) = \left(\inf_{\gamma \in \Gamma(P,Q)} \int \|\mathbf{x} - \mathbf{y}\|^2 d\gamma(\mathbf{x}, \mathbf{y})\right)^{1/2}$$

优化采样使得训练集速度分布接近目标分布。

### 主动多样性采集

**不确定性引导**：
模型预测不确定性高的区域需要更多数据：
$$U(\mathbf{x}) = H[p(y|\mathbf{x})] = -\sum_y p(y|\mathbf{x}) \log p(y|\mathbf{x})$$

**预期模型改变（EMC）**：
估计新样本对模型的影响：
$$\text{EMC}(\mathbf{x}) = \|\nabla_\theta \mathcal{L}(\mathbf{x})\|_2$$

梯度范数大表示该样本能显著改变模型。

**多样性奖励函数**：
$$R(\mathbf{x}) = \lambda_1 \cdot d_{min}(\mathbf{x}, \mathcal{D}) + \lambda_2 \cdot U(\mathbf{x}) + \lambda_3 \cdot \text{Rarity}(\mathbf{x})$$

其中$d_{min}$为到现有数据集的最小距离。

**Rule of Thumb**：
- 特征维度：512-2048维
- 场景聚类数：1000-5000类
- 稀有场景权重：10-100倍
- 不确定性阈值：熵的80分位数

## 2.3 高价值场景挖掘

### 价值定义与量化

高价值场景的多维度定义：

**安全关键性**：
```
场景类型              权重    示例
─────────────────────────────────────────
紧急制动             10.0    AEB触发场景
碰撞临界             8.0     TTC < 2s
违规行为             6.0     闯红灯、逆行
复杂交互             5.0     多车博弈
恶劣条件             4.0     雨雪雾、强光
```

**学习价值评估**：
$$V_{learn} = \alpha \cdot P_{error} + \beta \cdot I_{gradient} + \gamma \cdot R_{novel}$$

其中：
- $P_{error}$：模型预测错误率
- $I_{gradient}$：梯度信息量
- $R_{novel}$：新颖性评分

### 错误模式分析

**预测误差聚类**：
收集模型预测与真值的误差向量：
$$\mathbf{e}_t = \mathbf{y}_t - \hat{\mathbf{y}}_t$$

使用DBSCAN聚类发现系统性错误模式：
```
DBSCAN(ε, MinPts) ──> Error Clusters
         │                    │
         ↓                    ↓
   密度可达性            错误模式
```

**因果分析**：
构建错误因果图：
$$P(\text{Error}|\text{Conditions}) = \frac{P(\text{Conditions}|\text{Error}) \cdot P(\text{Error})}{P(\text{Conditions})}$$

识别导致错误的关键条件组合。

### 长尾分布挖掘

**重尾建模**：
使用Pareto分布建模长尾：
$$P(X > x) = \left(\frac{x_m}{x}\right)^\alpha$$

其中$\alpha$控制尾部厚度，典型值1.5-2.5。

**异常检测方法**：

1. **Isolation Forest**：
   通过随机划分隔离异常点：
   $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

   其中$h(x)$为隔离路径长度。

2. **VAE重构误差**：
   $$\mathcal{A}(\mathbf{x}) = \|\mathbf{x} - \text{Decoder}(\text{Encoder}(\mathbf{x}))\|_2$$

   重构误差大表示异常。

3. **OOD检测**：
   使用能量分数：
   $$E(\mathbf{x}) = -T \cdot \log \sum_y \exp(f_y(\mathbf{x})/T)$$

### 交互复杂度评估

**图复杂度度量**：
构建交通参与者交互图：
```
G_t = (V_t, E_t)
V_t: agents at time t
E_t: interactions (distance < threshold)
```

复杂度指标：
- 节点度分布熵：$H(deg) = -\sum_k p_k \log p_k$
- 聚类系数：$C = \frac{3 \times \text{triangles}}{\text{connected triples}}$
- 介数中心性：$BC(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$

**时序交互模式**：
使用注意力矩阵量化交互强度：
$$A_{ij}^t = \text{softmax}\left(\frac{Q_i^t \cdot K_j^t}{\sqrt{d_k}}\right)$$

交互复杂度：
$$\mathcal{C}_t = -\sum_{i,j} A_{ij}^t \log A_{ij}^t$$

### 仿真增强挖掘

**基于仿真的扰动**：
对真实场景施加扰动生成变体：
```
Real Scene ──> Perturbation ──> Variants
                   │                │
                   ↓                ↓
              速度±20%        生成10个变体
              位置±2m         评估鲁棒性
```

**反事实生成**：
$$\mathbf{x}_{cf} = \arg\min_{\mathbf{x}'} d(\mathbf{x}, \mathbf{x}') + \lambda \cdot \mathbb{1}[f(\mathbf{x}') \neq f(\mathbf{x})]$$

寻找最小改变导致不同结果的场景。

**场景插值**：
在latent space插值生成中间场景：
$$\mathbf{z}_{interp} = (1-\alpha) \cdot \mathbf{z}_1 + \alpha \cdot \mathbf{z}_2$$
$$\mathbf{x}_{new} = \text{Decoder}(\mathbf{z}_{interp})$$

**Rule of Thumb**：
- 错误聚类：DBSCAN ε=0.5, MinPts=10
- 异常阈值：99.5分位数
- 交互距离阈值：10米
- 扰动范围：关键参数的±20%

## 2.4 Fleet Trigger与主动学习

### Fleet Trigger架构

**分布式触发系统**：
```
Edge Model ──> Trigger Condition ──> Cloud Upload
     │              │                      │
     ↓              ↓                      ↓
 轻量推理      满足条件?            高价值数据
                    │
                    ├─> Uncertainty > τ
                    ├─> Anomaly Score > θ
                    └─> Safety Critical
```

**触发条件设计**：

1. **不确定性触发**：
   $$\text{Trigger}_{uncertainty} = \mathbb{1}[H(p(y|x)) > \tau_H]$$

2. **预测分歧触发**：
   多模型ensemble分歧：
   $$\text{Disagreement} = \text{Var}[\{f_i(x)\}_{i=1}^M] > \tau_{var}$$

3. **安全边界触发**：
   $$\text{Trigger}_{safety} = \mathbb{1}[\min(TTC, TTV, TTA) < \tau_{safe}]$$

4. **新颖性触发**：
   $$\text{Trigger}_{novel} = \mathbb{1}[d(x, \mathcal{D}_{train}) > \tau_{dist}]$$

### 边缘模型设计

**轻量化架构**：
```
Full Model (20B params) ──> Distillation ──> Edge Model (100M params)
                                │                     │
                                ↓                     ↓
                          知识蒸馏              10% size
                          特征对齐              50% accuracy
```

**多级决策树**：
触发策略采用分级决策树：首先检查是否为安全关键场景（如紧急制动、避障），若是则立即触发采集。其次评估模型不确定性，当超过0.8阈值时，如果计算预算允许则运行详细检查。最后对于新颖度分数超过0.9的样本，以0.5的概率进行采样。这种级联决策确保了安全优先、资源高效、兼顾探索的数据采集策略。

**资源约束优化**：
带宽限制下的优先级队列：
$$\text{Priority}(x) = w_1 \cdot \text{Safety} + w_2 \cdot \text{Learning} + w_3 \cdot \text{Novelty}$$

选择top-k样本满足：
$$\sum_{i=1}^k \text{Size}(x_i) \leq B_{max}$$

### 主动学习策略

**查询策略**：

1. **最大熵采样**：
   $$x^* = \arg\max_x H[p(y|x)]$$

2. **BALD（Bayesian Active Learning by Disagreement）**：
   $$x^* = \arg\max_x I(y; \theta|x) = H[p(y|x)] - E_{p(\theta|D)}[H[p(y|x,\theta)]]$$

3. **核心集选择**：
   最小化最大距离：
   $$\min_S \max_{x \in \mathcal{U}} \min_{s \in S} d(x, s)$$

**批量主动学习**：
避免冗余的多样性批选择：
$$S^* = \arg\max_S \sum_{x \in S} U(x) - \lambda \sum_{x_i, x_j \in S} \text{Sim}(x_i, x_j)$$

### 在线学习与更新

**增量学习框架**：
```
New Data ──> Validation ──> Incremental Training ──> A/B Test
                │                    │                    │
                ↓                    ↓                    ↓
           质量检查           避免灾难遗忘         性能验证
```

**Experience Replay Buffer**：
维护关键历史数据：
$$\mathcal{B} = \mathcal{B}_{old} \cup \mathcal{B}_{new}$$

采样策略：
$$p(x) \propto \text{Age}(x)^{-\alpha} \cdot \text{Importance}(x)^\beta$$

**持续学习正则化**：
EWC（Elastic Weight Consolidation）：
$$\mathcal{L}_{total} = \mathcal{L}_{new} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$$

其中$F_i$为Fisher信息矩阵对角元。

### 反馈闭环设计

**性能监控指标**：
```
Metric              Target    Alert Threshold
──────────────────────────────────────────────
Trigger Rate        1-5%      >10%
Upload Success      >95%      <90%
Label Quality       >98%      <95%
Model Improvement   >1%/week  <0.5%/week
```

**自适应阈值调整**：
基于历史性能动态调整触发阈值：
$$\tau_{t+1} = \tau_t + \eta \cdot (\text{Target} - \text{Actual})$$

使用PID控制器稳定调整：
$$\tau_{t+1} = \tau_t + K_p e_t + K_i \sum_{i=0}^t e_i + K_d (e_t - e_{t-1})$$

**Rule of Thumb**：
- 边缘模型大小：<200MB
- 推理延迟：<50ms
- 触发率：1-5%
- 批量大小：32-128样本
- 更新频率：周级别

## 2.5 数据飞轮与持续改进

### 数据飞轮机制

**闭环系统设计**：
```
Deploy ──> Collect ──> Label ──> Train ──> Evaluate
  ↑                                           │
  └───────────────────────────────────────────┘
                    Continuous Loop
```

**飞轮加速因子**：
1. **规模效应**：更多用户→更多数据→更好模型→吸引更多用户
2. **质量提升**：错误case→targeted collection→修复问题→减少错误
3. **效率优化**：自动标注→人工验证→标注模型改进→降低成本

### 自动标注系统

**多模型交叉验证**：
```python
predictions = [model_i.predict(x) for model_i in models]
confidence = agreement_score(predictions)
if confidence > threshold:
    auto_label = majority_vote(predictions)
else:
    send_to_human_labeling()
```

**伪标签质量控制**：
$$\text{Quality}(y_{pseudo}) = \text{Confidence} \times \text{Consistency} \times \text{Diversity}$$

其中：
- Confidence：模型置信度
- Consistency：时序一致性
- Diversity：与已有数据的差异

**教师-学生框架**：
$$\mathcal{L}_{student} = \mathcal{L}_{supervised} + \lambda \cdot \mathcal{L}_{distill}$$
$$\mathcal{L}_{distill} = \text{KL}(p_{student} || p_{teacher})$$

### 版本管理与回滚

**数据版本控制**：
```
Dataset_v1.0 ──> Dataset_v1.1 ──> Dataset_v2.0
      │              │                  │
      ↓              ↓                  ↓
  Model_v1.0    Model_v1.1        Model_v2.0
      │              │                  │
      └──> Metrics <─┴──────────────────┘
```

**增量更新策略**：
- Delta encoding：只存储变化部分
- Merkle tree：快速验证数据完整性
- Git-like branching：支持实验分支

**回滚机制**：
自动回滚系统持续监控模型性能指标，当检测到性能退化时立即触发回滚流程。系统首先恢复到上一个稳定版本，确保服务不中断。然后启动根因分析，检查是数据分布偏移、标注错误还是训练问题。最后修复问题并重新训练，只有在新版本通过全面验证后才会再次部署。这种机制确保了系统的稳定性和可靠性。

### 质量度量体系

**多层次评估**：
```
Level           Metrics                  Frequency
────────────────────────────────────────────────
Sample-level    Accuracy, Confidence     Real-time
Batch-level     Distribution shift       Hourly
Version-level   Overall performance      Daily
System-level    End-to-end metrics       Weekly
```

**分布偏移检测**：
使用MMD（Maximum Mean Discrepancy）：
$$\text{MMD}^2 = \|\mu_P - \mu_Q\|_{\mathcal{H}}^2$$

当MMD超过阈值时触发重训练。

**长期趋势分析**：
系统使用指数平滑算法对历史性能指标进行趋势预测，赋予近期数据更高权重。当检测到趋势斜率为负（性能下降）时，自动触发退化原因调查。调查包括分析数据分布变化、检查模型各层激活分布、对比不同时期的错误模式等。这种前瞻性监控能够在问题严重化之前及早发现和干预。

### 成本优化策略

**标注成本模型**：
$$C_{total} = C_{human} \cdot N_{human} + C_{compute} \cdot N_{auto} + C_{verify} \cdot N_{verify}$$

优化目标：
$$\min C_{total} \text{ s.t. } \text{Quality} > Q_{min}$$

**主动学习ROI**：
$$\text{ROI} = \frac{\Delta \text{Performance}}{C_{annotation} + C_{training}}$$

优先标注高ROI样本。

**计算资源调度**：
- 峰谷错峰训练
- Spot instance利用
- 模型并行与数据并行混合

**Rule of Thumb**：
- 自动标注比例：>70%
- 人工验证比例：10-20%
- 数据增长率：10%/月
- 模型更新周期：1-2周
- 性能提升目标：1%/月

## 高级话题：对抗样本生成与Corner Case合成

### 对抗性数据增强

**梯度对抗扰动**：
生成最坏情况扰动：
$$\mathbf{x}_{adv} = \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f(\mathbf{x}), y))$$

对于视频序列，考虑时序一致性：
$$\mathbf{x}_t^{adv} = \mathbf{x}_t + \epsilon \cdot \text{sign}(\nabla_{\mathbf{x}_t} \mathcal{L} + \lambda \cdot \|\mathbf{x}_t^{adv} - \mathbf{x}_{t-1}^{adv}\|_2)$$

**物理世界对抗样本**：
考虑真实世界约束：
```
Digital Perturbation ──> Physical Constraints ──> Realizable Attack
         │                        │                      │
         ↓                        ↓                      ↓
    像素级扰动              光照/遮挡限制          可实现攻击
```

约束优化问题：
$$\min_\delta \mathcal{L}_{adv} \text{ s.t. } \delta \in \mathcal{C}_{physical}$$

其中$\mathcal{C}_{physical}$包含：
- 光照变化范围
- 几何形变限制
- 运动模糊约束

### 生成式Corner Case合成

**条件VAE生成**：
$$p(z|y_{rare}) \sim \mathcal{N}(\mu_\phi(y_{rare}), \sigma_\phi^2(y_{rare}))$$
$$x_{synthetic} = g_\theta(z, y_{rare})$$

针对稀有类别$y_{rare}$生成新样本。

**场景组合生成**：
```python
# 组合不同元素生成新场景
weather = sample(['rain', 'fog', 'snow'])
lighting = sample(['night', 'dawn', 'backlight'])
traffic = sample(['crowded', 'empty', 'mixed'])
scenario = combine(weather, lighting, traffic)
```

**Diffusion模型引导生成**：
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t)) + \sigma_t \mathbf{z}$$

添加条件引导：
$$\epsilon_\theta(\mathbf{x}_t, t, c) = \epsilon_\theta(\mathbf{x}_t, t) + s \cdot \nabla_{\mathbf{x}_t} \log p(c|\mathbf{x}_t)$$

### 安全性验证生成

**形式化验证场景**：
定义安全属性$\phi$：
$$\phi ::= \text{Always}(\text{distance} > d_{min}) \land \text{Eventually}(\text{reach\_goal})$$

反例生成：
$$\mathbf{x}_{counter} = \arg\min_\mathbf{x} d(\mathbf{x}, \mathcal{D}_{real}) \text{ s.t. } \neg\phi(\mathbf{x})$$

**蒙特卡洛树搜索（MCTS）**：
MCTS系统地探索场景空间寻找危险案例。从当前场景配置（状态）出发，通过参数调整（动作）生成新场景（下一状态），并评估奖励（如碰撞给予-1000负奖励）。搜索过程迭代执行选择、扩展、模拟和回传四个步骤，逐步构建场景树，识别导致失败的关键场景配置。这种方法能够高效地发现稀有但关键的边界案例。

通过MCTS搜索危险场景：
$$UCB = \frac{Q(s,a)}{N(s,a)} + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$$

### 域随机化与泛化

**系统性域随机化**：
域随机化通过系统地变化环境参数来提升模型泛化能力。纹理强度在0.5到2.0倍之间均匀采样，模拟不同材质表面；色调偏移在-30到30度范围内调整，覆盖各种光照条件；噪声水平从0到0.1添加，模拟传感器噪声；运动模糊核大小在0到3像素变化，模拟不同速度下的成像；天气类型从预定义类别中随机选择，包括晴天、雨天、雾天等。这种多维度随机化策略确保模型在多样化条件下的鲁棒性。

**域插值与外推**：
$$\mathcal{D}_{new} = (1-\alpha) \cdot \mathcal{D}_{source} + \alpha \cdot \mathcal{D}_{target}, \alpha \in [-0.2, 1.2]$$

允许外推生成更极端场景。

**对抗域适应**：
$$\mathcal{L} = \mathcal{L}_{task} - \lambda \cdot \mathcal{L}_{domain}$$

使特征对域变化鲁棒。

## 本章小结

本章系统介绍了自动驾驶视频模型的数据工程体系：

**关键概念**：
1. **大规模去重**：多级去重架构，从帧级到轨迹级
2. **多样性优化**：多维度评估与主动采样
3. **高价值挖掘**：错误分析、长尾检测、复杂度评估
4. **Fleet Trigger**：边缘计算与分布式数据收集
5. **数据飞轮**：闭环优化与持续改进

**核心公式**：
- 感知哈希：$d_{hamming}(h_1, h_2) = \sum_{i=1}^{64} h_1[i] \oplus h_2[i]$
- 多样性度量：$H = -\int \hat{p}(\mathbf{z}) \log \hat{p}(\mathbf{z}) d\mathbf{z}$
- 主动学习：$x^* = \arg\max_x I(y; \theta|x)$
- 对抗生成：$\mathbf{x}_{adv} = \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L})$

**核心论文**：
1. [Active Learning] **Active Learning for Deep Object Detection**, ICCV 2019
2. [Data Efficiency] **Model-Agnostic Meta-Learning for Fast Adaptation**, ICML 2017
3. [Adversarial] **Robust Physical-World Attacks on Deep Learning Visual Classification**, CVPR 2018
4. [Continual Learning] **Elastic Weight Consolidation**, PNAS 2017
5. [Data Flywheel] **Tesla AI Day: Data Engine and Auto-labeling**, 2021

## 常见陷阱与错误 (Gotchas)

### 1. 过度去重导致多样性损失
**问题**：激进去重删除了有价值的细微差异
**症状**：模型泛化性能下降，对细节变化不敏感
**解决**：
- 使用软去重（相似度加权）而非硬去重
- 保留边界样本（相似度在阈值附近）
- 定期评估去重对下游任务的影响

### 2. 标注噪声累积
**问题**：自动标注错误通过数据飞轮放大
**症状**：模型性能先升后降，出现系统性偏差
**解决**：
- 设置置信度阈值，低置信样本人工复核
- 维护标注质量追踪系统
- 定期清洗历史数据

### 3. 分布漂移未检测
**问题**：真实世界分布缓慢变化未被察觉
**症状**：线上性能逐渐退化
**解决**：
- 部署分布监控系统（MMD、KL散度）
- 设置自动告警阈值
- 定期重新评估和校准

### 4. Fleet Trigger过于敏感
**问题**：触发率过高导致带宽和存储爆炸
**症状**：数据上传量超预期，成本失控
**解决**：
- 动态调整触发阈值
- 实施分级触发策略
- 本地预过滤和采样

### 5. 对抗样本过拟合
**问题**：过度优化对抗鲁棒性损害正常性能
**症状**：对抗测试分数高但实际部署效果差
**解决**：
- 控制对抗训练比例（10-20%）
- 使用多种对抗方法
- 在真实数据上验证

### 6. 版本依赖混乱
**问题**：数据版本与模型版本不匹配
**症状**：重现实验失败，性能不一致
**解决**：
- 严格版本对应关系记录
- 使用容器化环境
- 自动化版本兼容性检查

### 7. 采样偏差放大
**问题**：主动学习偏向某类场景
**症状**：模型在特定场景过拟合
**解决**：
- 多目标优化采样策略
- 设置各类场景配额
- 定期重置采样分布

### 8. 计算资源浪费
**问题**：重复计算相似样本的特征
**症状**：训练时间过长，GPU利用率低
**解决**：
- 特征缓存机制
- 增量特征更新
- 相似样本批处理