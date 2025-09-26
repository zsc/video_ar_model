（交流可以用英文，所有文档中文，保留这句）

## 项目目标
编写一份面向自动驾驶的视频自回归预测大模型（约 20B）的中文教科书markdown
开始先包括 Work breakdown structure，然后逐 chapter 深入讨论关键技术问题：
数据去重与筛选提升多样性，高质量数据 mining 与 trigger 下发至 fleet
原始视频的时域下采样（可以定帧率，也可以变帧率，考虑帧率感知的PE设计，考虑与 NTK context 外插的关系），
多视觉传感器融合为bev feature
空域下采样设计并与 gps trajectory 结合（类似 slam loop closure detection），
面向长 sequence 的 O(n log n) 或 O(n\sqrt{n}) attention 设计，
tokenizer/visual encoder设计（连续 vs. 离散，是否引入 diffusion 超分，bpe 等压缩方法，grounding, tokenizer loss 设计），
video/action 联合预测设计，思维链设计（纯文字，文字与视觉 token 交错，纯视觉）
MoE 架构设计（few expert, many expert, common expert 等），router aux loss等
loss 设计（是否加入object detection, semantic segmentation loss，是否用 lidar 做伪监督等）
pretrain 阶段 loss 监控，interpretability 设计（transformer-lens 可用？）, eval benchmark 设计
SFT / RL 设计（DPO/ PPO, long CoT RL）
量化剪枝等部署时加速
文件组织是 index.md + chapter1.md + ...

## Audience
verteran programmer and AI scientists

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **文字论述**：以文字论述为主，适当配上公式和 ASCII 图说明。如有数学公式，用 latex. 要有 rule-of-thumb. 不写代码。
3. **本章小结**：总结关键概念和公式，列出核心相关论文
4. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧

