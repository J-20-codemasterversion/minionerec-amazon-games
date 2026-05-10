# 基于 Amazon Review 数据的推荐系统 — 开发心路历程

## 一、项目背景

数据来源是 Amazon Review 全量数据集：
- **评论数据**: `All_Amazon_Review_5.json` (82GB, 5-core 过滤)
- **元数据**: `All_Amazon_Meta.json` (105GB)

目标：构建一个完整的工业级四阶段推荐流水线 **召回 → 粗排 → 精排 → 重排序**。

---

## 二、V1 阶段：小数据集上跑通流程

### 做了什么

第一版 `recommendation_pipeline.py` 使用的是 **AMAZON_FASHION** 这个小品类数据集，做了 3-core 过滤。目的是先在小数据上验证整个 pipeline 的正确性。

四阶段实现：
- **召回**: 热门 + ItemCF + UserCF + 内容召回（基于 also_buy/also_view）
- **粗排**: GradientBoostingRegressor，16 维轻量特征
- **精排**: FM (Factorization Machine)，SGD 训练
- **重排序**: 品牌多样性 + 新鲜度 + 来源加分

### 思考

小数据集跑通后，各阶段指标都能输出，验证了 pipeline 的端到端流程没问题。但 AMAZON_FASHION 数据量太小，不能代表真实场景。

---

## 三、V2 阶段（full 版）：切换到全量数据

### 问题

全量数据 82GB，不可能全部加载到内存。

### 做了什么

`recommendation_pipeline_full.py` 采用了采样策略：
- 前 100 万条全取，之后按 1% 采样率抽样
- 最多加载 200 万条评论
- 对加载后的数据做 3-core 过滤（用户/商品最少 3 条交互）

```python
MAX_REVIEWS = 2000000
SAMPLE_RATE = 0.01
```

同时，召回模块也做了模块化拆分，写成了独立文件：
- `recall_hot.py` — 热门召回
- `recall_itemcf.py` — ItemCF 召回
- `recall_usercf.py` — UserCF 召回
- `recall_swing.py` — Swing 召回（新增）
- `recall_fusion.py` — 多路融合（4 路：hot/itemcf/usercf/swing）
- `recall_data_loader.py` — 数据加载器

在 `recall_main_v2.py` 中又增加了 **ContentRecall（内容召回）** 通道，基于品牌/品类/标题关键词构建倒排索引，形成了 5 路召回融合的 `RecallFusionV2`。

### 思考

采样策略有个根本问题：**前 100 万条 + 随机 1% 采样** 意味着数据分布是偏斜的。前 100 万条可能集中在某些品类或时间段，随机采样又太稀疏，导致用户-商品交互矩阵非常稀疏，协同过滤效果差。

而且，这种策略无法控制最终选取的用户和商品的活跃度分布，低活跃用户太多会拉低整体效果。

---

## 四、V3 阶段：重新设计数据加载策略

### 核心问题

如何从 82GB 的全量 5-core 数据中，选出一个 **高质量、密度足够** 的子集？

### 思路演变

**最初想法**: 加载前 N 行，然后过滤掉低活跃用户（交互 < 5 的）。

**问题**: 前 N 行的数据分布不均匀，而且"先加载后过滤"会浪费大量 IO——可能加载了 500 万条，过滤完只剩很少。

**最终方案**: 两遍扫描（Two-Pass Scan）

1. **第 1 遍扫描**: 遍历全部 82GB，用 `Counter` 统计每个用户和每个商品的交互次数（只需记 ID 和计数，内存开销可控）
2. **选取 Top 集合**: 取交互次数最多的 10 万个用户 + 10 万个商品
3. **第 2 遍扫描**: 再遍历一次全文件，只保留 top 用户 × top 商品 之间的评论

```python
TARGET_USERS = 100000
TARGET_ITEMS = 100000
```

### 为什么这个方案好

- **数据密度高**: 选的都是高活跃用户和热门商品，它们之间的交互更密集，协同过滤信号更强
- **无采样偏差**: 不做随机采样，而是根据全局统计信息做确定性选取
- **数据本身已是 5-core**: 不需要再做额外的 core 过滤
- **可控性好**: 想要更大/更小的数据集，只需调整 `TARGET_USERS` 和 `TARGET_ITEMS`

### 代价

需要扫描两遍全量 82GB 文件，IO 时间较长。但这是一次性操作，且每遍扫描都是顺序读取，在 SSD 上可以接受。

---

## 五、V3 阶段：召回模块重构

### 问题

`recommendation_pipeline_v3.py` 最初把所有召回类（HotRecall、ItemCFRecall、UserCFRecall、SwingRecall、RecallFusionV2）都内联写在文件里，导致文件巨大（1500+ 行），可读性差。

而这些类的独立模块版本已经写好了（`recall_hot.py` 等），应该直接复用。

### 做了什么

1. **删除内联的 5 个召回类**（约 340 行代码）
2. **添加外部 import**:
   ```python
   from recall_hot import HotRecall
   from recall_itemcf import ItemCFRecall
   from recall_usercf import UserCFRecall
   from recall_swing import SwingRecall
   from recall_fusion import RecallFusion
   ```
3. **保留 ContentRecall 内联**: 因为外部 `RecallFusion` 只有 4 路（hot/itemcf/usercf/swing），不包含 content 通道
4. **手动合并 content 结果**: 在 `do_recall()` 中，先调用 4 路 `fusion_4ch.recall()`，再单独调用 `content_recall.recall()`，手动将结果 merge 进去：

```python
CONTENT_WEIGHT = 0.25

# 4路融合召回
base_results = fusion_4ch.recall(user_id, user_history, total_recall=total_k)

# content 通道补充
content_items = content_recall.recall(user_history, user_history_set, top_k=per_channel)
# ... merge content results into base_results ...

# 多路命中加分
for item in merged:
    n_src = len(merged[item]['sources'])
    if n_src > 1:
        merged[item]['score'] *= (1 + 0.1 * n_src)
```

### 思考

这样做的好处是：
- 独立召回模块可以在 `recall_main_v2.py` 中单独测试和评估
- `recommendation_pipeline_v3.py` 只关注 pipeline 的编排逻辑，代码量从 1500+ 行降到 1179 行
- ContentRecall 保留内联是因为它的存在是 pipeline 层面的决策（外部模块没有也不应该有），这种差异性属于 pipeline 本身

---

## 六、排序阶段的设计决策

### 粗排：GBT pointwise + pairwise 负采样

用 `GradientBoostingClassifier`，正样本是用户实际交互，负样本是随机采样未交互的商品。本质是一个二分类问题：用户会不会和这个商品交互。

训练样本量控制在 15000 对（正+负），因为 GBT 训练开销较大。

### 精排：FM + BPR pairwise 损失

FM 用 BPR (Bayesian Personalized Ranking) 损失训练，更适合排序任务。每个训练样本是 (user, pos_item, neg_item) 三元组，目标是让 pos_item 的分数高于 neg_item。

手写了 mini-batch SGD，包括：
- 梯度裁剪（`CLIP = 5.0`）
- 用户/商品 embedding + 特征交互
- 批量化计算加速

### 重排序：贪心选择

贪心地逐个选取：
- 品牌去重惩罚（-0.3）
- 新商品曝光加分（+0.05）
- 多路召回来源加分

---

## 七、评估设计

### 数据划分

采用 **leave-last-one-out per user**: 每个用户按时间排序，最后一条交互作为测试集，其余作为训练集。要求每个用户至少有 3 条交互（2 条训练 + 1 条测试）。

### 指标

| 阶段 | HitRate@10 | NDCG@10 | AUC |
|------|-----------|---------|-----|
| 召回 | ✓ | ✓ | — |
| 粗排 | ✓ | ✓ | ✓ |
| 精排 | ✓ | ✓ | ✓ |
| 重排序 | ✓ | ✓ | ✓ |

- **HitRate@10**: Top 10 推荐中是否命中了用户真实交互的商品
- **NDCG@10**: 考虑命中位置的排序质量
- **AUC**: 排序阶段的区分能力（正样本分数 > 负样本分数的概率）

---

## 八、文件结构总结

```
回归/
├── recommendation_pipeline.py       # V1: 小数据集 (AMAZON_FASHION)
├── recommendation_pipeline_full.py  # V2: 全量数据采样版
├── recommendation_pipeline_v3.py    # V3: 两遍扫描 + 外部召回模块 (当前主版本)
├── recall_hot.py                    # 热门召回模块
├── recall_itemcf.py                 # ItemCF 召回模块
├── recall_usercf.py                 # UserCF 召回模块
├── recall_swing.py                  # Swing 召回模块
├── recall_fusion.py                 # 4路召回融合模块
├── recall_data_loader.py            # 数据加载器
├── recall_main_v2.py                # 召回系统独立评估 (5路, 含ContentRecall)
├── 推荐系统方案.md                    # 系统设计文档
└── 推荐系统流水线报告.md              # 流水线报告
```

---

## 九、关键经验

1. **数据质量 > 数据量**: 与其随机采样 200 万条稀疏数据，不如精选 10 万活跃用户 × 10 万热门商品的密集子集
2. **模块化很重要**: 召回模块独立出去后，可以单独测试、单独调参，pipeline 文件也更清晰
3. **外部模块与 pipeline 的差异是正常的**: ContentRecall 保留内联是合理的设计——它是 pipeline 层面的决策，不应该强行塞进通用的 4 路融合模块
4. **两遍扫描虽然慢，但值得**: 全局统计信息能带来更好的数据选取决策，这个 IO 代价是值得付出的
5. **手写模型训练要注意数值稳定性**: FM 的 BPR 训练需要梯度裁剪、sigmoid 裁剪，否则容易数值爆炸
