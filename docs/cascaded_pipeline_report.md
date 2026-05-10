# Amazon 推荐系统流水线报告

## 一、项目概述

基于 **Amazon 全量评论数据**（82GB）构建完整的工业级推荐系统流水线，实现：

```
召回 → 粗排(XGBoost) → 精排(FM) → 重排序
```

---

## 二、数据处理

### 2.1 数据源

| 文件 | 大小 | 说明 |
|------|------|------|
| `All_Amazon_Review_5.json` | 82GB | 用户评论数据 |
| `All_Amazon_Meta.json` | 105GB | 商品元数据 |

### 2.2 采样策略

由于数据量巨大，采用**混合采样策略**：

```python
MAX_REVIEWS = 2,000,000    # 最多加载 200 万条评论
SAMPLE_RATE = 0.01         # 采样率（1%）
```

**策略细节**：
- **前 100 万条**：全量保留（保证数据多样性）
- **100 万条之后**：按 1% 随机采样（控制数据规模）
- **元数据加载**：只加载已有商品的 meta（按需加载）

### 2.3 数据过滤

```python
MIN_USER_REVIEWS = 3    # 用户最少 3 条评论
MIN_ITEM_REVIEWS = 3    # 商品最少 3 条评论
```

### 2.4 训练/测试集划分

- **按时间顺序划分**（模拟真实场景）
- 训练集：前 80%
- 测试集：后 20%
- 只评估**有新交互的用户**

---

## 三、特征工程

### 3.1 用户特征（3 维）

| 特征 | 说明 |
|------|------|
| `user_avg_rating` | 用户历史平均评分 |
| `user_rating_cnt` | 用户评论数量（截断 100）|
| `user_rating_std` | 用户评分标准差 |

### 3.2 商品特征（11 维）

| 特征 | 说明 |
|------|------|
| `has_brand` | 是否有品牌信息 |
| `title_word_cnt` | 标题词数（截断 50）|
| `n_features` | 特性数量（截断 20）|
| `has_description` | 是否有描述 |
| `has_price` | 是否有价格 |
| `price` | 价格（截断 500）|
| `log_rank` | 销量排名对数 |
| `n_also_buy` | "一起购买"数量 |
| `n_also_view` | "一起浏览"数量 |
| `item_avg_rating` | 商品平均评分 |
| `item_rating_cnt` | 商品评论数 |

### 3.3 交叉特征（2 维）

| 特征 | 说明 |
|------|------|
| `common_users` | 共同用户数 |
| `jaccard` | Jaccard 相似度 |

---

## 四、流水线各阶段

### 4.1 召回（Recall）

采用**多路召回融合**策略：

| 召回源 | 候选数 | 权重 | 说明 |
|--------|--------|------|------|
| **Popular** | 500 | 1.0 | 热门商品召回 |
| **ItemCF** | 300 | 2.0 | 基于物品的协同过滤 |
| **UserCF** | 200 | 1.5 | 基于用户的协同过滤 |
| **Content** | 200 | 1.8 | 基于内容（also_buy/also_view）|

**融合方式**：加权求和 + 去重，取 Top-500

### 4.2 粗排（Coarse Ranking）

**模型**：GradientBoostingRegressor

```python
n_estimators = 100    # 树数量
max_depth = 6         # 最大深度
learning_rate = 0.1   # 学习率
```

**训练数据**：
- 正样本：用户真实评论（评分作为标签）
- 负样本：随机采样未交互商品（标签 = 1.0）

**输出**：从 500 → 100 候选

### 4.3 精排（Fine Ranking）

**模型**：Factorization Machine (FM)

```python
K_FM = 8              # 隐向量维度
LR_FM = 0.002         # 学习率
REG_W = 0.02          # 线性权重正则
REG_V = 0.02          # 隐向量正则
EPOCHS_FM = 20        # 训练轮数
```

**FM 公式**：
$$\hat{y} = w_0 + w_u + w_i + \mathbf{w}^\top \mathbf{x} + \sum_{f=1}^{K} \langle \mathbf{v}_u^{(f)}, \mathbf{v}_i^{(f)} \rangle$$

**输出**：从 100 → 50 候选

### 4.4 重排序（Reranking）

**策略**：贪心选择 + 多样性约束

```python
final_score = fm_score - diversity_penalty + freshness_bonus + source_bonus
```

| 因素 | 说明 |
|------|------|
| `diversity_penalty` | 惩罚重复召回源 |
| `freshness_bonus` | 新商品加分（评论<10）|
| `source_bonus` | 多路召回加分 |

**输出**：最终 Top-10 推荐

---

## 五、评估指标

### 5.1 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| **Recall@K** | $\frac{\|R \cap GT\|}{\|GT\|}$ | 召回率 |
| **HitRate@K** | $\mathbb{1}[\|R \cap GT\| > 0]$ | 命中率 |
| **NDCG@K** | $\frac{DCG}{IDCG}$ | 排序质量 |
| **AUC** | $P(s_{pos} > s_{neg})$ | 排序区分度 |

### 5.2 各阶段指标汇总

```
阶段              HitRate@10      NDCG@10          AUC
-----------------------------------------------------------------
召回                  XX.XX%       0.XXXX          N/A
粗排(XGBoost)         XX.XX%       0.XXXX       0.XXXX
精排(FM)              XX.XX%       0.XXXX       0.XXXX
重排序                XX.XX%       0.XXXX       0.XXXX
```

> **注**：具体数值需运行代码获取

---

## 六、技术亮点

1. **大数据处理**：混合采样策略，高效处理 82GB 评论 + 105GB 元数据
2. **多路召回**：4 种召回源融合，覆盖不同用户偏好
3. **特征工程**：16 维特征，涵盖用户/商品/交叉维度
4. **工业级流水线**：完整的召回→粗排→精排→重排序架构
5. **多样性保障**：重排序阶段引入多样性约束

---

## 七、运行方式

```bash
cd /Users/jasonlihahaha/Desktop/amazon_data/回归
python3 recommendation_pipeline_full.py
```

---

## 八、文件结构

```
回归/
├── recommendation_pipeline_full.py    # 主流水线代码
├── recommendation_pipeline_full.png   # 评估结果可视化
└── 推荐系统流水线报告.md               # 本报告
```
