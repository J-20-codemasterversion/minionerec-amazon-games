# 推荐系统完整方案

## 基于 Amazon Review 数据的四阶段推荐流水线

---

## 一、系统架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户请求推荐                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 1: 召回 (Recall)                                               │
│  - 多路召回：热门 + ItemCF + UserCF + 内容召回                       │
│  - 输入: 全量商品库 (~1000+)                                         │
│  - 输出: 候选集 (~1000 个)                                           │
│  - 耗时: < 10ms                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 2: 粗排 (Coarse Ranking) — XGBoost / GradientBoosting          │
│  - 轻量级特征 + 快速打分                                             │
│  - 输入: 1000 个候选                                                 │
│  - 输出: Top 100                                                     │
│  - 耗时: < 50ms                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 3: 精排 (Fine Ranking) — FM                                    │
│  - 丰富特征 + 交互建模                                               │
│  - 输入: 100 个候选                                                  │
│  - 输出: Top 20                                                      │
│  - 耗时: < 100ms                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  阶段 4: 重排序 (Re-ranking)                                         │
│  - 多样性 + 新鲜度 + 业务规则                                        │
│  - 输入: 20 个候选                                                   │
│  - 输出: Top 10 最终推荐                                             │
│  - 耗时: < 10ms                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、各阶段详细设计

### 阶段 1: 召回 (Recall)

#### 1.1 热门召回 (Popular)
```python
def recall_popular(user_idx, k=500):
    """
    目的: 兜底策略，保证有足够候选
    逻辑: 返回用户没买过的、评论数最多的商品
    特点: 简单、快速、覆盖面广
    """
    bought = user2items[user_idx]
    candidates = [(i, item_rating_cnt[i]) for i in all_items if i not in bought]
    candidates.sort(key=lambda x: -x[1])
    return candidates[:k]
```

#### 1.2 ItemCF 召回
```python
def recall_itemcf(user_idx, k=300):
    """
    目的: 基于"买了 A 的人也买了 B"
    逻辑:
      1. 找用户买过的商品 → 买过这些商品的其他用户 → 他们还买了什么
      2. 用 IUF (Inverse User Frequency) 加权：热门商品权重低
    特点: 协同过滤信号强
    """
    bought = user2items[user_idx]
    candidate_scores = {}
    for i in bought:
        for other_user in item2users[i]:
            for j in user2items[other_user]:
                if j not in bought:
                    candidate_scores[j] += 1.0 / log(1 + len(item2users[j]))
    return sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:k]
```

#### 1.3 UserCF 召回
```python
def recall_usercf(user_idx, k=200):
    """
    目的: 找相似用户买过的商品
    逻辑:
      1. 找和当前用户共同购买商品最多的用户（相似用户）
      2. 返回相似用户买过、但当前用户没买的商品
    特点: 适合兴趣相近的用户群
    """
    similar_users = find_similar_users(user_idx, top_k=50)
    candidates = []
    for sim_user in similar_users:
        for item in user2items[sim_user]:
            if item not in user2items[user_idx]:
                candidates.append(item)
    return candidates[:k]
```

#### 1.4 内容召回 (Content-Based)
```python
def recall_content(user_idx, k=200):
    """
    目的: 利用商品元数据的 also_buy / also_view 关系
    逻辑:
      1. 用户买过的商品 → 商品的 also_buy/also_view 列表
      2. also_buy 权重更高（更强的购买意图）
    特点: 利用商品知识图谱
    """
    bought = user2items[user_idx]
    candidates = {}
    for item in bought:
        for related in meta[item].get('also_buy', []):
            candidates[related] = candidates.get(related, 0) + 2.0
        for related in meta[item].get('also_view', []):
            candidates[related] = candidates.get(related, 0) + 1.0
    return sorted(candidates, key=candidates.get, reverse=True)[:k]
```

#### 1.5 多路召回融合
```python
def multi_recall(user_idx, total_k=1000):
    """
    融合策略: 来源越多的候选，分数越高
    """
    results = {
        'popular': recall_popular(user_idx, 500),
        'itemcf': recall_itemcf(user_idx, 300),
        'usercf': recall_usercf(user_idx, 200),
        'content': recall_content(user_idx, 200),
    }
    
    # 统计每个候选来自哪些通道
    candidate_sources = defaultdict(set)
    for source, items in results.items():
        for item in items:
            candidate_sources[item].add(source)
    
    # 按来源数量排序
    final = sorted(candidate_sources.items(), 
                   key=lambda x: (len(x[1]), item_rating_cnt[x[0]]), 
                   reverse=True)
    return final[:total_k]
```

---

### 阶段 2: 粗排 (Coarse Ranking) — XGBoost

#### 2.1 特征设计（16 维轻量特征）

| 类别 | 特征名 | 说明 |
|------|--------|------|
| **用户特征** | user_avg_rating | 用户历史平均评分 |
| | user_rating_cnt | 用户评分数量 |
| | user_rating_std | 用户评分标准差 |
| **商品特征** | has_brand | 是否有品牌 |
| | title_word_cnt | 标题词数 |
| | n_features | 商品特性数量 |
| | has_description | 是否有描述 |
| | has_price | 是否有价格 |
| | price | 价格 |
| | log_rank | 销量排名(对数) |
| | n_also_buy | also_buy 数量 |
| | n_also_view | also_view 数量 |
| | item_avg_rating | 商品平均评分 |
| | item_rating_cnt | 商品评分数量 |
| **交叉特征** | co_interact | 协同交互强度 |
| | jaccard | Jaccard 相似度 |

#### 2.2 模型配置

```python
from sklearn.ensemble import GradientBoostingRegressor
# 或 import xgboost as xgb

coarse_model = GradientBoostingRegressor(
    n_estimators=100,      # 树的数量
    max_depth=6,           # 最大深度（控制复杂度）
    learning_rate=0.1,     # 学习率
    random_state=42
)

# XGBoost 版本
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    n_jobs=-1
)
```

#### 2.3 训练数据构建

```python
# 正样本: 用户实际评分过的 (user, item, rating)
# 负样本: 随机采样用户没买过的商品，标记为低分 (如 2.0)
```

---

### 阶段 3: 精排 (Fine Ranking) — FM

#### 3.1 FM 模型公式

$$\hat{y} = w_0 + \sum_{i} w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

- $w_0$: 全局偏置
- $w_i$: 线性权重
- $\mathbf{v}_i$: 隐向量（学习特征交互）

#### 3.2 特征设计（更丰富）

在粗排特征基础上，可以添加：
- 更精细的用户画像
- 商品属性交叉
- 上下文特征（时间、设备等）

#### 3.3 模型配置

```python
K = 8           # 隐向量维度
LR = 0.002      # 学习率
REG_W = 0.02    # 线性项正则
REG_V = 0.02    # 交互项正则
EPOCHS = 20
```

---

### 阶段 4: 重排序 (Re-ranking)

#### 4.1 重排序策略

```python
def rerank(user_idx, candidates, top_k=10):
    """
    策略:
    1. 多样性: 避免连续推荐同品牌/同类别
    2. 新鲜度: 提升新商品曝光
    3. 召回源加分: 多通道召回的更可信
    4. 业务规则: 如高毛利优先（可选）
    """
    seen_brands = set()
    result = []
    
    for item in candidates:
        brand = get_brand(item)
        
        # 多样性惩罚
        diversity_penalty = -0.5 if brand in seen_brands else 0.0
        
        # 新鲜度加分
        freshness_bonus = 0.1 if is_new_item(item) else 0.0
        
        # 召回源加分
        source_bonus = len(item.sources) * 0.1
        
        final_score = item.fm_score + diversity_penalty + freshness_bonus + source_bonus
        result.append((item, final_score))
        seen_brands.add(brand)
    
    return sorted(result, key=lambda x: -x[1])[:top_k]
```

---

## 三、Agent 执行清单

### Task 1: 数据准备
```
[ ] 加载 review 数据 (AMAZON_FASHION.json)
[ ] 加载 meta 数据 (meta_AMAZON_FASHION.json)
[ ] 执行 3-core 过滤
[ ] 构建 user2items, item2users 索引
[ ] 按时间划分 train/test (80/20)
```

### Task 2: 召回模块
```
[ ] 实现 recall_popular()
[ ] 实现 recall_itemcf()
[ ] 实现 recall_usercf()
[ ] 实现 recall_content()
[ ] 实现 multi_recall() 融合
[ ] 测试召回率 Recall@1000
```

### Task 3: 粗排模块 (XGBoost)
```
[ ] 提取 16 维特征
[ ] 构建正负样本
[ ] 特征标准化
[ ] 训练 GradientBoostingRegressor
[ ] 实现 coarse_rank() 函数
[ ] 输出 Top 100
```

### Task 4: 精排模块 (FM)
```
[ ] 提取精排特征
[ ] 初始化 FM 参数
[ ] 实现 SGD 训练
[ ] 添加梯度裁剪 + 早停
[ ] 实现 fine_rank() 函数
[ ] 输出 Top 20
```

### Task 5: 重排序模块
```
[ ] 实现多样性逻辑
[ ] 实现新鲜度加分
[ ] 实现召回源加分
[ ] 实现 rerank() 函数
[ ] 输出 Top 10
```

### Task 6: 评估 + 可视化
```
[ ] 计算 HitRate@10
[ ] 计算 NDCG@10
[ ] 绘制各阶段指标对比图
[ ] 绘制特征重要性图
[ ] 保存 recommendation_pipeline.png
```

---

## 四、代码文件

完整实现: `recommendation_pipeline.py`

运行命令:
```bash
cd /Users/jasonlihahaha/Desktop/amazon_data/回归
python3 recommendation_pipeline.py
```

---

## 五、预期输出

```
阶段          HitRate@10      NDCG@10
------------------------------------------
召回               XX.XX%       0.XXXX
粗排(XGBoost)      XX.XX%       0.XXXX
精排(FM)           XX.XX%       0.XXXX
重排序             XX.XX%       0.XXXX
------------------------------------------
```

---

## 六、扩展方向

1. **召回增强**: 加入向量召回 (Embedding-based)
2. **粗排升级**: 使用真正的 XGBoost 或 LightGBM
3. **精排升级**: 使用 DeepFM 或 DIN
4. **重排序升级**: 加入强化学习 (RL)
5. **实时特征**: 加入用户实时行为序列
