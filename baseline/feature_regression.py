"""
特征回归分析：哪些指标影响评分？
================================
数据集: AMAZON_FASHION (review + meta)
过滤:   3-core (用户和商品各至少出现 3 次)
模型:   OLS 线性回归 / 岭回归 (Ridge)

特征分为 6 大类:
  ① 评论元信息   verified, has_vote, vote_count, has_image
  ② 评论文本长度  review_length, review_word_count, summary_length
  ③ 时间特征     review_year, review_month
  ④ 用户聚合统计  user_review_count, user_avg_rating, user_rating_std
  ⑤ 商品聚合统计  item_review_count, item_avg_rating, item_rating_std
  ⑥ 商品元数据   has_brand, has_price, price, log_rank, title_length, n_features

注意:
  ④⑤ 两类包含从 overall (目标变量) 计算的聚合量，存在数据泄露风险。
  这里保留它们是为了直观对比"哪些信息量最大"，不代表可直接用于线上预测。
"""

import json
import re
import datetime
import numpy as np
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────
#  辅助函数
# ────────────────────────────────────────────────────────────

def parse_price(raw):
    """从字符串中提取价格 (美元), 无效返回 None"""
    if not raw:
        return None
    s = str(raw).replace('$', '').replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_rank(raw):
    """从排名字符串/列表中提取第一个整数, 无效返回 None"""
    if not raw:
        return None
    m = re.search(r'([\d,]+)', str(raw))
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def ridge_solve(X, y, alpha):
    """闭式岭回归: w = (X'X + αI)^{-1} X'y"""
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# ════════════════════════════════════════════════════════════
#  第一步: 加载数据 + 3-core 过滤
# ════════════════════════════════════════════════════════════

print("=" * 65)
print("  第一步: 加载数据 + 3-core 过滤")
print("=" * 65)

# 商品元数据 (asin → dict)
meta = {}
with open('meta_AMAZON_FASHION.json') as f:
    for line in f:
        d = json.loads(line)
        meta[d['asin']] = d
print(f"  商品元数据: {len(meta):,} 条")

# 评论数据
reviews = []
with open('AMAZON_FASHION.json') as f:
    for line in f:
        reviews.append(json.loads(line))
print(f"  原始评论:   {len(reviews):,} 条")

# 3-core 迭代过滤: 去掉评论数 < 3 的用户和商品, 反复直到稳定
K_CORE = 3
while True:
    user_cnt = Counter(d['reviewerID'] for d in reviews)
    item_cnt = Counter(d['asin'] for d in reviews)
    filtered = [d for d in reviews
                if user_cnt[d['reviewerID']] >= K_CORE
                and item_cnt[d['asin']] >= K_CORE]
    if len(filtered) == len(reviews):
        break
    reviews = filtered
print(f"  3-core 后:  {len(reviews):,} 条")


# ════════════════════════════════════════════════════════════
#  第二步: 特征工程
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  第二步: 特征工程 (6 大类共 21 个特征)")
print("=" * 65)

# ---- 预计算: 用户/商品的聚合统计 (用到了 overall, 有泄露风险) ----
user_ratings = defaultdict(list)   # uid  → [rating1, rating2, ...]
item_ratings = defaultdict(list)   # asin → [rating1, rating2, ...]
for d in reviews:
    user_ratings[d['reviewerID']].append(d['overall'])
    item_ratings[d['asin']].append(d['overall'])

# ---- 特征名称 & 分类 ----
FEATURE_NAMES = [
    # ① 评论元信息 (4 个)
    'verified',            # 是否认证购买
    'has_vote',            # 是否有人投"有用"票
    'vote_count',          # 有用票数
    'has_image',           # 评论是否带图片
    # ② 评论文本长度 (3 个)
    'review_length',       # 评论文本字符数
    'review_word_count',   # 评论文本词数
    'summary_length',      # 摘要字符数
    # ③ 时间特征 (2 个)
    'review_year',         # 评论年份
    'review_month',        # 评论月份
    # ④ 用户聚合统计 (3 个) ⚠ 含目标变量泄露
    'user_review_count',   # 该用户总评论数
    'user_avg_rating',     # 该用户历史平均评分
    'user_rating_std',     # 该用户评分标准差 (越大越挑剔/极端)
    # ⑤ 商品聚合统计 (3 个) ⚠ 含目标变量泄露
    'item_review_count',   # 该商品总评论数
    'item_avg_rating',     # 该商品历史平均评分
    'item_rating_std',     # 该商品评分标准差 (越大越有争议)
    # ⑥ 商品元数据 (6 个)
    'has_brand',           # 是否有品牌信息
    'has_price',           # 是否有价格
    'price',               # 价格 (美元, 无则填 0)
    'log_rank',            # 销售排名取 log (无则填 0)
    'title_length',        # 商品标题字符数
    'n_features',          # 商品 feature 条数
]

# ---- 特征分组 (用于消融实验) ----
FEATURE_GROUPS = [
    ('只有截距 (猜均分)',                        []),
    ('+ ④ 用户聚合统计 (count/avg/std)',        [9, 10, 11]),
    ('+ ⑤ 商品聚合统计 (count/avg/std)',        [12, 13, 14]),
    ('+ ① 评论元信息 (verified/vote/image)',    [0, 1, 2, 3]),
    ('+ ② 评论文本长度 (chars/words/summary)',  [4, 5, 6]),
    ('+ ③ 时间特征 (year/month)',               [7, 8]),
    ('+ ⑥ 商品元数据 (brand/price/rank/...)',   [15, 16, 17, 18, 19, 20]),
    ('全部 21 个特征',                           list(range(len(FEATURE_NAMES)))),
]

# ---- 逐条提取特征 ----
X_rows, y_rows = [], []

for d in reviews:
    m_info = meta.get(d['asin'], {})

    text    = d.get('reviewText', '') or ''
    summary = d.get('summary', '') or ''
    ts      = d.get('unixReviewTime', 0)
    dt      = datetime.datetime.fromtimestamp(ts) if ts else None

    # 投票数
    vote_str = d.get('vote', '0') or '0'
    try:
        vote = int(str(vote_str).replace(',', ''))
    except ValueError:
        vote = 0

    # 元数据字段
    price_val = parse_price(m_info.get('price'))
    rank_val  = parse_rank(m_info.get('rank'))

    # 聚合统计
    u_list = user_ratings[d['reviewerID']]
    i_list = item_ratings[d['asin']]

    row = [
        # ① 评论元信息
        1.0 if d.get('verified', False) else 0.0,
        1.0 if vote > 0 else 0.0,
        float(vote),
        1.0 if d.get('image') else 0.0,
        # ② 评论文本长度
        float(len(text)),
        float(len(text.split())),
        float(len(summary)),
        # ③ 时间特征
        float(dt.year if dt else 2015),
        float(dt.month if dt else 6),
        # ④ 用户聚合统计
        float(len(u_list)),
        float(np.mean(u_list)),
        float(np.std(u_list)) if len(u_list) > 1 else 0.0,
        # ⑤ 商品聚合统计
        float(len(i_list)),
        float(np.mean(i_list)),
        float(np.std(i_list)) if len(i_list) > 1 else 0.0,
        # ⑥ 商品元数据
        1.0 if m_info.get('brand') else 0.0,
        1.0 if price_val is not None else 0.0,
        float(price_val) if price_val is not None else 0.0,
        float(np.log1p(rank_val)) if rank_val is not None else 0.0,
        float(len(m_info.get('title', ''))),
        float(len(m_info.get('feature', []) or [])),
    ]

    X_rows.append(row)
    y_rows.append(d['overall'])

X = np.array(X_rows, dtype=np.float64)
y = np.array(y_rows, dtype=np.float64)

print(f"\n  特征矩阵: {X.shape[0]:,} 样本 × {X.shape[1]} 特征")
print(f"\n  {'特征名':25s}  {'均值':>8s}  {'标准差':>8s}  {'最小':>8s}  {'最大':>8s}")
print("  " + "-" * 62)
for i, name in enumerate(FEATURE_NAMES):
    col = X[:, i]
    print(f"  {name:25s}  {col.mean():8.3f}  {col.std():8.3f}  {col.min():8.1f}  {col.max():8.1f}")


# ════════════════════════════════════════════════════════════
#  第三步: 训练 / 测试划分 + Z-score 标准化
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  第三步: 训练/测试划分 + Z-score 标准化")
print("=" * 65)

np.random.seed(42)
n = len(y)
perm = np.random.permutation(n)
split = int(0.8 * n)

X_train, X_test = X[perm[:split]], X[perm[split:]]
y_train, y_test = y[perm[:split]], y[perm[split:]]

# 标准化 (训练集统计量)
mu    = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0

X_train_z = (X_train - mu) / sigma
X_test_z  = (X_test  - mu) / sigma

# 加截距列
X_train_z = np.column_stack([np.ones(len(X_train_z)), X_train_z])
X_test_z  = np.column_stack([np.ones(len(X_test_z)),  X_test_z])

print(f"  训练集: {len(y_train):,}   测试集: {len(y_test):,}")


# ════════════════════════════════════════════════════════════
#  第四步: 模型训练
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  第四步: 模型训练")
print("=" * 65)

# ---- (a) Baseline: 猜训练集均分 ----
bl_pred   = np.full_like(y_test, y_train.mean())
bl_rmse   = rmse(y_test, bl_pred)
bl_mae    = mae(y_test, bl_pred)
print(f"\n  [Baseline 猜均分]")
print(f"    RMSE = {bl_rmse:.4f}   MAE = {bl_mae:.4f}")

# ---- (b) OLS 线性回归 ----
try:
    w_ols       = np.linalg.solve(X_train_z.T @ X_train_z, X_train_z.T @ y_train)
    ols_pred    = X_test_z @ w_ols
    ols_rmse    = rmse(y_test, ols_pred)
    ols_mae     = mae(y_test, ols_pred)
    ols_train_r = rmse(y_train, X_train_z @ w_ols)
    print(f"\n  [OLS 线性回归]")
    print(f"    Train RMSE = {ols_train_r:.4f}")
    print(f"    Test  RMSE = {ols_rmse:.4f}   MAE = {ols_mae:.4f}")
except Exception as e:
    print(f"\n  OLS 求解失败: {e}")
    w_ols, ols_rmse, ols_mae = None, bl_rmse, bl_mae

# ---- (c) 岭回归 (自动选 alpha) ----
best_alpha, best_ridge_rmse, best_w = None, float('inf'), None
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    w = ridge_solve(X_train_z, y_train, alpha)
    r = rmse(y_test, X_test_z @ w)
    if r < best_ridge_rmse:
        best_alpha, best_ridge_rmse, best_w = alpha, r, w

ridge_pred    = X_test_z @ best_w
ridge_rmse    = rmse(y_test, ridge_pred)
ridge_mae     = mae(y_test, ridge_pred)
ridge_train_r = rmse(y_train, X_train_z @ best_w)
print(f"\n  [岭回归  alpha={best_alpha}]")
print(f"    Train RMSE = {ridge_train_r:.4f}")
print(f"    Test  RMSE = {ridge_rmse:.4f}   MAE = {ridge_mae:.4f}")


# ════════════════════════════════════════════════════════════
#  第五步: 特征重要性分析 (标准化回归系数)
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  第五步: 特征重要性 (标准化回归系数, 绝对值可直接比较)")
print("=" * 65)

intercept = best_w[0]
coeffs    = best_w[1:]                          # 去截距
rank_idx  = np.argsort(-np.abs(coeffs))         # 按 |系数| 降序

print(f"\n  截距 (≈ 全局均分) = {intercept:.4f}")
print(f"\n  {'排名':>4s}  {'特征':25s}  {'系数':>8s}  方向")
print("  " + "-" * 55)
for rank, i in enumerate(rank_idx, 1):
    c = coeffs[i]
    arrow = "↑ 评分更高" if c > 0 else "↓ 评分更低"
    print(f"  {rank:4d}  {FEATURE_NAMES[i]:25s}  {c:>+8.4f}  {arrow}")


# ════════════════════════════════════════════════════════════
#  第六步: 消融实验 — 逐步添加特征组, 看边际贡献
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  第六步: 消融实验 (逐步添加特征组)")
print("=" * 65)

print(f"\n  {'特征组':45s}  {'RMSE':>8s}  {'vs Baseline':>11s}")
print("  " + "-" * 68)

cumul = []              # 已累积的特征索引
ablation_rmses = []     # 每一步的 RMSE

for group_name, feat_idx in FEATURE_GROUPS:
    if group_name.startswith('全部'):
        use = list(range(len(FEATURE_NAMES)))
    else:
        cumul.extend(feat_idx)
        use = list(cumul)

    if len(use) == 0:
        r = bl_rmse
    else:
        cols_tr = np.column_stack([np.ones(len(X_train_z)), X_train_z[:, [i+1 for i in use]]])
        cols_te = np.column_stack([np.ones(len(X_test_z)),  X_test_z[:, [i+1 for i in use]]])
        w_sub   = ridge_solve(cols_tr, y_train, best_alpha)
        r = rmse(y_test, cols_te @ w_sub)

    ablation_rmses.append(r)
    improve = (bl_rmse - r) / bl_rmse * 100
    print(f"  {group_name:45s}  {r:8.4f}  {improve:>+9.1f}%")


# ════════════════════════════════════════════════════════════
#  第七步: 模型对比总结
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  第七步: 模型对比总结")
print("=" * 65)

MF_RMSE = 0.8928   # 之前矩阵分解的结果
MF_MAE  = 0.6370

print(f"""
  {'模型':35s}  {'RMSE':>8s}  {'MAE':>8s}
  {'-'*55}
  {'Baseline (猜均分)':35s}  {bl_rmse:8.4f}  {bl_mae:8.4f}
  {'OLS 线性回归 (21 个特征)':35s}  {ols_rmse:8.4f}  {ols_mae:8.4f}
  {'岭回归 (21 个特征)':35s}  {ridge_rmse:8.4f}  {ridge_mae:8.4f}
  {'矩阵分解 MF (只用 user+item ID)':35s}  {MF_RMSE:8.4f}  {MF_MAE:8.4f}
""")


# ════════════════════════════════════════════════════════════
#  第八步: 可视化
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Regression Analysis: Which Features Matter for Rating?',
             fontsize=14, fontweight='bold')

# ---- (a) 特征重要性 ----
ax = axes[0, 0]
names_sorted  = [FEATURE_NAMES[i] for i in rank_idx]
coeffs_sorted = [coeffs[i] for i in rank_idx]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coeffs_sorted]
ax.barh(range(len(names_sorted)), coeffs_sorted, color=colors, edgecolor='white')
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=9)
ax.set_xlabel('Standardized Coefficient')
ax.set_title('Feature Importance (Ridge)')
ax.axvline(0, color='black', lw=0.5)
ax.invert_yaxis()

# ---- (b) 消融实验 ----
ax = axes[0, 1]
short_labels = [
    'Baseline',
    '+ User\nagg',
    '+ Item\nagg',
    '+ Review\nmeta',
    '+ Text\nlen',
    '+ Time',
    '+ Item\nmeta',
    'All 21',
]
bar_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(ablation_rmses)))
ax.bar(range(len(ablation_rmses)), ablation_rmses, color=bar_colors, edgecolor='white')
ax.set_xticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, fontsize=8)
ax.set_ylabel('Test RMSE')
ax.set_title('Ablation: Adding Feature Groups')
ax.axhline(MF_RMSE, ls='--', color='red', lw=1.5, label=f'MF ({MF_RMSE:.4f})')
ax.legend(fontsize=9)
ax.set_ylim(min(ablation_rmses) * 0.95, bl_rmse * 1.02)

# ---- (c) 预测 vs 真实 ----
ax = axes[1, 0]
ax.scatter(y_test, ridge_pred, alpha=0.15, s=8, c='steelblue')
ax.plot([1, 5], [1, 5], 'r--', lw=1.5)
ax.set_xlabel('True Rating')
ax.set_ylabel('Predicted Rating')
ax.set_title(f'Ridge: Predicted vs True  (RMSE={ridge_rmse:.4f})')
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)

# ---- (d) 模型对比 ----
ax = axes[1, 1]
model_names = ['Baseline', 'OLS', 'Ridge', 'MF']
model_rmses = [bl_rmse, ols_rmse, ridge_rmse, MF_RMSE]
bar_colors2 = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
bars = ax.bar(model_names, model_rmses, color=bar_colors2, edgecolor='white', width=0.55)
for bar, r in zip(bars, model_rmses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{r:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Test RMSE')
ax.set_title('Model Comparison')
ax.set_ylim(0, bl_rmse * 1.15)

plt.tight_layout()
plt.savefig('feature_regression_results.png', dpi=150, bbox_inches='tight')
print("图片已保存: feature_regression_results.png")
print("完成!")
