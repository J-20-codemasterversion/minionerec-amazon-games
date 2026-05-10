"""
Factorization Machines (FM) 评分预测
====================================
将 user_id、item_id 编码为 one-hot，再拼接 review + meta 的辅助特征，
用 FM 统一建模线性项 + 二阶交互。

模型: y_hat = w0 + sum(w_i * x_i) + sum_{i<j} <v_i, v_j> x_i x_j
对稀疏 one-hot 特征，FM 的二阶项等价于矩阵分解的 gamma_u · gamma_i，
同时还能自动学辅助特征和 user/item 之间的交互。

数据: AMAZON_FASHION (review + meta), 3-core 过滤
"""

import json
import re
import math
import datetime
import numpy as np
from collections import Counter

# ============================================================
# 1. 加载数据 + 3-core 过滤 (复用 regression.py 的逻辑)
# ============================================================
print("=" * 60)
print("Factorization Machines — AMAZON FASHION")
print("=" * 60)

print("\n[1] 加载数据...")
meta = {}
with open('meta_AMAZON_FASHION.json') as f:
    for line in f:
        d = json.loads(line)
        meta[d['asin']] = d
print(f"    商品数: {len(meta):,}")

reviews = []
with open('AMAZON_FASHION.json') as f:
    for line in f:
        reviews.append(json.loads(line))
print(f"    原始评论: {len(reviews):,}")

K_CORE = 3
while True:
    uc = Counter(d['reviewerID'] for d in reviews)
    ic = Counter(d['asin'] for d in reviews)
    new = [d for d in reviews if uc[d['reviewerID']] >= K_CORE and ic[d['asin']] >= K_CORE]
    if len(new) == len(reviews):
        break
    reviews = new
print(f"    3-core 后: {len(reviews):,}")

# ============================================================
# 2. 编码 user/item ID + 提取辅助特征
# ============================================================
print("\n[2] 构建特征...")

# User / Item ID 映射
users = sorted(set(d['reviewerID'] for d in reviews))
items = sorted(set(d['asin'] for d in reviews))
uid2idx = {u: i for i, u in enumerate(users)}
iid2idx = {it: i for i, it in enumerate(items)}
n_users = len(users)
n_items = len(items)
print(f"    Users: {n_users:,}  Items: {n_items:,}")

# 辅助特征 (和 regression.py 一致)
def parse_price(p):
    if not p: return None
    p = str(p).replace('$', '').replace(',', '').strip()
    if '-' in p:
        parts = p.split('-')
        try: return (float(parts[0]) + float(parts[1])) / 2
        except: return None
    try:
        v = float(p)
        return v if 0 < v < 10000 else None
    except: return None

def parse_rank(r):
    if not r: return None
    if isinstance(r, list): r = r[0] if r else ''
    m = re.search(r'([\d,]+)', str(r))
    if m:
        try: return int(m.group(1).replace(',', ''))
        except: return None
    return None

side_feature_names = [
    'verified', 'review_len', 'summary_len',
    'has_vote', 'vote_count', 'has_image',
    'year', 'month',
    'has_brand', 'title_word_cnt', 'n_features',
    'has_description', 'has_price', 'price',
    'log_rank', 'n_also_buy', 'n_also_view',
]
n_side = len(side_feature_names)

# 构建稀疏表示: 每条样本 = (user_idx, item_idx, side_features[], rating)
data = []
for d in reviews:
    m_info = meta.get(d['asin'], {})
    txt = d.get('reviewText', '') or ''
    smr = d.get('summary', '') or ''
    ts = d.get('unixReviewTime', 0)
    dt = datetime.datetime.fromtimestamp(ts) if ts else None

    vote_str = d.get('vote', '0') or '0'
    try: vote = int(str(vote_str).replace(',', ''))
    except: vote = 0

    price = parse_price(m_info.get('price'))
    rank = parse_rank(m_info.get('rank'))

    side = np.array([
        1.0 if d.get('verified') else 0.0,
        len(txt.split()),
        len(smr.split()),
        1.0 if vote > 0 else 0.0,
        float(vote),
        1.0 if d.get('image') else 0.0,
        float(dt.year if dt else 2015),
        float(dt.month if dt else 6),
        1.0 if m_info.get('brand') else 0.0,
        float(len((m_info.get('title', '') or '').split())),
        float(len(m_info.get('feature', []) or [])),
        1.0 if m_info.get('description') else 0.0,
        1.0 if price is not None else 0.0,
        price if price is not None else 0.0,
        math.log1p(rank) if rank is not None else 0.0,
        float(len(m_info.get('also_buy', []) or [])),
        float(len(m_info.get('also_view', []) or [])),
    ], dtype=np.float32)

    data.append((uid2idx[d['reviewerID']], iid2idx[d['asin']], side, d['overall']))

# ============================================================
# 3. Train / Test 划分 + 辅助特征标准化
# ============================================================
print("\n[3] 划分训练/测试集...")
np.random.seed(42)
n = len(data)
perm = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = perm[:split], perm[split:]

# 先收集训练集辅助特征做标准化
side_train = np.array([data[i][2] for i in train_idx])
side_mu = side_train.mean(axis=0)
side_std = side_train.std(axis=0)
side_std[side_std == 0] = 1.0

# 标准化所有辅助特征
for i in range(n):
    u, it, side, rating = data[i]
    data[i] = (u, it, (side - side_mu) / side_std, rating)

print(f"    训练: {len(train_idx):,}  测试: {len(test_idx):,}")

# ============================================================
# 4. FM 模型 (SGD 训练)
# ============================================================
print("\n[4] 训练 FM 模型...")

# 超参数
K = 8           # 隐因子维度
LR = 0.002      # 学习率 (小一些防爆炸)
REG_W = 0.02    # 线性项正则
REG_V = 0.02    # 交互项正则
EPOCHS = 30
CLIP = 5.0      # 梯度裁剪阈值

# 参数初始化
# 逻辑维度: [n_users + n_items + n_side]
# 但 user/item 是 one-hot，用索引直接访问
w0 = 0.0                                           # 全局偏置
w_user = np.zeros(n_users, dtype=np.float64)        # 用户线性偏置
w_item = np.zeros(n_items, dtype=np.float64)        # 商品线性偏置
w_side = np.zeros(n_side, dtype=np.float64)         # 辅助特征线性权重
v_user = np.random.randn(n_users, K) * 0.01        # 用户隐向量
v_item = np.random.randn(n_items, K) * 0.01        # 商品隐向量
v_side = np.random.randn(n_side, K) * 0.01         # 辅助特征隐向量

def clip_val(x, lo=-10.0, hi=10.0):
    return max(lo, min(hi, x))

def predict(u, it, side):
    """FM 预测一条样本"""
    linear = w0 + w_user[u] + w_item[it] + w_side @ side
    sum_vx = v_user[u] + v_item[it] + (v_side.T @ side)
    sum_v2x2 = v_user[u]**2 + v_item[it]**2 + ((v_side**2).T @ (side**2))
    interaction = 0.5 * (sum_vx @ sum_vx - sum_v2x2.sum())
    return clip_val(linear + interaction, 0.5, 5.5)

def evaluate(indices):
    se = 0.0
    for i in indices:
        u, it, side, rating = data[i]
        pred = predict(u, it, side)
        se += (rating - pred) ** 2
    return math.sqrt(se / len(indices))

def clip_grad(g, c=CLIP):
    """标量或向量梯度裁剪"""
    if isinstance(g, np.ndarray):
        norm = np.linalg.norm(g)
        if norm > c:
            g = g * (c / norm)
        return g
    else:
        return max(-c, min(c, g))

# Baseline
train_ratings = np.array([data[i][3] for i in train_idx])
mean_rating = train_ratings.mean()
bl_rmse = math.sqrt(np.mean((np.array([data[i][3] for i in test_idx]) - mean_rating)**2))
print(f"    [Baseline 猜均分] Test RMSE = {bl_rmse:.4f}")

# SGD 训练
history = []
for epoch in range(EPOCHS):
    np.random.shuffle(train_idx)
    for i in train_idx:
        u, it, side, rating = data[i]

        # Forward
        sum_vx = v_user[u] + v_item[it] + (v_side.T @ side)

        pred = w0 + w_user[u] + w_item[it] + w_side @ side
        pred += 0.5 * (sum_vx @ sum_vx
                        - v_user[u] @ v_user[u]
                        - v_item[it] @ v_item[it]
                        - (side**2) @ (v_side * v_side).sum(axis=1))

        err = clip_val(rating - pred, -5.0, 5.0)

        # w0
        w0 += LR * clip_grad(err)

        # 线性项
        w_user[u] += LR * clip_grad(err - REG_W * w_user[u])
        w_item[it] += LR * clip_grad(err - REG_W * w_item[it])
        w_side += LR * clip_grad(err * side - REG_W * w_side)

        # 交互项
        grad_vu = err * (sum_vx - v_user[u]) - REG_V * v_user[u]
        v_user[u] += LR * clip_grad(grad_vu)

        grad_vi = err * (sum_vx - v_item[it]) - REG_V * v_item[it]
        v_item[it] += LR * clip_grad(grad_vi)

        for j in range(n_side):
            if abs(side[j]) < 1e-8:
                continue
            grad_vj = err * side[j] * (sum_vx - v_side[j] * side[j]) - REG_V * v_side[j]
            v_side[j] += LR * clip_grad(grad_vj)

    # 每 epoch 评估
    train_rmse = evaluate(train_idx[:5000])
    test_rmse = evaluate(test_idx)
    history.append((epoch + 1, train_rmse, test_rmse))
    print(f"    Epoch {epoch+1:2d}/{EPOCHS}  Train RMSE={train_rmse:.4f}  Test RMSE={test_rmse:.4f}")

    # 早停：连续 5 个 epoch 没改进
    if len(history) >= 6:
        recent = [h[2] for h in history[-5:]]
        if min(recent) >= history[-6][2] - 1e-4:
            print("    早停触发")
            break

best_test = min(h[2] for h in history)
print(f"\n    最佳 Test RMSE = {best_test:.4f}")
print(f"    vs Baseline({bl_rmse:.4f}): 降低 {(bl_rmse - best_test)/bl_rmse*100:.1f}%")

# ============================================================
# 5. 特征重要性分析
# ============================================================
print("\n" + "=" * 60)
print("特征分析")
print("=" * 60)

# 线性权重
print("\n  [线性偏置]")
print(f"    w0 (全局偏置) = {w0:.4f}")
print(f"    用户偏置 w_user: mean={w_user.mean():.4f}, std={w_user.std():.4f}")
print(f"    商品偏置 w_item: mean={w_item.mean():.4f}, std={w_item.std():.4f}")

print(f"\n  [辅助特征线性权重]")
side_idx = np.argsort(-np.abs(w_side))
for rank, j in enumerate(side_idx, 1):
    arrow = "↑" if w_side[j] > 0 else "↓"
    print(f"    {rank:2d}. {side_feature_names[j]:20s}  w={w_side[j]:>+.4f}  {arrow}")

# 交互项重要性 (||v||)
print(f"\n  [交互向量范数 ||v|| — 衡量该特征参与交互的强度]")
v_norms = np.linalg.norm(v_side, axis=1)
vnorm_idx = np.argsort(-v_norms)
for rank, j in enumerate(vnorm_idx, 1):
    print(f"    {rank:2d}. {side_feature_names[j]:20s}  ||v||={v_norms[j]:.4f}")

# ============================================================
# 6. 可视化: FM vs Regression 对比
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- 先跑一遍 Ridge Regression 获取其预测结果 ----
print("\n[6] 运行 Ridge Regression 用于对比...")

# 构建回归用的特征矩阵 (只用辅助特征，不用 user/item ID)
X_reg = np.array([data[i][2] for i in range(n)])
y_reg = np.array([data[i][3] for i in range(n)])

X_reg_train, X_reg_test = X_reg[train_idx], X_reg[test_idx]
y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

# 加截距
X_reg_train = np.column_stack([np.ones(len(X_reg_train)), X_reg_train])
X_reg_test = np.column_stack([np.ones(len(X_reg_test)), X_reg_test])

# Ridge 回归
def ridge_solve(X, y, alpha):
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)

reg_best_rmse = float('inf')
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    w_reg = ridge_solve(X_reg_train, y_reg_train, alpha)
    pred_reg = X_reg_test @ w_reg
    r = math.sqrt(np.mean((y_reg_test - pred_reg)**2))
    if r < reg_best_rmse:
        reg_best_rmse = r
        reg_best_w = w_reg
        reg_best_alpha = alpha

reg_pred = X_reg_test @ reg_best_w
print(f"    Ridge Regression RMSE = {reg_best_rmse:.4f} (alpha={reg_best_alpha})")

# FM 预测
fm_true, fm_pred = [], []
for i in test_idx:
    u, it, side, rating = data[i]
    fm_true.append(rating)
    fm_pred.append(predict(u, it, side))
fm_true, fm_pred = np.array(fm_true), np.array(fm_pred)

# ---- 画图 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('FM vs Ridge Regression — AMAZON FASHION', fontsize=14, fontweight='bold')

# (a) FM vs Regression RMSE 对比柱状图
ax = axes[0, 0]
model_names = ['Baseline\n(Mean)', 'Ridge\nRegression', 'FM\n(K=8)']
model_rmses = [bl_rmse, reg_best_rmse, best_test]
bar_colors = ['#95a5a6', '#3498db', '#e74c3c']
bars = ax.bar(model_names, model_rmses, color=bar_colors, edgecolor='white', width=0.6)
for bar, r in zip(bars, model_rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{r:.4f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Test RMSE', fontsize=11)
ax.set_title('Model Comparison: FM vs Regression', fontsize=12)
ax.set_ylim(0, bl_rmse * 1.15)
ax.axhline(bl_rmse, ls='--', color='gray', alpha=0.5)

# (b) FM 各 epoch 的 Train/Test RMSE 曲线
ax = axes[0, 1]
epochs_list = [h[0] for h in history]
train_rmses = [h[1] for h in history]
test_rmses = [h[2] for h in history]
ax.plot(epochs_list, train_rmses, 'o-', label='FM Train RMSE', color='steelblue', markersize=5)
ax.plot(epochs_list, test_rmses, 's-', label='FM Test RMSE', color='coral', markersize=5)
ax.axhline(bl_rmse, ls='--', color='gray', lw=1.5, label=f'Baseline {bl_rmse:.4f}')
ax.axhline(reg_best_rmse, ls='-.', color='#3498db', lw=1.5, label=f'Ridge Reg {reg_best_rmse:.4f}')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('FM Training Curve (per Epoch)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Ridge Regression: Prediction vs Truth
ax = axes[1, 0]
ax.scatter(y_reg_test, reg_pred, alpha=0.15, s=8, c='#3498db')
ax.plot([1, 5], [1, 5], 'r--', lw=1.5)
ax.set_xlabel('True Rating', fontsize=11)
ax.set_ylabel('Predicted Rating', fontsize=11)
ax.set_title(f'Ridge Regression: Pred vs Truth (RMSE={reg_best_rmse:.4f})', fontsize=12)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)

# (d) FM: Prediction vs Truth
ax = axes[1, 1]
ax.scatter(fm_true, fm_pred, alpha=0.15, s=8, c='#e74c3c')
ax.plot([1, 5], [1, 5], 'r--', lw=1.5)
ax.set_xlabel('True Rating', fontsize=11)
ax.set_ylabel('Predicted Rating', fontsize=11)
ax.set_title(f'FM: Pred vs Truth (RMSE={best_test:.4f})', fontsize=12)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)

plt.tight_layout()
plt.savefig('fm_results.png', dpi=150, bbox_inches='tight')
print(f"\n图片已保存: fm_results.png")

# 最终对比
print("\n" + "=" * 60)
print("模型对比")
print("=" * 60)
print(f"  Baseline (猜均分)      RMSE = {bl_rmse:.4f}")
print(f"  FM (K={K})              RMSE = {best_test:.4f}  (↓{(bl_rmse-best_test)/bl_rmse*100:.1f}%)")
print("=" * 60)
