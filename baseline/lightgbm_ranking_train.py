"""
LightGBM 排序模型训练脚本
=========================
目标: 给定一个用户，随机抽 5 个购买过的商品 + 全局随机 5 个商品，
      要求购买过的商品排序分值 > 未购买的商品排序分值。

特征: 用户特征 20 维 + 物品特征 20 维 = 40 维
模型: LightGBM lambdarank

运行环境: TiOne Notebook / CVM
数据来源: COS 上的 Video_Games.json + meta_Video_Games.json
"""

import os
import json
import re
import math
import time
import numpy as np
import pickle
from collections import Counter, defaultdict

# ============================================================
# 配置
# ============================================================
# 自动检测环境
if os.path.exists("/root/amazon_data"):
    DATA_DIR = "/root/amazon_data"
elif os.path.exists("/home/tione/notebook/data"):
    DATA_DIR = "/home/tione/notebook/data"
else:
    DATA_DIR = "/Users/jasonlihahaha/Desktop/amazon_data/数据"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else "."

REVIEW_FILE = os.path.join(DATA_DIR, "Video_Games.json")
META_FILE = os.path.join(DATA_DIR, "meta_Video_Games.json")

TARGET_USERS = 5000
MAX_REVIEWS_PER_USER = 80
MIN_REVIEWS_PER_USER = 10

print(f"数据目录: {DATA_DIR}")
print(f"评论文件: {REVIEW_FILE} (exists={os.path.exists(REVIEW_FILE)})")
print(f"Meta文件: {META_FILE} (exists={os.path.exists(META_FILE)})")

# ============================================================
# Step 1: 数据加载
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

try:
    import orjson
    _loads = orjson.loads
    print("  使用 orjson 解析（快速）")
except ImportError:
    _loads = json.loads
    print("  使用 json.loads 解析（标准）")

t0 = time.time()
user_reviews = defaultdict(list)
count = 0

with open(REVIEW_FILE, 'rb', buffering=8*1024*1024) as f:
    for raw_line in f:
        try:
            d = _loads(raw_line)
            uid = d.get('reviewerID')
            asin = d.get('asin')
            if uid and asin and 'overall' in d:
                ts = d.get('unixReviewTime', 0)
                rating = d['overall']
                user_reviews[uid].append((asin, rating, ts))
                count += 1
        except Exception:
            continue
        if count % 500000 == 0 and count > 0:
            print(f"  已加载 {count/1e6:.1f}M 条, {len(user_reviews):,} 用户")

print(f"  总计: {count:,} 条评论, {len(user_reviews):,} 用户, 耗时 {time.time()-t0:.1f}s")

# 选取活跃用户
user_counts = [(uid, len(revs)) for uid, revs in user_reviews.items()]
user_counts.sort(key=lambda x: -x[1])
qualified = [(uid, cnt) for uid, cnt in user_counts if cnt >= MIN_REVIEWS_PER_USER]
selected_uids = set(uid for uid, cnt in qualified[:TARGET_USERS])

print(f"  交互 >= {MIN_REVIEWS_PER_USER} 的用户: {len(qualified):,}")
print(f"  选取 top {len(selected_uids):,} 用户")

# 构建评论列表
reviews = []
for uid in selected_uids:
    revs = user_reviews[uid]
    revs_sorted = sorted(revs, key=lambda x: x[2])[:MAX_REVIEWS_PER_USER]
    for asin, rating, ts in revs_sorted:
        reviews.append({'reviewerID': uid, 'asin': asin, 'overall': rating, 'unixReviewTime': ts})

del user_reviews, user_counts, qualified

item_set = set(d['asin'] for d in reviews)
print(f"  最终: {len(reviews):,} 条评论, {len(selected_uids):,} 用户, {len(item_set):,} 商品")

# ============================================================
# Step 2: 加载 Meta
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 加载商品元数据")
print("=" * 60)

t0 = time.time()
item_bytes_set = {a.encode('utf-8') for a in item_set}
meta = {}

with open(META_FILE, 'rb', buffering=8*1024*1024) as f:
    for raw_line in f:
        if b'"asin"' not in raw_line:
            continue
        idx = raw_line.find(b'"asin"')
        if idx < 0: continue
        start = raw_line.find(b'"', idx + 6)
        if start < 0: continue
        start += 1
        end = raw_line.find(b'"', start)
        if end < 0: continue
        asin_bytes = raw_line[start:end]
        if asin_bytes not in item_bytes_set:
            continue
        try:
            d = _loads(raw_line)
            asin = d.get('asin')
            if asin and asin in item_set:
                meta[asin] = d
        except Exception:
            continue

print(f"  加载 {len(meta):,} 个商品元数据 (覆盖率: {len(meta)/max(len(item_set),1)*100:.1f}%)")
print(f"  耗时: {time.time()-t0:.1f}s")

# 填充缺失 meta
for asin in item_set:
    if asin not in meta:
        meta[asin] = {'asin': asin, 'title': '', 'brand': '', 'category': 'unknown'}

# ============================================================
# Step 3: 构建索引 + 统计特征
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 构建索引和统计特征")
print("=" * 60)

users = sorted(set(d['reviewerID'] for d in reviews))
items_list = sorted(item_set)
uid2idx = {u: i for i, u in enumerate(users)}
iid2idx = {it: i for i, it in enumerate(items_list)}
idx2uid = {i: u for u, i in uid2idx.items()}
idx2iid = {i: it for it, i in iid2idx.items()}
n_users = len(users)
n_items = len(items_list)

user2items = defaultdict(set)
item2users = defaultdict(set)
user2ratings = defaultdict(dict)
item_ratings = defaultdict(list)
user_ratings_list = defaultdict(list)
user_items_ts = defaultdict(list)  # uid -> [(iid, rating, ts)]

for d in reviews:
    u, i = uid2idx[d['reviewerID']], iid2idx[d['asin']]
    r = d['overall']
    ts = d.get('unixReviewTime', 0)
    user2items[u].add(i)
    item2users[i].add(u)
    user2ratings[u][i] = r
    item_ratings[i].append(r)
    user_ratings_list[u].append(r)
    user_items_ts[u].append((i, r, ts))

# 统计特征
user_avg_rating = {u: np.mean(rs) for u, rs in user_ratings_list.items()}
user_rating_cnt = {u: len(rs) for u, rs in user_ratings_list.items()}
user_rating_std = {u: np.std(rs) if len(rs) > 1 else 0 for u, rs in user_ratings_list.items()}
user_max_rating = {u: max(rs) for u, rs in user_ratings_list.items()}
user_min_rating = {u: min(rs) for u, rs in user_ratings_list.items()}

item_avg_rating = {i: np.mean(rs) for i, rs in item_ratings.items()}
item_rating_cnt = {i: len(rs) for i, rs in item_ratings.items()}
item_rating_std = {i: np.std(rs) if len(rs) > 1 else 0 for i, rs in item_ratings.items()}
item_max_rating = {i: max(rs) for i, rs in item_ratings.items()}
item_min_rating = {i: min(rs) for i, rs in item_ratings.items()}
global_avg = np.mean([d['overall'] for d in reviews])

# 商品热门度排名
_item_pop_sorted = sorted(item_rating_cnt.items(), key=lambda x: -x[1])
_item_pop_rank = {iid: rank for rank, (iid, _) in enumerate(_item_pop_sorted)}

# 品类/品牌统计
def extract_category(meta_dict):
    cats = meta_dict.get('category', [])
    if isinstance(cats, list) and cats:
        return cats[-1]
    if isinstance(cats, str) and cats:
        return cats
    return 'unknown'

def parse_price(p):
    if not p: return 0.0
    p = str(p).replace('$', '').replace(',', '').strip()
    if '-' in p:
        parts = p.split('-')
        try: return (float(parts[0]) + float(parts[1])) / 2
        except: return 0.0
    try:
        v = float(p)
        return v if 0 < v < 10000 else 0.0
    except: return 0.0

def parse_rank(r):
    if not r: return 0
    if isinstance(r, list): r = r[0] if r else ''
    m = re.search(r'([\d,]+)', str(r))
    if m:
        try: return int(m.group(1).replace(',', ''))
        except: return 0
    return 0

# 品类索引
item_category = {}
category_items = defaultdict(set)
item_brand = {}
brand_items = defaultdict(set)

for asin in item_set:
    m = meta.get(asin, {})
    cat = extract_category(m)
    brand = (m.get('brand', '') or '').strip().lower()
    item_category[asin] = cat
    category_items[cat].add(asin)
    item_brand[asin] = brand
    if brand and brand != 'unknown':
        brand_items[brand].add(asin)

# 用户品类/品牌偏好
user_cat_dist = {}  # uid_idx -> Counter
user_brand_dist = {}
for u in range(n_users):
    cat_cnt = Counter()
    brand_cnt = Counter()
    for i in user2items[u]:
        asin = idx2iid[i]
        cat_cnt[item_category.get(asin, 'unknown')] += 1
        br = item_brand.get(asin, '')
        if br: brand_cnt[br] += 1
    user_cat_dist[u] = cat_cnt
    user_brand_dist[u] = brand_cnt

print(f"  Users: {n_users:,}  Items: {n_items:,}")
print(f"  品类数: {len(category_items):,}  品牌数: {len(brand_items):,}")

# ============================================================
# Step 4: 特征工程 — 20 维用户特征 + 20 维物品特征
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 特征工程 (20 用户特征 + 20 物品特征)")
print("=" * 60)

def extract_user_features(uid_idx):
    """20 维用户特征"""
    n_items_u = len(user2items.get(uid_idx, set()))
    avg_r = user_avg_rating.get(uid_idx, global_avg)
    cnt_r = user_rating_cnt.get(uid_idx, 0)
    std_r = user_rating_std.get(uid_idx, 0)
    max_r = user_max_rating.get(uid_idx, global_avg)
    min_r = user_min_rating.get(uid_idx, global_avg)
    range_r = max_r - min_r

    # 品类多样性
    cat_dist = user_cat_dist.get(uid_idx, Counter())
    n_cats = len(cat_dist)
    top_cat_ratio = max(cat_dist.values()) / max(sum(cat_dist.values()), 1) if cat_dist else 0

    # 品牌多样性
    brand_dist = user_brand_dist.get(uid_idx, Counter())
    n_brands = len(brand_dist)
    top_brand_ratio = max(brand_dist.values()) / max(sum(brand_dist.values()), 1) if brand_dist else 0

    # 评分偏好
    high_ratio = sum(1 for r in user_ratings_list.get(uid_idx, []) if r >= 4) / max(cnt_r, 1)
    low_ratio = sum(1 for r in user_ratings_list.get(uid_idx, []) if r <= 2) / max(cnt_r, 1)

    # 时间跨度
    ts_list = [ts for _, _, ts in user_items_ts.get(uid_idx, []) if ts > 0]
    ts_span = (max(ts_list) - min(ts_list)) / 86400.0 if len(ts_list) > 1 else 0  # 天
    avg_interval = ts_span / max(len(ts_list) - 1, 1)

    # 邻居密度
    neighbor_users = set()
    for j in user2items.get(uid_idx, set()):
        neighbor_users |= item2users.get(j, set())
    n_neighbors = len(neighbor_users)

    return np.array([
        avg_r,                              # 1. 平均评分
        float(min(cnt_r, 200)),             # 2. 评分数量
        std_r,                              # 3. 评分标准差
        max_r,                              # 4. 最高评分
        min_r,                              # 5. 最低评分
        range_r,                            # 6. 评分范围
        high_ratio,                         # 7. 高分(>=4)比例
        low_ratio,                          # 8. 低分(<=2)比例
        float(n_items_u),                   # 9. 交互商品数
        float(n_cats),                      # 10. 品类数
        top_cat_ratio,                      # 11. 最爱品类占比
        float(n_brands),                    # 12. 品牌数
        top_brand_ratio,                    # 13. 最爱品牌占比
        math.log1p(ts_span),               # 14. 活跃时间跨度(log天)
        math.log1p(avg_interval),           # 15. 平均购买间隔(log天)
        math.log1p(n_neighbors),            # 16. 邻居用户数(log)
        float(n_items_u) / max(n_cats, 1),  # 17. 每品类平均购买数
        avg_r - global_avg,                 # 18. 评分偏差(vs全局)
        float(cnt_r >= 20),                 # 19. 是否活跃用户
        float(cnt_r >= 50),                 # 20. 是否重度用户
    ], dtype=np.float32)


def extract_item_features(iid_idx):
    """20 维物品特征"""
    asin = idx2iid.get(iid_idx, '')
    m = meta.get(asin, {})
    price = parse_price(m.get('price'))
    rank = parse_rank(m.get('rank'))
    n_also_buy = len(m.get('also_buy', []) or [])
    n_also_view = len(m.get('also_view', []) or [])
    avg_r = item_avg_rating.get(iid_idx, global_avg)
    cnt_r = item_rating_cnt.get(iid_idx, 0)
    std_r = item_rating_std.get(iid_idx, 0)
    max_r = item_max_rating.get(iid_idx, global_avg)
    min_r = item_min_rating.get(iid_idx, global_avg)
    pop_rank = _item_pop_rank.get(iid_idx, len(_item_pop_rank))

    title = m.get('title', '') or ''
    title_len = len(title.split())
    has_brand = float(bool((m.get('brand', '') or '').strip()))
    has_price = float(price > 0)
    has_desc = float(bool(m.get('description')))

    cat = item_category.get(asin, 'unknown')
    cat_size = len(category_items.get(cat, set()))

    # 评分分布
    ratings = item_ratings.get(iid_idx, [])
    high_ratio = sum(1 for r in ratings if r >= 4) / max(len(ratings), 1)

    # 购买者多样性
    buyers = item2users.get(iid_idx, set())
    buyer_avg_activity = np.mean([len(user2items.get(bu, set())) for bu in buyers]) if buyers else 0

    return np.array([
        min(price, 500),                    # 1. 价格
        math.log1p(rank),                   # 2. 排名(log)
        float(min(n_also_buy, 200)),        # 3. also_buy数量
        float(min(n_also_view, 200)),       # 4. also_view数量
        avg_r,                              # 5. 平均评分
        float(min(cnt_r, 200)),             # 6. 评价数量
        std_r,                              # 7. 评分标准差
        max_r,                              # 8. 最高评分
        min_r,                              # 9. 最低评分
        math.log1p(pop_rank),               # 10. 热门度排名(log)
        float(title_len),                   # 11. 标题词数
        has_brand,                          # 12. 是否有品牌
        has_price,                          # 13. 是否有价格
        has_desc,                           # 14. 是否有描述
        math.log1p(cat_size),               # 15. 所属品类大小(log)
        high_ratio,                         # 16. 好评(>=4)比例
        avg_r - global_avg,                 # 17. 评分偏差(vs全局)
        math.log1p(buyer_avg_activity),     # 18. 购买者平均活跃度(log)
        float(cnt_r >= 5),                  # 19. 是否热门(>=5评价)
        float(cnt_r >= 20),                 # 20. 是否爆款(>=20评价)
    ], dtype=np.float32)


USER_FEAT_NAMES = [
    'u_avg_rating', 'u_rating_cnt', 'u_rating_std', 'u_max_rating', 'u_min_rating',
    'u_rating_range', 'u_high_ratio', 'u_low_ratio', 'u_n_items', 'u_n_cats',
    'u_top_cat_ratio', 'u_n_brands', 'u_top_brand_ratio', 'u_ts_span',
    'u_avg_interval', 'u_n_neighbors', 'u_items_per_cat', 'u_rating_bias',
    'u_is_active', 'u_is_heavy',
]

ITEM_FEAT_NAMES = [
    'i_price', 'i_log_rank', 'i_n_also_buy', 'i_n_also_view', 'i_avg_rating',
    'i_rating_cnt', 'i_rating_std', 'i_max_rating', 'i_min_rating', 'i_pop_rank',
    'i_title_len', 'i_has_brand', 'i_has_price', 'i_has_desc', 'i_cat_size',
    'i_high_ratio', 'i_rating_bias', 'i_buyer_activity', 'i_is_popular', 'i_is_hot',
]

ALL_FEAT_NAMES = USER_FEAT_NAMES + ITEM_FEAT_NAMES
print(f"  用户特征: {len(USER_FEAT_NAMES)} 维")
print(f"  物品特征: {len(ITEM_FEAT_NAMES)} 维")
print(f"  总计: {len(ALL_FEAT_NAMES)} 维")

# ============================================================
# Step 5: 构建 LightGBM 排序训练数据
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 构建排序训练数据")
print("=" * 60)

# Leave-last-one-out 划分
user_reviews_grouped = defaultdict(list)
for d in reviews:
    user_reviews_grouped[d['reviewerID']].append(d)

train_reviews = []
test_reviews = []
for uid, rvs in user_reviews_grouped.items():
    rvs_sorted = sorted(rvs, key=lambda x: x.get('unixReviewTime', 0))
    if len(rvs_sorted) >= 3:
        train_reviews.extend(rvs_sorted[:-1])
        test_reviews.append(rvs_sorted[-1])
    else:
        train_reviews.extend(rvs_sorted)

train_user2items = defaultdict(set)
for d in train_reviews:
    u = uid2idx[d['reviewerID']]
    i = iid2idx[d['asin']]
    train_user2items[u].add(i)

print(f"  训练集: {len(train_reviews):,}, 测试集: {len(test_reviews):,}")

# 构建排序训练数据：每个用户一个 query group
# 每个 group: 5 个正样本(购买过的) + 5 个负样本(随机的)
N_POS = 5
N_NEG = 5
np.random.seed(42)

X_train = []
y_train = []       # relevance label: 正样本=1, 负样本=0
groups = []         # 每个 group 的大小

train_users_list = [u for u in range(n_users) if len(train_user2items.get(u, set())) >= N_POS]
np.random.shuffle(train_users_list)
train_users_list = train_users_list[:3000]  # 取 3000 个用户

t0 = time.time()
for idx, u in enumerate(train_users_list):
    if idx % 500 == 0:
        print(f"  构建训练数据: {idx}/{len(train_users_list)} ({time.time()-t0:.1f}s)")

    user_feat = extract_user_features(u)
    pos_items = list(train_user2items[u])

    # 随机抽 5 个正样本
    if len(pos_items) >= N_POS:
        selected_pos = np.random.choice(pos_items, size=N_POS, replace=False)
    else:
        selected_pos = pos_items

    # 随机抽 5 个负样本
    neg_items = []
    while len(neg_items) < N_NEG:
        neg_i = np.random.randint(0, n_items)
        if neg_i not in train_user2items.get(u, set()) and neg_i not in neg_items:
            neg_items.append(neg_i)

    group_size = len(selected_pos) + len(neg_items)
    groups.append(group_size)

    for i in selected_pos:
        item_feat = extract_item_features(i)
        X_train.append(np.concatenate([user_feat, item_feat]))
        y_train.append(1)

    for i in neg_items:
        item_feat = extract_item_features(i)
        X_train.append(np.concatenate([user_feat, item_feat]))
        y_train.append(0)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

print(f"\n  训练数据: {X_train.shape[0]:,} 样本, {X_train.shape[1]} 维特征")
print(f"  正样本: {(y_train==1).sum():,}, 负样本: {(y_train==0).sum():,}")
print(f"  Query groups: {len(groups):,}")
print(f"  构建耗时: {time.time()-t0:.1f}s")

# 划分验证集（按 group 划分，不打散）
n_val_groups = max(int(len(groups) * 0.2), 10)
n_train_groups = len(groups) - n_val_groups

train_end = sum(groups[:n_train_groups])
val_start = train_end

X_tr, X_va = X_train[:train_end], X_train[val_start:]
y_tr, y_va = y_train[:train_end], y_train[val_start:]
groups_tr = groups[:n_train_groups]
groups_va = groups[n_train_groups:]

print(f"  训练: {len(X_tr):,} 样本, {len(groups_tr):,} groups")
print(f"  验证: {len(X_va):,} 样本, {len(groups_va):,} groups")

# ============================================================
# Step 6: 训练 LightGBM 排序模型
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 训练 LightGBM 排序模型")
print("=" * 60)

import lightgbm as lgb

lgb_train = lgb.Dataset(X_tr, label=y_tr, group=groups_tr, feature_name=ALL_FEAT_NAMES, free_raw_data=False)
lgb_val = lgb.Dataset(X_va, label=y_va, group=groups_va, feature_name=ALL_FEAT_NAMES, free_raw_data=False, reference=lgb_train)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': [5, 10],
    'max_depth': 6,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'lambda_l2': 1.0,
    'lambda_l1': 0.1,
    'seed': 42,
    'verbose': -1,
    'num_threads': 4,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=20, verbose=True),
    lgb.log_evaluation(period=50),
]

t0 = time.time()
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=callbacks,
)

print(f"\n  训练完成! 最佳迭代: {model.best_iteration}, 耗时: {time.time()-t0:.1f}s")

# 特征重要性
importances = model.feature_importance(importance_type='gain')
sorted_imp = sorted(zip(ALL_FEAT_NAMES, importances), key=lambda x: -x[1])
print(f"\n  特征重要性 (Top 15):")
for fname, imp in sorted_imp[:15]:
    bar = '█' * int(imp / max(importances) * 30)
    print(f"    {fname:<25s} {imp:>10.1f}  {bar}")

# ============================================================
# Step 7: 验证 Demo — 指定用户，验证排序效果
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 验证 Demo")
print("=" * 60)

def demo_user_ranking(user_idx, n_pos=5, n_neg=5):
    """给定用户，抽正负样本，用模型打分，检查排序是否正确"""
    user_feat = extract_user_features(user_idx)
    pos_items = list(train_user2items.get(user_idx, set()))

    if len(pos_items) < n_pos:
        print(f"  用户 {idx2uid[user_idx]} 购买商品不足 {n_pos} 个，跳过")
        return None

    selected_pos = list(np.random.choice(pos_items, size=n_pos, replace=False))
    neg_items = []
    while len(neg_items) < n_neg:
        neg_i = np.random.randint(0, n_items)
        if neg_i not in train_user2items.get(user_idx, set()):
            neg_items.append(neg_i)

    all_items = selected_pos + neg_items
    labels = [1] * len(selected_pos) + [0] * len(neg_items)

    feats = []
    for i in all_items:
        item_feat = extract_item_features(i)
        feats.append(np.concatenate([user_feat, item_feat]))
    feats = np.array(feats, dtype=np.float32)

    scores = model.predict(feats)

    # 按分数排序
    ranked = sorted(zip(all_items, labels, scores), key=lambda x: -x[2])

    pos_scores = [s for _, l, s in zip(all_items, labels, scores) if l == 1]
    neg_scores = [s for _, l, s in zip(all_items, labels, scores) if l == 0]
    avg_pos = np.mean(pos_scores)
    avg_neg = np.mean(neg_scores)
    all_correct = all(ps > ns for ps in pos_scores for ns in neg_scores)

    return {
        'user_id': idx2uid[user_idx],
        'ranked': ranked,
        'avg_pos_score': avg_pos,
        'avg_neg_score': avg_neg,
        'all_correct': all_correct,
    }


# 测试 20 个用户
test_user_pool = [u for u in range(n_users) if len(train_user2items.get(u, set())) >= 10]
np.random.seed(123)
demo_users = np.random.choice(test_user_pool, size=min(20, len(test_user_pool)), replace=False)

n_correct = 0
n_total = 0

for u in demo_users:
    result = demo_user_ranking(u)
    if result is None:
        continue
    n_total += 1
    if result['all_correct']:
        n_correct += 1

    print(f"\n  用户: {result['user_id']}")
    print(f"  {'排名':<4} {'商品ASIN':<15} {'标签':>4} {'分数':>10}")
    print(f"  " + "-" * 40)
    for item_idx, label, score in result['ranked']:
        asin = idx2iid[item_idx]
        tag = "✓买过" if label == 1 else "✗随机"
        print(f"  {result['ranked'].index((item_idx, label, score))+1:<4} {asin:<15} {tag:>6} {score:>10.4f}")
    print(f"  正样本均分: {result['avg_pos_score']:.4f}, 负样本均分: {result['avg_neg_score']:.4f}")
    status = "✓ 全部正确" if result['all_correct'] else "⚠ 存在排序错误"
    print(f"  排序状态: {status}")

print(f"\n" + "=" * 60)
print(f"  验证结果: {n_correct}/{n_total} 用户的排序完全正确 ({n_correct/max(n_total,1)*100:.1f}%)")
print("=" * 60)

# ============================================================
# Step 8: 保存模型
# ============================================================
model_path = os.path.join(OUTPUT_DIR, "lightgbm_ranking_model.txt")
model.save_model(model_path)
print(f"\n模型已保存: {model_path}")

# 保存特征名和元数据
meta_info = {
    'user_feat_names': USER_FEAT_NAMES,
    'item_feat_names': ITEM_FEAT_NAMES,
    'all_feat_names': ALL_FEAT_NAMES,
    'n_users': n_users,
    'n_items': n_items,
    'global_avg': global_avg,
    'best_iteration': model.best_iteration,
}
meta_path = os.path.join(OUTPUT_DIR, "lightgbm_ranking_meta.pkl")
with open(meta_path, 'wb') as f:
    pickle.dump(meta_info, f)
print(f"元数据已保存: {meta_path}")

print("\n" + "=" * 60)
print("全部完成!")
print("=" * 60)
