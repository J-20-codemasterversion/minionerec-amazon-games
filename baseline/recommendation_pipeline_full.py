"""
完整推荐系统流水线 - 全量 Amazon 数据版本
==========================================
基于 All_Amazon_Review_5.json (82GB) 和 All_Amazon_Meta.json (105GB)
实现：召回 → 粗排(XGBoost) → 精排(FM) → 重排序

由于数据量巨大，采用采样策略
"""

import os
import json
import re
import math
import random
import numpy as np
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_DIR = "/Users/jasonlihahaha/Desktop/amazon_data/数据"
OUTPUT_DIR = "/Users/jasonlihahaha/Desktop/amazon_data/回归"

# 全量数据文件
REVIEW_FILE = os.path.join(DATA_DIR, "All_Amazon_Review_5.json")
META_FILE = os.path.join(DATA_DIR, "All_Amazon_Meta.json")

# 采样配置
MAX_REVIEWS = 2000000      # 最多加载 200 万条评论（增加数据量）
SAMPLE_RATE = 0.01         # 采样率（用于大文件快速扫描）
MIN_USER_REVIEWS = 3       # 用户最少评论数
MIN_ITEM_REVIEWS = 3       # 商品最少评论数

# ============================================================
# 数据加载函数
# ============================================================

def load_reviews_sampled(filepath: str, max_reviews: int = MAX_REVIEWS, 
                         sample_rate: float = SAMPLE_RATE, seed: int = 42):
    """
    从大文件中采样加载评论
    
    策略：
    1. 先快速扫描，随机采样
    2. 最多加载 max_reviews 条
    """
    print(f"正在加载评论数据 (最多 {max_reviews:,} 条)...")
    print(f"  文件: {filepath}")
    
    random.seed(seed)
    reviews = []
    line_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            
            # 进度显示
            if line_count % 1000000 == 0:
                print(f"  已扫描 {line_count/1000000:.0f}M 行, 已采集 {len(reviews):,} 条")
            
            # 采样策略：前 100 万条全取，之后按 sample_rate 采样
            if line_count <= 1000000 or random.random() < sample_rate:
                try:
                    d = json.loads(line.strip())
                    # 确保必要字段存在
                    if 'reviewerID' in d and 'asin' in d and 'overall' in d:
                        reviews.append(d)
                except:
                    continue
            
            # 达到上限
            if len(reviews) >= max_reviews:
                print(f"  达到上限 {max_reviews:,}，停止加载")
                break
    
    print(f"  总共加载: {len(reviews):,} 条评论")
    return reviews


def load_meta_for_items(filepath: str, item_set: set):
    """
    只加载已有商品的元数据
    
    由于 meta 文件很大(105GB)，且匹配商品集中在后半段（约第1370万行开始），
    需要跳过前面的无用数据。策略：
    1. 先快速跳过前 1300 万行（这些行几乎无匹配）
    2. 然后仔细扫描后续行寻找匹配
    3. 如果连续 200 万行无新匹配则停止
    """
    print(f"正在加载商品元数据...")
    print(f"  需要加载: {len(item_set):,} 个商品")
    
    meta = {}
    line_count = 0
    found_count = 0
    SKIP_LINES = 13000000       # 跳过前 1300 万行
    MAX_SCAN_AFTER = 5000000    # 匹配区之后最多再扫描 500 万行
    no_match_streak = 0         # 连续无匹配行数
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            
            # 快速跳过前面的无用行
            if line_count <= SKIP_LINES:
                if line_count % 2000000 == 0:
                    print(f"  快速跳过: {line_count/1000000:.0f}M 行...")
                continue
            
            if (line_count - SKIP_LINES) % 500000 == 0:
                print(f"  已扫描 {line_count/1000000:.1f}M 行, 已找到 {found_count:,} 个")
            
            try:
                d = json.loads(line.strip())
                asin = d.get('asin')
                if asin and asin in item_set:
                    meta[asin] = d
                    found_count += 1
                    no_match_streak = 0
                    
                    # 找完了就停止
                    if found_count >= len(item_set):
                        break
                else:
                    no_match_streak += 1
            except:
                no_match_streak += 1
                continue
            
            # 如果已经找到一些，且连续 200 万行无匹配，说明匹配区已过
            if found_count > 0 and no_match_streak > 2000000:
                print(f"  连续 200 万行无新匹配，停止扫描")
                break
            
            # 安全上限
            if line_count - SKIP_LINES > MAX_SCAN_AFTER:
                print(f"  达到扫描上限，停止")
                break
    
    print(f"  总共加载: {len(meta):,} 个商品元数据 (覆盖率: {len(meta)/max(len(item_set),1)*100:.1f}%)")
    return meta



# ============================================================
# 主流水线函数
# ============================================================

def run_pipeline(verbose: bool = True):
    """
    运行完整推荐系统流水线
    """
    if verbose:
        print("\n" + "=" * 70)
        print("全量 Amazon 数据 - 推荐系统流水线")
        print("召回 → 粗排(XGBoost) → 精排(FM) → 重排序")
        print("=" * 70)
    
    # ========================================
    # 1. 数据加载 + 预处理
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("[1] 数据加载")
        print("=" * 70)
    
    # 加载评论
    reviews = load_reviews_sampled(REVIEW_FILE, MAX_REVIEWS, SAMPLE_RATE)
    
    # ---- 核心：过滤冷启动用户和商品（多轮迭代） ----
    # 原始数据太稀疏（150万用户×50万商品），必须过滤
    if verbose:
        print(f"\n  过滤前: {len(reviews):,} 条评论")
    
    for round_i in range(3):  # 多轮迭代过滤
        user_cnt = Counter(d['reviewerID'] for d in reviews)
        item_cnt = Counter(d['asin'] for d in reviews)
        reviews = [d for d in reviews 
                   if user_cnt[d['reviewerID']] >= MIN_USER_REVIEWS 
                   and item_cnt[d['asin']] >= MIN_ITEM_REVIEWS]
        if verbose:
            u_cnt = len(set(d['reviewerID'] for d in reviews))
            i_cnt = len(set(d['asin'] for d in reviews))
            print(f"  过滤轮次 {round_i+1}: {len(reviews):,} 条, {u_cnt:,} 用户, {i_cnt:,} 商品")
    
    # 收集商品 ID
    item_set = set(d['asin'] for d in reviews)
    
    # 加载元数据
    meta = load_meta_for_items(META_FILE, item_set)
    
    if verbose:
        print(f"\n最终数据规模:")
        print(f"  评论数: {len(reviews):,}")
        print(f"  用户数: {len(set(d['reviewerID'] for d in reviews)):,}")
        print(f"  商品数: {len(item_set):,}")
        print(f"  元数据覆盖: {len(meta):,} ({len(meta)/len(item_set)*100:.1f}%)")
    
    # ========================================
    # 2. 构建用户-商品交互矩阵 + 统计特征
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("[2] 构建交互矩阵")
        print("=" * 70)
    
    users = sorted(set(d['reviewerID'] for d in reviews))
    items = sorted(set(d['asin'] for d in reviews))
    uid2idx = {u: i for i, u in enumerate(users)}
    iid2idx = {it: i for i, it in enumerate(items)}
    idx2uid = {i: u for u, i in uid2idx.items()}
    idx2iid = {i: it for it, i in iid2idx.items()}
    n_users = len(users)
    n_items = len(items)
    
    if verbose:
        print(f"  Users: {n_users:,}  Items: {n_items:,}")
    
    # 构建交互关系
    user2items = defaultdict(set)
    item2users = defaultdict(set)
    user2ratings = defaultdict(dict)
    item_ratings = defaultdict(list)
    user_ratings = defaultdict(list)
    
    for d in reviews:
        u, i = uid2idx[d['reviewerID']], iid2idx[d['asin']]
        r = d['overall']
        user2items[u].add(i)
        item2users[i].add(u)
        user2ratings[u][i] = r
        item_ratings[i].append(r)
        user_ratings[u].append(r)
    
    # 统计特征
    user_avg_rating = {u: np.mean(rs) for u, rs in user_ratings.items()}
    user_rating_cnt = {u: len(rs) for u, rs in user_ratings.items()}
    user_rating_std = {u: np.std(rs) if len(rs) > 1 else 0 for u, rs in user_ratings.items()}
    item_avg_rating = {i: np.mean(rs) for i, rs in item_ratings.items()}
    item_rating_cnt = {i: len(rs) for i, rs in item_ratings.items()}
    
    global_avg = np.mean([d['overall'] for d in reviews])
    
    # ========================================
    # 3. 划分训练/测试集
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("[3] 划分训练/测试集")
        print("=" * 70)
    
    reviews_sorted = sorted(reviews, key=lambda x: x.get('unixReviewTime', 0))
    n_reviews = len(reviews_sorted)
    split_idx = int(0.8 * n_reviews)
    
    train_reviews = reviews_sorted[:split_idx]
    test_reviews = reviews_sorted[split_idx:]
    
    # 基于训练集构建交互关系
    train_user2items = defaultdict(set)
    train_item2users = defaultdict(set)
    train_user_ratings = defaultdict(dict)
    
    for d in train_reviews:
        u, i = uid2idx[d['reviewerID']], iid2idx[d['asin']]
        r = d['overall']
        train_user2items[u].add(i)
        train_item2users[i].add(u)
        train_user_ratings[u][i] = r
    
    # 测试集真实标签
    test_ground_truth = {}
    for d in test_reviews:
        u, i = uid2idx[d['reviewerID']], iid2idx[d['asin']]
        if u not in test_ground_truth:
            test_ground_truth[u] = {}
        test_ground_truth[u][i] = d['overall']
    
    # 过滤有效测试用户
    valid_test_users = {}
    for u, gt_items in test_ground_truth.items():
        if u in train_user2items and len(train_user2items[u]) >= 2:
            new_items = {i: r for i, r in gt_items.items() if i not in train_user2items[u]}
            if new_items:
                valid_test_users[u] = new_items
    
    test_ground_truth = valid_test_users
    
    if verbose:
        print(f"  训练集: {len(train_reviews):,}")
        print(f"  测试集: {len(test_reviews):,}")
        print(f"  有效测试用户: {len(test_ground_truth):,}")
    
    if len(test_ground_truth) < 50:
        print("⚠️ 有效测试用户太少")
        return None
    
    # ========================================
    # 特征提取函数
    # ========================================
    def extract_user_features(user_idx):
        return np.array([
            user_avg_rating.get(user_idx, global_avg),
            min(user_rating_cnt.get(user_idx, 0), 100),
            user_rating_std.get(user_idx, 0),
        ], dtype=np.float32)

    def extract_item_features(asin):
        m = meta.get(asin, {})
        item_idx = iid2idx.get(asin, -1)
        
        has_brand = 1 if m.get('brand') else 0
        title = m.get('title', '')
        title_word_cnt = len(title.split()) if title else 0
        n_features = len(m.get('feature', []) or [])
        has_description = 1 if m.get('description') else 0
        has_price = 1 if m.get('price') else 0
        price = 0.0
        if m.get('price'):
            try:
                p = re.sub(r'[^\d.]', '', str(m['price']))
                price = float(p) if p else 0
            except:
                price = 0
        price = min(price, 500)
        
        rank = m.get('rank')
        log_rank = 0.0
        if rank:
            try:
                if isinstance(rank, list):
                    rank = rank[0]
                nums = re.findall(r'[\d,]+', str(rank))
                if nums:
                    log_rank = math.log1p(int(nums[0].replace(',', '')))
            except:
                log_rank = 0
        
        n_also_buy = len(m.get('also_buy', []) or [])
        n_also_view = len(m.get('also_view', []) or [])
        
        return np.array([
            has_brand, min(title_word_cnt, 50), min(n_features, 20), has_description,
            has_price, price, min(log_rank, 15), min(n_also_buy, 50), min(n_also_view, 50),
            item_avg_rating.get(item_idx, global_avg) if item_idx >= 0 else global_avg,
            min(item_rating_cnt.get(item_idx, 0), 100) if item_idx >= 0 else 0,
        ], dtype=np.float32)
    
    def extract_cross_features(uid_idx, iid_idx):
        user_items = train_user2items.get(uid_idx, set())
        item_users = train_item2users.get(iid_idx, set())
        
        common_users = 0
        if user_items:
            neighbor_users = set()
            for j in user_items:
                neighbor_users |= train_item2users.get(j, set())
            common_users = len(item_users & neighbor_users)
        
        neighbor_items = set()
        for other_u in item_users:
            neighbor_items |= train_user2items.get(other_u, set())
        jaccard = len(user_items & neighbor_items) / max(1, len(user_items | neighbor_items))
        
        return np.array([
            float(min(common_users, 100)),
            jaccard,
        ], dtype=np.float32)
    
    # ========================================
    # 阶段 1: 多路召回
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 1: 多路召回")
        print("=" * 70)
    
    def recall_popular(user_idx, k=500):
        bought = train_user2items.get(user_idx, set())
        candidates = [(i, item_rating_cnt.get(i, 0)) for i in range(n_items) if i not in bought]
        candidates.sort(key=lambda x: -x[1])
        return [i for i, _ in candidates[:k]]
    
    def recall_cf_itemcf(user_idx, k=300):
        bought = train_user2items.get(user_idx, set())
        if not bought:
            return []
        
        candidate_scores = defaultdict(float)
        for i in bought:
            for other_u in train_item2users.get(i, set()):
                for j in train_user2items.get(other_u, set()):
                    if j not in bought:
                        candidate_scores[j] += 1.0 / math.log1p(len(train_item2users.get(j, set())))
        
        candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return [i for i, _ in candidates[:k]]
    
    def recall_cf_usercf(user_idx, k=200):
        bought = train_user2items.get(user_idx, set())
        if not bought:
            return []
        
        user_similarity = defaultdict(int)
        for i in bought:
            for other_u in train_item2users.get(i, set()):
                if other_u != user_idx:
                    user_similarity[other_u] += 1
        
        similar_users = sorted(user_similarity.items(), key=lambda x: -x[1])[:50]
        
        candidate_scores = defaultdict(float)
        for other_u, sim in similar_users:
            for j in train_user2items.get(other_u, set()):
                if j not in bought:
                    candidate_scores[j] += sim * user_avg_rating.get(other_u, 3.5)
        
        candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return [i for i, _ in candidates[:k]]
    
    def recall_content_based(user_idx, k=200):
        bought = train_user2items.get(user_idx, set())
        if not bought:
            return []
        
        candidate_scores = defaultdict(float)
        for i in bought:
            asin = idx2iid[i]
            m = meta.get(asin, {})
            
            for related_asin in (m.get('also_buy', []) or [])[:20]:
                if related_asin in iid2idx:
                    j = iid2idx[related_asin]
                    if j not in bought:
                        candidate_scores[j] += 2.0
            
            for related_asin in (m.get('also_view', []) or [])[:20]:
                if related_asin in iid2idx:
                    j = iid2idx[related_asin]
                    if j not in bought:
                        candidate_scores[j] += 1.0
        
        candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return [i for i, _ in candidates[:k]]
    
    def multi_recall(user_idx, total_k=1000):
        """多路召回融合 - 与原版一致"""
        recall_results = {
            'popular': set(recall_popular(user_idx, k=500)),
            'itemcf': set(recall_cf_itemcf(user_idx, k=300)),
            'usercf': set(recall_cf_usercf(user_idx, k=200)),
            'content': set(recall_content_based(user_idx, k=200)),
        }
        
        # 融合：每个来源给分数，来源越多分越高
        candidate_sources = defaultdict(set)
        for source, items_set in recall_results.items():
            for i in items_set:
                candidate_sources[i].add(source)
        
        # 按来源数量 + 热度排序（与原版一致）
        candidates_with_score = []
        for i, sources in candidate_sources.items():
            score = len(sources) * 10 + item_rating_cnt.get(i, 0) * 0.01
            candidates_with_score.append((i, score, sources))
        
        candidates_with_score.sort(key=lambda x: -x[1])
        return candidates_with_score[:total_k]
    
    # 测试召回
    test_sample = list(test_ground_truth.keys())[:100]
    recall_hits = 0
    total_gt = 0
    for u in test_sample:
        recalled = set(i for i, _, _ in multi_recall(u, 1000))
        gt = set(test_ground_truth[u].keys())
        recall_hits += len(recalled & gt)
        total_gt += len(gt)
    
    recall_rate = recall_hits / total_gt if total_gt > 0 else 0
    if verbose:
        print(f"  召回率 (Recall@1000): {recall_rate*100:.2f}%")
    
    # ========================================
    # 阶段 2: 粗排 (XGBoost/GradientBoosting)
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 2: 粗排 (XGBoost)")
        print("=" * 70)
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    # 构建粗排训练数据
    coarse_X = []
    coarse_y = []
    
    train_sample = train_reviews[:min(20000, len(train_reviews))]
    for d in train_sample:
        u = uid2idx[d['reviewerID']]
        i = iid2idx[d['asin']]
        
        user_feat = extract_user_features(u)
        item_feat = extract_item_features(d['asin'])
        cross_feat = extract_cross_features(u, i)
        
        features = np.concatenate([user_feat, item_feat, cross_feat])
        coarse_X.append(features)
        coarse_y.append(d['overall'])
    
    # 负样本
    np.random.seed(42)
    neg_sample = train_reviews[:min(10000, len(train_reviews))]
    for d in neg_sample:
        u = uid2idx[d['reviewerID']]
        bought = user2items.get(u, set())
        
        neg_i = np.random.randint(0, n_items)
        while neg_i in bought:
            neg_i = np.random.randint(0, n_items)
        
        user_feat = extract_user_features(u)
        item_feat = extract_item_features(idx2iid[neg_i])
        cross_feat = extract_cross_features(u, neg_i)
        
        features = np.concatenate([user_feat, item_feat, cross_feat])
        coarse_X.append(features)
        coarse_y.append(2.0)  # 负样本设为 2.0（低于平均，与原版一致）
    
    coarse_X = np.array(coarse_X)
    coarse_y = np.array(coarse_y)
    
    coarse_mean = coarse_X.mean(axis=0)
    coarse_std = coarse_X.std(axis=0)
    coarse_std[coarse_std == 0] = 1.0
    coarse_X_norm = (coarse_X - coarse_mean) / coarse_std
    
    if verbose:
        print(f"  粗排训练样本: {len(coarse_X):,}")
    
    coarse_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    coarse_model.fit(coarse_X_norm, coarse_y)
    
    if verbose:
        print("  粗排模型训练完成!")
    
    def coarse_rank(user_idx, candidates, top_k=100):
        if not candidates:
            return []
        
        user_feat = extract_user_features(user_idx)
        
        scores = []
        for item_idx, recall_score, sources in candidates:
            item_feat = extract_item_features(idx2iid[item_idx])
            cross_feat = extract_cross_features(user_idx, item_idx)
            features = np.concatenate([user_feat, item_feat, cross_feat])
            features_norm = (features - coarse_mean) / coarse_std
            
            pred = coarse_model.predict(features_norm.reshape(1, -1))[0]
            scores.append((item_idx, pred, sources))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    # ========================================
    # 阶段 3: 精排 (FM)
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 3: 精排 (FM)")
        print("=" * 70)
    
    K_FM = 8
    LR_FM = 0.005       # 提高学习率，补偿 epoch 和样本量减少
    REG_W = 0.02
    REG_V = 0.02
    EPOCHS_FM = 10       # 从 20 减少到 10
    CLIP = 5.0
    
    n_fine_features = 16
    
    def extract_fine_features(user_idx, item_idx):
        user_feat = extract_user_features(user_idx)
        item_feat = extract_item_features(idx2iid[item_idx])
        cross_feat = extract_cross_features(user_idx, item_idx)
        return np.concatenate([user_feat, item_feat, cross_feat])
    
    # 构建精排训练数据（采样，避免全量训练太慢）
    MAX_FINE_SAMPLES = 50000
    fine_train_sample = train_reviews[:MAX_FINE_SAMPLES] if len(train_reviews) > MAX_FINE_SAMPLES else train_reviews
    
    fine_data = []
    for d in fine_train_sample:
        u = uid2idx[d['reviewerID']]
        i = iid2idx[d['asin']]
        feat = extract_fine_features(u, i)
        fine_data.append((u, i, feat, d['overall']))
    
    fine_feats = np.array([x[2] for x in fine_data])
    fine_mu = fine_feats.mean(axis=0)
    fine_std = fine_feats.std(axis=0)
    fine_std[fine_std == 0] = 1.0
    
    for idx in range(len(fine_data)):
        u, i, feat, rating = fine_data[idx]
        fine_data[idx] = (u, i, (feat - fine_mu) / fine_std, rating)
    
    if verbose:
        print(f"  精排训练样本: {len(fine_data):,}")
    
    # FM 模型初始化
    w0_fm = 0.0
    w_user_fm = np.zeros(n_users, dtype=np.float64)
    w_item_fm = np.zeros(n_items, dtype=np.float64)
    w_fine = np.zeros(n_fine_features, dtype=np.float64)
    v_user_fm = np.random.randn(n_users, K_FM) * 0.01
    v_item_fm = np.random.randn(n_items, K_FM) * 0.01
    v_fine = np.random.randn(n_fine_features, K_FM) * 0.01
    
    def clip_val(x, lo=-10.0, hi=10.0):
        return max(lo, min(hi, x))
    
    def clip_grad(g, c=CLIP):
        if isinstance(g, np.ndarray):
            norm = np.linalg.norm(g)
            if norm > c:
                g = g * (c / norm)
            return g
        return max(-c, min(c, g))
    
    def fm_predict(u, it, feat):
        linear = w0_fm + w_user_fm[u] + w_item_fm[it] + w_fine @ feat
        sum_vx = v_user_fm[u] + v_item_fm[it] + (v_fine.T @ feat)
        sum_v2x2 = v_user_fm[u]**2 + v_item_fm[it]**2 + ((v_fine**2).T @ (feat**2))
        interaction = 0.5 * (sum_vx @ sum_vx - sum_v2x2.sum())
        return clip_val(linear + interaction, 0.5, 5.5)
    
    # 训练 FM
    np.random.seed(42)
    fine_indices = np.arange(len(fine_data))
    
    for epoch in range(EPOCHS_FM):
        np.random.shuffle(fine_indices)
        
        for idx in fine_indices:
            u, it, feat, rating = fine_data[idx]
            
            sum_vx = v_user_fm[u] + v_item_fm[it] + (v_fine.T @ feat)
            
            pred = w0_fm + w_user_fm[u] + w_item_fm[it] + w_fine @ feat
            pred += 0.5 * (sum_vx @ sum_vx
                           - v_user_fm[u] @ v_user_fm[u]
                           - v_item_fm[it] @ v_item_fm[it]
                           - (feat**2) @ (v_fine * v_fine).sum(axis=1))
            
            err = clip_val(rating - pred, -5.0, 5.0)
            
            w0_fm += LR_FM * clip_grad(err)
            w_user_fm[u] += LR_FM * clip_grad(err - REG_W * w_user_fm[u])
            w_item_fm[it] += LR_FM * clip_grad(err - REG_W * w_item_fm[it])
            w_fine += LR_FM * clip_grad(err * feat - REG_W * w_fine)
            
            grad_vu = err * (sum_vx - v_user_fm[u]) - REG_V * v_user_fm[u]
            v_user_fm[u] += LR_FM * clip_grad(grad_vu)
            
            grad_vi = err * (sum_vx - v_item_fm[it]) - REG_V * v_item_fm[it]
            v_item_fm[it] += LR_FM * clip_grad(grad_vi)
            
            for j in range(n_fine_features):
                if abs(feat[j]) < 1e-8:
                    continue
                grad_vj = err * feat[j] * (sum_vx - v_fine[j] * feat[j]) - REG_V * v_fine[j]
                v_fine[j] += LR_FM * clip_grad(grad_vj)
        
        if verbose:
            print(f"    Epoch {epoch+1}/{EPOCHS_FM} 完成 ({len(fine_data):,} 样本)")
    
    if verbose:
        print("  FM 模型训练完成!")
    
    def fine_rank(user_idx, candidates, top_k=20):
        if not candidates:
            return []
        
        scores = []
        for item_idx, coarse_score, sources in candidates:
            feat = extract_fine_features(user_idx, item_idx)
            feat_norm = (feat - fine_mu) / fine_std
            
            fm_pred = fm_predict(user_idx, item_idx, feat_norm)
            coarse_normalized = (coarse_score - 2.0) / 3.0
            final_score = 0.6 * fm_pred + 0.4 * (coarse_normalized * 5.0)
            
            scores.append((item_idx, final_score, coarse_score, sources))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    # ========================================
    # 阶段 4: 重排序
    # ========================================
    def rerank(user_idx, candidates, top_k=10):
        """重排序 - 与原版一致：品牌多样性 + 新鲜度 + 来源加分"""
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k] if candidates else []
        
        result = []
        seen_brands = set()
        remaining = list(candidates)
        
        for _ in range(min(len(remaining), top_k)):
            best_idx = 0
            best_score = -float('inf')
            
            for i, (item_idx, fm_score, coarse_score, sources) in enumerate(remaining):
                asin = idx2iid[item_idx]
                m = meta.get(asin, {})
                brand = m.get('brand', 'unknown')
                
                # 多样性惩罚（按品牌）
                diversity_penalty = -0.5 if brand in seen_brands else 0.0
                
                # 新鲜度加分
                freshness_bonus = 0.1 if item_rating_cnt.get(item_idx, 0) < 10 else 0.0
                
                # 召回来源加分（多通道更可信）
                source_bonus = len(sources) * 0.1
                
                final_score = fm_score + diversity_penalty + freshness_bonus + source_bonus
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = i
            
            item_idx, fm_score, coarse_score, sources = remaining.pop(best_idx)
            asin = idx2iid[item_idx]
            brand = meta.get(asin, {}).get('brand', 'unknown')
            seen_brands.add(brand)
            
            result.append((item_idx, best_score, fm_score, sources))
        
        return result
    
    # ========================================
    # 评估
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("评估")
        print("=" * 70)
    
    def ndcg_at_k(ranked_items, ground_truth, k=10):
        dcg = 0.0
        for i, item_idx in enumerate(ranked_items[:k]):
            if item_idx in ground_truth:
                rel = ground_truth[item_idx] / 5.0
                dcg += (2**rel - 1) / math.log2(i + 2)
        
        ideal_rels = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = sum((2**(r/5.0) - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def hit_rate_at_k(ranked_items, ground_truth, k=10):
        return 1.0 if any(i in ground_truth for i in ranked_items[:k]) else 0.0
    
    def compute_auc(scored_items, ground_truth):
        if not scored_items or not ground_truth:
            return 0.5
        
        pos_scores = []
        neg_scores = []
        
        for item_idx, score, *_ in scored_items:
            if item_idx in ground_truth:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        
        if not pos_scores or not neg_scores:
            return 0.5
        
        auc_sum = 0
        for ps in pos_scores:
            for ns in neg_scores:
                if ps > ns:
                    auc_sum += 1
                elif ps == ns:
                    auc_sum += 0.5
        
        return auc_sum / (len(pos_scores) * len(neg_scores))
    
    # 评估
    metrics = {
        'recall_hitrate': [], 'coarse_hitrate': [], 'fine_hitrate': [], 'final_hitrate': [],
        'recall_ndcg': [], 'coarse_ndcg': [], 'fine_ndcg': [], 'final_ndcg': [],
        'coarse_auc': [], 'fine_auc': [], 'final_auc': []
    }
    
    test_users_eval = list(test_ground_truth.keys())[:200]
    
    for idx, u in enumerate(test_users_eval):
        if verbose and idx % 50 == 0:
            print(f"  评估进度: {idx}/{len(test_users_eval)}")
        
        gt = test_ground_truth[u]
        
        recalled = multi_recall(u, total_k=1000)
        coarse_ranked = coarse_rank(u, recalled, top_k=100)
        fine_ranked = fine_rank(u, coarse_ranked, top_k=50)
        final_ranked = rerank(u, fine_ranked, top_k=10)
        
        recall_items = [i for i, _, _ in recalled[:10]]
        coarse_items = [i for i, _, _ in coarse_ranked[:10]]
        fine_items = [i for i, _, _, _ in fine_ranked[:10]]
        final_items = [i for i, _, _, _ in final_ranked]
        
        metrics['recall_hitrate'].append(hit_rate_at_k(recall_items, gt, 10))
        metrics['coarse_hitrate'].append(hit_rate_at_k(coarse_items, gt, 10))
        metrics['fine_hitrate'].append(hit_rate_at_k(fine_items, gt, 10))
        metrics['final_hitrate'].append(hit_rate_at_k(final_items, gt, 10))
        
        metrics['recall_ndcg'].append(ndcg_at_k(recall_items, gt, 10))
        metrics['coarse_ndcg'].append(ndcg_at_k(coarse_items, gt, 10))
        metrics['fine_ndcg'].append(ndcg_at_k(fine_items, gt, 10))
        metrics['final_ndcg'].append(ndcg_at_k(final_items, gt, 10))
        
        metrics['coarse_auc'].append(compute_auc(coarse_ranked, gt))
        metrics['fine_auc'].append(compute_auc(fine_ranked, gt))
        metrics['final_auc'].append(compute_auc(final_ranked, gt))
    
    # 汇总
    result = {
        'dataset': 'All_Amazon',
        'n_users': n_users,
        'n_items': n_items,
        'n_reviews': len(reviews),
        'recall_rate': recall_rate,
        'recall_hitrate': np.mean(metrics['recall_hitrate']),
        'coarse_hitrate': np.mean(metrics['coarse_hitrate']),
        'fine_hitrate': np.mean(metrics['fine_hitrate']),
        'final_hitrate': np.mean(metrics['final_hitrate']),
        'recall_ndcg': np.mean(metrics['recall_ndcg']),
        'coarse_ndcg': np.mean(metrics['coarse_ndcg']),
        'fine_ndcg': np.mean(metrics['fine_ndcg']),
        'final_ndcg': np.mean(metrics['final_ndcg']),
        'coarse_auc': np.mean(metrics['coarse_auc']),
        'fine_auc': np.mean(metrics['fine_auc']),
        'final_auc': np.mean(metrics['final_auc']),
    }
    
    if verbose:
        print(f"\n" + "=" * 70)
        print("📊 评估结果")
        print("=" * 70)
        print(f"\n数据规模: {n_users:,} 用户 × {n_items:,} 商品 × {len(reviews):,} 评论")
        print(f"\n各阶段指标:")
        print("-" * 65)
        print(f"{'阶段':<15} {'HitRate@10':>12} {'NDCG@10':>12} {'AUC':>12}")
        print("-" * 65)
        print(f"{'召回':<15} {result['recall_hitrate']*100:>11.2f}% {result['recall_ndcg']:>12.4f} {'N/A':>12}")
        print(f"{'粗排(XGBoost)':<15} {result['coarse_hitrate']*100:>11.2f}% {result['coarse_ndcg']:>12.4f} {result['coarse_auc']:>12.4f}")
        print(f"{'精排(FM)':<15} {result['fine_hitrate']*100:>11.2f}% {result['fine_ndcg']:>12.4f} {result['fine_auc']:>12.4f}")
        print(f"{'重排序':<15} {result['final_hitrate']*100:>11.2f}% {result['final_ndcg']:>12.4f} {result['final_auc']:>12.4f}")
        print("-" * 65)
    
    return result


# ============================================================
# 可视化
# ============================================================
def visualize_result(result: dict, output_path: str):
    """生成可视化图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Amazon Full Dataset - Recommendation Pipeline Results\n'
                 f'({result["n_users"]:,} users × {result["n_items"]:,} items × {result["n_reviews"]:,} reviews)',
                 fontsize=12, fontweight='bold')
    
    stages = ['Recall', 'Coarse\n(XGB)', 'Fine\n(FM)', 'Rerank']
    
    # (a) HitRate@10
    ax = axes[0]
    hitrates = [
        result['recall_hitrate'] * 100,
        result['coarse_hitrate'] * 100,
        result['fine_hitrate'] * 100,
        result['final_hitrate'] * 100
    ]
    colors = ['#3498db', '#e67e22', '#e74c3c', '#2ecc71']
    bars = ax.bar(stages, hitrates, color=colors)
    ax.set_ylabel('HitRate@10 (%)')
    ax.set_title('(a) HitRate@10 by Stage')
    ax.set_ylim(0, max(hitrates) * 1.3)
    for bar, val in zip(bars, hitrates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=10)
    
    # (b) NDCG@10
    ax = axes[1]
    ndcgs = [
        result['recall_ndcg'],
        result['coarse_ndcg'],
        result['fine_ndcg'],
        result['final_ndcg']
    ]
    bars = ax.bar(stages, ndcgs, color=colors)
    ax.set_ylabel('NDCG@10')
    ax.set_title('(b) NDCG@10 by Stage')
    ax.set_ylim(0, max(ndcgs) * 1.3)
    for bar, val in zip(bars, ndcgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', fontsize=10)
    
    # (c) AUC
    ax = axes[2]
    aucs = [
        0.5,  # 召回无 AUC
        result['coarse_auc'],
        result['fine_auc'],
        result['final_auc']
    ]
    bars = ax.bar(stages, aucs, color=colors)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel('AUC')
    ax.set_title('(c) AUC by Stage')
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right')
    for bar, val in zip(bars, aucs):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 图片已保存: {output_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("全量 Amazon 数据 - 推荐系统流水线")
    print("=" * 70)
    
    result = run_pipeline(verbose=True)
    
    if result:
        output_path = os.path.join(OUTPUT_DIR, 'recommendation_pipeline_full.png')
        visualize_result(result, output_path)
        
        print("\n" + "=" * 70)
        print("✅ 全量数据流水线完成!")
        print("=" * 70)
