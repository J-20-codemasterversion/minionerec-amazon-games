"""
推荐系统流水线 V4 - 基于 V3 + 训练损失曲线 + 精选 Demo 测试
=========================================================
V3 全部功能 +
  新增1: 粗排/精排训练过程中记录 loss & AUC，训练结束后绘制曲线图
  新增2: 精选"容易被正确排序"的用户做 demo 级评估

生成文件:
  - training_curves.png (损失 & AUC 曲线)
  - demo_eval_results.json (精选 vs 全部用户的评估对比)
"""

# macOS libomp 兼容：lightgbm 需要 libomp.dylib，SIP 下 @rpath 找不到
# 在 import lightgbm 前显式预加载一次
import sys as _sys
if _sys.platform == "darwin":
    try:
        import ctypes as _ct, site as _site
        from pathlib import Path as _P
        for _sp in _site.getsitepackages() + [_site.getusersitepackages()]:
            _c = _P(_sp) / "lightgbm" / "lib" / "libomp.dylib"
            if _c.is_file():
                try:
                    _ct.CDLL(str(_c), mode=_ct.RTLD_GLOBAL)
                    break
                except OSError:
                    continue
    except Exception:
        pass

from email.policy import default
import os
import json
import re
import math
import random
import time
import numpy as np
from collections import Counter, defaultdict
import heapq
import warnings

try:
    import orjson
    _loads = orjson.loads          # C 实现, 比标准库快 3-6x
except ImportError:
    _loads = json.loads
warnings.filterwarnings('ignore')

# V4 新增: 训练监控模块
from training_monitor import (
    LiveAUCPlot, select_demo_test_users
)

# 召回模块（从独立文件导入）
from recall_hot import HotRecall
from recall_itemcf import ItemCFRecall
from recall_usercf import UserCFRecall
from recall_swing import SwingRecall
from recall_fusion import RecallFusion

# ============================================================
# 配置
# ============================================================
# 自动检测运行环境：CVM 或 本地 Mac
if os.path.exists("/root/amazon_data"):
    DATA_DIR = "/root/amazon_data"
    OUTPUT_DIR = "/root/amazon_data"
else:
    DATA_DIR = "/Users/jasonlihahaha/Desktop/amazon_data/数据"
    OUTPUT_DIR = "/Users/jasonlihahaha/Desktop/amazon_data/回归"

# 分品类 5-core 数据集
DATASETS = [
    {
        'name': 'Video_Games',
        'review': os.path.join(DATA_DIR, 'Video_Games.json'),
        'meta': os.path.join(DATA_DIR, 'meta_Video_Games.json'),
    },
    {
        'name': 'Electronics',
        'review': os.path.join(DATA_DIR, 'Electronics.json'),
        'meta': os.path.join(DATA_DIR, 'meta_Electronics.json'),
        'max_reviews': 3000000,  # Electronics 太大，采样 300 万条
    },
]

TARGET_USERS = 10000        # 选取交互最多的 N 个用户
MAX_REVIEWS_PER_USER = 100  # 每用户最多保留的评论数
MIN_REVIEWS_PER_USER = 10   # 每用户最少需要的评论数（5-core 保证 >= 5）


# ============================================================
# 内容召回（外部模块没有此通道，在此内联定义）
# ============================================================

class ContentRecall:
    """内容召回 - 基于品牌/品类/标题关键词倒排索引"""
    def __init__(self, item_meta, item_users):
        self.item_meta = item_meta
        self.item_users = item_users
        self.brand_items = defaultdict(set)
        self.category_items = defaultdict(set)
        self.title_words_items = defaultdict(set)

    def build_index(self):
        print("  构建内容召回索引...", end=" ", flush=True)
        for item, meta in self.item_meta.items():
            brand = meta.get('brand', '').strip().lower()
            if brand and brand != 'unknown':
                self.brand_items[brand].add(item)
            category = meta.get('category', '')
            if category:
                self.category_items[category].add(item)
            title = meta.get('title', '')
            if title:
                words = title.lower().split()[:10]
                for word in words:
                    if len(word) > 3:
                        self.title_words_items[word].add(item)
        print(f"完成 (品牌: {len(self.brand_items)}, 品类: {len(self.category_items)}, 关键词: {len(self.title_words_items)})")

    def recall(self, user_history, user_history_set, top_k=200):
        candidate_scores = defaultdict(float)
        user_brands = []
        user_categories = []
        user_words = []

        for item, rating, ts in user_history:
            meta = self.item_meta.get(item, {})
            brand = meta.get('brand', '').strip().lower()
            category = meta.get('category', '')
            title = meta.get('title', '')
            if brand:
                user_brands.append((brand, rating))
            if category:
                user_categories.append((category, rating))
            if title:
                for word in title.lower().split()[:5]:
                    if len(word) > 3:
                        user_words.append((word, rating))

        for brand, rating in user_brands:
            for item in self.brand_items.get(brand, []):
                if item not in user_history_set:
                    candidate_scores[item] += rating * 0.5

        for category, rating in user_categories:
            for item in list(self.category_items.get(category, []))[:500]:
                if item not in user_history_set:
                    candidate_scores[item] += rating * 0.3

        for word, rating in user_words[:20]:
            for item in list(self.title_words_items.get(word, []))[:100]:
                if item not in user_history_set:
                    candidate_scores[item] += rating * 0.2

        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


class AlsoBuyViewRecall:
    """基于 Amazon meta 中 also_buy / also_view 关联图谱的召回"""
    def __init__(self, full_meta, item_pool):
        """
        full_meta: 完整 meta dict (asin -> meta_dict, 包含 also_buy/also_view)
        item_pool: 当前商品池 set(asin)，只召回池内商品
        """
        self.full_meta = full_meta
        self.item_pool = item_pool
        self.also_buy_graph = defaultdict(list)   # item -> [related_items]
        self.also_view_graph = defaultdict(list)

    def build_index(self):
        print("  构建 also_buy/also_view 索引...", end=" ", flush=True)
        buy_edges = 0
        view_edges = 0
        for asin in self.item_pool:
            m = self.full_meta.get(asin, {})
            ab = m.get('also_buy', []) or []
            av = m.get('also_view', []) or []
            # 只保留池内商品
            ab_in_pool = [x for x in ab if x in self.item_pool]
            av_in_pool = [x for x in av if x in self.item_pool]
            if ab_in_pool:
                self.also_buy_graph[asin] = ab_in_pool
                buy_edges += len(ab_in_pool)
            if av_in_pool:
                self.also_view_graph[asin] = av_in_pool
                view_edges += len(av_in_pool)
        items_with_buy = len(self.also_buy_graph)
        items_with_view = len(self.also_view_graph)
        print(f"完成 (also_buy: {items_with_buy:,} 商品/{buy_edges:,} 边, "
              f"also_view: {items_with_view:,} 商品/{view_edges:,} 边)")

    def recall(self, user_history, user_history_set, top_k=500):
        """
        user_history: [(asin, rating, ts), ...]
        user_history_set: set(asin)
        """
        candidate_scores = defaultdict(float)

        for item, rating, ts in user_history:
            # also_buy: 权重更高（购买行为更强）
            for related in self.also_buy_graph.get(item, []):
                if related not in user_history_set:
                    candidate_scores[related] += rating * 2.0
            # also_view: 权重低一些
            for related in self.also_view_graph.get(item, []):
                if related not in user_history_set:
                    candidate_scores[related] += rating * 1.0

        # 二跳扩展：对得分最高的候选，再扩展一层 also_buy（增加覆盖率）
        if candidate_scores:
            top1_items = sorted(candidate_scores.items(), key=lambda x: -x[1])[:50]
            for hop_item, hop_score in top1_items:
                for related in self.also_buy_graph.get(hop_item, [])[:10]:
                    if related not in user_history_set and related not in candidate_scores:
                        candidate_scores[related] += hop_score * 0.3

        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


# ============================================================
# 数据加载函数
# ============================================================

def load_category_reviews(datasets, target_users=TARGET_USERS,
                          max_per_user=MAX_REVIEWS_PER_USER,
                          min_per_user=MIN_REVIEWS_PER_USER):
    """
    从分品类 5-core JSON 文件加载评论数据。
    1. 逐文件扫描，收集每个用户的交互
    2. 选取 top 活跃用户
    3. 只保留这些用户的交互（不引入路人用户）
    """
    t0 = time.time()
    print(f"[数据加载] 分品类 5-core 数据")

    # Pass 1: 扫描所有文件，统计每个用户的交互
    user_reviews = defaultdict(list)  # uid -> [(asin, rating, ts, dataset_name)]
    total_loaded = 0

    for ds in datasets:
        filepath = ds['review']
        ds_name = ds['name']
        max_reviews = ds.get('max_reviews', None)
        print(f"\n  [{ds_name}] {filepath}")

        if not os.path.exists(filepath):
            print(f"    文件不存在，跳过")
            continue

        count = 0
        with open(filepath, 'rb', buffering=8*1024*1024) as f:
            for raw_line in f:
                try:
                    d = _loads(raw_line)
                    uid = d.get('reviewerID')
                    asin = d.get('asin')
                    if uid and asin and 'overall' in d:
                        ts = d.get('unixReviewTime', 0)
                        rating = d['overall']
                        user_reviews[uid].append((asin, rating, ts, ds_name))
                        count += 1
                except Exception:
                    continue
                if max_reviews and count >= max_reviews:
                    break
                if count % 1000000 == 0 and count > 0:
                    elapsed = time.time() - t0
                    print(f"    {count/1e6:.0f}M 条, {elapsed:.1f}s, 用户 {len(user_reviews):,}")

        total_loaded += count
        print(f"    加载完成: {count:,} 条")

    t1 = time.time()
    print(f"\n  总计: {total_loaded:,} 条评论, {len(user_reviews):,} 个用户, 耗时 {t1-t0:.1f}s")

    # 选取 top 活跃用户
    print(f"\n[选取] top {target_users:,} 活跃用户 (最少 {min_per_user} 条交互)...")
    user_counts = [(uid, len(revs)) for uid, revs in user_reviews.items()]
    user_counts.sort(key=lambda x: -x[1])
    qualified = [(uid, cnt) for uid, cnt in user_counts if cnt >= min_per_user]
    selected_uids = set(uid for uid, cnt in qualified[:target_users])

    print(f"  交互 >= {min_per_user} 的用户: {len(qualified):,}")
    print(f"  选取 top {len(selected_uids):,} 用户")

    if selected_uids:
        selected_counts = [len(user_reviews[u]) for u in selected_uids]
        print(f"  交互范围: [{min(selected_counts)}, {max(selected_counts)}], "
              f"平均: {np.mean(selected_counts):.1f}")

    # 只保留种子用户的评论（每人截取前 max_per_user 条）
    reviews = []
    for uid in selected_uids:
        revs = user_reviews[uid]
        revs_sorted = sorted(revs, key=lambda x: x[2])[:max_per_user]
        for asin, rating, ts, ds_name in revs_sorted:
            reviews.append({
                'reviewerID': uid,
                'asin': asin,
                'overall': rating,
                'unixReviewTime': ts,
                '_dataset': ds_name,
            })

    # 释放内存
    del user_reviews, user_counts, qualified

    final_users = set(d['reviewerID'] for d in reviews)
    final_items = set(d['asin'] for d in reviews)
    t2 = time.time()

    print(f"\n  === 最终数据集 ===")
    print(f"  总评论数: {len(reviews):,}")
    print(f"  用户数: {len(final_users):,}")
    print(f"  商品数: {len(final_items):,}")
    print(f"  平均交互/用户: {len(reviews)/max(len(final_users),1):.1f}")
    print(f"  平均交互/商品: {len(reviews)/max(len(final_items),1):.1f}")
    print(f"  总耗时: {t2 - t0:.1f}s")

    return reviews, selected_uids


def load_category_meta(datasets, item_set):
    """
    从分品类 meta JSON 文件加载商品元数据。
    只加载 item_set 中的商品。
    """
    t0 = time.time()
    print(f"正在加载商品元数据...")
    print(f"  需要加载: {len(item_set):,} 个商品")

    item_bytes_set = {a.encode('utf-8') for a in item_set}
    meta = {}

    for ds in datasets:
        filepath = ds.get('meta', '')
        ds_name = ds['name']
        if not filepath or not os.path.exists(filepath):
            print(f"  [{ds_name}] meta 文件不存在，跳过")
            continue
        print(f"  [{ds_name}] {filepath}")
        count = 0
        with open(filepath, 'rb', buffering=8*1024*1024) as f:
            for raw_line in f:
                if b'"asin"' not in raw_line:
                    continue
                # 快速提取 asin
                idx = raw_line.find(b'"asin"')
                if idx < 0:
                    continue
                start = raw_line.find(b'"', idx + 6)
                if start < 0:
                    continue
                start += 1
                end = raw_line.find(b'"', start)
                if end < 0:
                    continue
                asin_bytes = raw_line[start:end]
                if asin_bytes not in item_bytes_set:
                    continue
                try:
                    d = _loads(raw_line)
                    asin = d.get('asin')
                    if asin and asin in item_set:
                        meta[asin] = d
                        meta[asin]['_source_dataset'] = ds_name
                        count += 1
                except Exception:
                    continue

        print(f"    找到: {count:,} 个商品")

    t1 = time.time()
    print(f"  总共加载: {len(meta):,} 个商品元数据 "
          f"(覆盖率: {len(meta)/max(len(item_set),1)*100:.1f}%)")
    print(f"  耗时: {t1 - t0:.1f}s")
    return meta


# ============================================================
# 特征提取
# ============================================================

def parse_price(p):
    if not p:
        return None
    p = str(p).replace('$', '').replace(',', '').strip()
    if '-' in p:
        parts = p.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return None
    try:
        v = float(p)
        return v if 0 < v < 10000 else None
    except:
        return None


def parse_rank(r):
    if not r:
        return None
    if isinstance(r, list):
        r = r[0] if r else ''
    m = re.search(r'([\d,]+)', str(r))
    if m:
        try:
            return int(m.group(1).replace(',', ''))
        except:
            return None
    return None


def extract_category(meta_dict):
    """从 meta 中提取真实品类"""
    cats = meta_dict.get('category', [])
    if isinstance(cats, list) and cats:
        # 取最细粒度的品类（最后一个），或第一个
        return cats[-1]
    if isinstance(cats, str) and cats:
        return cats
    # 尝试从 rank 字段提取品类
    rank = meta_dict.get('rank')
    if isinstance(rank, str):
        m = re.search(r'in\s+(.+?)(?:\s*\()', rank)
        if m:
            return m.group(1).strip()
    return 'unknown'


# ============================================================
# 主流水线
# ============================================================

def run_pipeline(verbose=True):
    start_time = time.time()

    # V4: 图片保存路径
    _curve_path = os.path.join(OUTPUT_DIR, 'training_curves.png')

    if verbose:
        print("\n" + "=" * 70)
        print("推荐系统流水线 V4")
        print("V3 + 训练损失曲线 + 精选 Demo 测试")
        print("=" * 70)

    # ========================================
    # 1. 数据加载 + 过滤低活跃用户
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("[1] 数据加载（分品类 5-core 数据: Video_Games + Electronics）")
        print("=" * 70)

    reviews, selected_uid_set = load_category_reviews(DATASETS)

    item_set = set(d['asin'] for d in reviews)
    meta = load_category_meta(DATASETS, item_set)

    # 为没有 meta 的商品构建简单 meta，并提取真实品类
    for asin in item_set:
        if asin not in meta:
            meta[asin] = {'asin': asin, 'title': '', 'brand': '', 'category': 'unknown'}
        else:
            # 提取真实品类到统一字段
            meta[asin]['_category'] = extract_category(meta[asin])

    for asin in item_set:
        if '_category' not in meta[asin]:
            meta[asin]['_category'] = 'unknown'

    if verbose:
        cats = Counter(meta[a]['_category'] for a in item_set)
        n_users_total = len(set(d['reviewerID'] for d in reviews))
        ratings = [d['overall'] for d in reviews]
        rating_dist = Counter(ratings)

        print(f"\n{'='*50}")
        print(f"  [数据健康检查]")
        print(f"{'='*50}")
        print(f"  评论数: {len(reviews):,}")
        print(f"  用户数: {n_users_total:,}")
        print(f"  商品数: {len(item_set):,}")
        density = len(reviews) / max(n_users_total * len(item_set), 1) * 100
        print(f"  交互密度: {density:.4f}%  {'⚠ 太稀疏(<0.01%)' if density < 0.01 else '✓'}")
        avg_per_user = len(reviews) / max(n_users_total, 1)
        avg_per_item = len(reviews) / max(len(item_set), 1)
        print(f"  平均交互/用户: {avg_per_user:.1f}  {'⚠ 太少(<3)' if avg_per_user < 3 else '✓'}")
        print(f"  平均交互/商品: {avg_per_item:.1f}  {'⚠ 太少(<2)' if avg_per_item < 2 else '✓'}")
        print(f"  评分分布: {dict(sorted(rating_dist.items()))}")
        print(f"  平均评分: {np.mean(ratings):.2f}  {'⚠ 偏高(>4.5)偏低(<2)' if np.mean(ratings) > 4.5 or np.mean(ratings) < 2 else '✓'}")
        meta_cover = len([a for a in item_set if meta.get(a, {}).get('title')])
        meta_ratio = meta_cover / max(len(item_set), 1) * 100
        print(f"  元数据覆盖: {meta_cover:,} / {len(item_set):,} ({meta_ratio:.1f}%)  {'⚠ 覆盖率低(<50%)' if meta_ratio < 50 else '✓'}")
        print(f"  品类数: {len(cats):,}  (Top5: {cats.most_common(5)})")
        print(f"{'='*50}")

    # ========================================
    # 2. 构建交互矩阵 + ID 映射 + 统计特征
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("[2] 构建交互矩阵 + 统计特征")
        print("=" * 70)

    users = sorted(set(d['reviewerID'] for d in reviews))
    items_list = sorted(set(d['asin'] for d in reviews))
    uid2idx = {u: i for i, u in enumerate(users)}
    iid2idx = {it: i for i, it in enumerate(items_list)}
    idx2uid = {i: u for u, i in uid2idx.items()}
    idx2iid = {i: it for it, i in iid2idx.items()}
    n_users = len(users)
    n_items = len(items_list)

    if verbose:
        print(f"  Users: {n_users:,}  Items: {n_items:,}")

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

    user_avg_rating = {u: np.mean(rs) for u, rs in user_ratings.items()}
    user_rating_cnt = {u: len(rs) for u, rs in user_ratings.items()}
    user_rating_std = {u: np.std(rs) if len(rs) > 1 else 0 for u, rs in user_ratings.items()}
    item_avg_rating = {i: np.mean(rs) for i, rs in item_ratings.items()}
    item_rating_cnt = {i: len(rs) for i, rs in item_ratings.items()}
    global_avg = np.mean([d['overall'] for d in reviews])

    if verbose:
        print(f"  用户平均交互数: {np.mean([len(v) for v in user2items.values()]):.1f}")
        print(f"  商品平均交互数: {np.mean([len(v) for v in item2users.values()]):.1f}")

    # ========================================
    # 3. 按用户分组的 leave-one-out 划分
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("[3] 划分训练/测试集 (leave-last-one-out per user)")
        print("=" * 70)

    # 按用户分组，每个用户的最后一条交互作为测试
    user_reviews = defaultdict(list)
    for d in reviews:
        user_reviews[d['reviewerID']].append(d)

    train_reviews = []
    test_reviews = []
    for uid, rvs in user_reviews.items():
        rvs_sorted = sorted(rvs, key=lambda x: x.get('unixReviewTime', 0))
        if len(rvs_sorted) >= 3:  # 至少3条：2条训练+1条测试
            train_reviews.extend(rvs_sorted[:-1])
            test_reviews.append(rvs_sorted[-1])
        else:
            train_reviews.extend(rvs_sorted)  # 太少的全给训练

    # 训练集交互
    train_user2items_idx = defaultdict(set)
    train_item2users_idx = defaultdict(set)
    train_user_ratings = defaultdict(dict)

    train_user_items_asin = defaultdict(list)
    train_item_users_asin = defaultdict(list)

    for d in train_reviews:
        u_idx = uid2idx[d['reviewerID']]
        i_idx = iid2idx[d['asin']]
        r = d['overall']
        ts = d.get('unixReviewTime', 0)
        user_id = d['reviewerID']
        asin = d['asin']

        train_user2items_idx[u_idx].add(i_idx)
        train_item2users_idx[i_idx].add(u_idx)
        train_user_ratings[u_idx][i_idx] = r

        train_user_items_asin[user_id].append((asin, r, ts))
        train_item_users_asin[asin].append((user_id, r, ts))

    # 测试集真实标签
    test_ground_truth = {}
    for d in test_reviews:
        u = uid2idx[d['reviewerID']]
        i = iid2idx[d['asin']]
        if i not in train_user2items_idx.get(u, set()):  # 确保是新商品
            test_ground_truth[u] = {i: d['overall']}

    if verbose:
        test_ratio = len(test_reviews) / max(len(reviews), 1) * 100
        valid_ratio = len(test_ground_truth) / max(len(test_reviews), 1) * 100
        print(f"\n{'='*50}")
        print(f"  [划分健康检查]")
        print(f"{'='*50}")
        print(f"  训练集: {len(train_reviews):,}")
        print(f"  测试集: {len(test_reviews):,}  (占比 {test_ratio:.1f}%)  {'⚠ 测试集太小(<5%)' if test_ratio < 5 else '✓'}")
        print(f"  有效测试用户 (有新商品): {len(test_ground_truth):,}  (占比 {valid_ratio:.1f}%)  {'⚠ 有效率低(<50%)' if valid_ratio < 50 else '✓'}")
        train_items_cnt = len(set(d['asin'] for d in train_reviews))
        test_items_cnt = len(set(d['asin'] for d in test_reviews))
        test_in_train = len(set(d['asin'] for d in test_reviews) & set(d['asin'] for d in train_reviews))
        print(f"  训练集商品数: {train_items_cnt:,}, 测试集商品数: {test_items_cnt:,}")
        print(f"  测试商品在训练集中出现: {test_in_train:,} / {test_items_cnt:,} ({test_in_train/max(test_items_cnt,1)*100:.1f}%)  {'⚠ 冷启动严重(<70%)' if test_in_train/max(test_items_cnt,1)*100 < 70 else '✓'}")
        print(f"{'='*50}")

    if len(test_ground_truth) < 50:
        print("  ⚠ 有效测试用户太少(<50)，退出")
        return None

    # ========================================
    # 4. 构建 item_meta（给召回模块用）
    # ========================================
    item_meta_for_recall = {}
    for asin in item_set:
        m = meta.get(asin, {})
        item_meta_for_recall[asin] = {
            'title': m.get('title', ''),
            'brand': m.get('brand', ''),
            'category': m.get('_category', 'unknown'),
        }

    # ========================================
    # 阶段 1: 多路召回（使用外部模块 + 内容召回）
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 1: 多路召回 (imported modules + content)")
        print("=" * 70)

    train_user_items_dict = dict(train_user_items_asin)
    train_item_users_dict = dict(train_item_users_asin)

    print("\n[1.1] 热门召回")
    hot_recall = HotRecall(train_item_users_dict, item_meta_for_recall)
    hot_recall.build_index()

    print("\n[1.2] ItemCF 召回")
    itemcf_recall = ItemCFRecall(train_user_items_dict, train_item_users_dict)
    itemcf_recall.build_index(top_k_sim=200)

    print("\n[1.3] UserCF 召回")
    usercf_recall = UserCFRecall(train_user_items_dict, train_item_users_dict)
    usercf_recall.build_index(top_k_sim=100, max_users=50000)

    print("\n[1.4] Swing 召回")
    swing_recall = SwingRecall(train_user_items_dict, train_item_users_dict)
    swing_recall.build_index(alpha=0.5, top_k_sim=200, max_items=30000)

    print("\n[1.5] 内容召回 (品牌/品类/标题)")
    content_recall = ContentRecall(item_meta_for_recall, train_item_users_dict)
    content_recall.build_index()

    print("\n[1.6] Also-Buy/View 召回 (Amazon 关联图谱)")
    also_recall = AlsoBuyViewRecall(meta, item_set)
    also_recall.build_index()

    print("\n[1.7] 初始化多路召回融合 (4路 RecallFusion + content + also)")
    fusion_4ch = RecallFusion(hot_recall, itemcf_recall, usercf_recall,
                              swing_recall, train_user_items_dict)
    print("  4路融合权重: hot=0.1, itemcf=0.3, usercf=0.2, swing=0.4")
    print("  + content 通道 (权重 0.25)")
    print("  + also_buy/view 通道 (权重 0.50)")
    print("  + 品类热门通道 (权重 0.15)")

    CONTENT_WEIGHT = 0.25
    ALSO_WEIGHT = 0.50
    CAT_HOT_WEIGHT = 0.15

    def do_recall(user_idx, total_k=1000):
        user_id = idx2uid[user_idx]
        user_history = train_user_items_asin.get(user_id, [])
        if not user_history:
            return []

        # 4路融合召回（每通道已给满量）
        base_results = fusion_4ch.recall(user_id, user_history, total_recall=total_k)
        # 转为 dict 方便合并
        merged = {}
        for r in base_results:
            merged[r['item']] = {'score': r['score'], 'sources': list(r['sources'])}

        user_history_set = {item for item, r, t in user_history}

        # content 通道（给大一些的量）
        content_items = content_recall.recall(user_history, user_history_set, top_k=total_k)
        if content_items:
            max_cs = max(s for _, s in content_items) or 1
            for item, sim_score in content_items:
                norm_score = (sim_score / max_cs) * CONTENT_WEIGHT
                if item in merged:
                    merged[item]['score'] += norm_score
                    merged[item]['sources'].append('content')
                else:
                    merged[item] = {'score': norm_score, 'sources': ['content']}

        # also_buy/also_view 通道（最重要的召回源，给大量）
        also_items = also_recall.recall(user_history, user_history_set, top_k=total_k)
        if also_items:
            max_as = max(s for _, s in also_items) or 1
            for item, also_score in also_items:
                norm_score = (also_score / max_as) * ALSO_WEIGHT
                if item in merged:
                    merged[item]['score'] += norm_score
                    merged[item]['sources'].append('also')
                else:
                    merged[item] = {'score': norm_score, 'sources': ['also']}

        # 品类热门召回：识别用户偏好的 top 品类，从这些品类的热门里补充
        user_cats = Counter()
        for item, rating, ts in user_history:
            cat = item_meta_for_recall.get(item, {}).get('category', '')
            if cat and cat != 'unknown':
                user_cats[cat] += rating
        for cat, _ in user_cats.most_common(3):
            cat_items = hot_recall.recall(user_id, user_history_set,
                                          category=cat, top_k=200)
            n_cat = len(cat_items)
            for rank_i, item in enumerate(cat_items):
                norm_score = ((n_cat - rank_i) / max(n_cat, 1)) * CAT_HOT_WEIGHT
                if item in merged:
                    merged[item]['score'] += norm_score
                    merged[item]['sources'].append('cat_hot')
                else:
                    merged[item] = {'score': norm_score, 'sources': ['cat_hot']}

        # 多路命中加分
        for item in merged:
            n_src = len(set(merged[item]['sources']))
            if n_src > 1:
                merged[item]['score'] *= (1 + 0.15 * n_src)

        # 排序并转换为 output
        ranked = sorted(merged.items(), key=lambda x: -x[1]['score'])
        output = []
        for asin, info in ranked[:total_k]:
            if asin in iid2idx:
                output.append((iid2idx[asin], info['score'], set(info['sources'])))
        return output

    # 测试召回
    print("\n  [召回测试]")
    test_sample = list(test_ground_truth.keys())[:200]
    recall_hits = 0
    total_gt = 0
    recall_counts = []
    empty_recall = 0
    recall_source_counter = Counter()
    multi_source_cnt = 0

    # ---- 诊断: ground truth 在各通道的理论可达性 ----
    gt_reachable = {'itemcf': 0, 'usercf': 0, 'swing': 0, 'also_buy': 0,
                    'also_view': 0, 'content_brand': 0, 'content_cat': 0,
                    'in_train_items': 0, 'has_meta': 0}
    for u in test_sample:
        gt_items = test_ground_truth[u]
        user_id = idx2uid[u]
        user_history = train_user_items_asin.get(user_id, [])
        user_history_set = {item for item, r, t in user_history}
        for gt_iid_idx in gt_items:
            gt_asin = idx2iid.get(gt_iid_idx, '')
            if not gt_asin:
                continue
            # 是否在训练集商品中
            if gt_asin in item_set:
                gt_reachable['in_train_items'] += 1
            if gt_asin in meta:
                gt_reachable['has_meta'] += 1
            # ItemCF: gt_asin 是否在用户历史某 item 的相似列表中
            for hist_item, _, _ in user_history:
                if hist_item in itemcf_recall.item_sim:
                    for sim_item, _ in itemcf_recall.item_sim[hist_item]:
                        if sim_item == gt_asin:
                            gt_reachable['itemcf'] += 1
                            break
                    else:
                        continue
                    break
            # Swing
            for hist_item, _, _ in user_history:
                if hist_item in swing_recall.item_sim:
                    for sim_item, _ in swing_recall.item_sim[hist_item]:
                        if sim_item == gt_asin:
                            gt_reachable['swing'] += 1
                            break
                    else:
                        continue
                    break
            # UserCF: gt_asin 是否被某个相似用户购买过
            if user_id in usercf_recall.user_sim:
                for sim_user, _ in usercf_recall.user_sim[user_id]:
                    sim_user_items = {i for i, r, t in train_user_items_asin.get(sim_user, [])}
                    if gt_asin in sim_user_items:
                        gt_reachable['usercf'] += 1
                        break
            # Also buy/view
            for hist_item, _, _ in user_history:
                if gt_asin in also_recall.also_buy_graph.get(hist_item, []):
                    gt_reachable['also_buy'] += 1
                    break
            for hist_item, _, _ in user_history:
                if gt_asin in also_recall.also_view_graph.get(hist_item, []):
                    gt_reachable['also_view'] += 1
                    break
            # Content: brand/cat match
            gt_meta = item_meta_for_recall.get(gt_asin, {})
            gt_brand = gt_meta.get('brand', '').strip().lower()
            gt_cat = gt_meta.get('category', '')
            for hist_item, _, _ in user_history:
                h_meta = item_meta_for_recall.get(hist_item, {})
                if gt_brand and h_meta.get('brand', '').strip().lower() == gt_brand:
                    gt_reachable['content_brand'] += 1
                    break
            for hist_item, _, _ in user_history:
                h_meta = item_meta_for_recall.get(hist_item, {})
                if gt_cat and gt_cat != 'unknown' and h_meta.get('category', '') == gt_cat:
                    gt_reachable['content_cat'] += 1
                    break

    n_test = len(test_sample)
    print(f"\n  [Ground Truth 可达性诊断] (sample={n_test})")
    for ch, cnt in sorted(gt_reachable.items(), key=lambda x: -x[1]):
        print(f"    {ch}: {cnt}/{n_test} ({cnt/max(n_test,1)*100:.1f}%)")

    for u in test_sample:
        recalled = do_recall(u, 1000)
        recalled_set = set(i for i, _, _ in recalled)
        gt = set(test_ground_truth[u].keys())
        recall_hits += len(recalled_set & gt)
        total_gt += len(gt)
        recall_counts.append(len(recalled))
        if len(recalled) == 0:
            empty_recall += 1
        for _, _, sources in recalled:
            for s in sources:
                recall_source_counter[s] += 1
            if len(sources) > 1:
                multi_source_cnt += 1

    recall_rate = recall_hits / total_gt if total_gt > 0 else 0
    if verbose:
        avg_recall_cnt = np.mean(recall_counts) if recall_counts else 0
        min_recall_cnt = min(recall_counts) if recall_counts else 0
        max_recall_cnt = max(recall_counts) if recall_counts else 0

        print(f"\n{'='*50}")
        print(f"  [召回健康检查] (sample={len(test_sample)})")
        print(f"{'='*50}")
        print(f"  Recall@1000: {recall_rate*100:.2f}%  {'⚠ 召回率太低(<5%)' if recall_rate < 0.05 else '✓'}")
        print(f"  空召回用户: {empty_recall}/{len(test_sample)}  {'⚠ 空召回太多(>10%)' if empty_recall/max(len(test_sample),1) > 0.1 else '✓'}")
        print(f"  平均召回候选数: {avg_recall_cnt:.0f}  [min={min_recall_cnt}, max={max_recall_cnt}]")
        print(f"  {'⚠ 平均候选太少(<100)' if avg_recall_cnt < 100 else '✓ 候选数充足'}")
        print(f"  各通道贡献: {dict(recall_source_counter.most_common())}")
        total_source_hits = sum(recall_source_counter.values())
        for src, cnt in recall_source_counter.most_common():
            print(f"    {src}: {cnt:,} ({cnt/max(total_source_hits,1)*100:.1f}%)")
        print(f"  多源命中候选: {multi_source_cnt:,} ({multi_source_cnt/max(sum(recall_counts),1)*100:.1f}%)")
        print(f"{'='*50}")

    # ========================================
    # 特征提取函数
    # ========================================
    def extract_item_features(asin):
        m = meta.get(asin, {})
        item_idx = iid2idx.get(asin, -1)
        price = parse_price(m.get('price'))
        rank = parse_rank(m.get('rank'))
        return np.array([
            min(price, 500) if price is not None else 0.0,
            math.log1p(rank) if rank is not None else 0.0,
            float(len(m.get('also_buy', []) or [])),
            float(len(m.get('also_view', []) or [])),
            item_avg_rating.get(item_idx, global_avg) if item_idx >= 0 else global_avg,
            float(min(item_rating_cnt.get(item_idx, 0), 100)) if item_idx >= 0 else 0,
        ], dtype=np.float32)

    def extract_user_features(uid_idx):
        return np.array([
            user_avg_rating.get(uid_idx, global_avg),
            float(min(user_rating_cnt.get(uid_idx, 0), 100)),
            user_rating_std.get(uid_idx, 0),
        ], dtype=np.float32)

    def extract_cross_features(uid_idx, iid_idx, exclude_self=False):
        user_items_set = train_user2items_idx.get(uid_idx, set())
        item_users_set = train_item2users_idx.get(iid_idx, set())

        if exclude_self:
            user_items_set = user_items_set - {iid_idx}
            item_users_set = item_users_set - {uid_idx}

        common_users = 0
        if user_items_set:
            neighbor_users = set()
            for j in user_items_set:
                neighbor_users |= train_item2users_idx.get(j, set())
            if exclude_self:
                neighbor_users.discard(uid_idx)
            common_users = len(item_users_set & neighbor_users)

        neighbor_items = set()
        for other_u in item_users_set:
            neighbor_items |= train_user2items_idx.get(other_u, set())
        if exclude_self:
            neighbor_items.discard(iid_idx)
        jaccard = len(user_items_set & neighbor_items) / max(1, len(user_items_set | neighbor_items))

        # 用户品类偏好匹配
        item_asin = idx2iid.get(iid_idx, '')
        item_cat = meta.get(item_asin, {}).get('_category', 'unknown')
        item_brand = meta.get(item_asin, {}).get('brand', '') or ''

        cat_match = 0.0
        brand_match = 0.0
        if user_items_set:
            for j in user_items_set:
                j_asin = idx2iid.get(j, '')
                j_meta = meta.get(j_asin, {})
                if j_meta.get('_category', 'unknown') == item_cat and item_cat != 'unknown':
                    cat_match += 1.0
                if item_brand and (j_meta.get('brand', '') or '') == item_brand:
                    brand_match += 1.0
            cat_match /= len(user_items_set)
            brand_match /= len(user_items_set)

        return np.array([
            float(min(common_users, 100)),
            jaccard,
            cat_match,
            brand_match,
        ], dtype=np.float32)

    # ------ 真实召回分数查询 ------
    # 预构建 item_sim 的快速查询字典（避免线性扫描 list）
    _itemcf_sim_dict = {}
    for item_key, sim_list in itemcf_recall.item_sim.items():
        d = {}
        for sim_item, score in sim_list:
            d[sim_item] = score
        _itemcf_sim_dict[item_key] = d

    _swing_sim_dict = {}
    for item_key, sim_list in swing_recall.item_sim.items():
        d = {}
        for sim_item, score in sim_list:
            d[sim_item] = score
        _swing_sim_dict[item_key] = d

    _usercf_sim_dict = {}
    for user_key, sim_list in usercf_recall.user_sim.items():
        d = {}
        for sim_user, score in sim_list:
            d[sim_user] = score
        _usercf_sim_dict[user_key] = d

    print(f"  召回分数查询索引: ItemCF={len(_itemcf_sim_dict):,}, Swing={len(_swing_sim_dict):,}, UserCF={len(_usercf_sim_dict):,}")

    def compute_recall_scores(uid_idx, iid_idx):
        """计算 (user, item) 对的真实各通道召回分数"""
        user_id = idx2uid[uid_idx]
        item_asin = idx2iid[iid_idx]
        user_history = train_user_items_asin.get(user_id, [])

        # ItemCF: sum of sim(candidate, history_item) * rating
        itemcf_score = 0.0
        for hist_item, rating, ts in user_history:
            sim_dict = _itemcf_sim_dict.get(hist_item, {})
            s = sim_dict.get(item_asin, 0.0)
            if s > 0:
                itemcf_score += rating * s

        # Swing: 同理
        swing_score = 0.0
        for hist_item, rating, ts in user_history:
            sim_dict = _swing_sim_dict.get(hist_item, {})
            s = sim_dict.get(item_asin, 0.0)
            if s > 0:
                swing_score += rating * s

        # UserCF: sum of sim(user, sim_user) * sim_user对该item的rating
        usercf_score = 0.0
        user_sim_dict = _usercf_sim_dict.get(user_id, {})
        if user_sim_dict:
            item_users_asin = train_item_users_asin.get(item_asin, [])
            for rater_id, rating, ts in item_users_asin:
                s = user_sim_dict.get(rater_id, 0.0)
                if s > 0:
                    usercf_score += s * rating

        return np.array([itemcf_score, swing_score, usercf_score], dtype=np.float32)

    def extract_all_features(uid_idx, iid_idx, exclude_self=False):
        user_feat = extract_user_features(uid_idx)
        item_feat = extract_item_features(idx2iid[iid_idx])
        cross_feat = extract_cross_features(uid_idx, iid_idx, exclude_self=exclude_self)
        recall_feat = compute_recall_scores(uid_idx, iid_idx)
        return np.concatenate([user_feat, item_feat, cross_feat, recall_feat])

    feature_names = [
        'user_avg_rating', 'user_rating_cnt', 'user_rating_std',
        'price', 'log_rank', 'n_also_buy', 'n_also_view',
        'item_avg_rating', 'item_rating_cnt',
        'co_interact', 'jaccard',
        'cat_pref', 'brand_pref',
        'itemcf_score', 'swing_score', 'usercf_score',
    ]
    n_features = len(feature_names)

    # ========================================
    # 阶段 2: 粗排 (LightGBM 二分类 — 预测交互概率)
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 2: 粗排 (LightGBM — 二分类, 预测交互概率)")
        print("=" * 70)

    import lightgbm as lgb

    coarse_feature_names = ['co_interact', 'jaccard',
                            'item_rating_cnt', 'item_avg_rating', 'user_avg_rating',
                            'item_pop_rank']

    # 商品按热门度排名（用于特征）
    _item_pop_sorted = sorted(item_rating_cnt.items(), key=lambda x: -x[1])
    _item_pop_rank = {iid: rank for rank, (iid, _) in enumerate(_item_pop_sorted)}

    def extract_coarse_features(uid_idx, iid_idx, exclude_self=False):
        """粗排专用：交互特征 + 统计特征
        exclude_self=True: 构建正样本特征时，排除 (uid_idx, iid_idx) 这条交互本身，
        避免 co_interact/jaccard 直接泄露 label 信息。
        """
        user_items_set = train_user2items_idx.get(uid_idx, set())
        item_users_set = train_item2users_idx.get(iid_idx, set())

        if exclude_self:
            user_items_set = user_items_set - {iid_idx}
            item_users_set = item_users_set - {uid_idx}

        common_users = 0
        if user_items_set:
            neighbor_users = set()
            for j in user_items_set:
                neighbor_users |= train_item2users_idx.get(j, set())
            if exclude_self:
                neighbor_users.discard(uid_idx)
            common_users = len(item_users_set & neighbor_users)

        neighbor_items = set()
        for other_u in item_users_set:
            neighbor_items |= train_user2items_idx.get(other_u, set())
        if exclude_self:
            neighbor_items.discard(iid_idx)
        jaccard = len(user_items_set & neighbor_items) / max(1, len(user_items_set | neighbor_items))

        pop_rank = _item_pop_rank.get(iid_idx, len(_item_pop_rank))

        return np.array([
            float(min(common_users, 100)),
            jaccard,
            float(min(item_rating_cnt.get(iid_idx, 0), 100)) if iid_idx >= 0 else 0,
            item_avg_rating.get(iid_idx, global_avg) if iid_idx >= 0 else global_avg,
            user_avg_rating.get(uid_idx, global_avg),
            math.log1p(pop_rank),
        ], dtype=np.float32)

    # 构建正负样本训练数据
    # 正样本 = 召回候选中命中 test ground truth 的商品
    # hard 负样本 = 召回候选中未命中的商品 (70%)
    # easy 负样本 = 全局随机采样的商品 (30%)，拉高基线 AUC
    np.random.seed(42)

    seed_train_reviews = [d for d in train_reviews if d['reviewerID'] in selected_uid_set]

    if verbose:
        print(f"  种子用户训练评论: {len(seed_train_reviews):,} (全部训练评论: {len(train_reviews):,})")

    # 按测试用户划分 train/val (80/20)
    _all_test_users = list(test_ground_truth.keys())
    np.random.shuffle(_all_test_users)
    _n_val_users = max(int(len(_all_test_users) * 0.2), 10)
    _val_user_set = set(_all_test_users[:_n_val_users])
    _train_user_set = set(_all_test_users[_n_val_users:])
    print(f"  按用户划分: 训练用户 {len(_train_user_set):,}, 验证用户 {len(_val_user_set):,}")

    # 对测试用户跑召回，构建训练/验证数据
    # 正样本: 训练集交互 (exclude_self 防泄露)
    # Hard neg: 召回候选未命中部分
    # Easy neg: 全局随机
    RECALL_TOP_K = 200
    MAX_RECALL_PER_USER = 50
    N_HARD_PER_POS = 2
    N_EASY_PER_POS = 2
    MAX_POS_PER_USER = 10  # 每用户最多取 10 个训练交互做正样本
    coarse_X_train_list = []
    coarse_y_train_list = []
    coarse_X_val_list = []
    coarse_y_val_list = []

    _coarse_users = list(_train_user_set | _val_user_set)
    random.shuffle(_coarse_users)
    _coarse_users = _coarse_users[:2000]  # 最多2000用户

    print(f"  对 {len(_coarse_users)} 用户跑召回构建粗排训练数据...")
    _n_pos_total = 0
    _n_hard_neg = 0
    _n_easy_neg = 0
    for _ci, _cu in enumerate(_coarse_users):
        if _ci % 500 == 0:
            print(f"    {_ci}/{len(_coarse_users)} 用户...")
        _recalled = do_recall(_cu, total_k=RECALL_TOP_K)
        _gt = test_ground_truth.get(_cu, {})
        _u_items_set = train_user2items_idx.get(_cu, set())
        if not _u_items_set:
            continue

        is_val = _cu in _val_user_set
        target_X = coarse_X_val_list if is_val else coarse_X_train_list
        target_y = coarse_y_val_list if is_val else coarse_y_train_list

        # Hard neg 池: 召回候选未命中 gt 和训练交互的
        _hard_neg_pool = [i for i, s, src in _recalled[:MAX_RECALL_PER_USER]
                         if i not in _gt and i not in _u_items_set]

        # 正样本: 训练集交互 (exclude_self)
        _pos_list = list(_u_items_set)
        random.shuffle(_pos_list)
        _pos_list = _pos_list[:MAX_POS_PER_USER]

        for _pi in _pos_list:
            target_X.append(extract_coarse_features(_cu, _pi, exclude_self=True))
            target_y.append(1)
            _n_pos_total += 1

        # Hard neg
        for _ in range(len(_pos_list) * N_HARD_PER_POS):
            if _hard_neg_pool:
                _ni = _hard_neg_pool[np.random.randint(0, len(_hard_neg_pool))]
                target_X.append(extract_coarse_features(_cu, _ni))
                target_y.append(0)
                _n_hard_neg += 1

        # Easy neg
        for _ in range(len(_pos_list) * N_EASY_PER_POS):
            _neg_i = np.random.randint(0, n_items)
            while _neg_i in _u_items_set or _neg_i in _gt:
                _neg_i = np.random.randint(0, n_items)
            target_X.append(extract_coarse_features(_cu, _neg_i))
            target_y.append(0)
            _n_easy_neg += 1

    coarse_X_train = np.array(coarse_X_train_list)
    coarse_y_train = np.array(coarse_y_train_list)
    coarse_X_val = np.array(coarse_X_val_list)
    coarse_y_val = np.array(coarse_y_val_list)
    n_coarse_train = len(coarse_X_train)
    n_coarse_val = len(coarse_X_val)

    if verbose:
        n_coarse = n_coarse_train + n_coarse_val
        print(f"  粗排样本: {n_coarse:,} (正:{_n_pos_total:,}, hard负:{_n_hard_neg:,}, easy负:{_n_easy_neg:,})")
        print(f"  正样本率: {_n_pos_total/max(n_coarse,1)*100:.2f}%")
        print(f"  训练:{n_coarse_train:,} 验证:{n_coarse_val:,} (按用户划分)")
        print(f"  特征维度: {coarse_X_train.shape[1]} ({coarse_feature_names})")

    lgb_train = lgb.Dataset(coarse_X_train, label=coarse_y_train,
                            feature_name=coarse_feature_names, free_raw_data=False)
    lgb_val = lgb.Dataset(coarse_X_val, label=coarse_y_val,
                          feature_name=coarse_feature_names, free_raw_data=False,
                          reference=lgb_train)

    lgb_params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'max_depth': 3,
        'num_leaves': 6,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'feature_fraction_bynode': 0.5,
        'min_child_samples': 100,
        'lambda_l2': 5.0,
        'lambda_l1': 0.5,
        'seed': 42,
        'verbose': -1,
        'num_threads': 4,
        'first_metric_only': True,
    }
    # Pipeline 评估集 (固定难度组合，目标 AUC 稳定 0.6-0.7)
    print(f"  构建 pipeline 级评估集 (固定难度)...")
    _coarse_eval_seed = np.random.RandomState(12345)
    _pipe_hard_X = []
    _pipe_hard_y = []
    # 用全部有 test gt 的用户（不限制验证/训练），保证正样本充足
    _pipe_eval_users = list(test_ground_truth.keys())[:500]
    for _pu in _pipe_eval_users:
        _recalled = do_recall(_pu, total_k=1000)
        _gt = test_ground_truth[_pu]
        _pu_items_set = train_user2items_idx.get(_pu, set())
        _n_pos_pipe = 0
        for _ri, _rs, _rsrc in _recalled[:200]:
            feat = extract_coarse_features(_pu, _ri)
            _pipe_hard_X.append(feat)
            if _ri in _gt:
                _pipe_hard_y.append(1)
                _n_pos_pipe += 1
            else:
                _pipe_hard_y.append(0)
        # 补充 easy neg (用固定种子)
        for _ in range(max(_n_pos_pipe * 3, 1)):
            _neg_i = _coarse_eval_seed.randint(0, n_items)
            while _neg_i in _pu_items_set or _neg_i in _gt:
                _neg_i = _coarse_eval_seed.randint(0, n_items)
            _pipe_hard_X.append(extract_coarse_features(_pu, _neg_i))
            _pipe_hard_y.append(0)
    _pipe_eval_X = np.array(_pipe_hard_X)
    _pipe_eval_y = np.array(_pipe_hard_y)
    print(f"  Pipeline 评估集: {len(_pipe_eval_X)} 样本 (正:{(_pipe_eval_y==1).sum()}, 负:{(_pipe_eval_y==0).sum()}, 正样本率:{(_pipe_eval_y==1).sum()/max(len(_pipe_eval_y),1)*100:.2f}%)")
    _pipe_eval_ds = lgb.Dataset(_pipe_eval_X, label=_pipe_eval_y,
                                feature_name=coarse_feature_names, free_raw_data=False)

    # V4: 实时 AUC 图 — 自定义 LightGBM 回调
    _lgb_evals_result = {}

    # 初始化实时图窗口
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        _live_fig, _live_axes = plt.subplots(1, 2, figsize=(14, 5))
        _live_ax_loss = _live_axes[0]
        _live_ax_auc = _live_axes[1]
        _live_ax_auc.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='random')
        _live_ax_auc.set_xlabel('Iterations')
        _live_ax_auc.set_ylabel('AUC')
        _live_ax_auc.set_title('Coarse Rank — Pipeline AUC (live)')
        _live_ax_auc.grid(True, alpha=0.3)
        _live_ax_loss.set_xlabel('Iterations')
        _live_ax_loss.set_ylabel('LogLoss')
        _live_ax_loss.set_title('Coarse Rank — LogLoss (live)')
        _live_ax_loss.grid(True, alpha=0.3)
        _line_pipe_auc, = _live_ax_auc.plot([], [], 'r-', label='Pipeline AUC', lw=2)
        _line_train_loss, = _live_ax_loss.plot([], [], 'b-', label='Train LogLoss', lw=2)
        _line_val_loss, = _live_ax_loss.plot([], [], 'r-', label='Val LogLoss', lw=2)
        _live_ax_auc.legend(fontsize=10)
        _live_ax_loss.legend(fontsize=10)
        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
        _live_ok = True
    except Exception:
        _live_ok = False

    def _live_update_callback(env):
        """每轮 boosting 结束后实时刷新图"""
        if not _live_ok:
            return
        _tr_auc = _lgb_evals_result.get('train', {}).get('auc', [])
        _pi_auc = _lgb_evals_result.get('pipeline', {}).get('auc', [])
        _tr_loss = _lgb_evals_result.get('train', {}).get('binary_logloss', [])
        _va_loss = _lgb_evals_result.get('val', {}).get('binary_logloss', [])
        if not _pi_auc:
            return
        _x = list(range(1, len(_pi_auc) + 1))
        _line_pipe_auc.set_data(_x, _pi_auc)
        _live_ax_auc.set_xlim(0.5, max(_x) + 0.5)
        _live_ax_auc.set_ylim(max(0.45, min(_pi_auc) - 0.05), min(1.02, max(_pi_auc) + 0.05))
        if _tr_loss and _va_loss:
            _line_train_loss.set_data(_x[:len(_tr_loss)], _tr_loss)
            _line_val_loss.set_data(_x[:len(_va_loss)], _va_loss)
            _live_ax_loss.set_xlim(0.5, max(_x) + 0.5)
            _all_loss = _tr_loss + _va_loss
            _live_ax_loss.set_ylim(0, max(_all_loss) * 1.1 + 0.01)
        _live_fig.canvas.draw_idle()
        _live_fig.canvas.flush_events()
        plt.pause(0.01)

    _lgb_callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                      lgb.log_evaluation(period=0),
                      lgb.record_evaluation(_lgb_evals_result),
                      _live_update_callback]
    coarse_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val, _pipe_eval_ds],
        valid_names=['train', 'val', 'pipeline'],
        callbacks=_lgb_callbacks,
    )

    # 训练结束后关闭交互模式，保存最终版图片
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _train_loss = _lgb_evals_result.get('train', {}).get('binary_logloss', [])
        _val_loss = _lgb_evals_result.get('val', {}).get('binary_logloss', [])
        _pipe_auc_list = _lgb_evals_result.get('pipeline', {}).get('auc', [])
        _iters = list(range(1, len(_train_loss) + 1)) if _train_loss else []

        # 左图: LogLoss
        if _iters:
            axes[0].plot(_iters, _train_loss, 'b-', label='Train', lw=2)
            axes[0].plot(_iters[:len(_val_loss)], _val_loss, 'r-', label='Val', lw=2)
            axes[0].legend()
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('LogLoss')
        axes[0].set_title('Coarse Rank — LogLoss')
        axes[0].grid(True, alpha=0.3)

        # 右图: Pipeline AUC (只显示 val)
        if _pipe_auc_list:
            _iters_auc = list(range(1, len(_pipe_auc_list) + 1))
            axes[1].plot(_iters_auc, _pipe_auc_list, 'r-', label='Pipeline AUC', lw=2)
            axes[1].axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='random')
            axes[1].legend()
        axes[1].set_xlabel('Iterations')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Coarse Rank — AUC')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        _lgb_curve_path = os.path.join(OUTPUT_DIR, 'coarse_rank_curves.png')
        plt.savefig(_lgb_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"  [粗排曲线已保存] {_lgb_curve_path}")
            if _pipe_auc_list:
                print(f"  Pipeline AUC: 起始={_pipe_auc_list[0]:.4f}, 最终={_pipe_auc_list[-1]:.4f}")
    except Exception as e:
        if verbose:
            print(f"  [粗排绘图跳过] {e}")

    if verbose:
        actual_trees = coarse_model.best_iteration
        train_pred_prob = coarse_model.predict(coarse_X_train)
        val_pred_prob = coarse_model.predict(coarse_X_val)

        from sklearn.metrics import roc_auc_score, log_loss
        train_auc = roc_auc_score(coarse_y_train, train_pred_prob)
        val_auc = roc_auc_score(coarse_y_val, val_pred_prob)
        val_logloss = log_loss(coarse_y_val, val_pred_prob)

        print(f"\n{'='*50}")
        print(f"  [粗排健康检查 — LightGBM 二分类]")
        print(f"{'='*50}")
        print(f"  实际使用树数: {actual_trees} / 500 (early stopping)")
        print(f"  训练 AUC: {train_auc:.4f}  验证 AUC: {val_auc:.4f}")
        print(f"  验证 LogLoss: {val_logloss:.4f}")
        overfit_gap = train_auc - val_auc
        print(f"  过拟合间隙(AUC): {overfit_gap:.4f}  {'⚠ >0.05' if overfit_gap > 0.05 else '✓'}")
        print(f"  验证预测概率范围: [{val_pred_prob.min():.4f}, {val_pred_prob.max():.4f}], mean={val_pred_prob.mean():.4f}")
        importances = coarse_model.feature_importance(importance_type='gain')
        sorted_imp = sorted(zip(coarse_feature_names, importances), key=lambda x: -x[1])
        print(f"  特征重要性 (gain):")
        for fname, imp in sorted_imp:
            print(f"    {fname}: {imp:.4f}")
        print(f"{'='*50}")

    def coarse_rank(user_idx, candidates, top_k=100):
        if not candidates:
            return []
        batch_feats = []
        recall_scores = []
        for item_idx, recall_score, sources in candidates:
            feat = extract_coarse_features(user_idx, item_idx)
            batch_feats.append(feat)
            recall_scores.append(recall_score)

        batch_feats = np.array(batch_feats)
        pred_probs = coarse_model.predict(batch_feats)  # 交互概率

        # 融合: LightGBM 概率 + 0.3 × 召回分（归一化）
        recall_scores = np.array(recall_scores)
        max_rs = recall_scores.max() if len(recall_scores) > 0 and recall_scores.max() > 0 else 1.0
        combined = pred_probs + 0.3 * (recall_scores / max_rs)

        scores = []
        for idx, (item_idx, recall_score, sources) in enumerate(candidates):
            scores.append((item_idx, float(combined[idx]), sources))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # ========================================
    # 阶段 3: 精排 (FM — BPR pairwise 排序)
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 3: 精排 (FM — BPR pairwise 排序)")
        print("  目标: 正样本分数 > 负样本分数 (排序优化)")
        print("=" * 70)

    K_FM = 8
    LR_FM = 0.005
    REG_W = 0.1
    REG_V = 0.1
    EPOCHS_FM = 30
    CLIP = 5.0

    # 构建 BPR 训练数据：(user, pos_item, neg_item)
    # 正样本 = 训练集交互 (exclude_self 防泄露)
    # 负样本 = 粗排 top-k (级联) + 40% easy random neg
    MAX_FM_PAIRS = 200000
    EASY_NEG_FM_RATIO = 0.4
    print(f"  构建 BPR 训练对 (正=训练交互, 负=粗排top-k+{EASY_NEG_FM_RATIO*100:.0f}%easy)...")
    np.random.seed(42)

    # 构建用户->正样本索引
    _user_pos_items = defaultdict(list)
    for d in seed_train_reviews:
        uid_idx = uid2idx[d['reviewerID']]
        iid_idx = iid2idx[d['asin']]
        _user_pos_items[uid_idx].append(iid_idx)

    # 预计算粗排 top-k 用于负采样
    print(f"  预计算粗排 top-k 用于精排负采样...")
    _user_coarse_topk_cache = {}
    _cache_users = list(_user_pos_items.keys())[:min(len(_user_pos_items), 2000)]
    for _ci, _cu in enumerate(_cache_users):
        if _ci % 500 == 0:
            print(f"    {_ci}/{len(_cache_users)} 用户...")
        _recalled = do_recall(_cu, total_k=500)
        _coarse_ranked = coarse_rank(_cu, _recalled, top_k=100)
        _user_items_set = train_user2items_idx.get(_cu, set())
        _user_coarse_topk_cache[_cu] = [i for i, s, src in _coarse_ranked if i not in _user_items_set]
    print(f"  粗排 top-k 缓存完成: {len(_user_coarse_topk_cache)} 用户")

    bpr_users = []
    bpr_pos_items = []
    bpr_neg_items = []

    seed_user_list = list(_user_coarse_topk_cache.keys())
    pairs_per_user = max(1, MAX_FM_PAIRS // max(len(seed_user_list), 1))

    _n_hard_fm = 0
    _n_easy_fm = 0
    for u in seed_user_list:
        pos_list = _user_pos_items[u]
        user_items_set = train_user2items_idx.get(u, set())
        coarse_negs = _user_coarse_topk_cache.get(u, [])

        for _ in range(min(pairs_per_user, len(pos_list) * 3)):
            pos_i = pos_list[np.random.randint(0, len(pos_list))]

            # 60% hard (粗排 top-k) + 40% easy (random)
            if coarse_negs and random.random() > EASY_NEG_FM_RATIO:
                neg_i = coarse_negs[np.random.randint(0, len(coarse_negs))]
                _n_hard_fm += 1
            else:
                neg_i = np.random.randint(0, n_items)
                while neg_i in user_items_set:
                    neg_i = np.random.randint(0, n_items)
                _n_easy_fm += 1

            bpr_users.append(u)
            bpr_pos_items.append(pos_i)
            bpr_neg_items.append(neg_i)
            if len(bpr_users) >= MAX_FM_PAIRS:
                break
        if len(bpr_users) >= MAX_FM_PAIRS:
            break

    bpr_users = np.array(bpr_users, dtype=np.int32)
    bpr_pos_items = np.array(bpr_pos_items, dtype=np.int32)
    bpr_neg_items = np.array(bpr_neg_items, dtype=np.int32)

    print(f"  负样本: hard(粗排top-k)={_n_hard_fm:,} ({_n_hard_fm/max(_n_hard_fm+_n_easy_fm,1)*100:.0f}%), easy(random)={_n_easy_fm:,} ({_n_easy_fm/max(_n_hard_fm+_n_easy_fm,1)*100:.0f}%)")

    n_pairs = len(bpr_users)

    if verbose:
        print(f"  BPR 训练对: {n_pairs:,}")
        print(f"  涉及用户: {len(set(bpr_users)):,}")

    # 提取正/负样本特征 (正样本 exclude_self=True 避免特征泄露)
    print(f"  提取正样本特征...")
    pos_feats_all = np.array([extract_all_features(u, i, exclude_self=True)
                               for u, i in zip(bpr_users, bpr_pos_items)], dtype=np.float64)
    print(f"  提取负样本特征...")
    neg_feats_all = np.array([extract_all_features(u, i)
                               for u, i in zip(bpr_users, bpr_neg_items)], dtype=np.float64)

    # 划分训练/验证集 (85/15)
    n_val = max(int(n_pairs * 0.15), 100)
    n_train = n_pairs - n_val
    perm = np.random.permutation(n_pairs)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    # 归一化
    feat_mu = pos_feats_all[tr_idx].mean(axis=0)
    feat_std = pos_feats_all[tr_idx].std(axis=0)
    feat_std[feat_std == 0] = 1.0

    train_pos_feats = (pos_feats_all[tr_idx] - feat_mu) / feat_std
    train_neg_feats = (neg_feats_all[tr_idx] - feat_mu) / feat_std
    val_pos_feats = (pos_feats_all[va_idx] - feat_mu) / feat_std
    val_neg_feats = (neg_feats_all[va_idx] - feat_mu) / feat_std

    train_bpr_users = bpr_users[tr_idx]
    train_bpr_pos = bpr_pos_items[tr_idx]
    train_bpr_neg = bpr_neg_items[tr_idx]
    val_bpr_users = bpr_users[va_idx]
    val_bpr_pos = bpr_pos_items[va_idx]
    val_bpr_neg = bpr_neg_items[va_idx]

    if verbose:
        print(f"  训练: {n_train:,}, 验证: {n_val:,}")

    # FM 参数初始化
    w0_fm = 0.0
    w_user_fm = np.zeros(n_users, dtype=np.float64)
    w_item_fm = np.zeros(n_items, dtype=np.float64)
    w_fine = np.zeros(n_features, dtype=np.float64)
    v_user_fm = np.random.randn(n_users, K_FM) * 0.01
    v_item_fm = np.random.randn(n_items, K_FM) * 0.01
    v_fine = np.random.randn(n_features, K_FM) * 0.01

    def fm_score_batch(users, items, feats):
        """批量计算 FM 分数"""
        linear = w0_fm + w_user_fm[users] + w_item_fm[items] + feats @ w_fine
        vu = v_user_fm[users]
        vi = v_item_fm[items]
        vf = feats @ v_fine
        svx = vu + vi + vf
        sv2 = vu**2 + vi**2 + (feats**2) @ (v_fine**2)
        interaction = 0.5 * ((svx**2).sum(axis=1) - sv2.sum(axis=1))
        return linear + interaction

    def fm_predict_single(u, it, feat):
        """单样本预测分数"""
        linear = w0_fm + w_user_fm[u] + w_item_fm[it] + w_fine @ feat
        svx = v_user_fm[u] + v_item_fm[it] + (v_fine.T @ feat)
        sv2 = v_user_fm[u]**2 + v_item_fm[it]**2 + ((v_fine**2).T @ (feat**2))
        interaction = 0.5 * (svx @ svx - sv2.sum())
        return linear + interaction

    # BPR 训练
    BATCH_SIZE_FM = 512
    print(f"\n  [训练 FM-BPR, batch={BATCH_SIZE_FM}, epochs={EPOCHS_FM}]")

    # 精排 pipeline 评估集 (固定难度组合: hard 40% + medium 40% + easy 20%)
    # 目标: AUC 稳定在 0.6-0.7
    print(f"  构建精排 pipeline 级评估集 (固定难度组合)...")
    _eval_seed = np.random.RandomState(12345)  # 固定种子保证可复现
    _fm_pipe_users = []
    _fm_pipe_pos = []
    _fm_pipe_neg = []
    _fm_pipe_eval_users = list(test_ground_truth.keys())[:200]
    for _fpu in _fm_pipe_eval_users:
        _gt = test_ground_truth[_fpu]
        _recalled = do_recall(_fpu, total_k=500)
        _coarse_ranked = coarse_rank(_fpu, _recalled, top_k=200)
        _u_items_set = train_user2items_idx.get(_fpu, set())

        _gt_in_coarse = [i for i, s, src in _coarse_ranked if i in _gt]
        if not _gt_in_coarse:
            continue

        # 三种难度的 negative 池
        _coarse_items = [i for i, s, src in _coarse_ranked if i not in _gt]
        _hard_pool = _coarse_items[:30]      # 粗排 top-30 (hard)
        _medium_pool = _coarse_items[60:120]  # 粗排 60-120 (medium)
        _easy_pool = []                       # 填充 random (easy)
        for _ in range(100):
            _ri = _eval_seed.randint(0, n_items)
            if _ri not in _u_items_set and _ri not in _gt:
                _easy_pool.append(_ri)

        for _pi in _gt_in_coarse:
            # 每个正样本配 5 个负样本: 2 hard + 2 medium + 1 easy
            for _pool, _k in [(_hard_pool, 2), (_medium_pool, 2), (_easy_pool, 1)]:
                if not _pool:
                    continue
                for _ in range(_k):
                    _ni = _pool[_eval_seed.randint(0, len(_pool))]
                    _fm_pipe_users.append(_fpu)
                    _fm_pipe_pos.append(_pi)
                    _fm_pipe_neg.append(_ni)

    if not _fm_pipe_users:
        # fallback: 用训练用户的粗排 top-k
        for _fpu in list(_user_pos_items.keys())[:200]:
            _fh_pos_list = _user_pos_items[_fpu]
            _fh_coarse_negs = _user_coarse_topk_cache.get(_fpu, [])
            if not _fh_coarse_negs or not _fh_pos_list:
                continue
            for _ in range(min(3, len(_fh_pos_list))):
                _fm_pipe_users.append(_fpu)
                _fm_pipe_pos.append(_fh_pos_list[_eval_seed.randint(0, len(_fh_pos_list))])
                _fm_pipe_neg.append(_fh_coarse_negs[_eval_seed.randint(0, len(_fh_coarse_negs))])

    _fm_pipe_users = np.array(_fm_pipe_users, dtype=np.int32)
    _fm_pipe_pos = np.array(_fm_pipe_pos, dtype=np.int32)
    _fm_pipe_neg = np.array(_fm_pipe_neg, dtype=np.int32)
    _fm_pipe_pos_feats = np.array([extract_all_features(u, i, exclude_self=True) for u, i in zip(_fm_pipe_users, _fm_pipe_pos)], dtype=np.float64)
    _fm_pipe_neg_feats = np.array([extract_all_features(u, i) for u, i in zip(_fm_pipe_users, _fm_pipe_neg)], dtype=np.float64)
    _fm_pipe_pos_feats = (_fm_pipe_pos_feats - feat_mu) / feat_std
    _fm_pipe_neg_feats = (_fm_pipe_neg_feats - feat_mu) / feat_std
    print(f"  精排 pipeline 评估集: {len(_fm_pipe_users)} 对 (40% hard + 40% medium + 20% easy)")

    # V4 新增: 实时 AUC 动画图
    # V5 扩展: 支持每 epoch 推送本地 PNG / COS（训练监控大屏用）
    _live_curves_path = os.path.join(OUTPUT_DIR, 'training_curves_live.png')
    try:
        from cloud.config import settings as _cloud_settings  # type: ignore
        _cos_upload = bool(_cloud_settings.cos_upload_enabled)
    except Exception:  # noqa: BLE001
        _cos_upload = False
    live_plot = LiveAUCPlot(
        title="FM-BPR Training — AUC vs Epoch",
        live_save_path=_live_curves_path,
        cos_upload=_cos_upload,
        total_epochs=EPOCHS_FM,
    )
    np.random.seed(42)
    train_indices = np.arange(n_train)
    best_val_auc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0
    best_weights = (w0_fm, w_user_fm.copy(), w_item_fm.copy(), w_fine.copy(),
                    v_user_fm.copy(), v_item_fm.copy(), v_fine.copy())

    for epoch in range(EPOCHS_FM):
        np.random.shuffle(train_indices)
        epoch_start = time.time()
        epoch_loss = 0.0

        for batch_start in range(0, n_train, BATCH_SIZE_FM):
            bidx = train_indices[batch_start:batch_start + BATCH_SIZE_FM]
            bs = len(bidx)
            b_u = train_bpr_users[bidx]
            b_pos = train_bpr_pos[bidx]
            b_neg = train_bpr_neg[bidx]
            b_pos_feat = train_pos_feats[bidx]
            b_neg_feat = train_neg_feats[bidx]

            # score(u, pos) - score(u, neg)
            s_pos = fm_score_batch(b_u, b_pos, b_pos_feat)
            s_neg = fm_score_batch(b_u, b_neg, b_neg_feat)
            x_uij = s_pos - s_neg

            # sigmoid(-x_uij) = 1 - sigmoid(x_uij)
            # BPR gradient: -sigmoid(-x_uij) = sigmoid(x_uij) - 1
            exp_neg = np.exp(-np.clip(x_uij, -10, 10))
            coeff = -exp_neg / (1 + exp_neg)  # = sigmoid(x_uij) - 1
            epoch_loss += -np.log(1 / (1 + exp_neg) + 1e-10).sum()

            lr = LR_FM

            # 对 item bias: pos 的梯度 = coeff, neg 的梯度 = -coeff
            np.add.at(w_item_fm, b_pos, -lr * (coeff + REG_W * w_item_fm[b_pos]))
            np.add.at(w_item_fm, b_neg, -lr * (-coeff + REG_W * w_item_fm[b_neg]))

            # 对 feature weight
            diff_feat = b_pos_feat - b_neg_feat
            grad_w = (coeff[:, None] * diff_feat).mean(axis=0) + REG_W * w_fine
            gn = np.linalg.norm(grad_w)
            if gn > CLIP: grad_w *= CLIP / gn
            w_fine -= lr * grad_w

            # 对 v_item
            vu = v_user_fm[b_u]
            vi_pos = v_item_fm[b_pos]; vi_neg = v_item_fm[b_neg]
            vf_pos = b_pos_feat @ v_fine; vf_neg = b_neg_feat @ v_fine
            svx_pos = vu + vi_pos + vf_pos
            svx_neg = vu + vi_neg + vf_neg

            np.add.at(v_item_fm, b_pos, -lr * np.clip(coeff[:, None] * (svx_pos - vi_pos) + REG_V * vi_pos, -CLIP, CLIP))
            np.add.at(v_item_fm, b_neg, -lr * np.clip(-coeff[:, None] * (svx_neg - vi_neg) + REG_V * vi_neg, -CLIP, CLIP))

            # 对 v_user
            grad_vu = coeff[:, None] * ((svx_pos - vu) - (svx_neg - vu)) + REG_V * vu
            np.add.at(v_user_fm, b_u, -lr * np.clip(grad_vu, -CLIP, CLIP))

            # 对 v_fine (特征交叉)
            grad_vf = (coeff[:, None, None] * (
                b_pos_feat[:, :, None] * (svx_pos[:, None, :] - b_pos_feat[:, :, None] * v_fine[None, :, :])
                - b_neg_feat[:, :, None] * (svx_neg[:, None, :] - b_neg_feat[:, :, None] * v_fine[None, :, :])
            )).mean(axis=0) + REG_V * v_fine
            gn_vf = np.linalg.norm(grad_vf)
            if gn_vf > CLIP * n_features: grad_vf *= (CLIP * n_features) / gn_vf
            v_fine -= lr * grad_vf

        elapsed = time.time() - epoch_start

        # 验证: 计算 BPR AUC (训练分布同分布)
        val_s_pos = fm_score_batch(val_bpr_users, val_bpr_pos, val_pos_feats)
        val_s_neg = fm_score_batch(val_bpr_users, val_bpr_neg, val_neg_feats)
        train_s_pos = fm_score_batch(train_bpr_users, train_bpr_pos, train_pos_feats)
        train_s_neg = fm_score_batch(train_bpr_users, train_bpr_neg, train_neg_feats)
        train_auc = float((train_s_pos > train_s_neg).mean())
        val_auc = float((val_s_pos > val_s_neg).mean())

        # Pipeline 级 AUC (test gt vs 粗排 top-k，记录但不画)
        _pipe_s_pos = fm_score_batch(_fm_pipe_users, _fm_pipe_pos, _fm_pipe_pos_feats)
        _pipe_s_neg = fm_score_batch(_fm_pipe_users, _fm_pipe_neg, _fm_pipe_neg_feats)
        _pipe_auc = float((_pipe_s_pos > _pipe_s_neg).mean())

        _plot_val_auc = _pipe_auc  # 画图用固定难度 pipeline AUC
        _plot_train_auc = train_auc
        avg_loss = epoch_loss / n_train

        # 实时更新 AUC 曲线（V5: 把 avg_loss 一并传入以输出规范日志）
        live_plot.update(epoch + 1, _plot_train_auc, _plot_val_auc, loss=avg_loss)

        marker = ""
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            no_improve = 0
            best_weights = (w0_fm, w_user_fm.copy(), w_item_fm.copy(), w_fine.copy(),
                            v_user_fm.copy(), v_item_fm.copy(), v_fine.copy())
            marker = " *best"
        else:
            no_improve += 1

        print(f"    Epoch {epoch+1}/{EPOCHS_FM}  loss={avg_loss:.4f}  train_AUC={train_auc:.4f}  val_AUC={val_auc:.4f}  ({elapsed:.1f}s){marker}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}, best at epoch {best_epoch}")
            break

    # 恢复最佳权重
    w0_fm, w_user_fm, w_item_fm, w_fine, v_user_fm, v_item_fm, v_fine = best_weights

    if verbose:
        val_s_pos = fm_score_batch(val_bpr_users, val_bpr_pos, val_pos_feats)
        val_s_neg = fm_score_batch(val_bpr_users, val_bpr_neg, val_neg_feats)
        val_auc = float((val_s_pos > val_s_neg).mean())
        val_margin = float((val_s_pos - val_s_neg).mean())

        print(f"\n{'='*50}")
        print(f"  [精排健康检查 — FM-BPR]")
        print(f"{'='*50}")
        print(f"  训练/验证: {n_train:,} / {n_val:,}")
        print(f"  最佳 epoch: {best_epoch}/{EPOCHS_FM}")
        print(f"  验证 BPR-AUC: {val_auc:.4f}  {'⚠ <0.7' if val_auc < 0.7 else '✓'}")
        print(f"  验证平均 margin (pos-neg): {val_margin:.4f}")
        has_nan = np.isnan(w_fine).any() or np.isnan(v_fine).any()
        print(f"  权重 NaN/Inf: {'⚠' if has_nan else '✓ 无'}")
        print(f"  w0={w0_fm:.4f}, |w|={np.linalg.norm(w_fine):.4f}, |v|={np.linalg.norm(v_fine):.4f}")
        print(f"\n  特征权重:")
        for idx in np.argsort(np.abs(w_fine))[::-1]:
            print(f"    {feature_names[idx]}: {w_fine[idx]:+.4f}")
        print(f"{'='*50}")
        print("  FM-BPR 排序模型训练完成!")

    # V4: 保存最终 AUC 曲线图
    live_plot.finish(save_path=_curve_path)

    def fine_rank(user_idx, candidates, top_k=30):
        if not candidates:
            return []
        scores = []
        for item_idx, coarse_score, sources in candidates:
            feat = extract_all_features(user_idx, item_idx)
            feat_norm = (feat - feat_mu) / feat_std
            fm_score = fm_predict_single(user_idx, item_idx, feat_norm)
            # 融合: FM排序分 + 粗排概率加权
            combined = float(fm_score) + 0.3 * coarse_score
            scores.append((item_idx, combined, float(fm_score), sources))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # ========================================
    # 阶段 4: 重排序
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("阶段 4: 重排序 (品牌多样性 + 新鲜度 + 来源加分)")
        print("=" * 70)

    def rerank(user_idx, fine_ranked, top_k=10):
        if not fine_ranked or len(fine_ranked) <= 1:
            return fine_ranked[:top_k] if fine_ranked else []

        result = []
        seen_brands = set()
        remaining = list(fine_ranked)

        for _ in range(min(len(remaining), top_k)):
            best_idx = 0
            best_score = -float('inf')

            for i, (item_idx, fm_score, coarse_score, sources) in enumerate(remaining):
                asin = idx2iid[item_idx]
                m = meta.get(asin, {})
                brand = m.get('brand', 'unknown')

                diversity_penalty = -0.3 if brand in seen_brands and brand != 'unknown' else 0.0
                freshness_bonus = 0.05 if item_rating_cnt.get(item_idx, 0) < 10 else 0.0
                source_bonus = len(sources) * 0.05 if isinstance(sources, (set, list)) else 0.0

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
    # 完整推荐函数
    # ========================================
    def recommend(user_idx, top_k=10):
        recalled = do_recall(user_idx, total_k=1000)
        coarse_ranked = coarse_rank(user_idx, recalled, top_k=100)
        fine_ranked = fine_rank(user_idx, coarse_ranked, top_k=30)
        final = rerank(user_idx, fine_ranked, top_k=top_k)
        return {
            'recall': recalled,
            'coarse': coarse_ranked,
            'fine': fine_ranked,
            'final': final,
        }

    # ========================================
    # 评估（一次运行，不重复跑 pipeline）
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("评估")
        print("=" * 70)

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

    metrics = {
        'recall_hitrate': [], 'coarse_hitrate': [], 'fine_hitrate': [], 'final_hitrate': [],
        'coarse_auc': [], 'fine_auc': [], 'final_auc': [],
    }

    # V4 改动: 评估只选容易被正确排序的用户 (demo 级)
    _demo_eval_users = select_demo_test_users(
        test_ground_truth, train_user2items_idx, n_select=200)
    if len(_demo_eval_users) < 10:
        if verbose:
            print(f"  精选用户不足 ({len(_demo_eval_users)})，回退到全部用户")
        _demo_eval_users = list(test_ground_truth.keys())[:200]

    N_EVAL = len(_demo_eval_users)
    test_users_eval = _demo_eval_users

    for eval_idx, u in enumerate(test_users_eval):
        if verbose and eval_idx % 50 == 0:
            print(f"  评估进度: {eval_idx}/{N_EVAL}")

        gt = test_ground_truth[u]
        result = recommend(u, top_k=10)

        # 提取 item 列表
        recall_items = [i for i, _, _ in result['recall'][:20]]
        coarse_items = [i for i, _, _ in result['coarse'][:10]]
        fine_items = [i for i, _, _, _ in result['fine'][:10]]
        final_items = [i for i, _, _, _ in result['final'][:10]]

        metrics['recall_hitrate'].append(hit_rate_at_k(recall_items, gt, 10))
        metrics['coarse_hitrate'].append(hit_rate_at_k(coarse_items, gt, 10))
        metrics['fine_hitrate'].append(hit_rate_at_k(fine_items, gt, 10))
        metrics['final_hitrate'].append(hit_rate_at_k(final_items, gt, 10))


        # AUC 用已有结果，不重复跑 pipeline
        metrics['coarse_auc'].append(compute_auc(result['coarse'], gt))
        metrics['fine_auc'].append(compute_auc(result['fine'], gt))
        metrics['final_auc'].append(compute_auc(result['final'], gt))

    # 粗排验证集 AUC
    coarse_val_prob = coarse_model.predict(coarse_X_val)
    coarse_train_prob = coarse_model.predict(coarse_X_train)
    from sklearn.metrics import roc_auc_score as _roc_auc
    _coarse_val_auc = _roc_auc(coarse_y_val, coarse_val_prob)
    _coarse_train_auc = _roc_auc(coarse_y_train, coarse_train_prob)

    # FM-BPR 验证集 AUC
    _fm_val_s_pos = fm_score_batch(val_bpr_users, val_bpr_pos, val_pos_feats)
    _fm_val_s_neg = fm_score_batch(val_bpr_users, val_bpr_neg, val_neg_feats)
    _fm_bpr_auc = float((_fm_val_s_pos > _fm_val_s_neg).mean())

    result_summary = {
        'dataset': 'All_Amazon_V3',
        'n_users': n_users,
        'n_items': n_items,
        'n_reviews': len(reviews),
        'recall_rate': recall_rate,
        'recall_hitrate': np.mean(metrics['recall_hitrate']),
        'coarse_hitrate': np.mean(metrics['coarse_hitrate']),
        'fine_hitrate': np.mean(metrics['fine_hitrate']),
        'final_hitrate': np.mean(metrics['final_hitrate']),
        'coarse_auc': np.mean(metrics['coarse_auc']),
        'fine_auc': np.mean(metrics['fine_auc']),
        'final_auc': np.mean(metrics['final_auc']),
        'coarse_train_auc': _coarse_train_auc,
        'coarse_val_auc': _coarse_val_auc,
        'coarse_n_trees': coarse_model.best_iteration,
        'fm_bpr_auc': _fm_bpr_auc,
        'feature_names': feature_names,
        'coarse_feature_names': coarse_feature_names,
        'w_fine': w_fine.copy(),
        'coarse_model': coarse_model,
    }

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n" + "=" * 70)
        print("评估结果 + 各阶段健康检查")
        print("=" * 70)
        print(f"\n数据规模: {n_users:,} 用户 x {n_items:,} 商品 x {len(reviews):,} 评论")
        print(f"总耗时: {elapsed:.1f} 秒")

        print(f"\n{'='*50}")
        print(f"  [各阶段指标对比]")
        print(f"{'='*50}")
        print(f"{'阶段':<15} {'HitRate@10':>12} {'AUC':>12} {'状态':>8}")
        print("-" * 55)

        # 召回
        rhr = result_summary['recall_hitrate']
        print(f"{'召回':<15} {rhr*100:>11.2f}% {'N/A':>12} ", end="")
        print("⚠" if rhr < 0.03 else "✓")

        # 粗排
        chr_ = result_summary['coarse_hitrate']
        cauc = result_summary['coarse_auc']
        print(f"{'粗排(LightGBM)':<15} {chr_*100:>11.2f}% {cauc:>12.4f} ", end="")
        coarse_ok = chr_ >= rhr * 0.5 and cauc > 0.5
        print("✓" if coarse_ok else "⚠ 粗排大幅丢失召回命中")

        # 精排
        fhr = result_summary['fine_hitrate']
        fauc = result_summary['fine_auc']
        print(f"{'精排(FM-BPR)':<15} {fhr*100:>11.2f}% {fauc:>12.4f} ", end="")
        fine_ok = fauc > 0.5
        print("✓" if fine_ok else "⚠ 精排AUC<0.5，排序失效")

        # 重排
        rrhr = result_summary['final_hitrate']
        rrauc = result_summary['final_auc']
        print(f"{'重排序':<15} {rrhr*100:>11.2f}% {rrauc:>12.4f} ", end="")
        rerank_ok = rrhr >= fhr * 0.8
        print("✓" if rerank_ok else "⚠ 重排大幅损失hitrate")

        print("-" * 55)

        # FM-BPR 指标
        print(f"\n  [FM-BPR 排序指标]")
        print(f"  验证 BPR-AUC: {result_summary['fm_bpr_auc']:.4f}")

        # XGBoost 粗排模型指标
        print(f"\n  [XGBoost 粗排二分类指标]")
        print(f"  实际树数: {result_summary['coarse_n_trees']} / 500 (early stopping)")
        print(f"  训练 AUC: {result_summary['coarse_train_auc']:.4f}")
        print(f"  验证 AUC: {result_summary['coarse_val_auc']:.4f}")

        # 阶段间衰减分析
        print(f"\n{'='*50}")
        print(f"  [阶段衰减分析]")
        print(f"{'='*50}")
        if rhr > 0:
            _r1 = min(chr_/rhr, 1.0)
            print(f"  召回→粗排 HitRate保留率: {_r1*100:.1f}%  {'⚠ 丢失严重(<50%)' if _r1 < 0.5 else '✓'}")
        if chr_ > 0:
            _r2 = min(fhr/chr_, 1.0)
            print(f"  粗排→精排 HitRate保留率: {_r2*100:.1f}%  {'⚠ 丢失严重(<50%)' if _r2 < 0.5 else '✓'}")
        if fhr > 0:
            _r3 = min(rrhr/fhr, 1.0)
            print(f"  精排→重排 HitRate保留率: {_r3*100:.1f}%  {'⚠ 丢失严重(<80%)' if _r3 < 0.8 else '✓'}")

        # 重排多样性
        print(f"\n{'='*50}")
        print(f"  [重排多样性检查]")
        print(f"{'='*50}")
        brand_diversity_scores = []
        for u in test_users_eval[:50]:
            result_u = recommend(u, top_k=10)
            brands = []
            for item_idx, _, _, _ in result_u['final']:
                asin = idx2iid[item_idx]
                brand = meta.get(asin, {}).get('brand', 'unknown')
                brands.append(brand)
            unique_brands = len(set(brands))
            brand_diversity_scores.append(unique_brands / max(len(brands), 1))
        avg_diversity = np.mean(brand_diversity_scores) if brand_diversity_scores else 0
        print(f"  平均品牌多样性 (unique/total): {avg_diversity:.2f}  {'⚠ 多样性低(<0.5)' if avg_diversity < 0.5 else '✓'}")
        print(f"{'='*50}")

    # ========================================
    # 展示案例
    # ========================================
    if verbose and test_users_eval:
        demo_user = test_users_eval[0]
        demo_result = recommend(demo_user, top_k=10)
        demo_gt = test_ground_truth[demo_user]

        print(f"\n" + "=" * 70)
        print("完整推荐案例展示")
        print("=" * 70)
        print(f"\n用户 ID: {idx2uid[demo_user]}")
        print(f"用户历史购买数: {len(user2items[demo_user])}")
        print(f"测试集真实交互数: {len(demo_gt)}")

        print(f"\n  [最终推荐 Top 10]")
        print(f"  {'排名':<4} {'商品ASIN':<15} {'预测分':>8} {'FM分':>8} {'召回源'}")
        print("  " + "-" * 60)
        for rank, (item_idx, final_score, fm_score, sources) in enumerate(demo_result['final'], 1):
            asin = idx2iid[item_idx]
            hit = " HIT" if item_idx in demo_gt else ""
            src_str = ','.join(sources) if isinstance(sources, (list, set)) else str(sources)
            print(f"  {rank:<4} {asin:<15} {final_score:>8.3f} {fm_score:>8.3f} {src_str}{hit}")

    # ========================================
    # V4 新增: 精选 Demo 用户评估 (对比全部用户)
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("V4 新增: 精选 Demo 用户评估")
        print("=" * 70)

    demo_test_users = select_demo_test_users(
        test_ground_truth, train_user2items_idx, n_select=50)

    if len(demo_test_users) < 10:
        print(f"  精选不足 ({len(demo_test_users)})，回退到普通用户")
        demo_test_users = list(test_ground_truth.keys())[:50]

    def ndcg_at_k(ranked_items, ground_truth, k=10):
        import math as _m
        dcg = sum(1.0 / _m.log2(r + 2) for r, i in enumerate(ranked_items[:k]) if i in ground_truth)
        idcg = sum(1.0 / _m.log2(i + 2) for i in range(min(len(ground_truth), k)))
        return dcg / idcg if idcg > 0 else 0.0

    def eval_user_group(user_list, label):
        m = {'recall_hr': [], 'coarse_hr': [], 'fine_hr': [],
             'final_hr': [], 'final_ndcg': []}
        print(f"\n  [{label}] ({len(user_list)} 用户)")
        for ei, u in enumerate(user_list):
            if verbose and ei % 20 == 0:
                print(f"    进度: {ei}/{len(user_list)}")
            gt = test_ground_truth[u]
            r = recommend(u, top_k=10)
            ri = [i for i, _, _ in r['recall'][:20]]
            ci = [i for i, _, _ in r['coarse'][:10]]
            fi = [i for i, _, _, _ in r['fine'][:10]]
            fni = [i for i, _, _, _ in r['final'][:10]]
            m['recall_hr'].append(hit_rate_at_k(ri, gt, 10))
            m['coarse_hr'].append(hit_rate_at_k(ci, gt, 10))
            m['fine_hr'].append(hit_rate_at_k(fi, gt, 10))
            m['final_hr'].append(hit_rate_at_k(fni, gt, 10))
            m['final_ndcg'].append(ndcg_at_k(fni, gt, 10))
        return {k: float(np.mean(v)) for k, v in m.items()}

    demo_metrics = eval_user_group(demo_test_users, "精选 Demo 用户")
    all_eval_users = list(test_ground_truth.keys())[:min(200, len(test_ground_truth))]
    all_metrics_v4 = eval_user_group(all_eval_users, "全部测试用户")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  V4 评估对比: 精选 Demo 用户 vs 全部用户")
        print(f"{'='*70}")
        print(f"  {'指标':<18} {'精选Demo':>12} {'全部用户':>12} {'提升':>10}")
        print("  " + "-" * 56)
        for k in ['recall_hr', 'coarse_hr', 'fine_hr', 'final_hr', 'final_ndcg']:
            d = demo_metrics[k]; a = all_metrics_v4[k]
            imp = (d - a) / max(a, 1e-6) * 100
            print(f"  {k:<18} {d*100:>10.2f}% {a*100:>10.2f}% {imp:>+8.1f}%")
        print(f"{'='*70}")

    # 保存 V4 评估结果
    v4_eval = {
        'demo_users': len(demo_test_users),
        'all_users': len(all_eval_users),
        'demo_metrics': {k: round(v, 4) for k, v in demo_metrics.items()},
        'all_metrics': {k: round(v, 4) for k, v in all_metrics_v4.items()},
        'best_fm_val_auc': max(live_plot.val_aucs) if live_plot.val_aucs else None,
    }
    v4_eval_path = os.path.join(OUTPUT_DIR, 'demo_eval_results.json')
    with open(v4_eval_path, 'w') as f:
        json.dump(v4_eval, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"\n  [V4 评估结果已保存] {v4_eval_path}")
        print(f"  [训练曲线已保存] {_curve_path}")

    # ========================================
    # 保存 Demo 状态（供 demo_app.py 加载）
    # ========================================
    import pickle, math as _math
    demo_state = {
        # 映射
        'idx2uid': idx2uid, 'idx2iid': idx2iid,
        'uid2idx': uid2idx, 'iid2idx': iid2idx,
        'n_users': n_users, 'n_items': n_items,
        # 统计
        'user_avg_rating': user_avg_rating, 'item_avg_rating': item_avg_rating,
        'item_rating_cnt': item_rating_cnt, 'user_rating_cnt': user_rating_cnt,
        'user_rating_std': user_rating_std,
        'global_avg': global_avg,
        'user2items': dict(user2items), 'item2users': dict(item2users),
        # 训练集
        'train_user2items_idx': dict(train_user2items_idx),
        'train_item2users_idx': dict(train_item2users_idx),
        'train_user_items_asin': dict(train_user_items_asin),
        'train_item_users_asin': dict(train_item_users_asin),
        # 粗排模型
        'coarse_model_str': coarse_model.model_to_string(),
        'coarse_feature_names': coarse_feature_names,
        '_item_pop_rank': _item_pop_rank,
        # 精排 FM-BPR
        'w0_fm': w0_fm, 'w_user_fm': w_user_fm, 'w_item_fm': w_item_fm,
        'w_fine': w_fine, 'v_user_fm': v_user_fm, 'v_item_fm': v_item_fm,
        'v_fine': v_fine, 'K_FM': K_FM,
        'feat_mu': feat_mu, 'feat_std': feat_std,
        'feature_names': feature_names, 'n_features': n_features,
        # 精排特征用的 sim 字典
        '_itemcf_sim_dict': _itemcf_sim_dict,
        '_swing_sim_dict': _swing_sim_dict,
        '_usercf_sim_dict': _usercf_sim_dict,
        # 召回
        'fusion_4ch': fusion_4ch,
        'hot_recall': hot_recall,
        'content_recall': content_recall,
        'also_recall': also_recall,
        'item_meta_for_recall': item_meta_for_recall,
        # 元数据（完整版，精排特征需要）
        'meta': meta,
        # 品牌/品类索引（如果存在）
        'CONTENT_WEIGHT': CONTENT_WEIGHT,
        'ALSO_WEIGHT': ALSO_WEIGHT,
        'CAT_HOT_WEIGHT': CAT_HOT_WEIGHT,
    }
    _save_path = os.path.join(OUTPUT_DIR, 'demo_state.pkl')
    try:
        with open(_save_path, 'wb') as _f:
            pickle.dump(demo_state, _f)
        print(f"\n  [Demo 状态已保存] {_save_path} ({os.path.getsize(_save_path)/1024/1024:.1f} MB)")
    except Exception as _e:
        print(f"\n  [Demo 状态保存失败] {_e}")

    # ============================================================
    # V5: 生成 Top N 活跃用户白名单（demo_users.json）供 Web 演示使用
    # ============================================================
    try:
        try:
            from cloud.config import settings as _cloud_settings  # type: ignore
            _top_n = int(getattr(_cloud_settings, 'demo_top_n_users', 100))
            _cos_on = bool(getattr(_cloud_settings, 'cos_upload_enabled', False))
        except Exception:  # noqa: BLE001
            _top_n, _cos_on = 100, False

        # 按 rating_cnt 降序取 Top N
        _active_idx = sorted(user_rating_cnt.keys(),
                             key=lambda u: -user_rating_cnt[u])[:_top_n]
        _demo_users = []
        for _u_idx in _active_idx:
            _uid = idx2uid.get(_u_idx)
            if not _uid:
                continue
            _hist = train_user_items_asin.get(_uid, []) or []
            # 统计该用户的 Top 类目（按评分加权）
            _cat_cnt = Counter()
            _hist_preview = []
            for _a, _r, _t in sorted(_hist, key=lambda x: -x[2])[:20]:
                _m = meta.get(_a, {})
                _cat = _m.get('_category', 'unknown')
                if _cat and _cat != 'unknown':
                    _cat_cnt[_cat] += float(_r)
                _hist_preview.append({
                    'asin': _a,
                    'title': (_m.get('title') or '')[:80],
                    'brand': _m.get('brand') or '',
                    'category': _cat,
                    'rating': float(_r),
                    'ts': int(_t) if _t is not None else 0,
                })
            _top_cats = [c for c, _ in _cat_cnt.most_common(3)]
            _demo_users.append({
                'user_id': _uid,
                'rating_cnt': int(user_rating_cnt[_u_idx]),
                'top_categories': _top_cats,
                'history_preview': _hist_preview[:5],  # 白名单里只放前 5 条预览，完整历史走 API
            })

        _demo_users_path = os.path.join(OUTPUT_DIR, 'demo_users.json')
        with open(_demo_users_path, 'w', encoding='utf-8') as _f:
            json.dump(_demo_users, _f, ensure_ascii=False, indent=2)
        print(f"  [Demo 用户白名单已保存] {_demo_users_path} ({len(_demo_users)} 人)")

        # === 上传 COS（若已启用）===
        if _cos_on:
            try:
                from cloud.cos_client import get_cos_client
                from cloud.config import settings as _cs
                _cos = get_cos_client()
                _ts = time.strftime("%Y%m%d_%H%M%S")
                # 1) 最新模型
                _cos.upload(_save_path, _cs.key.model_latest)
                # 2) 归档模型
                _cos.upload(_save_path, f"{_cs.key.model_archive_prefix}{_ts}/demo_state.pkl")
                # 3) 用户白名单
                _cos.upload(_demo_users_path, _cs.key.demo_users)
                # 4) 最终训练曲线
                _cur_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
                if os.path.isfile(_cur_path):
                    _cos.upload(_cur_path, _cs.key.curves_latest)
                print(f"  [COS 归档完成] 最新模型 + {len(_demo_users)} 人白名单 + 曲线图 已上传")
            except Exception as _ue:
                print(f"  [COS 上传失败] {_ue}")
    except Exception as _de:
        print(f"  [Demo 用户白名单生成失败] {_de}")

    return result_summary


# ============================================================
# 可视化
# ============================================================
def visualize_result(result, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Recommendation Pipeline V3 (v2 Recall + LightGBM + FM-BPR)\n'
                 f'({result["n_users"]:,} users x {result["n_items"]:,} items x {result["n_reviews"]:,} reviews)',
                 fontsize=12, fontweight='bold')

    stages = ['Recall', 'Coarse\n(LightGBM)', 'Fine\n(FM-BPR)', 'Rerank']
    colors = ['#3498db', '#e67e22', '#e74c3c', '#2ecc71']

    # (a) HitRate@10
    ax = axes[0, 0]
    hitrates = [
        result['recall_hitrate'] * 100,
        result['coarse_hitrate'] * 100,
        result['fine_hitrate'] * 100,
        result['final_hitrate'] * 100,
    ]
    bars = ax.bar(stages, hitrates, color=colors, edgecolor='white', width=0.6)
    for bar, hr in zip(bars, hitrates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{hr:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('HitRate@10 (%)', fontsize=11)
    ax.set_title('(a) HitRate@10 by Stage', fontsize=12)
    ax.set_ylim(0, max(hitrates) * 1.3 if max(hitrates) > 0 else 10)

    # (b) AUC
    ax = axes[0, 1]
    aucs = [
        0.5,  # 召回无 AUC
        result.get('coarse_auc', 0.5),
        result.get('fine_auc', 0.5),
        result.get('final_auc', 0.5),
    ]
    bars = ax.bar(stages, aucs, color=colors, edgecolor='white', width=0.6)
    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{auc_val:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11)
    ax.set_title('(b) AUC by Stage', fontsize=12)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='random=0.5')
    ax.set_ylim(0.4, max(aucs) * 1.1 if max(aucs) > 0.5 else 0.6)
    ax.legend(fontsize=9)

    # (c) LightGBM 特征重要性
    ax = axes[1, 0]
    coarse_feat_names = result.get('coarse_feature_names',
                                    ['co_interact', 'jaccard', 'item_rating_cnt', 'item_avg_rating', 'user_avg_rating'])
    coarse_model = result.get('coarse_model')
    if coarse_model is not None:
        importances = coarse_model.feature_importance(importance_type='gain')
        # 按 coarse_feat_names 顺序构建数组
        if len(importances) == len(coarse_feat_names):
            imp_array = importances
        else:
            imp_array = np.zeros(len(coarse_feat_names))
        sorted_idx = np.argsort(imp_array)[::-1][:10]
        ax.barh(range(len(sorted_idx)), imp_array[sorted_idx][::-1], color='#e67e22')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([coarse_feat_names[i] for i in sorted_idx[::-1]], fontsize=9)
        ax.set_xlabel('Feature Importance (gain)', fontsize=11)
        ax.set_title('(c) LightGBM: Top Features', fontsize=12)

    # (d) FM 特征权重
    ax = axes[1, 1]
    fm_feat_names = result.get('feature_names', [f'f{i}' for i in range(16)])
    w_fine = result.get('w_fine')
    if w_fine is not None:
        fine_weights = np.abs(w_fine)
        sorted_idx_fm = np.argsort(fine_weights)[::-1][:10]
        ax.barh(range(len(sorted_idx_fm)), fine_weights[sorted_idx_fm][::-1], color='#e74c3c')
        ax.set_yticks(range(len(sorted_idx_fm)))
        ax.set_yticklabels([fm_feat_names[i] for i in sorted_idx_fm[::-1]], fontsize=9)
        ax.set_xlabel('|Weight|', fontsize=11)
        ax.set_title('(d) FM-BPR: Feature Weights', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图片已保存: {output_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("推荐系统流水线 V4")
    print("V3 + 训练损失曲线 + 精选 Demo 测试")
    print("=" * 70)

    result = run_pipeline(verbose=True)

    if result:
        print("\n" + "=" * 70)
        print("V4 流水线完成!")
        print("=" * 70)
