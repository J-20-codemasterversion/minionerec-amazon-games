"""
召回系统 V2 - 使用 Metadata 特征
(已移除召回阶段不适用的 AUC 指标)
"""

import json
import random
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import math

# 导入所有召回模块
from recall_data_loader import RecallDataLoader
from recall_hot import HotRecall
from recall_itemcf import ItemCFRecall
from recall_usercf import UserCFRecall
from recall_swing import SwingRecall
from recall_fusion import RecallFusion


# ============================================
# 新增: 基于内容的召回 (使用 Metadata)
# ============================================
class ContentRecall:
    """
    基于商品元数据的内容召回
    使用 brand, category, title 等特征
    """
    def __init__(self, item_meta: dict, item_users: dict):
        self.item_meta = item_meta
        self.item_users = item_users
        
        # 倒排索引
        self.brand_items = defaultdict(set)      # brand -> items
        self.category_items = defaultdict(set)   # category -> items
        self.title_words_items = defaultdict(set)  # word -> items
        
    def build_index(self):
        """构建内容倒排索引"""
        print("构建内容召回索引...", end=" ", flush=True)
        
        for item, meta in self.item_meta.items():
            # 品牌索引
            brand = meta.get('brand', '').strip().lower()
            if brand and brand != 'unknown':
                self.brand_items[brand].add(item)
            
            # 品类索引
            category = meta.get('category', '')
            if category:
                self.category_items[category].add(item)
            
            # 标题关键词索引 (简单分词)
            title = meta.get('title', '')
            if title:
                words = title.lower().split()[:10]  # 取前10个词
                for word in words:
                    if len(word) > 3:  # 过滤短词
                        self.title_words_items[word].add(item)
        
        print(f"完成 (品牌: {len(self.brand_items)}, 品类: {len(self.category_items)}, 关键词: {len(self.title_words_items)})")
    
    def recall(self, user_history: list, user_history_set: set, top_k: int = 200):
        """基于用户历史商品的内容特征召回"""
        candidate_scores = defaultdict(float)
        
        # 收集用户历史商品的特征
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
        
        # 基于品牌召回
        for brand, rating in user_brands:
            for item in self.brand_items.get(brand, []):
                if item not in user_history_set:
                    candidate_scores[item] += rating * 0.5
        
        # 基于品类召回
        for category, rating in user_categories:
            for item in list(self.category_items.get(category, []))[:500]:
                if item not in user_history_set:
                    candidate_scores[item] += rating * 0.3
        
        # 基于标题关键词召回
        for word, rating in user_words[:20]:
            for item in list(self.title_words_items.get(word, []))[:100]:
                if item not in user_history_set:
                    candidate_scores[item] += rating * 0.2
        
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


# ============================================
# 改进的融合模块 (加入内容召回)
# ============================================
class RecallFusionV2:
    """多路召回融合 V2 - 包含内容召回"""
    
    def __init__(self, hot, itemcf, usercf, swing, content, user_items):
        self.hot = hot
        self.itemcf = itemcf
        self.usercf = usercf
        self.swing = swing
        self.content = content
        self.user_items = user_items
        
        # 更新权重，加入内容召回
        self.weights = {
            'hot': 0.1,
            'itemcf': 0.25,
            'usercf': 0.15,
            'swing': 0.25,
            'content': 0.25  # 新增内容召回
        }
        
    def recall(self, user_id: str, user_history: list, total_recall: int = 500):
        user_history_set = {item for item, r, t in user_history}
        per_channel = total_recall // 4
        
        results = defaultdict(lambda: {'score': 0, 'sources': []})
        
        # 1. 热门召回
        hot_items = self.hot.recall(user_id, user_history_set, top_k=per_channel)
        for i, item in enumerate(hot_items):
            score = (per_channel - i) / per_channel * self.weights['hot']
            results[item]['score'] += score
            results[item]['sources'].append('hot')
        
        # 2. ItemCF 召回
        itemcf_items = self.itemcf.recall(user_id, user_history, top_k=per_channel)
        if itemcf_items:
            max_score = max(s for _, s in itemcf_items) or 1
            for item, sim_score in itemcf_items:
                score = (sim_score / max_score) * self.weights['itemcf']
                results[item]['score'] += score
                results[item]['sources'].append('itemcf')
        
        # 3. UserCF 召回
        usercf_items = self.usercf.recall(user_id, user_history_set, top_k=per_channel)
        if usercf_items:
            max_score = max(s for _, s in usercf_items) or 1
            for item, sim_score in usercf_items:
                score = (sim_score / max_score) * self.weights['usercf']
                results[item]['score'] += score
                results[item]['sources'].append('usercf')
        
        # 4. Swing 召回
        swing_items = self.swing.recall(user_id, user_history, top_k=per_channel)
        if swing_items:
            max_score = max(s for _, s in swing_items) or 1
            for item, sim_score in swing_items:
                score = (sim_score / max_score) * self.weights['swing']
                results[item]['score'] += score
                results[item]['sources'].append('swing')
        
        # 5. 内容召回 (新增)
        content_items = self.content.recall(user_history, user_history_set, top_k=per_channel)
        if content_items:
            max_score = max(s for _, s in content_items) or 1
            for item, sim_score in content_items:
                score = (sim_score / max_score) * self.weights['content']
                results[item]['score'] += score
                results[item]['sources'].append('content')
        
        # 多路命中加分
        for item in results:
            num_sources = len(results[item]['sources'])
            if num_sources > 1:
                results[item]['score'] *= (1 + 0.1 * num_sources)
        
        ranked = sorted(results.items(), key=lambda x: -x[1]['score'])
        
        output = []
        for item, info in ranked[:total_recall]:
            output.append({
                'item': item,
                'score': info['score'],
                'sources': info['sources'],
                'num_sources': len(info['sources'])
            })
        
        return output


# ============================================
# 评估函数 (召回阶段只用 Recall 和 HitRate)
# ============================================
def evaluate_recall(recall_results: list, ground_truth: set, k_list: list = [50, 100, 200, 500]):
    """
    评估召回效果
    
    召回阶段的核心指标:
    - Recall@K: 召回了多少比例的真实购买
    - HitRate@K: 是否至少命中一个
    
    注: AUC 更适合排序/精排阶段，召回阶段已移除
    """
    metrics = {}
    
    recalled_items = [r['item'] for r in recall_results]
    
    for k in k_list:
        top_k = set(recalled_items[:k])
        hits = len(top_k & ground_truth)
        
        metrics[f'Recall@{k}'] = hits / len(ground_truth) if ground_truth else 0
        metrics[f'HitRate@{k}'] = 1 if hits > 0 else 0
    
    return metrics


def split_by_time(user_items: dict, train_ratio: float = 0.8):
    """按时间划分训练集和测试集"""
    train_user_items = defaultdict(list)
    test_user_items = defaultdict(list)
    
    for user, items in user_items.items():
        if len(items) < 2:
            train_user_items[user] = items
            continue
        sorted_items = sorted(items, key=lambda x: x[2])
        split_idx = max(1, int(len(sorted_items) * train_ratio))
        train_user_items[user] = sorted_items[:split_idx]
        test_user_items[user] = sorted_items[split_idx:]
    
    return dict(train_user_items), dict(test_user_items)


def main():
    print("=" * 60)
    print("🎯 多路召回系统 V2 - 增加 AUC + Metadata 特征")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: 加载数据
    print("\n📦 [Step 1/6] 加载数据...")
    data_dir = "/Users/jasonlihahaha/Desktop/amazon_data/数据"
    if not os.path.exists(data_dir):
        data_dir = "/Users/jasonlihahaha/Desktop/amazon_data/回归"
    
    loader = RecallDataLoader(data_dir, sample_size=300000)
    loader.load_all()
    
    # Step 2: 划分训练/测试集
    print("\n✂️  [Step 2/6] 划分训练/测试集...")
    train_user_items, test_user_items = split_by_time(dict(loader.user_items))
    
    train_item_users = defaultdict(list)
    for user, items in train_user_items.items():
        for item, rating, ts in items:
            train_item_users[item].append((user, rating, ts))
    
    print(f"  训练集: {len(train_user_items):,} 用户, {len(train_item_users):,} 商品")
    
    # Step 3: 构建召回索引
    print("\n🔨 [Step 3/6] 构建召回索引...")
    
    print("\n[3.1] 热门召回")
    hot_recall = HotRecall(dict(train_item_users), loader.item_meta)
    hot_recall.build_index()
    
    print("\n[3.2] ItemCF 召回")
    itemcf_recall = ItemCFRecall(train_user_items, dict(train_item_users))
    itemcf_recall.build_index(top_k_sim=50)
    
    print("\n[3.3] UserCF 召回")
    usercf_recall = UserCFRecall(train_user_items, dict(train_item_users))
    usercf_recall.build_index(top_k_sim=30, max_users=50000)
    
    print("\n[3.4] Swing 召回")
    swing_recall = SwingRecall(train_user_items, dict(train_item_users))
    swing_recall.build_index(alpha=0.5, top_k_sim=50, max_items=30000)
    
    print("\n[3.5] 内容召回 (使用 Metadata)")
    content_recall = ContentRecall(loader.item_meta, dict(train_item_users))
    content_recall.build_index()
    
    # Step 4: 融合召回
    print("\n🔗 [Step 4/6] 初始化多路召回融合 V2...")
    fusion = RecallFusionV2(hot_recall, itemcf_recall, usercf_recall, swing_recall, content_recall, train_user_items)
    print("  融合权重: hot=0.1, itemcf=0.25, usercf=0.15, swing=0.25, content=0.25")
    
    # Step 5: 评估
    print("\n📊 [Step 5/6] 评估召回效果...")
    
    test_users_valid = [
        u for u in test_user_items.keys() 
        if len(test_user_items[u]) > 0 and len(train_user_items.get(u, [])) >= 3
    ]
    
    sample_size = min(2000, len(test_users_valid))
    sample_users = random.sample(test_users_valid, sample_size)
    
    all_metrics = defaultdict(list)
    source_stats = defaultdict(int)
    
    print(f"  评估样本: {sample_size} 用户")
    
    for idx, user in enumerate(sample_users):
        if idx % 500 == 0:
            print(f"  进度: {idx}/{sample_size}")
            
        train_history = train_user_items.get(user, [])
        test_items = {item for item, r, t in test_user_items[user]}
        
        if not train_history or not test_items:
            continue
        
        recall_results = fusion.recall(user, train_history, total_recall=500)
        
        for r in recall_results[:100]:
            for src in r['sources']:
                source_stats[src] += 1
        
        # 评估 (只用 Recall 和 HitRate)
        metrics = evaluate_recall(recall_results, test_items)
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    # Step 6: 输出结果
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📈 召回效果评估结果 V2")
    print("=" * 60)
    
    print("\n【召回率指标 Recall】")
    print("  含义: 召回命中数 / 用户实际购买数")
    for metric in ['Recall@50', 'Recall@100', 'Recall@200', 'Recall@500']:
        if metric in all_metrics:
            values = all_metrics[metric]
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric}: {avg:.4f} ({avg*100:.2f}%)")
    
    print("\n【命中率指标 HitRate】")
    print("  含义: 是否至少命中1个用户购买 (0或1)")
    for metric in ['HitRate@50', 'HitRate@100', 'HitRate@200', 'HitRate@500']:
        if metric in all_metrics:
            values = all_metrics[metric]
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric}: {avg:.4f} ({avg*100:.2f}%)")
    
    print("\n【召回来源分布】")
    total_sources = sum(source_stats.values())
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        pct = count / total_sources * 100 if total_sources > 0 else 0
        print(f"  {src}: {count:,} ({pct:.1f}%)")
    
    print(f"\n⏱️  总耗时: {elapsed:.1f} 秒")
    
    # 可视化
    print("\n📊 生成可视化图表...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1: Recall@K
    ax1 = axes[0]
    k_values = [50, 100, 200, 500]
    recall_values = [sum(all_metrics.get(f'Recall@{k}', [0])) / max(len(all_metrics.get(f'Recall@{k}', [1])), 1) for k in k_values]
    bars1 = ax1.bar(range(len(k_values)), recall_values, color='steelblue')
    ax1.set_xticks(range(len(k_values)))
    ax1.set_xticklabels([f'@{k}' for k in k_values])
    ax1.set_ylabel('Recall')
    ax1.set_title('Recall@K\n(召回命中数 / 实际购买数)')
    ax1.set_ylim(0, max(recall_values) * 1.3 if max(recall_values) > 0 else 0.5)
    for i, v in enumerate(recall_values):
        ax1.text(i, v + 0.005, f'{v:.3f}', ha='center')
    
    # 子图2: HitRate@K
    ax2 = axes[1]
    hitrate_values = [sum(all_metrics.get(f'HitRate@{k}', [0])) / max(len(all_metrics.get(f'HitRate@{k}', [1])), 1) for k in k_values]
    bars2 = ax2.bar(range(len(k_values)), hitrate_values, color='coral')
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([f'@{k}' for k in k_values])
    ax2.set_ylabel('HitRate')
    ax2.set_title('HitRate@K\n(是否至少命中1个)')
    ax2.set_ylim(0, 1.0)
    for i, v in enumerate(hitrate_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 子图3: 来源分布 (含 content)
    ax3 = axes[2]
    if source_stats:
        sources = list(source_stats.keys())
        counts = [source_stats[s] for s in sources]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'][:len(sources)]
        ax3.pie(counts, labels=sources, autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Recall Source Distribution\n(含 Content 内容召回)')
    
    plt.tight_layout()
    
    output_path = '/Users/jasonlihahaha/Desktop/amazon_data/回归/recall_evaluation_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 结果已保存到: {output_path}")
    print("\n" + "=" * 60)
    print("🎉 召回系统 V2 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
