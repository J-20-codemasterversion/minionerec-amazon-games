"""
召回系统 - 热门召回模块
基于评论数和评分的全局/品类热门召回
"""

from collections import Counter
import time
from typing import List, Dict, Set

class HotRecall:
    def __init__(self, item_users: Dict, item_meta: Dict):
        self.item_users = item_users
        self.item_meta = item_meta
        self.global_hot = []
        self.category_hot = {}
        
    def build_index(self, time_decay_days: int = 365):
        """构建热门索引"""
        print("构建热门索引...", end=" ", flush=True)
        
        now = time.time()
        decay_threshold = now - time_decay_days * 86400
        
        item_scores = Counter()
        category_scores = {}
        
        for item, interactions in self.item_users.items():
            score = 0
            for user, rating, ts in interactions:
                # 时间衰减: 最近的交互权重更高
                if ts > decay_threshold:
                    weight = 1.0 + (ts - decay_threshold) / (now - decay_threshold + 1)
                else:
                    weight = 0.5
                score += rating * weight
            
            item_scores[item] = score
            
            # 按品类统计
            cat = self.item_meta.get(item, {}).get('category', 'unknown')
            if cat not in category_scores:
                category_scores[cat] = Counter()
            category_scores[cat][item] = score
        
        # 全局热门
        self.global_hot = [item for item, _ in item_scores.most_common(10000)]
        
        # 品类热门
        for cat, scores in category_scores.items():
            self.category_hot[cat] = [item for item, _ in scores.most_common(2000)]
        
        print(f"完成 (全局 {len(self.global_hot)} 商品, {len(self.category_hot)} 品类)")
        
    def recall(self, user_id: str, user_history: Set[str], 
               category: str = None, top_k: int = 200) -> List[str]:
        """召回热门商品"""
        if category and category in self.category_hot:
            candidates = self.category_hot[category]
        else:
            candidates = self.global_hot
        
        # 过滤用户已交互商品
        result = [item for item in candidates if item not in user_history]
        return result[:top_k]
