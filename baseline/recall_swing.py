"""
召回系统 - Swing 召回模块
Swing 算法: 解决 ItemCF 中热门商品的问题
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import heapq

class SwingRecall:
    """
    Swing 算法: 解决 ItemCF 中热门商品的问题
    核心思想: 如果 item_i 和 item_j 被用户 u 和 v 共同购买，
    但 u 和 v 之间交集很小，则 i 和 j 更相似
    """
    def __init__(self, user_items: Dict, item_users: Dict):
        self.user_items = user_items
        self.item_users = item_users
        self.item_sim = defaultdict(list)
        
    def build_index(self, alpha: float = 0.5, top_k_sim: int = 50, max_items: int = 50000):
        """构建 Swing 相似度索引"""
        print("构建 Swing 索引...")
        
        # 用户-商品集合
        user_item_set = {
            user: set([item for item, r, t in items])
            for user, items in self.user_items.items()
            if 3 <= len(items) <= 500  # 过滤异常用户
        }
        
        # 商品-用户集合 (只选择有一定交互量的商品)
        item_user_count = [(item, len(users)) for item, users in self.item_users.items()]
        item_user_count.sort(key=lambda x: -x[1])
        
        # 选择适中热度的商品 (太热门的计算量太大)
        selected_items = set()
        for item, count in item_user_count:
            if 5 <= count <= 500:  # 至少5人购买，但不超过500人
                selected_items.add(item)
                if len(selected_items) >= max_items:
                    break
        
        print(f"  选择 {len(selected_items):,} 商品计算 Swing")
        
        item_user_set = {
            item: set([user for user, r, t in users])
            for item, users in self.item_users.items()
            if item in selected_items
        }
        
        # 计算 Swing 相似度
        item_sim_dict = defaultdict(lambda: defaultdict(float))
        
        items_list = list(selected_items)
        total = len(items_list)
        
        for idx, item_i in enumerate(items_list):
            if idx % 5000 == 0:
                print(f"  进度: {idx:,}/{total:,}")
                
            users_i = item_user_set.get(item_i, set())
            if len(users_i) < 2:
                continue
            
            # 找到与 item_i 有共同用户的商品
            co_items = defaultdict(set)
            for u in users_i:
                if u in user_item_set:
                    for other_item in user_item_set[u]:
                        if other_item != item_i and other_item in selected_items:
                            co_items[other_item].add(u)
            
            for item_j, common_users in co_items.items():
                if len(common_users) < 1:
                    continue
                    
                # Swing 公式
                swing_score = 0
                common_users_list = list(common_users)
                
                # 限制计算量
                if len(common_users_list) > 50:
                    common_users_list = common_users_list[:50]
                
                for i in range(len(common_users_list)):
                    for j in range(i + 1, len(common_users_list)):
                        u, v = common_users_list[i], common_users_list[j]
                        if u in user_item_set and v in user_item_set:
                            # 用户 u 和 v 的交集大小（惩罚项）
                            overlap = len(user_item_set[u] & user_item_set[v])
                            swing_score += 1.0 / (alpha + overlap)
                
                if swing_score > 0:
                    item_sim_dict[item_i][item_j] = swing_score
        
        # 保留 Top-K
        for item, sims in item_sim_dict.items():
            if sims:
                top_sims = heapq.nlargest(top_k_sim, sims.items(), key=lambda x: x[1])
                self.item_sim[item] = top_sims
        
        print(f"  Swing 索引完成: {len(self.item_sim):,} 商品有相似项")
        
    def recall(self, user_id: str, user_history: List[Tuple],
               top_k: int = 200) -> List[Tuple[str, float]]:
        """召回"""
        user_items_set = {item for item, r, t in user_history}
        candidate_scores = defaultdict(float)
        
        for item, rating, ts in user_history:
            if item not in self.item_sim:
                continue
            for sim_item, sim_score in self.item_sim[item]:
                if sim_item not in user_items_set:
                    candidate_scores[sim_item] += rating * sim_score
        
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]
