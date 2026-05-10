"""
召回系统 - ItemCF 召回模块
基于物品协同过滤的召回
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import heapq

class ItemCFRecall:
    def __init__(self, user_items: Dict, item_users: Dict):
        self.user_items = user_items
        self.item_users = item_users
        self.item_sim = defaultdict(list)  # item -> [(sim_item, score), ...]
        
    def build_index(self, top_k_sim: int = 50):
        """构建物品相似度索引"""
        print("构建 ItemCF 索引...")
        
        # 统计商品被多少用户购买
        item_user_count = {item: len(users) for item, users in self.item_users.items()}
        
        # 计算商品共现矩阵
        item_pair_count = defaultdict(float)
        
        processed = 0
        for user, items in self.user_items.items():
            item_list = list(set([i for i, r, t in items]))  # 去重
            n = len(item_list)
            
            if n > 500 or n < 2:  # 交互太多或太少的用户跳过
                continue
                
            # IUF 权重: 惩罚活跃用户
            weight = 1.0 / math.log(1 + n)
            
            for i in range(n):
                for j in range(i + 1, n):
                    item_i, item_j = item_list[i], item_list[j]
                    item_pair_count[(item_i, item_j)] += weight
                    item_pair_count[(item_j, item_i)] += weight
            
            processed += 1
            if processed % 100000 == 0:
                print(f"  处理用户: {processed:,}")
        
        print(f"  共现对数: {len(item_pair_count):,}")
        
        # 计算余弦相似度
        item_sim_dict = defaultdict(dict)
        for (i, j), count in item_pair_count.items():
            if i in item_user_count and j in item_user_count:
                denom = math.sqrt(item_user_count[i] * item_user_count[j])
                if denom > 0:
                    sim = count / denom
                    item_sim_dict[i][j] = sim
        
        # 保留 Top-K 相似商品
        for item, sims in item_sim_dict.items():
            top_sims = heapq.nlargest(top_k_sim, sims.items(), key=lambda x: x[1])
            self.item_sim[item] = top_sims
        
        print(f"  ItemCF 索引完成: {len(self.item_sim):,} 商品有相似项")
        
    def recall(self, user_id: str, user_history: List[Tuple], 
               top_k: int = 200) -> List[Tuple[str, float]]:
        """基于用户历史召回相似商品"""
        # 用户历史商品
        user_items_set = {item for item, rating, ts in user_history}
        
        # 计算候选商品分数
        candidate_scores = defaultdict(float)
        
        for item, rating, ts in user_history:
            if item not in self.item_sim:
                continue
            for sim_item, sim_score in self.item_sim[item]:
                if sim_item not in user_items_set:
                    # 评分 * 相似度
                    candidate_scores[sim_item] += rating * sim_score
        
        # 排序返回
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]
