"""
召回系统 - UserCF 召回模块
基于用户协同过滤的召回
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import heapq

class UserCFRecall:
    def __init__(self, user_items: Dict, item_users: Dict):
        self.user_items = user_items
        self.item_users = item_users
        self.user_sim = defaultdict(list)  # user -> [(sim_user, score), ...]
        
    def build_index(self, top_k_sim: int = 30, max_users: int = 100000):
        """构建用户相似度索引"""
        print("构建 UserCF 索引...")
        
        # 用户-商品集合 (只处理部分用户以节省内存)
        all_users = list(self.user_items.keys())
        if len(all_users) > max_users:
            # 优先选择有更多交互的用户
            user_counts = [(u, len(items)) for u, items in self.user_items.items()]
            user_counts.sort(key=lambda x: -x[1])
            selected_users = set([u for u, c in user_counts[:max_users] if 3 <= c <= 500])
        else:
            selected_users = set([u for u in all_users if 3 <= len(self.user_items[u]) <= 500])
        
        print(f"  选择 {len(selected_users):,} 用户计算相似度")
        
        user_item_set = {
            user: set([item for item, r, t in items])
            for user, items in self.user_items.items()
            if user in selected_users
        }
        
        # 倒排: 商品 -> 用户列表
        item_to_users = defaultdict(set)
        for user, items in user_item_set.items():
            for item in items:
                item_to_users[item].add(user)
        
        # 计算用户共现
        user_pair_count = defaultdict(int)
        processed = 0
        
        for item, users in item_to_users.items():
            users = list(users)
            if len(users) > 500 or len(users) < 2:  # 太热门或太冷门的商品跳过
                continue
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    user_pair_count[(users[i], users[j])] += 1
                    user_pair_count[(users[j], users[i])] += 1
            
            processed += 1
            if processed % 50000 == 0:
                print(f"  处理商品: {processed:,}")
        
        print(f"  用户共现对数: {len(user_pair_count):,}")
        
        # 计算 Jaccard 相似度
        user_sim_dict = defaultdict(dict)
        for (u1, u2), count in user_pair_count.items():
            if u1 in user_item_set and u2 in user_item_set:
                intersection = count
                union = len(user_item_set[u1]) + len(user_item_set[u2]) - count
                if union > 0:
                    sim = intersection / union
                    user_sim_dict[u1][u2] = sim
        
        # 保留 Top-K 相似用户
        for user, sims in user_sim_dict.items():
            top_sims = heapq.nlargest(top_k_sim, sims.items(), key=lambda x: x[1])
            self.user_sim[user] = top_sims
        
        print(f"  UserCF 索引完成: {len(self.user_sim):,} 用户有相似用户")
        
    def recall(self, user_id: str, user_history_set: Set[str],
               top_k: int = 200) -> List[Tuple[str, float]]:
        """基于相似用户召回商品"""
        if user_id not in self.user_sim:
            return []
        
        candidate_scores = defaultdict(float)
        
        for sim_user, sim_score in self.user_sim[user_id]:
            # 相似用户的交互商品
            for item, rating, ts in self.user_items.get(sim_user, []):
                if item not in user_history_set:
                    candidate_scores[item] += sim_score * rating
        
        ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]
