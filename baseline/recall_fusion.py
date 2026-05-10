"""
召回系统 - 多路召回融合模块
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

class RecallFusion:
    """多路召回融合"""
    
    def __init__(self, 
                 hot_recall, 
                 itemcf_recall, 
                 usercf_recall, 
                 swing_recall,
                 user_items: Dict):
        self.hot = hot_recall
        self.itemcf = itemcf_recall
        self.usercf = usercf_recall
        self.swing = swing_recall
        self.user_items = user_items
        
        # 各通道权重
        self.weights = {
            'hot': 0.1,
            'itemcf': 0.3,
            'usercf': 0.2,
            'swing': 0.4
        }
        
    def recall(self, user_id: str, 
               user_history: List[Tuple],
               total_recall: int = 500) -> List[Dict]:
        """
        多路召回融合
        
        Returns:
            List[Dict]: [{'item': item_id, 'score': float, 'sources': list}, ...]
        """
        user_history_set = {item for item, r, t in user_history}
        
        # 各通道召回数量（每通道都给满量，最后通过分数融合排序截断）
        per_channel = total_recall
        
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
            max_score = max(s for _, s in itemcf_items) if itemcf_items else 1
            for item, sim_score in itemcf_items:
                score = (sim_score / max_score) * self.weights['itemcf']
                results[item]['score'] += score
                results[item]['sources'].append('itemcf')
        
        # 3. UserCF 召回
        usercf_items = self.usercf.recall(user_id, user_history_set, top_k=per_channel)
        if usercf_items:
            max_score = max(s for _, s in usercf_items) if usercf_items else 1
            for item, sim_score in usercf_items:
                score = (sim_score / max_score) * self.weights['usercf']
                results[item]['score'] += score
                results[item]['sources'].append('usercf')
        
        # 4. Swing 召回
        swing_items = self.swing.recall(user_id, user_history, top_k=per_channel)
        if swing_items:
            max_score = max(s for _, s in swing_items) if swing_items else 1
            for item, sim_score in swing_items:
                score = (sim_score / max_score) * self.weights['swing']
                results[item]['score'] += score
                results[item]['sources'].append('swing')
        
        # 多路命中加分
        for item in results:
            num_sources = len(results[item]['sources'])
            if num_sources > 1:
                results[item]['score'] *= (1 + 0.1 * num_sources)
        
        # 排序并格式化输出
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
