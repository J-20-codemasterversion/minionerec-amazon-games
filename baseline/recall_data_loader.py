"""
召回系统 - 数据加载模块
支持加载 Amazon Fashion, Video Games, Electronics, Books 四个数据集
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import os

class RecallDataLoader:
    def __init__(self, data_dir: str, sample_size: int = 500000):
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.datasets = ['AMAZON_FASHION', 'Video_Games', 'Electronics', 'Books']
        
        # 核心数据结构
        self.user_items = defaultdict(list)  # user -> [(item, rating, time)]
        self.item_users = defaultdict(list)  # item -> [(user, rating, time)]
        self.item_meta = {}                   # item -> {title, brand, category}
        self.category_items = defaultdict(set)  # category -> {items}
        
    def load_reviews(self, dataset_name: str, sample: bool = False):
        """加载评论数据"""
        filepath = os.path.join(self.data_dir, f"{dataset_name}.json")
        if not os.path.exists(filepath):
            print(f"  警告: {filepath} 不存在，跳过")
            return 0
            
        records = []
        print(f"  读取 {dataset_name}...", end=" ", flush=True)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    records.append(json.loads(line))
                except:
                    continue
                # 对大文件提前采样
                if sample and len(records) >= self.sample_size * 2:
                    break
        
        if sample and len(records) > self.sample_size:
            records = random.sample(records, self.sample_size)
        
        for r in records:
            user = r.get('reviewerID')
            item = r.get('asin')
            rating = r.get('overall', 3.0)
            timestamp = r.get('unixReviewTime', 0)
            
            if user and item:
                self.user_items[user].append((item, rating, timestamp))
                self.item_users[item].append((user, rating, timestamp))
        
        print(f"{len(records):,} 条")
        return len(records)
    
    def load_meta(self, dataset_name: str):
        """加载商品元数据"""
        filepath = os.path.join(self.data_dir, f"meta_{dataset_name}.json")
        if not os.path.exists(filepath):
            print(f"  警告: {filepath} 不存在，跳过")
            return 0
            
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    asin = item.get('asin')
                    if asin:
                        self.item_meta[asin] = {
                            'title': item.get('title', '')[:100],  # 截断标题
                            'brand': item.get('brand', ''),
                            'category': dataset_name
                        }
                        self.category_items[dataset_name].add(asin)
                        count += 1
                except:
                    continue
        return count
    
    def load_all(self):
        """加载所有数据"""
        print("=" * 50)
        print("加载数据集")
        print("=" * 50)
        
        stats = {}
        for ds in self.datasets:
            print(f"\n[{ds}]")
            # 大数据集采样
            sample = ds in ['Electronics', 'Books']
            n_reviews = self.load_reviews(ds, sample=sample)
            n_meta = self.load_meta(ds)
            stats[ds] = {'reviews': n_reviews, 'meta': n_meta}
            print(f"  商品元数据: {n_meta:,}")
        
        print("\n" + "=" * 50)
        print(f"总计: {len(self.user_items):,} 用户, {len(self.item_users):,} 商品")
        print("=" * 50)
        return stats


if __name__ == "__main__":
    # 测试
    loader = RecallDataLoader("/Users/jasonlihahaha/Desktop/amazon_data/数据")
    loader.load_all()
