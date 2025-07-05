#!/usr/bin/env python3
import requests
import json
import os
from datetime import datetime

CACHE_FILE = "cache.json"
API_BASE_URL = "http://45.78.231.212:8001"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            print("缓存文件损坏，重新创建")
    
    return {"total_records_at_last_run": 0, "results": []}

def save_cache(data):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ 缓存已保存")

def get_total_records():
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=30)
        if response.status_code == 200:
            return response.json().get('total_records', 0)
    except Exception as e:
        print(f"获取总数失败: {e}")
    return 0

def fetch_element(index):
    try:
        response = requests.get(f"{API_BASE_URL}/elements/by-index/{index}", timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"获取索引 {index} 失败: {e}")
    return None

def main():
    print("🔄 开始更新数据...")
    
    cache = load_cache()
    current_total = get_total_records()
    
    if current_total == 0:
        print("❌ 无法连接API")
        return
    
    last_total = cache.get("total_records_at_last_run", 0)
    results = cache.get("results", [])
    
    print(f"API总记录: {current_total}, 缓存记录: {last_total}")
    
    if current_total <= last_total:
        print("📝 没有新数据")
        return
    
    # 获取新数据
    new_count = 0
    for i in range(last_total + 1, current_total + 1):
        print(f"获取索引 {i}...")
        data = fetch_element(i)
        
        if data and 'result' in data:
            entry = {
                "index": i,
                "name": data.get('name', f'model_{i}'),
                "parent": data.get('parent'),
                "test": data['result'].get('test', ''),
                "train": data['result'].get('train', ''),
                "timestamp": datetime.now().isoformat()
            }
            results.append(entry)
            new_count += 1
            print(f"✅ {entry['name']}")
    
    # 保存缓存
    cache.update({
        "total_records_at_last_run": current_total,
        "results": results,
        "last_update": datetime.now().isoformat()
    })
    
    save_cache(cache)
    print(f"🎉 新增 {new_count} 条记录，总计 {len(results)} 条")

if __name__ == "__main__":
    main()
