#!/usr/bin/env python3
"""
综合缓存更新脚本 - 同时更新两个数据库
数据库1: http://45.78.231.212:8001
数据库2: http://10.252.176.14:8001
"""
import requests
import json
import os
from datetime import datetime
import argparse

# 数据库配置
DATABASES = {
    "db1": {
        "name": "数据库1",
        "cache_file": "cache_db1.json",
        "api_url": "http://45.78.231.212:8001"
    },
    "db2": {
        "name": "数据库2", 
        "cache_file": "cache_db2.json",
        "api_url": "http://10.252.176.14:8001"
    }
}

def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"缓存文件 {cache_file} 损坏: {e}，重新创建")
    
    return {"total_records_at_last_run": 0, "results": []}

def save_cache(data, cache_file, db_name):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ {db_name}缓存已保存到 {cache_file}")

def get_total_records(api_url):
    try:
        response = requests.get(f"{api_url}/stats", timeout=30)
        if response.status_code == 200:
            return response.json().get('total_records', 0)
    except Exception as e:
        print(f"获取总数失败 ({api_url}): {e}")
    return 0

def fetch_element(api_url, index):
    try:
        response = requests.get(f"{api_url}/elements/with-score/by-index/{index}", timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"获取索引 {index} 失败 ({api_url}): {e}")
    return None

def update_database(db_key):
    """更新单个数据库"""
    config = DATABASES[db_key]
    db_name = config["name"]
    cache_file = config["cache_file"]
    api_url = config["api_url"]
    
    print(f"\n🔄 开始更新{db_name}...")
    
    cache = load_cache(cache_file)
    current_total = get_total_records(api_url)
    
    if current_total == 0:
        print(f"❌ 无法连接{db_name} API: {api_url}")
        return False
    
    last_total = cache.get("total_records_at_last_run", 0)
    results = cache.get("results", [])
    
    print(f"{db_name} API总记录: {current_total}, 缓存记录: {last_total}")
    
    if current_total <= last_total:
        print(f"📝 {db_name}没有新数据")
        return True
    
    # 获取新数据
    new_count = 0
    for i in range(last_total + 1, current_total + 1):
        print(f"获取{db_name}索引 {i}...")
        data = fetch_element(api_url, i)
        
        if data and 'result' in data:
            entry = {
                "index": i,
                "name": data.get('name', f'model_{i}'),
                "parent": data.get('parent'),
                "test": data['result'].get('test', ''),
                "train": data['result'].get('train', ''),
                "score": data.get('score'),
                "timestamp": datetime.now().isoformat()
            }
            results.append(entry)
            new_count += 1
            print(f"✅ {db_name}: {entry['name']}")
        else:
            print(f"⚠️ {db_name}索引 {i} 数据无效")
    
    # 保存缓存
    cache.update({
        "total_records_at_last_run": current_total,
        "results": results,
        "last_update": datetime.now().isoformat()
    })
    
    save_cache(cache, cache_file, db_name)
    print(f"🎉 {db_name}新增 {new_count} 条记录，总计 {len(results)} 条")
    return True

def main():
    parser = argparse.ArgumentParser(description='更新AI模型数据库缓存')
    parser.add_argument('--db', choices=['db1', 'db2', 'all'], default='all',
                        help='选择要更新的数据库 (db1, db2, or all)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新获取所有数据')
    
    args = parser.parse_args()
    
    if args.force:
        print("⚠️ 强制模式：将重新获取所有数据")
        for db_key in DATABASES.keys():
            cache_file = DATABASES[db_key]["cache_file"]
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️ 已删除 {cache_file}")
    
    # 执行更新
    success_count = 0
    total_count = 0
    
    if args.db == 'all':
        databases_to_update = list(DATABASES.keys())
    else:
        databases_to_update = [args.db]
    
    for db_key in databases_to_update:
        total_count += 1
        if update_database(db_key):
            success_count += 1
    
    # 总结
    print(f"\n{'='*50}")
    print(f"📊 更新完成: {success_count}/{total_count} 个数据库成功更新")
    
    if success_count == total_count:
        print("🎉 所有数据库更新成功！")
    else:
        print("⚠️ 部分数据库更新失败，请检查网络连接和API状态")
    
    # 显示文件状态
    print(f"\n📁 生成的缓存文件:")
    for db_key in databases_to_update:
        cache_file = DATABASES[db_key]["cache_file"]
        if os.path.exists(cache_file):
            size = os.path.getsize(cache_file)
            print(f"  - {cache_file}: {size:,} bytes")
        else:
            print(f"  - {cache_file}: 文件不存在")

if __name__ == "__main__":
    main()