#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import requests
from datetime import datetime
import numpy as np

# 配置页面
st.set_page_config(
    page_title="AI模型结果表格",
    page_icon="📊",
    layout="wide"
)

# GitHub Raw URL - 这里使用你的仓库地址
GITHUB_RAW_URL = "https://raw.githubusercontent.com/HZxCzar/FrontEnd/main/cache.json"

@st.cache_data(ttl=300)
def load_data_from_github():
    try:
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"无法获取数据，状态码: {response.status_code}")
            return {"results": []}
    except Exception as e:
        st.error(f"加载数据错误: {str(e)}")
        return {"results": []}

def parse_test_results(test_string):
    if not test_string:
        return {}
    
    lines = test_string.strip().split('\n')
    if len(lines) < 2:
        return {}
    
    try:
        headers = [h.strip() for h in lines[0].split(',')]
        values = [v.strip() for v in lines[1].split(',')]
        
        result = {}
        for header, value in zip(headers, values):
            if header == 'Model':
                continue
            try:
                result[header] = float(value)
            except ValueError:
                result[header] = value
        return result
    except:
        return {}

def get_min_loss(train_string):
    if not train_string:
        return None
    
    lines = train_string.strip().split('\n')
    if len(lines) < 2:
        return None
    
    try:
        loss_values = []
        parts = lines[1].split(',')[1:]  # 跳过第一个标签
        
        for part in parts:
            try:
                loss_values.append(float(part.strip()))
            except ValueError:
                continue
        
        return min(loss_values) if loss_values else None
    except:
        return None

def main():
    st.title("📊 AI模型结果表格")
    st.markdown("显示每个模型的最低loss、12个测试集结果和测试集均值")
    st.markdown("---")
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        cache = load_data_from_github()
    
    results = cache.get("results", [])
    
    if not results:
        st.error("❌ 没有找到数据")
        st.info("请先运行数据更新脚本: python update_cache.py")
        return
    
    # 创建表格数据
    table_data = []
    
    for result in results:
        if not result.get('name'):
            continue
            
        row = {'模型名称': result['name']}
        
        # 最低loss
        min_loss = get_min_loss(result.get('train', ''))
        row['最低Loss'] = f"{min_loss:.4f}" if min_loss else 'N/A'
        
        # 测试结果
        test_results = parse_test_results(result.get('test', ''))
        
        # 12个测试集
        test_datasets = [
            'ARC Challenge', 'ARC Easy', 'BoolQ', 'FDA', 'HellaSwag', 
            'LAMBDA OpenAI', 'OpenBookQA', 'PIQA', 'Social IQA', 
            'SQuAD Completion', 'SWDE', 'WinoGrande'
        ]
        
        test_values = []
        for dataset in test_datasets:
            if dataset in test_results:
                value = test_results[dataset]
                row[dataset] = f"{value:.3f}" if isinstance(value, (int, float)) else value
                if isinstance(value, (int, float)):
                    test_values.append(value)
            else:
                row[dataset] = 'N/A'
        
        # 测试集均值
        if test_values:
            row['测试集均值'] = f"{np.mean(test_values):.4f}"
        elif 'Average' in test_results:
            row['测试集均值'] = f"{test_results['Average']:.4f}"
        else:
            row['测试集均值'] = 'N/A'
        
        table_data.append(row)
    
    if not table_data:
        st.error("没有可显示的数据")
        return
    
    # 显示统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔢 模型总数", len(table_data))
    with col2:
        st.metric("📊 记录数", len(results))
    with col3:
        last_update = cache.get("last_update", "未知")
        if last_update != "未知":
            try:
                last_update = datetime.fromisoformat(last_update).strftime('%m-%d %H:%M')
            except:
                pass
        st.metric("🕒 最后更新", last_update)
    
    # 显示表格
    df = pd.DataFrame(table_data)
    st.subheader("📋 模型结果表格")
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 下载按钮
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载CSV",
        data=csv,
        file_name=f'model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
