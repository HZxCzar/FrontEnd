#!/usr/bin/env python3
"""
多数据库AI模型结果表格展示
支持两个数据库：
1. 数据库1: http://45.78.231.212:8001
2. 数据库2: http://10.252.176.14:8001
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
import os
from datetime import datetime
import numpy as np

# 配置页面
st.set_page_config(
    page_title="AI模型结果表格",
    page_icon="📊",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 美化表格 */
    .stDataFrame > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* 标题样式 */
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* 第二个数据库的卡片样式 */
    .metric-card-db2 {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card-db2:hover {
        transform: translateY(-5px);
    }
    
    /* 数据源信息 */
    .data-source {
        background-color: #e7f3ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    .data-source-db2 {
        background-color: #ffe7e7;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff6b6b;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    /* 说明文字 */
    .legend {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 15px 0;
    }
    
    .legend-db2 {
        background-color: #fff5f5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 15px 0;
    }
    
    /* 页面切换按钮 */
    .page-nav {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }
    
    .page-button {
        padding: 10px 20px;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .page-button-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .page-button-inactive {
        background: #f8f9fa;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# 数据库配置
DB_CONFIGS = {
    "database1": {
        "name": "数据库1",
        "cache_file": "cache_db1.json",
        "github_url": "https://raw.githubusercontent.com/HZxCzar/FrontEnd/main/cache.json",
        "api_url": "http://45.78.231.212:8001",
        "color_scheme": "blue"
    },
    "database2": {
        "name": "数据库2", 
        "cache_file": "cache_db2.json",
        "github_url": "https://raw.githubusercontent.com/HZxCzar/FrontEnd/main/cache_db2.json",
        "api_url": "http://10.252.176.14:8001",
        "color_scheme": "red"
    }
}

@st.cache_data(ttl=300)
def load_data(db_key):
    """智能加载数据：优先本地，后备远程"""
    config = DB_CONFIGS[db_key]
    data_source = ""
    
    # 首先尝试加载本地文件
    if os.path.exists(config["cache_file"]):
        try:
            with open(config["cache_file"], 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_source = f"📁 本地文件: {config['cache_file']}"
                return data, data_source
        except Exception as e:
            st.warning(f"读取本地文件失败: {e}")
    
    # 如果本地文件不存在，尝试从GitHub加载
    try:
        response = requests.get(config["github_url"], timeout=10)
        if response.status_code == 200:
            data = response.json()
            data_source = f"🌐 远程GitHub: {config['github_url']}"
            return data, data_source
        else:
            st.error(f"GitHub数据加载失败，状态码: {response.status_code}")
    except requests.exceptions.Timeout:
        st.error("GitHub连接超时，请检查网络连接")
    except Exception as e:
        st.error(f"GitHub数据加载错误: {str(e)}")
    
    data_source = "❌ 无法加载数据"
    return {"results": []}, data_source

def normalize_column_name(col_name):
    """标准化列名，处理大小写和格式差异"""
    column_mapping = {
        'arc_challenge': 'ARC Challenge',
        'arc challenge': 'ARC Challenge',
        'arc_easy': 'ARC Easy', 
        'arc easy': 'ARC Easy',
        'boolq': 'BoolQ',
        'fda': 'FDA',
        'hellaswag': 'HellaSwag',
        'lambada_openai': 'LAMBDA OpenAI',
        'lambda openai': 'LAMBDA OpenAI',
        'openbookqa': 'OpenBookQA',
        'piqa': 'PIQA',
        'social_iqa': 'Social IQA',
        'social iqa': 'Social IQA',
        'squad_completion': 'SQuAD Completion',
        'squad completion': 'SQuAD Completion',
        'swde': 'SWDE',
        'winogrande': 'WinoGrande',
        'average': 'Average'
    }
    
    clean_name = col_name.strip().lower()
    return column_mapping.get(clean_name, col_name.strip().title())

def parse_test_results(test_string):
    """解析测试结果，处理不同的格式"""
    if not test_string:
        return {}
    
    # 清理字符串，处理\r\n
    test_string = test_string.replace('\r\n', '\n').replace('\r', '\n')
    lines = test_string.strip().split('\n')
    
    if len(lines) < 2:
        return {}
    
    try:
        headers = [h.strip() for h in lines[0].split(',')]
        values = [v.strip() for v in lines[1].split(',')]
        
        result = {}
        
        # 跳过第一列（模型名称）
        for i, (header, value) in enumerate(zip(headers, values)):
            if i == 0 or header.lower() in ['model', '']:
                continue
                
            standard_header = normalize_column_name(header)
            
            try:
                numeric_value = float(value)
                result[standard_header] = numeric_value
            except ValueError:
                result[standard_header] = value
        
        return result
    except Exception as e:
        print(f"解析测试结果时出错: {e}")
        return {}

def get_loss_at_step_2000(train_string):
    """从训练数据中提取2000步时的loss，如果没有2000步则返回None"""
    if not train_string:
        return None
    
    train_string = train_string.replace('\r\n', '\n').replace('\r', '\n')
    lines = train_string.strip().split('\n')
    
    if len(lines) < 2:
        return None
    
    try:
        step_line = lines[0]
        loss_line = lines[1]
        
        steps = [s.strip() for s in step_line.split(',')[1:]]  # 跳过第一个标签
        losses = [l.strip() for l in loss_line.split(',')[1:]]  # 跳过第一个标签
        
        # 查找2000步对应的loss
        for step, loss in zip(steps, losses):
            try:
                if int(step) == 2000:
                    return float(loss)
            except ValueError:
                continue
        
        # 如果没找到2000步，返回None
        return None
    except Exception as e:
        print(f"解析训练数据时出错: {e}")
        return None

def create_performance_summary(df):
    """创建性能摘要"""
    summary_columns = ['Score', 'Loss', '测试集均值']
    benchmark_columns = [
        'ARC Challenge', 'ARC Easy', 'BoolQ', 'FDA', 'HellaSwag', 
        'LAMBDA OpenAI', 'OpenBookQA', 'PIQA', 'Social IQA', 
        'SQuAD Completion', 'SWDE', 'WinoGrande'
    ]
    
    all_columns = summary_columns + benchmark_columns
    summary_data = []
    
    for col in all_columns:
        if col in df.columns:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            valid_data = numeric_series.dropna()
            
            if len(valid_data) > 0:
                if col == 'Loss':
                    best_value = valid_data.min()
                    best_idx = numeric_series.idxmin()
                    direction = "↓"
                else:
                    best_value = valid_data.max()
                    best_idx = numeric_series.idxmax()
                    direction = "↑"
                
                best_model = df.loc[best_idx, '模型名称']
                
                # 格式化显示
                if col == 'Score':
                    formatted_value = f"{best_value:.6f}"
                else:
                    formatted_value = f"{best_value:.4f}"
                
                summary_data.append({
                    '指标': col,
                    '最优值': formatted_value,
                    '最优模型': best_model,
                    '趋势': direction
                })
    
    return pd.DataFrame(summary_data)

def render_database_page(db_key):
    """渲染数据库页面"""
    config = DB_CONFIGS[db_key]
    is_db2 = db_key == "database2"
    
    # 页面标题
    emoji = "🔥" if is_db2 else "🏆"
    title_class = "main-title"
    st.markdown(f'<h1 class="{title_class}">{emoji} {config["name"]}模型性能排行榜</h1>', unsafe_allow_html=True)
    
    # 加载数据
    with st.spinner("🔄 正在智能加载数据..."):
        cache, data_source = load_data(db_key)
    
    # 数据源信息
    source_class = "data-source-db2" if is_db2 else "data-source"
    st.markdown(f'<div class="{source_class}">📊 数据源: {data_source}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    results = cache.get("results", [])
    
    if not results:
        st.error("❌ 没有找到数据")
        st.info("**数据加载指南:**")
        st.markdown(f"""
        **本地测试:**
        1. 确保当前目录有 `{config['cache_file']}` 文件
        2. 运行相应的更新脚本生成数据
        
        **远程部署:**
        1. 推送 `{config['cache_file']}` 到GitHub仓库
        2. 应用会自动从GitHub加载数据
        """)
        return
    
    # 处理数据，按新的列顺序组织
    table_data = []
    delta_net_row = None
    
    for result in results:
        if not result.get('name'):
            continue
        
        # Index
        row = {'Index': result['index']}
        
        # 按照要求的列顺序：名字、Score、Loss(2000步)、测试集均值、各个benchmark
        row['模型名称'] = result['name']
        
        # Score（新增字段）
        score = result.get('score')
        row['Score'] = score if score is not None else np.nan
        
        # 2000步的loss
        loss_2000 = get_loss_at_step_2000(result.get('train', ''))
        row['Loss'] = loss_2000 if loss_2000 is not None else np.nan
        
        # 测试结果解析
        test_results = parse_test_results(result.get('test', ''))
        
        # 先计算测试集均值，放在第4列
        benchmark_datasets = [
            'ARC Challenge', 'ARC Easy', 'BoolQ', 'FDA', 'HellaSwag', 
            'LAMBDA OpenAI', 'OpenBookQA', 'PIQA', 'Social IQA', 
            'SQuAD Completion', 'SWDE', 'WinoGrande'
        ]
        
        test_values = []
        benchmark_data = {}
        
        # 收集benchmark数据
        for dataset in benchmark_datasets:
            if dataset in test_results:
                value = test_results[dataset]
                benchmark_data[dataset] = value
                if isinstance(value, (int, float)):
                    test_values.append(value)
            else:
                benchmark_data[dataset] = np.nan
        
        # 计算测试集均值
        if test_values:
            row['测试集均值'] = np.mean(test_values)
        elif 'Average' in test_results:
            row['测试集均值'] = test_results['Average']
        else:
            row['测试集均值'] = np.nan
        
        # 添加各个benchmark（第5列开始）
        for dataset in benchmark_datasets:
            row[dataset] = benchmark_data[dataset]
        
        # 检查是否是delta_net，如果是则单独保存
        if str(result['name']).lower() == 'delta_net':
            delta_net_row = row
        else:
            table_data.append(row)
    
    # 手动添加gated_delta_net作为SOTA模型（第一行）
    gated_delta_net_row = {
        'Index': 'SOTA',
        '模型名称': 'gated_delta_net',
        'Score': np.nan,  # 没有提供Score数据
        'Loss': 4.377,
        '测试集均值': 0.226,
        'ARC Challenge': 0.168,
        'ARC Easy': 0.374,
        'BoolQ': 0.370,
        'FDA': 0.000,
        'HellaSwag': 0.282,
        'LAMBDA OpenAI': 0.002,
        'OpenBookQA': 0.144,
        'PIQA': 0.562,
        'Social IQA': 0.350,
        'SQuAD Completion': 0.004,
        'SWDE': 0.002,
        'WinoGrande': 0.456
    }
    
    # 重新组织数据：gated_delta_net在第一行，delta_net在第二行，其他按原顺序
    final_table_data = [gated_delta_net_row]
    
    if delta_net_row is not None:
        final_table_data.append(delta_net_row)
    
    final_table_data.extend(table_data)
    
    if not final_table_data:
        st.error("没有可显示的数据")
        return
    
    df = pd.DataFrame(final_table_data)
    
    # 调试：打印前几行数据确认gated_delta_net是否存在
    if len(df) > 0:
        print("前3行数据:")
        print(df.head(3)['模型名称'].tolist())
    
    # 统计卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    card_class = "metric-card-db2" if is_db2 else "metric-card"
    
    with col1:
        st.markdown(f"""
        <div class="{card_class}">
            <h3>🔢 模型总数</h3>
            <h2>{len(table_data)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        complete_count = sum(1 for _, row in df.iterrows() 
                           if sum(pd.isna(row[col]) for col in ['ARC Challenge', 'ARC Easy', 'BoolQ'] if col in row) == 0)
        st.markdown(f"""
        <div class="{card_class}">
            <h3>✅ 完整数据</h3>
            <h2>{complete_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # 最高Score
        if 'Score' in df.columns:
            numeric_score = pd.to_numeric(df['Score'], errors='coerce')
            if not numeric_score.isna().all():
                max_score = numeric_score.max()
                st.markdown(f"""
                <div class="{card_class}">
                    <h3>🎯 最高Score</h3>
                    <h2>{max_score:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="{card_class}">
                    <h3>🎯 最高Score</h3>
                    <h2>N/A</h2>
                </div>
                """, unsafe_allow_html=True)
    
    with col4:
        last_update = cache.get("last_update", "未知")
        if last_update != "未知":
            try:
                last_update = datetime.fromisoformat(last_update).strftime('%m-%d %H:%M')
            except:
                pass
        st.markdown(f"""
        <div class="{card_class}">
            <h3>🕒 最后更新</h3>
            <h2>{last_update}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        # 显示SOTA模型性能而不是baseline
        sota_perf = "0.226"  # gated_delta_net的测试集均值
        st.markdown(f"""
        <div class="{card_class}">
            <h3>🥇 SOTA</h3>
            <h2>{sota_perf}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 说明
    st.markdown("---")
    legend_class = "legend-db2" if is_db2 else "legend"
    st.markdown(f"""
    <div class="{legend_class}">
        <strong>📖 图例说明:</strong><br>
        🟢 <strong>绿色高亮</strong>: 该列最优值 (Score、测试集均值、各benchmark越高越好，Loss越低越好)<br>
        🥇 <strong>金色背景 + 橙色左边框</strong>: gated_delta_net (SOTA模型)<br>
        🟡 <strong>黄色背景 + 黄色左边框</strong>: delta_net (Baseline模型)<br>
        📊 <strong>列顺序</strong>: 模型名称 → Score → Loss(2000步) → 测试集均值 → 12个benchmark详情<br>
        📋 <strong>排序</strong>: gated_delta_net (SOTA) → delta_net (Baseline) → 其他模型
    </div>
    """, unsafe_allow_html=True)
    
    # 主表格
    st.markdown("### 📊 模型性能详细对比表")
    
    # 侧边栏选项
    with st.sidebar:
        st.header(f"🎛️ {config['name']}显示选项")
        
        # 排序选项
        sort_options = ["模型名称", "Score", "Loss", "测试集均值"]
        sort_by = st.selectbox("排序依据", sort_options, index=1, key=f"sort_{db_key}")
        sort_ascending = st.checkbox("升序排列", value=False, key=f"asc_{db_key}")
        
        # 筛选选项
        show_only_complete = st.checkbox("只显示完整数据", value=False, key=f"complete_{db_key}")
        
        # Score范围筛选
        if 'Score' in df.columns:
            score_series = pd.to_numeric(df['Score'], errors='coerce')
            if not score_series.isna().all():
                min_score = float(score_series.min())
                max_score = float(score_series.max())
                score_range = st.slider(
                    "Score范围筛选", 
                    min_value=min_score, 
                    max_value=max_score, 
                    value=(min_score, max_score),
                    step=0.001,
                    key=f"score_range_{db_key}"
                )
            else:
                score_range = None
        else:
            score_range = None
        
        if st.button("🔄 重新加载数据", key=f"reload_{db_key}"):
            st.cache_data.clear()
            st.rerun()
    
    # 应用筛选 - 但要确保gated_delta_net和delta_net不被筛选掉
    display_df = df.copy()
    
    # 先分离特殊模型
    gated_row = display_df[display_df['模型名称'].str.lower() == 'gated_delta_net']
    delta_row = display_df[display_df['模型名称'].str.lower() == 'delta_net']
    other_rows = display_df[~display_df['模型名称'].str.lower().isin(['gated_delta_net', 'delta_net'])]
    
    if show_only_complete:
        benchmark_cols = ['ARC Challenge', 'ARC Easy', 'BoolQ', 'FDA', 'HellaSwag', 
                         'LAMBDA OpenAI', 'OpenBookQA', 'PIQA', 'Social IQA', 
                         'SQuAD Completion', 'SWDE', 'WinoGrande']
        complete_mask = other_rows[benchmark_cols].notna().all(axis=1)
        other_rows = other_rows[complete_mask]
    
    # Score范围筛选（只应用于其他模型）
    if score_range and 'Score' in other_rows.columns:
        score_series = pd.to_numeric(other_rows['Score'], errors='coerce')
        score_mask = (score_series >= score_range[0]) & (score_series <= score_range[1])
        other_rows = other_rows[score_mask]
    
    # 应用排序，但保持gated_delta_net和delta_net在前两行
    # 对其他模型应用排序
    if sort_by in other_rows.columns and len(other_rows) > 0:
        if sort_by in ['Score', 'Loss', '测试集均值']:
            numeric_col = pd.to_numeric(other_rows[sort_by], errors='coerce')
            other_rows = other_rows.loc[numeric_col.sort_values(ascending=sort_ascending).index]
        else:
            other_rows = other_rows.sort_values(sort_by, ascending=sort_ascending)
    
    # 重新组合：gated_delta_net -> delta_net -> 其他模型
    display_df = pd.concat([gated_row, delta_row, other_rows], ignore_index=True)
    
    # 调试：打印最终显示的前几行
    if len(display_df) > 0:
        print("最终显示的前3行:")
        print(display_df.head(3)['模型名称'].tolist())
    
    # 高亮函数
    def highlight_cells(data):
        """高亮最优值、SOTA模型和baseline行"""
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        
        # 定义需要高亮的列
        score_column = 'Score'
        loss_column = 'Loss'
        avg_column = '测试集均值'
        benchmark_columns = [
            'ARC Challenge', 'ARC Easy', 'BoolQ', 'FDA', 'HellaSwag', 
            'LAMBDA OpenAI', 'OpenBookQA', 'PIQA', 'Social IQA', 
            'SQuAD Completion', 'SWDE', 'WinoGrande'
        ]
        
        # 高亮Score最高值
        if score_column in data.columns:
            numeric_series = pd.to_numeric(data[score_column], errors='coerce')
            if not numeric_series.isna().all():
                max_idx = numeric_series.idxmax()
                styles.loc[max_idx, score_column] = 'background-color: #28a745; color: white; font-weight: bold; border-radius: 3px'
        
        # 高亮Loss最低值  
        if loss_column in data.columns:
            numeric_series = pd.to_numeric(data[loss_column], errors='coerce')
            if not numeric_series.isna().all():
                min_idx = numeric_series.idxmin()
                styles.loc[min_idx, loss_column] = 'background-color: #28a745; color: white; font-weight: bold; border-radius: 3px'
        
        # 高亮测试集均值最高值
        if avg_column in data.columns:
            numeric_series = pd.to_numeric(data[avg_column], errors='coerce')
            if not numeric_series.isna().all():
                max_idx = numeric_series.idxmax()
                styles.loc[max_idx, avg_column] = 'background-color: #28a745; color: white; font-weight: bold; border-radius: 3px'
        
        # 高亮各个benchmark的最优值
        for col in benchmark_columns:
            if col in data.columns:
                numeric_series = pd.to_numeric(data[col], errors='coerce')
                if not numeric_series.isna().all():
                    max_idx = numeric_series.idxmax()
                    styles.loc[max_idx, col] = 'background-color: #28a745; color: white; font-weight: bold; border-radius: 3px'
        
        # 高亮gated_delta_net（SOTA模型）- 金色
        for idx in data.index:
            model_name = str(data.loc[idx, '模型名称']).lower()
            if model_name == 'gated_delta_net':
                for col in data.columns:
                    if styles.loc[idx, col] == '':
                        styles.loc[idx, col] = 'background-color: #ffd700; color: #000; font-weight: bold; border-left: 4px solid #ff8c00; box-shadow: 0 2px 4px rgba(255, 215, 0, 0.3)'
                    else:
                        styles.loc[idx, col] += '; border-left: 4px solid #ff8c00; box-shadow: 0 2px 4px rgba(255, 215, 0, 0.3)'
        
        # 高亮delta_net（baseline行）- 黄色
        for idx in data.index:
            model_name = str(data.loc[idx, '模型名称']).lower()
            if model_name == 'delta_net':
                for col in data.columns:
                    if styles.loc[idx, col] == '':
                        styles.loc[idx, col] = 'background-color: #fff3cd; border-left: 4px solid #ffc107; font-weight: bold'
                    else:
                        styles.loc[idx, col] += '; border-left: 4px solid #ffc107'
        
        return styles
    
    # 显示表格
    st.dataframe(
        display_df.style.apply(highlight_cells, axis=None),
        use_container_width=True,
        height=min(700, len(display_df) * 45 + 100),
        hide_index=True
    )
    
    # 性能摘要和图表
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 各指标最优模型")
        summary_df = create_performance_summary(df)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Top 5 模型对比")
        
        # 按Score排序显示Top 5
        if 'Score' in df.columns:
            chart_df = df.dropna(subset=['Score']).nlargest(5, 'Score')
            
            if not chart_df.empty:
                color_scale = 'reds' if is_db2 else 'viridis'
                fig = px.bar(
                    chart_df,
                    x='Score',
                    y='模型名称',
                    orientation='h',
                    color='Score',
                    color_continuous_scale=color_scale,
                    title="Top 5 模型Score排行"
                )
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("缺少Score数据")
    
    # 下载功能
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载当前表格 (CSV)",
            data=csv_data,
            file_name=f'{config["name"]}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            key=f"download_{db_key}"
        )
    
    with col2:
        if st.button("📊 显示数据统计", key=f"stats_{db_key}"):
            st.info(f"""
            **{config['name']}数据统计:**
            - 总模型数: {len(df)}
            - 显示模型数: {len(display_df)}
            - 数据来源: {data_source}
            - 最后更新: {cache.get('last_update', '未知')}
            - 有Score数据: {len(df.dropna(subset=['Score']))} 个模型
            """)

def main():
    # 初始化页面状态
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "database1"
    
    # 页面导航
    st.markdown("### 🗂️ 数据库选择")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🏆 数据库1 ", 
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "database1" else "secondary"):
            st.session_state.current_page = "database1"
    
    with col2:
        if st.button("🔥 数据库2 ", 
                    use_container_width=True,
                    type="primary" if st.session_state.current_page == "database2" else "secondary"):
            st.session_state.current_page = "database2"
    
    st.markdown("---")
    
    # 渲染当前选中的页面
    render_database_page(st.session_state.current_page)

if __name__ == "__main__":
    main()