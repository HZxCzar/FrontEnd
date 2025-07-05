#!/usr/bin/env python3
"""
更新版AI模型结果表格展示
新增Score列，重新调整列顺序：
1. 模型名称 2. Score 3. 最低Loss 4. 测试集均值 5. 各个benchmark
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
    
    /* 数据源信息 */
    .data-source {
        background-color: #e7f3ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #007bff;
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
    
    /* 列标题样式 */
    .column-header {
        font-weight: bold;
        text-align: center;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# 数据源配置
LOCAL_CACHE_FILE = "cache.json"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/HZxCzar/FrontEnd/main/cache.json"

@st.cache_data(ttl=300)
def load_data():
    """智能加载数据：优先本地，后备远程"""
    data_source = ""
    
    # 首先尝试加载本地文件
    if os.path.exists(LOCAL_CACHE_FILE):
        try:
            with open(LOCAL_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_source = f"📁 本地文件: {LOCAL_CACHE_FILE}"
                st.session_state['data_source'] = data_source
                return data
        except Exception as e:
            st.warning(f"读取本地文件失败: {e}")
    
    # 如果本地文件不存在，尝试从GitHub加载
    try:
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            data_source = f"🌐 远程GitHub: {GITHUB_RAW_URL}"
            st.session_state['data_source'] = data_source
            return data
        else:
            st.error(f"GitHub数据加载失败，状态码: {response.status_code}")
    except requests.exceptions.Timeout:
        st.error("GitHub连接超时，请检查网络连接")
    except Exception as e:
        st.error(f"GitHub数据加载错误: {str(e)}")
    
    st.session_state['data_source'] = "❌ 无法加载数据"
    return {"results": []}

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

def get_min_loss(train_string):
    """从训练数据中提取最低loss"""
    if not train_string:
        return None
    
    train_string = train_string.replace('\r\n', '\n').replace('\r', '\n')
    lines = train_string.strip().split('\n')
    
    if len(lines) < 2:
        return None
    
    try:
        loss_line = lines[1]
        parts = loss_line.split(',')
        
        loss_values = []
        for part in parts[1:]:  # 跳过第一个元素（标签）
            try:
                loss_values.append(float(part.strip()))
            except ValueError:
                continue
        
        return min(loss_values) if loss_values else None
    except Exception as e:
        print(f"解析训练数据时出错: {e}")
        return None

def create_styled_table(df):
    """创建带样式的表格，突出显示最优值"""
    
    def highlight_cells(data):
        """高亮最优值和baseline行"""
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        
        # 定义需要高亮的列
        score_column = 'Score'
        loss_column = '最低Loss'
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
        
        # 高亮baseline行（delta_net）
        for idx in data.index:
            model_name = str(data.loc[idx, '模型名称']).lower()
            if model_name == 'delta_net':
                for col in data.columns:
                    if styles.loc[idx, col] == '':
                        styles.loc[idx, col] = 'background-color: #fff3cd; border-left: 4px solid #ffc107; font-weight: bold'
                    else:
                        styles.loc[idx, col] += '; border-left: 4px solid #ffc107'
        
        return styles
    
    # 应用样式并格式化
    styled_df = df.style.apply(highlight_cells, axis=None)
    
    # 格式化数值显示
    format_dict = {}
    for col in df.columns:
        if col != '模型名称':
            if col == 'Score':
                format_dict[col] = lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else ("N/A" if pd.isna(x) else str(x))
            else:
                format_dict[col] = lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else ("N/A" if pd.isna(x) else str(x))
    
    styled_df = styled_df.format(format_dict)
    return styled_df

def create_performance_summary(df):
    """创建性能摘要"""
    summary_columns = ['Score', '最低Loss', '测试集均值']
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
                if col == '最低Loss':
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

def main():
    # 页面标题
    st.markdown('<h1 class="main-title">🏆 AI模型性能排行榜</h1>', unsafe_allow_html=True)
    
    # 数据源信息
    data_source = st.session_state.get('data_source', '')
    if data_source:
        st.markdown(f'<div class="data-source">📊 数据源: {data_source}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 加载数据
    with st.spinner("🔄 正在智能加载数据..."):
        cache = load_data()
    
    results = cache.get("results", [])
    
    if not results:
        st.error("❌ 没有找到数据")
        st.info("**数据加载指南:**")
        st.markdown("""
        **本地测试:**
        1. 确保当前目录有 `cache.json` 文件
        2. 运行: `python update_cache.py` 生成数据
        
        **远程部署:**
        1. 推送 `cache.json` 到GitHub仓库
        2. 应用会自动从GitHub加载数据
        """)
        return
    
    # 处理数据，按新的列顺序组织
    table_data = []
    
    for result in results:
        if not result.get('name'):
            continue
            
        # 按照要求的列顺序：名字、Score、Loss、测试集均值、各个benchmark
        row = {'模型名称': result['name']}
        
        # Score（新增字段）
        score = result.get('score')
        row['Score'] = score if score is not None else np.nan
        
        # 最低loss
        min_loss = get_min_loss(result.get('train', ''))
        row['最低Loss'] = min_loss
        
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
        
        table_data.append(row)
    
    if not table_data:
        st.error("没有可显示的数据")
        return
    
    df = pd.DataFrame(table_data)
    
    # 统计卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 模型总数</h3>
            <h2>{len(table_data)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        complete_count = sum(1 for _, row in df.iterrows() 
                           if sum(pd.isna(row[col]) for col in ['ARC Challenge', 'ARC Easy', 'BoolQ'] if col in row) == 0)
        st.markdown(f"""
        <div class="metric-card">
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
                <div class="metric-card">
                    <h3>🎯 最高Score</h3>
                    <h2>{max_score:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
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
        <div class="metric-card">
            <h3>🕒 最后更新</h3>
            <h2>{last_update}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        baseline_perf = "N/A"
        for _, row in df.iterrows():
            if str(row['模型名称']).lower() == 'delta_net':
                if '测试集均值' in row and pd.notna(row['测试集均值']):
                    baseline_perf = f"{row['测试集均值']:.3f}"
                break
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏁 Baseline</h3>
            <h2>{baseline_perf}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 说明
    st.markdown("---")
    st.markdown("""
    <div class="legend">
        <strong>📖 图例说明:</strong><br>
        🟢 <strong>绿色高亮</strong>: 该列最优值 (Score、测试集均值、各benchmark越高越好，Loss越低越好)<br>
        🟡 <strong>黄色背景 + 橙色左边框</strong>: delta_net (Baseline模型)<br>
        📊 <strong>列顺序</strong>: 模型名称 → Score → 最低Loss → 测试集均值 → 12个benchmark详情
    </div>
    """, unsafe_allow_html=True)
    
    # 主表格
    st.markdown("### 📊 模型性能详细对比表")
    
    # 侧边栏选项
    with st.sidebar:
        st.header("🎛️ 显示选项")
        
        # 排序选项
        sort_options = ["模型名称", "Score", "最低Loss", "测试集均值"]
        sort_by = st.selectbox("排序依据", sort_options, index=1)  # 默认按Score排序
        sort_ascending = st.checkbox("升序排列", value=False)
        
        # 筛选选项
        show_only_complete = st.checkbox("只显示完整数据", value=False)
        
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
                    step=0.001
                )
            else:
                score_range = None
        else:
            score_range = None
        
        if st.button("🔄 重新加载数据"):
            st.cache_data.clear()
            st.rerun()
    
    # 应用筛选
    display_df = df.copy()
    
    if show_only_complete:
        benchmark_cols = ['ARC Challenge', 'ARC Easy', 'BoolQ', 'FDA', 'HellaSwag', 
                         'LAMBDA OpenAI', 'OpenBookQA', 'PIQA', 'Social IQA', 
                         'SQuAD Completion', 'SWDE', 'WinoGrande']
        complete_mask = display_df[benchmark_cols].notna().all(axis=1)
        display_df = display_df[complete_mask]
    
    # Score范围筛选
    if score_range and 'Score' in display_df.columns:
        score_series = pd.to_numeric(display_df['Score'], errors='coerce')
        score_mask = (score_series >= score_range[0]) & (score_series <= score_range[1])
        display_df = display_df[score_mask]
    
    # 应用排序
    if sort_by in display_df.columns:
        if sort_by in ['Score', '最低Loss', '测试集均值']:
            numeric_col = pd.to_numeric(display_df[sort_by], errors='coerce')
            display_df = display_df.loc[numeric_col.sort_values(ascending=sort_ascending).index]
        else:
            display_df = display_df.sort_values(sort_by, ascending=sort_ascending)
    
    # 显示表格
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(700, len(display_df) * 45 + 100)
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
                fig = px.bar(
                    chart_df,
                    x='Score',
                    y='模型名称',
                    orientation='h',
                    color='Score',
                    color_continuous_scale='viridis',
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
            file_name=f'ai_model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
    
    with col2:
        if st.button("📊 显示数据统计"):
            st.info(f"""
            **数据统计:**
            - 总模型数: {len(df)}
            - 显示模型数: {len(display_df)}
            - 数据来源: {st.session_state.get('data_source', '未知')}
            - 最后更新: {cache.get('last_update', '未知')}
            - 有Score数据: {len(df.dropna(subset=['Score']))} 个模型
            """)

if __name__ == "__main__":
    main()