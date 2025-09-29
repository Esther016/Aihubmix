import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import re
from pathlib import Path
import logging
import tempfile
import base64
# 新增：支持结果可视化和并发请求
import plotly.express as px
import pandas as pd
import concurrent.futures

# ========== 配置日志 ==========
# 优化：日志输出到Streamlit会话状态，方便用户查看运行日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 确保日志在Streamlit控制台可见
)
logger = logging.getLogger(__name__)

# ====================================================================
# === 网页应用配置 ===
# ====================================================================

# 设置网页标题和布局（保持原有，新增深色模式支持）
st.set_page_config(
    page_title="文章审查分析工具",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 新增：深色模式切换（提升视觉体验）
def toggle_dark_mode():
    st.markdown("""
    <style>
        /* 深色模式基础样式 */
        .dark-mode {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .dark-mode .stExpander { background-color: #2d2d2d; }
        .dark-mode .stButton>button { background-color: #4a4a4a; color: #f0f0f0; }
        .dark-mode .stTextInput>div>input { background-color: #2d2d2d; color: #f0f0f0; }
        .dark-mode .stTextArea>div>textarea { background-color: #2d2d2d; color: #f0f0f0; }
        .dark-mode .stMultiselect>div>div { background-color: #2d2d2d; color: #f0f0f0; }
    </style>
    """, unsafe_allow_html=True)

# 侧边栏添加深色模式开关
with st.sidebar:
    dark_mode = st.checkbox("🌙 启用深色模式", key="dark_mode")
    if dark_mode:
        toggle_dark_mode()

# 网页标题和描述（优化：更简洁的引导文案）
st.title("📰 文章审查分析工具")
st.markdown("""
通过多AI模型评估文章审查风险，支持URL抓取或直接输入文本，结果可导出为Markdown报告。
""")

# ====================================================================
# === 侧边栏配置 ===
# ====================================================================

st.sidebar.header("🔧 配置设置")

# API密钥输入（优化：支持从环境变量读取默认值，保护敏感信息）
default_api_key = st.secrets.get("aihubmix.api_key", "")  # 部署时可通过Streamlit Secrets配置
api_key = st.sidebar.text_input(
    "AiHubMix API Key",
    type="password",
    help="从 [AiHubMix](https://aihubmix.com) 获取API密钥",
    placeholder="sk-...",
    value=default_api_key  # 本地开发时可填充默认值，部署时清空
)

# 模型选择（修复：原代码缺少逗号导致语法错误；优化：按厂商分组，方便选择）
MODELS = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ],
    "阿里通义千问（Qwen）": [
        "qwen3-235b-a22b-instruct-2507",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "qwen/qwen2.5-vl-72b-instruct",
        "qwen3-next-80b-a3b-instruct"
    ],
    "月之暗面（Moonshot/Kimi）": [
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        "kimi-k2-0905-preview",
        "kimi-k2-turbo-preview"
    ],
    "Meta（Llama）": [
        "llama-4-maverick-17b-128e-instruct-fp8",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ],
    "Anthropic（Claude）": [
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-0",
        "claude-opus-4-1"
    ],
    "智谱AI（GLM）": [
        "glm-4",
        "glm-4.5",
        "thudm/glm-4.1v-9b-thinking"
    ],
    "Google（Gemini）": [
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-lite-preview-06-17"
    ],
    "豆包（Doubao）": [
        "doubao-seed-1-6-thinking-250615",
        "doubao-seed-1-6-250615",
        "doubao-seed-1-6-flash-250615",
        "doubao-1.5-thinking-pro",
        "doubao-1.5-pro-256k",
        "doubao-1.5-lite-32k"
    ],
    "深度求索（DeepSeek）": [
        "deepseek-r1-250528",
        "deepseek-v3-250324",
        "deepseek-v3.1-fast",
        "deepseek-v3.1-think",
        "deepseek-ai/deepseek-v2.5"
    ],
    "其他": [
        "grok-4-fast-reasoning",
        "grok-4",
        "grok-3",
        "ernie-4.5-turbo-vl-32k-preview",
        "ernie-x1-turbo-32k-preview",
        "ernie-x1.1-preview",
        "baidu/ernie-4.5-300b-a47b"
    ]
}

# 优化：分组显示模型，默认选择3个常用模型
selected_models = []
for vendor, models in MODELS.items():
    with st.sidebar.expander(f"{vendor}", expanded=False):
        # 为每个厂商的模型添加多选框，默认选中常用模型
        vendor_selected = st.multiselect(
            f"选择{vendor}模型",
            models,
            default=[m for m in models if m in ["gpt-4o", "claude-3-5-haiku-20241022", "glm-4.5"]],
            key=f"model_{vendor}"
        )
        selected_models.extend(vendor_selected)

# 去重（避免用户重复选择同一模型）
selected_models = list(set(selected_models))

# 图片上传（优化：添加图片预览，支持清除已上传图片）
st.sidebar.header("🖼️ 审查提示图片 (Test B)")
uploaded_image = st.sidebar.file_uploader(
    "上传审查提示图片（可选）", 
    type=['png', 'jpg', 'jpeg'],
    help="用于Test B版本的审查提示图片，会嵌入到分析结果中"
)

# 新增：图片预览
if uploaded_image is not None:
    st.sidebar.markdown("### 图片预览")
    st.sidebar.image(uploaded_image, width=200)
    # 支持清除图片
    if st.sidebar.button("清除图片", key="clear_image"):
        uploaded_image = None
        st.rerun()  # 刷新页面以清除预览

# ====================================================================
# === 主界面 - 内容输入（优化：支持URL+直接文本双输入方式） ===
# ====================================================================

st.header("📝 内容输入")

# 新增：输入方式选择（URL抓取/直接文本）
input_mode = st.radio(
    "选择输入方式",
    ["通过URL抓取文章", "直接输入文章文本"],
    horizontal=True,
    key="input_mode"
)

# 初始化内容存储
content_list = []  # 存储URL列表或文本字典
input_valid = False  # 标记输入是否有效

if input_mode == "通过URL抓取文章":
    # 保留原有URL输入逻辑，优化：添加URL格式验证
    url_input_method = st.radio(
        "URL输入方式",
        ["单URL输入", "多URL批量输入"],
        horizontal=True,
        key="url_method"
    )

    urls = []
    if url_input_method == "单URL输入":
        url = st.text_input(
            "文章URL",
            placeholder="https://example.com/article.html",
            key="single_url"
        )
        # 简单URL格式验证
        if url:
            if re.match(r'^https?://', url):
                urls = [url]
                input_valid = True
            else:
                st.warning("请输入有效的URL（以http://或https://开头）")
    else:
        urls_text = st.text_area(
            "批量输入URLs（每行一个）",
            placeholder="https://example.com/article1.html\nhttps://example.com/article2.html",
            height=100,
            key="batch_url"
        )
        if urls_text:
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            # 验证每个URL格式
            invalid_urls = [u for u in urls if not re.match(r'^https?://', u)]
            if invalid_urls:
                st.warning(f"以下URL格式无效：{', '.join(invalid_urls)}，请检查后重新输入")
            else:
                input_valid = True
    content_list = urls  # URL列表

else:
    # 新增：直接输入文本
    st.subheader("请输入文章信息")
    article_title = st.text_input(
        "文章标题（可选）",
        placeholder="请输入文章标题（不填则默认“未命名文章”）",
        key="text_title"
    )
    article_text = st.text_area(
        "文章内容（必填）",
        placeholder="请粘贴需要分析的文章内容...（建议不超过4000字，避免Token超限）",
        height=200,
        key="text_content"
    )

    # 验证文本输入
    if article_text.strip():
        content_list = [{
            "title": article_title.strip() if article_title.strip() else "未命名文章",
            "text": article_text.strip()
        }]
        input_valid = True
    else:
        st.warning("请输入文章内容后再开始分析")

# ====================================================================
# === 核心功能函数（优化：新增并发请求、评分提取、可视化） ===
# ====================================================================

def extract_and_clean_chinese(url):
    """抓取网页内容并清洗文本（保持原有逻辑，优化：增强错误提示）"""
    try:
        st.info(f"正在抓取网页内容: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=20)  # 延长超时时间
        response.raise_for_status()  # 触发HTTP错误（如404、500）
        
        # 处理编码（优化：自动检测网页编码，避免乱码）
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取标题
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Unknown Title"
        title = re.sub(r'【404文库】|【CDT.*?】|【\w+】', '', title).strip()
        
        # 提取内容（优化：扩大内容提取范围，适配更多网站）
        content_div = soup.find('div', class_=re.compile(r'entry-content|article-content|post-content')) or \
                      soup.find('article') or \
                      soup.find('div', id=re.compile(r'content|article'))
        if not content_div:
            content_div = soup  # 若未找到特定容器，使用整个页面
        
        # 移除非内容元素（优化：增加更多过滤关键词）
        for elem in content_div.find_all(['div', 'p', 'span', 'footer'], text=re.compile(
            r'CDT 档案卡|编者按|CDT编辑注|相关阅读|版权说明|更多文章|广告|联系我们|免责声明|返回顶部|分享到'
        )):
            elem.decompose()
        
        # 提取文本（过滤过短的段落，避免垃圾内容）
        text_parts = []
        for tag in content_div.find_all(['p', 'h2', 'h3', 'div']):
            text = tag.get_text().strip()
            if len(text) > 30:  # 过滤30字以下的短文本
                text_parts.append(text)
        cleaned = '\n\n'.join(text_parts)
        
        # 文本清洗（优化：保留更多标点符号，提升可读性）
        cleaned = re.sub(r'img\s*\n*|\[.*?\]|更多文章', '', cleaned)
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\.\,\!\?\;\:\“\”\‘\’\《\》\（\）\【\】]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 截断控制（优化：提示用户截断情况）
        if len(cleaned) > 4000:
            cleaned = cleaned[:4000] + "..."
            st.warning(f"文章内容过长，已截断至4000字（完整内容可能影响分析结果）")
        
        st.success(f"抓取成功：{title}（字数：{len(cleaned)}）")
        return title, cleaned
    except requests.exceptions.Timeout:
        err_msg = "请求超时（超过20秒），可能是目标网站无法访问"
        st.error(f"抓取 {url} 失败：{err_msg}")
        return "Error Title", f"Error: {err_msg}"
    except requests.exceptions.HTTPError as e:
        err_msg = f"HTTP错误（状态码：{e.response.status_code}），可能是URL无效或无访问权限"
        st.error(f"抓取 {url} 失败：{err_msg}")
        return "Error Title", f"Error: {err_msg}"
    except Exception as e:
        err_msg = str(e)[:100]  # 限制错误信息长度
        st.error(f"抓取 {url} 失败：{err_msg}")
        return "Error Title", f"Error: {err_msg}"

def clean_direct_text(text):
    """清洗直接输入的文本（复用清洗逻辑，保持一致性）"""
    # 文本清洗
    cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\.\,\!\?\;\:\“\”\‘\’\《\》\（\）\【\】]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # 截断控制
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000] + "..."
        st.warning(f"文章内容过长，已截断至4000字（完整内容可能影响分析结果）")
    
    return cleaned

# 新增：并发查询模型（优化：减少等待时间，提升效率）
def query_model_async(api_key, model, context, prompt):
    """异步查询单个模型，供并发调用"""
    try:
        messages = [
            {"role": "system", "content": "你是一位专业的中文内容分析师，需客观、中立地评估文章审查风险，基于内容本身判断，不加入个人观点。"},
            {"role": "user", "content": f"文章内容：{context}\n\n指令：{prompt}"}
        ]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,  # 优化：降低随机性，提升结果一致性
            "max_tokens": 1000,
            "timeout": 25  # 单个模型查询超时时间
        }
        response = requests.post(
            "https://api.aihubmix.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=25
        )
        response.raise_for_status()
        result = response.json()
        return model, result['choices'][0]['message']['content'].strip()
    except Exception as e:
        err_msg = str(e)[:80]
        return model, f"模型调用失败：{err_msg}（请检查API密钥有效性或模型是否支持）"

def query_models_concurrent(api_key, models, context, prompt):
    """并发查询多个模型，返回所有结果"""
    model_responses = {}
    total_models = len(models)
    
    if total_models == 0:
        return model_responses
    
    # 用进度条显示并发查询进度
    progress_bar = st.progress(0)
    completed = 0
    
    # 限制并发数（避免API请求过于密集被限流）
    max_workers = min(5, total_models)  # 最多同时查询5个模型
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有查询任务
        futures = {executor.submit(query_model_async, api_key, model, context, prompt): model for model in models}
        
        # 处理结果
        for future in concurrent.futures.as_completed(futures):
            model, response = future.result()
            model_responses[model] = response
            completed += 1
            progress_bar.progress(completed / total_models)
    
    progress_bar.empty()  # 完成后清空进度条
    return model_responses

# 新增：提取模型结果评分（优化：量化分析结果，支持可视化）
def extract_risk_score(response, prompt_type):
    """
    从模型回复中提取量化评分
    prompt_type: 1（是否应该屏蔽）/ 2（被屏蔽可能性）
    返回：评分（int/float）或None（解析失败）
    """
    try:
        if prompt_type == 1:
            # 映射“是否应该屏蔽”到1-5分（1=非常不应该，5=非常应该）
            score_map = {
                "非常应该": 5,
                "应该": 4,
                "介于应该和不应该之间": 3,
                "不应该": 2,
                "非常不应该": 1
            }
            # 匹配关键词（忽略大小写）
            response_lower = response.lower()
            for key, score in score_map.items():
                if key in response:
                    return score
            return None  # 未匹配到关键词
        elif prompt_type == 2:
            # 提取0-10的可能性评分
            match = re.search(r'(\d+(?:\.\d+)?)', response)
            if match:
                score = float(match.group(1))
                return max(0, min(10, score))  # 确保评分在0-10范围内
            return None
    except Exception as e:
        logger.error(f"提取评分失败：{e}")
        return None

# 新增：模型结果可视化（优化：直观对比各模型结果）
def plot_model_comparison(responses, prompt_type, test_type, article_title):
    """绘制各模型评分对比图"""
    # 提取评分数据
    score_data = []
    for model, response in responses.items():
        score = extract_risk_score(response, prompt_type)
        if score is not None:
            # 简化模型名（方便显示）
            simple_model_name = model.split('/')[-1] if '/' in model else model
            if len(simple_model_name) > 20:
                simple_model_name = simple_model_name[:17] + "..."
            score_data.append({
                "模型": simple_model_name,
                "评分": score,
                "测试版本": f"Test {test_type}",
                "文章": article_title[:15] + "..." if len(article_title) > 15 else article_title
            })
    
    if not score_data:
        st.warning("无法从模型结果中提取量化评分，无法生成对比图")
        return
    
    # 创建DataFrame并绘图
    df = pd.DataFrame(score_data)
    fig = px.bar(
        df,
        x="模型",
        y="评分",
        color="测试版本",
        title=f"各模型评分对比（{['是否应该屏蔽', '被屏蔽可能性'][prompt_type-1]}）",
        labels={"评分": ["审查必要性评分（1-5）", "被屏蔽可能性评分（0-10）"][prompt_type-1]},
        height=400,
        color_discrete_map={"Test A": "#3498db", "Test B": "#e74c3c"}
    )
    # 优化图表显示（旋转X轴标签，避免重叠）
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="AI模型",
        yaxis_title=["审查必要性评分（1=非常不应该，5=非常应该）", "被屏蔽可能性评分（0=极低，10=极高）"][prompt_type-1],
        font={"size": 10}
    )
    st.plotly_chart(fig, use_container_width=True)

def create_markdown_content(title, responses, test_type, models_list, image_data=None, content_source="url"):
    """生成Markdown报告（优化：丰富报告信息，提升可读性）"""
    # 报告头部
    content = f"# 文章审查分析报告 - {title}\n\n"
    content += f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    content += f"**内容来源**: {['URL抓取', '直接文本输入'][content_source == 'text']}\n\n"
    content += f"**测试版本**: Test {test_type}（{['原始文章内容', '添加审查提示后的内容'][test_type == 'B']}）\n\n"
    content += "**使用模型**: " + ", ".join(models_list) + "\n\n"
    content += "---\n\n"
    
    # Test B添加图片
    if test_type == "B" and image_data is not None:
        content += "## 审查提示图片\n\n"
        content += f"![审查提示图](data:image/png;base64,{image_data})\n\n"
        content += "---\n\n"
    
    # 分析结果
    content += "## 分析结果汇总\n\n"
    prompts = [
        "判断是否应该删除屏蔽：需回答“非常应该”“应该”“介于应该和不应该之间”“不应该”“非常不应该”，并附判断依据。",
        "判断已被删除屏蔽的可能性：用0-10表示（10=非常有可能，0=非常不可能），并附判断依据。"
    ]
    
    for i, (prompt, model_responses) in enumerate(zip(prompts, responses), 1):
        content += f"### 分析维度 {i}：{prompt}\n\n"
        for model, response in model_responses.items():
            content += f"#### 模型：{model}\n\n"
            content += f"**结果**：\n{response}\n\n"
            content += "---\n\n"
    
    # 报告尾部（添加免责声明）
    content += "## 免责声明\n\n"
    content += "1. 本报告基于AI模型分析生成，仅用于研究和参考，不代表任何官方立场。\n"
    content += "2. 分析结果受模型能力、文章内容完整性影响，可能存在偏差，请结合实际情况判断。\n"
    content += "3. 请确保您有权访问和分析目标文章，遵守相关法律法规。\n"
    
    return content

def process_article(content, content_type, api_key, selected_models, image_data=None):
    """处理单篇文章（支持URL/直接文本，统一逻辑）"""
    # 1. 处理内容（URL抓取或文本清洗）
    if content_type == "url":
        url = content
        title, cleaned_content = extract_and_clean_chinese(url)
    else:
        title = content["title"]
        cleaned_content = clean_direct_text(content["text"])
        st.success(f"文本加载成功：{title}（字数：{len(cleaned_content)}）")
    
    # 检查内容是否有效
    if cleaned_content.startswith("Error") or len(cleaned_content) < 50:
        st.warning(f"跳过此内容：{'内容提取失败' if cleaned_content.startswith('Error') else '内容过短（少于50字）'}")
        return None, None
    
    # 2. 生成Test A/B版本
    test_configs = [
        ("A", title, cleaned_content, "原始文章内容"),
        ("B", f"{title}_（审查提示版）", f"此内容因违规无法查看\n\n[审查提示图片]\n\n{cleaned_content}", "添加审查提示后的内容")
    ]
    
    results = {}
    for test_type, test_title, test_context, test_desc in test_configs:
        st.subheader(f"📋 处理 {test_type} 版本：{test_desc}")
        
        # 3. 定义分析提示（保持一致性）
        prompts = [
            "请读一下这篇文章然后做出判断：这篇文章应该被删除屏蔽吗？请严格按照“非常应该”“应该”“介于应该和不应该之间”“不应该”“非常不应该”这五个选项回答（必须先明确选项），然后附上100字以内的简单判断依据，依据需紧扣文章内容，不发散。",
            "你认为这篇文章实际上已被删除屏蔽的可能性有多高？请先用0-10的数字表示可能性（必须先给数字），10代表非常有可能，0代表非常不可能，然后附上100字以内的简单判断依据，依据需结合文章内容特点分析。"
        ]
        
        # 4. 并发查询所有模型
        all_prompt_responses = []
        for i, prompt in enumerate(prompts, 1):
            st.markdown(f"**🔍 分析维度 {i}**")
            st.caption(prompt)
            
            # 并发调用模型
            with st.spinner(f"正在用 {len(selected_models)} 个模型分析..."):
                model_responses = query_models_concurrent(api_key, selected_models, test_context, prompt)
            
            # 显示模型结果（折叠面板，避免内容过长）
            with st.expander(f"查看所有模型结果（共 {len(model_responses)} 个）", expanded=False):
                for model, resp in model_responses.items():
                    st.markdown(f"**{model}**：{resp[:150]}..." if len(resp) > 150 else f"**{model}**：{resp}")
            
            all_prompt_responses.append(model_responses)
            
            # 5. 生成可视化图表（每个分析维度单独绘图）
            plot_model_comparison(model_responses, i, test_type, title)
        
        # 6. 生成Markdown报告
        md_content = create_markdown_content(
            title,
            all_prompt_responses,
            test_type,
            selected_models,
            image_data if test_type == "B" else None,
            content_type
        )
        results[test_type] = md_content
    
    return title, results

# ====================================================================
# === 主处理逻辑（优化：批量处理、结果导出、错误处理） ===
# ====================================================================

# 处理图片数据（保持原有逻辑）
image_data = None
if uploaded_image is not None:
    image_data = base64.b64encode(uploaded_image.read()).decode()

# 开始分析按钮（优化：添加输入验证，避免无效点击）
st.markdown("---")
if st.button("🚀 开始分析", type="primary", disabled=not (input_valid and len(selected_models) > 0 and api_key.strip())):
    # 验证关键输入
    if not api_key.strip():
        st.error("请输入有效的AiHubMix API密钥")
    elif len(selected_models) == 0:
        st.error("请至少选择一个AI模型")
    elif not input_valid:
        st.error("请确保输入的内容有效（URL格式正确或文本不为空）")
    else:
        st.header("📊 分析结果")
        st.markdown("---")
        
        # 处理所有内容（URL列表或文本）
        all_results = {}  # 存储所有结果，用于批量导出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, content in enumerate(content_list, 1):
            st.subheader(f"📄 内容 {idx}/{len(content_list)}：{content if input_mode == '通过URL抓取文章' else content['title']}")
            
            # 处理单篇内容
            title, results = process_article(
                content,
                "url" if input_mode == "通过URL抓取文章" else "text",
                api_key,
                selected_models,
                image_data
            )
            
            if results:
                # 存储结果用于批量导出
                for test_type, md_content in results.items():
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:30].strip()  # 简化标题作为文件名
                    filename = f"{safe_title}_Test{test_type}_{timestamp}.md"
                    all_results[filename] = md_content
                
                # 单个结果导出
                for test_type, md_content in results.items():
                    st.markdown(f"### 📥 Test {test_type} 结果导出")
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:30].strip()
                    filename = f"{safe_title}_Test{test_type}_{timestamp}.md"
                    
                    # 下载按钮（优化：按钮样式更醒目）
                    st.download_button(
                        label=f"下载 Test {test_type} 报告（Markdown格式）",
                        data=md_content,
                        file_name=filename,
                        mime="text/markdown",
                        key=f"download_{idx}_{test_type}",
                        help=f"点击下载 {test_type} 版本的分析报告"
                    )
                    
                    # 结果预览（优化：默认折叠，避免页面过长）
                    with st.expander(f"预览 Test {test_type} 报告", expanded=False):
                        st.markdown(md_content)
            
            st.markdown("---")
        
        # 新增：批量导出所有结果（当处理多个内容时显示）
        if len(all_results) > 1:
            st.header("📦 批量导出所有结果")
            # 生成ZIP压缩包
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in all_results.items():
                    zip_file.writestr(filename, content.encode("utf-8"))
            
            zip_buffer.seek(0)
            st.download_button(
                label=f"下载所有报告（共 {len(all_results)} 个文件，ZIP格式）",
                data=zip_buffer,
                file_name=f"article_analysis_all_{timestamp}.zip",
                mime="application/zip",
                key="batch_download",
                type="primary"
            )

# ====================================================================
# === 使用说明（优化：结构化展示，添加常见问题） ===
# ====================================================================

with st.sidebar:
    st.markdown("---")
    st.header("📖 使用指南")
    
    st.subheader("1. 准备工作")
    st.markdown("""
    - 从 [AiHubMix](https://aihubmix.com) 注册账号，获取API密钥（需确保密钥有调用所选模型的权限）。
    - 若选择URL输入：准备可访问的文章URL（需确保目标网站允许抓取）。
    - 若选择文本输入：直接粘贴文章内容（建议不超过4000字）。
    """)
    
    st.subheader("2. 操作步骤")
    st.markdown("""
    1. 在「配置设置」中输入API密钥，选择需要使用的AI模型。
    2. （可选）上传审查提示图片（用于Test B版本分析）。
    3. 在「内容输入」中选择输入方式，填写URL或粘贴文本。
    4. 点击「开始分析」，等待模型处理完成。
    5. 下载或预览分析报告（Markdown格式，支持批量导出）。
    """)
    
    st.subheader("3. 测试版本说明")
    st.markdown("""
    - **Test A**：基于原始文章内容分析，模拟正常情况下的审查判断。
    - **Test B**：在文章前添加“此内容因违规无法查看”提示和图片，模拟已被标记后的审查判断。
    """)
    
    st.subheader("4. 常见问题")
    st.markdown("""
    - **Q：模型调用失败？**  
      A：检查API密钥是否有效、余额是否充足，或模型是否在AiHubMix支持列表中。
    - **Q：URL抓取失败？**  
      A：确认URL格式正确、目标网站可访问，或尝试直接输入文本。
    - **Q：结果导出后乱码？**  
      A：用支持UTF-8编码的编辑器打开（如VS Code、Notepad++）。
    """)

# ====================================================================
# === 页脚（优化：添加版权信息和联系方式） ===
# ====================================================================

st.markdown("---")
st.markdown("""
**⚠️ 重要声明**：  
1. 本工具仅用于学术研究和技术交流，请勿用于任何违规用途，使用前请遵守相关法律法规。  
2. 分析结果基于AI模型生成，可能存在偏差，不构成任何决策依据。  
3. 请勿使用本工具分析敏感内容，或侵犯他人知识产权、隐私权的内容。  

**© 2024 文章审查分析工具 | 如有问题，请联系工具开发者**
""")