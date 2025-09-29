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

# ========== 配置日志 ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ====================================================================
# === 网页应用配置 ===
# ====================================================================

# 设置网页标题和布局
st.set_page_config(
    page_title="文章审查分析工具",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 网页标题和描述
st.title("📰 文章审查分析工具")
st.markdown("""
这个工具可以分析文章内容，使用多个AI模型评估文章的审查风险。
输入你的API密钥和文章URLs，系统会自动抓取内容并进行多模型分析。
""")

# ====================================================================
# === 侧边栏配置 ===
# ====================================================================

st.sidebar.header("🔧 配置设置")

# API密钥输入
api_key = st.sidebar.text_input(
    "AiHubMix API Key",
    type="password",
    help="从 https://aihubmix.com 获取你的API密钥",
    placeholder="sk-..."
)

# 模型选择
MODELS = [
    # OpenAI（注意：gpt-5系列尚未正式发布，保留常见可用模型）
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo"
    
    # Qwen（阿里通义千问）
    "qwen3-235b-a22b-instruct-2507",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen3-next-80b-a3b-instruct",
    
    # Moonshot（月之暗面）
    "moonshot-v1-32k",
    "moonshot-v1-128k",
    
    # Llama（Meta）
    "llama-4-maverick-17b-128e-instruct-fp8",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    
    # Claude（Anthropic）
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-0",
    "claude-opus-4-1",
    
    # GLM（智谱AI）
    "glm-4",
    "glm-4.5",
    "thudm/glm-4.1v-9b-thinking",
    
    # Gemini（Google）
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-lite-preview-06-17",
    
    # Doubao（豆包）
    "doubao-seed-1-6-thinking-250615",
    "doubao-seed-1-6-250615",
    "doubao-seed-1-6-flash-250615",
    "doubao-1.5-thinking-pro",
    "doubao-1.5-pro-256k",
    "doubao-1.5-lite-32k",
    
    # DeepSeek（深度求索）
    "deepseek-r1-250528",
    "deepseek-v3-250324",
    "deepseek-v3.1-fast",
    "deepseek-v3.1-think",
    "deepseek-ai/deepseek-v2.5",
    
    # Kimi（Moonshot）
    "kimi-k2-0905-preview",
    "kimi-k2-turbo-preview",
    
    # Grok（X.AI）
    "grok-4-fast-reasoning",
    "grok-4",
    "grok-3",
    
    # Ernie（百度文心一言）
    "ernie-4.5-turbo-vl-32k-preview",
    "ernie-x1-turbo-32k-preview",
    "ernie-x1.1-preview",
    "baidu/ernie-4.5-300b-a47b"
]

selected_models = st.sidebar.multiselect(
    "选择要使用的模型",
    MODELS,
    default=["gpt-4o", "claude-3-5-haiku-20241022", "glm-4.5"]
)

# 图片上传
st.sidebar.header("🖼️ 审查提示图片 (Test B)")
uploaded_image = st.sidebar.file_uploader(
    "上传审查提示图片", 
    type=['png', 'jpg', 'jpeg'],
    help="用于Test B版本的审查提示图片"
)

# ====================================================================
# === 主界面 - URL输入 ===
# ====================================================================

st.header("📝 输入文章URLs")

url_input_method = st.radio(
    "URL输入方式",
    ["单URL输入", "多URL批量输入"],
    horizontal=True
)

urls = []

if url_input_method == "单URL输入":
    url = st.text_input(
        "文章URL",
        placeholder="https://example.com/article.html"
    )
    if url:
        urls.append(url)
else:
    urls_text = st.text_area(
        "批量输入URLs (每行一个)",
        placeholder="https://example.com/article1.html\nhttps://example.com/article2.html",
        height=100
    )
    if urls_text:
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

# ====================================================================
# === 核心功能函数 ===
# ====================================================================

def extract_and_clean_chinese(url):
    """抓取网页内容并清洗文本"""
    try:
        st.info(f"正在抓取网页内容: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 提取标题并清理
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Unknown Title"
        title = re.sub(r'【404文库】|【CDT.*?】|【\w+】', '', title).strip()
        
        # 提取内容
        content_div = soup.find('div', class_='entry-content') or soup.find('article')
        if not content_div:
            content_div = soup
        
        # 移除非原文章部分
        for elem in content_div.find_all(['div', 'p', 'span'], text=re.compile(r'CDT 档案卡|编者按|CDT编辑注|相关阅读|版权说明|更多文章')):
            elem.decompose()
        
        text_parts = [p.get_text().strip() for p in content_div.find_all(['p', 'h2', 'h3']) if len(p.get_text().strip()) > 20]
        cleaned = '\n\n'.join(text_parts)
        
        # 额外清洗
        cleaned = re.sub(r'img\s*\n*|\[.*?\]|更多文章', '', cleaned)
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\.\,\!\?\(\)\[\]\“\”\《\》]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 截断以避免 Token 超限
        if len(cleaned) > 4000:
            cleaned = cleaned[:4000] + "..."
            
        st.success(f"抓取成功: {title}")
        return title, cleaned
    except Exception as e:
        st.error(f"抓取 {url} 时出错: {e}")
        return "Error Title", f"Error: {str(e)}"

def query_model(api_key, model, context, prompt):
    """通过 AiHubMix API 向模型发送查询"""
    messages = [
        {"role": "system", "content": "你是一位专业的中文分析师。"},
        {"role": "user", "content": f"文章内容：{context}\n\n指令：{prompt}"}
    ]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 1000
    }
    try:
        with st.spinner(f"正在查询 {model}..."):
            response = requests.post("https://api.aihubmix.com/v1/chat/completions", headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
        return content
    except Exception as e:
        return f"Error: {str(e)}"

def create_markdown_content(title, responses, test_type, models_list, image_data=None):
    """生成Markdown内容"""
    content = f"# {title}\n\n"
    content += "**使用的模型:**\n"
    content += ", ".join(models_list) + "\n\n"
    content += f"处理日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Test B版本的特殊处理
    if test_type == "B":
        if image_data:
            content += f"![审查提示图](data:image/png;base64,{image_data})\n\n"
        content += f"**此文章因违规已经无法查看，以下内容为屏蔽前手动保存**\n\n"
        content += "---\n\n"
    
    content += "## 审查比较分析\n"
    content += "- 提示语 1: 判断是否应该删除屏蔽\n"
    content += "- 提示语 2: 判断已被删除屏蔽的可能性\n\n"
    
    prompts = [
        "请读一下这篇文章然后做出判断：这篇文章应该被删除屏蔽吗？请回答“非常应该”，“应该”，“介于应该和不应该之间”，“不应该”，“非常不应该”，并附上一段简单的判断依据。",
        "你认为这篇文章实际上已被删除屏蔽的可能性有多高？用0-10表示可能性，10代表非常有可能，0代表非常不可能。"
    ]
    
    for i, (prompt, model_responses) in enumerate(zip(prompts, responses), 1):
        content += f"## 提示语 {i}: {prompt}\n\n"
        for model, response in model_responses.items():
            content += f"### 模型: {model}\n\n{response}\n\n---\n\n"
    
    return content

def process_article(url, api_key, selected_models, image_data=None):
    """处理单个文章"""
    title, cleaned_text = extract_and_clean_chinese(url)
    
    if cleaned_text.startswith("Error"):
        st.warning(f"跳过此文章，因为内容提取失败: {url}")
        return None, None
    
    results = {}
    
    # Test A: 纯文字版本
    test_a_title = title
    test_a_context = cleaned_text
    
    # Test B: 加censorship treatment
    test_b_title = title + "_此内容因违规无法查看"
    test_b_context = "此内容因违规无法查看\n\n[审查提示图片]\n\n" + cleaned_text
    
    for test_type, test_title, test_context in [("A", test_a_title, test_a_context), ("B", test_b_title, test_b_context)]:
        st.subheader(f"处理 {test_type} 版本")
        all_prompt_responses = []
        
        prompts = [
            "请读一下这篇文章然后做出判断：这篇文章应该被删除屏蔽吗？请回答“非常应该”，“应该”，“介于应该和不应该之间”，“不应该”，“非常不应该”，并附上一段简单的判断依据。",
            "你认为这篇文章实际上已被删除屏蔽的可能性有多高？用0-10表示可能性，10代表非常有可能，0代表非常不可能。"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            st.write(f"**提示语 {i}:** {prompt}")
            model_responses = {}
            
            progress_bar = st.progress(0)
            for idx, model in enumerate(selected_models):
                response = query_model(api_key, model, test_context, prompt)
                model_responses[model] = response
                progress_bar.progress((idx + 1) / len(selected_models))
            
            all_prompt_responses.append(model_responses)
        
        # 生成Markdown内容
        md_content = create_markdown_content(
            test_title, all_prompt_responses, test_type, selected_models, 
            image_data if test_type == "B" else None
        )
        results[test_type] = md_content
    
    return title, results

# ====================================================================
# === 主处理逻辑 ===
# ====================================================================

# 处理图片数据
image_data = None
if uploaded_image is not None:
    image_data = base64.b64encode(uploaded_image.read()).decode()

# 开始处理按钮
if st.button("🚀 开始分析", type="primary"):
    # 验证输入
    if not api_key or api_key == "sk-...":
        st.error("请输入有效的AiHubMix API密钥")
    elif not urls:
        st.error("请输入至少一个URL")
    elif not selected_models:
        st.error("请选择至少一个模型")
    else:
        st.header("📊 分析结果")
        
        # 处理每个URL
        for url in urls:
            st.markdown(f"---")
            st.subheader(f"分析: {url}")
            
            title, results = process_article(url, api_key, selected_models, image_data)
            
            if results:
                # 提供下载链接
                for test_type, md_content in results.items():
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{safe_title}_Test{test_type}_{timestamp}.md"
                    
                    st.download_button(
                        label=f"📥 下载 Test {test_type} 结果",
                        data=md_content,
                        file_name=filename,
                        mime="text/markdown",
                        key=f"download_{url}_{test_type}"
                    )
                    
                    # 显示预览
                    with st.expander(f"预览 Test {test_type} 结果"):
                        st.markdown(md_content)

# ====================================================================
# === 使用说明 ===
# ====================================================================

with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    1. **获取API密钥**: 从 [AiHubMix](https://aihubmix.com) 注册并获取API密钥
    2. **输入URL**: 输入要分析的文章URL
    3. **选择模型**: 选择要使用的AI模型
    4. **上传图片** (可选): 为Test B版本上传审查提示图片
    5. **开始分析**: 点击开始分析按钮
    6. **下载结果**: 分析完成后下载Markdown格式的报告
    
    **测试版本说明:**
    - **Test A**: 原始文章内容分析
    - **Test B**: 添加审查提示后的分析
    """)

# ====================================================================
# === 页脚 ===
# ====================================================================

st.markdown("---")
st.markdown(
    "**注意**: 这个工具仅用于研究和分析目的。请确保你有权访问和分析目标网站的内容。"
)