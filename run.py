import streamlit as st
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import re
import sqlite3
import hashlib
import secrets
from typing import Optional
from urllib.parse import urlparse
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# -------------------------
# Global Session for reuse
# -------------------------
session = requests.Session()
retries = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# -------------------------
# Auth (persistent via SQLite)
# -------------------------
DB_PATH = Path(__file__).with_name("auth.db")


def get_db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            );
            """
        )
        conn.commit()


def hash_password(password: str, salt: Optional[str] = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(stored: str, provided_password: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    return hash_password(provided_password, salt) == stored


def sign_up(username: str, password: str) -> bool:
    if not username or not password:
        return False
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password)),
            )
            conn.commit()
        return True
    except Exception:
        return False


def log_in(username: str, password: str) -> bool:
    if not username or not password:
        return False
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if not row:
                return False
            if verify_password(row[0], password):
                return True
    except Exception:
        pass
    return False


# -------------------------
# Utilities
# -------------------------
def extract_and_clean_chinese(url: str):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = session.get(url, headers=headers, timeout=(5, 20))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Unknown Title"
        title = re.sub(r'【404文库】|【CDT.*?】|【\w+】', '', title).strip()
        content_div = soup.find('div', class_='entry-content') or soup.find('article')
        if not content_div:
            content_div = soup
        for elem in content_div.find_all(['div', 'p', 'span'], text=re.compile(r'CDT 档案卡|编者按|CDT编辑注|相关阅读|版权说明|更多文章')):
            elem.decompose()
        text_parts = [p.get_text().strip() for p in content_div.find_all(['p', 'h2', 'h3']) if len(p.get_text().strip()) > 20]
        cleaned = '\n\n'.join(text_parts)
        cleaned = re.sub(r'img\s*\n*|\[.*?\]|更多文章', '', cleaned)
        cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\.\,\!\?\(\)\[\]\"\"\《\》]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > 5000:
            cleaned = cleaned[:5000] + "..."
        return title, cleaned
    except Exception as e:
        return "Error Title", f"网页抓取失败: {str(e)}"


def query_single_model(api_url: str, api_key: str, model: str, context: str, prompt: str, max_retries: int = 2):
    """Query a single model with retry mechanism"""
    messages = [{"role": "user", "content": f"文章内容：{context}\n\n指令：{prompt}"}]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Special handling for different model types
    models_need_max_completion = ["gpt-5", "gpt-5-mini", "o3", "o4-mini"]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
    }
    
    if model in models_need_max_completion:
        payload["max_completion_tokens"] = 800
    else:
        payload["max_tokens"] = 800
    
    backoff = 2
    for retry in range(max_retries + 1):
        try:
            start = time.perf_counter()
            resp = session.post(api_url, headers=headers, json=payload, timeout=(5, 25))
            resp.raise_for_status()
            data = resp.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content'].strip()
            elif 'content' in data:
                content = data['content'].strip()
            else:
                content = f"Error: Unexpected response format: {data}"
            
            elapsed = time.perf_counter() - start
            return model, content
            
        except Exception as e:
            if retry < max_retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                return model, f"模型 {model} 调用失败: {str(e)}"


def call_provider_concurrent(api_url: str, api_key: str, models: list[str], context: str, prompt: str, provider_name: str):
    """Call multiple models concurrently for a single prompt"""
    results = OrderedDict((m, "未响应") for m in models)
    
    max_workers = min(8, len(models))  # Limit concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(query_single_model, api_url, api_key, m, context, prompt): m 
            for m in models
        }
        
        for future in as_completed(futures):
            model, response = future.result()
            results[model] = response
    
    return results


def create_zip_from_files(files: list[tuple[str, str]]) -> bytes:
    """Create a ZIP file in memory from a list of (filename, content) tuples."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for fname, content in files:
            zip_file.writestr(fname, content.encode('utf-8'))
    return zip_buffer.getvalue()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Censorship Compare", layout="wide")

# Initialize database
init_db()

# Initialize session state
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "generated_files" not in st.session_state:
    st.session_state.generated_files = []
if "show_results" not in st.session_state:
    st.session_state.show_results = False

st.title("Censorship Compare - AiHubMix & Hunyuan")

# -------------------------
# Authentication UI
# -------------------------
if st.session_state.auth_user:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.success(f"✓ Logged in as **{st.session_state.auth_user}**")
    with col_b:
        if st.button("Log out", key="logout_btn"):
            st.session_state.auth_user = None
            st.session_state.generated_files = []
            st.session_state.show_results = False
            st.rerun()
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sign Up")
        with st.form("signup_form"):
            su_user = st.text_input("New Username", key="su_user")
            su_pass = st.text_input("New Password", type="password", key="su_pass")
            signup_btn = st.form_submit_button("Create Account")
            if signup_btn:
                if sign_up(su_user, su_pass):
                    st.success("✓ Account created! Please log in.")
                else:
                    st.error("✗ Sign up failed: username already exists or invalid input")
    
    with col2:
        st.subheader("Log In")
        with st.form("login_form"):
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            login_btn = st.form_submit_button("Log in")
            if login_btn:
                if log_in(li_user, li_pass):
                    st.session_state.auth_user = li_user
                    st.rerun()
                else:
                    st.error("✗ Login failed: invalid username or password")

if not st.session_state.auth_user:
    st.stop()

# -------------------------
# Main Application
# -------------------------
st.header("Inputs")
col_left, col_right = st.columns(2)

with col_left:
    urls_text = st.text_area("Article URLs (one per line)", key="urls_input")
    prompts_text = st.text_area("Prompts (one per line)", key="prompts_input")

with col_right:
    st.markdown("**AiHubMix**")
    aihubmix_key = st.text_input("AIHUBMIX_API_KEY", type="password", key="aihubmix_key")
    aihubmix_models_all = [
        # OpenAI
        "gpt-5",
        "gpt-5-mini",
        "o3",
        "o4-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        # Qwen
        "Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen3-Next-80B-A3B-Instruct",
        # Moonshot
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        # Llama
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        # Claude
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-0",
        "claude-opus-4-1",
        # GLM
        "glm-4",
        "glm-4.5",
        "THUDM/GLM-4.1V-9B-Thinking",
        # Gemini
        "gemini-2.0-flash",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-flash-lite-preview-06-17",
        # Doubao
        "doubao-seed-1-6-250615",
        "doubao-seed-1-6-flash-250615",
        "Doubao-1.5-thinking-pro",
        "Doubao-1.5-pro-256k",
        "Doubao-1.5-lite-32k",
        # DeepSeek
        "deepseek-r1-250528",
        "deepseek-v3-250324",
        "DeepSeek-V3.1-Fast",
        "deepseek-ai/DeepSeek-V2.5",
        # Kimi
        "kimi-k2-0905-preview",
        "kimi-k2-turbo-preview",
        # Grok
        "grok-4-fast-reasoning",
        "grok-4",
        "grok-3",
        # Ernie
        "ernie-4.5-turbo-vl-32k-preview",
        "ernie-x1-turbo-32k-preview",
        "ernie-x1.1-preview",
        "baidu/ERNIE-4.5-300B-A47B"
    ]
    aihubmix_models = st.multiselect("Select AiHubMix models", aihubmix_models_all, default=aihubmix_models_all[:2], key="aihubmix_models")

    st.markdown("**Hunyuan (Cherry-Studio)**")
    hunyuan_key = st.text_input("CHERRY_API_KEY", type="password", key="hunyuan_key")
    hunyuan_models_all = ["hunyuan-pro", "hunyuan-standard", "hunyuan-turbos-latest", "hunyuan-t1-latest"]
    hunyuan_models = st.multiselect("Select Hunyuan models", hunyuan_models_all, default=hunyuan_models_all[:2], key="hunyuan_models")

if st.button("Run Analysis", key="run_btn"):
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    
    if not urls or not prompts:
        st.warning("⚠ Please input at least one URL and one prompt.")
    else:
        st.session_state.generated_files = []
        st.session_state.show_results = False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(urls)
        
        for idx, url in enumerate(urls):
            status_text.text(f"Processing URL {idx+1}/{total_steps}: {url[:50]}...")
            progress_bar.progress((idx) / total_steps)
            
            try:
                title, cleaned = extract_and_clean_chinese(url)
            except Exception as e:
                st.error(f"✗ Fetch failed for {url}: {e}")
                continue
            
            try:
                netloc = urlparse(url).netloc
                source = netloc.replace("www.", "").split(":")[0]
            except Exception:
                source = "来源"

            def sanitize_filename(s: str) -> str:
                s = re.sub(r"[\\/:*?\"<>|]", "", s)
                s = re.sub(r"\s+", "", s)
                return s[:100]  # Limit filename length

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')

            provider_names = []
            if aihubmix_models:
                provider_names.append("aihubmix")
            if hunyuan_models:
                provider_names.append("hunyuan")
            providers_line = ", ".join(provider_names) if provider_names else "(none)"

            called_models_lines = []
            aihubmix_list = ", ".join(aihubmix_models) if aihubmix_models else "(none)"
            hunyuan_list = ", ".join(hunyuan_models) if hunyuan_models else "(none)"
            called_models_lines.append(f"- aihubmix: {aihubmix_list}")
            called_models_lines.append(f"- hunyuan: {hunyuan_list}")

            tests = [
                ("A", title, cleaned, False),
                ("B", f"{title}_此内容因违规无法查看", cleaned, True),
            ]

            for test_type, test_title, test_context, is_b in tests:
                blocks: list[str] = []
                content_title = f"{source}｜{title}"
                if is_b:
                    content_title = content_title + "_此内容因违规无法查看"
                blocks.append(f"# {content_title}")

                blocks.append("")
                blocks.append("**Providers:**")
                blocks.append(providers_line)
                blocks.append("")
                blocks.append("**Called Models:**")
                blocks.extend(called_models_lines)
                blocks.append("")
                blocks.append(f"处理日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                blocks.append("")

                if is_b:
                    img_path = Path("D:/My/aihubmix/censorship.png").absolute()
                    if img_path.exists():
                        blocks.append(f"![审查提示图]({img_path})")
                    blocks.append("")
                    blocks.append("**此文章因违规已经无法查看，以下内容为屏蔽前手动保存**")
                    blocks.append("")
                    blocks.append("---")
                    blocks.append("")

                blocks.append("## 审查比较分析")
                blocks.append("- 提示语 1: 判断是否应该删除屏蔽")
                blocks.append("- 提示语 2: 判断已被删除屏蔽的可能性")
                blocks.append("")

                if aihubmix_key and aihubmix_models:
                    blocks.append("## Provider: aihubmix")
                    blocks.append("")
                    for pidx, prompt in enumerate(prompts, start=1):
                        blocks.append(f"### 提示语 {pidx}: {prompt}")
                        blocks.append("")
                        aihubmix_res = call_provider_concurrent(
                            api_url="https://api.aihubmix.com/v1/chat/completions",
                            api_key=aihubmix_key,
                            models=aihubmix_models,
                            context=test_context,
                            prompt=prompt,
                            provider_name="aihubmix",
                        )
                        for m in aihubmix_models:
                            r = aihubmix_res.get(m, "")
                            blocks.append(f"#### 模型: {m}")
                            blocks.append("")
                            blocks.append(r)
                            blocks.append("")
                            blocks.append("---")
                            blocks.append("")
                        # Small delay between prompts
                        time.sleep(1)
                elif aihubmix_models:
                    blocks.append("## Provider: aihubmix")
                    blocks.append("")
                    blocks.append("(skipped: missing key)")
                    blocks.append("")

                if hunyuan_key and hunyuan_models:
                    blocks.append("## Provider: hunyuan")
                    blocks.append("")
                    for pidx, prompt in enumerate(prompts, start=1):
                        blocks.append(f"### 提示语 {pidx}: {prompt}")
                        blocks.append("")
                        hunyuan_res = call_provider_concurrent(
                            api_url="https://api.hunyuan.cloud.tencent.com/v1/chat/completions",
                            api_key=hunyuan_key,
                            models=hunyuan_models,
                            context=test_context,
                            prompt=prompt,
                            provider_name="hunyuan",
                        )
                        for m in hunyuan_models:
                            r = hunyuan_res.get(m, "")
                            blocks.append(f"#### 模型: {m}")
                            blocks.append("")
                            blocks.append(r)
                            blocks.append("")
                            blocks.append("---")
                            blocks.append("")
                        # Small delay between prompts
                        time.sleep(1)
                elif hunyuan_models:
                    blocks.append("## Provider: hunyuan")
                    blocks.append("")
                    blocks.append("(skipped: missing key)")
                    blocks.append("")

                safe_title = sanitize_filename(f"{source}{title}")
                fname = f"{safe_title}_{test_type}_{ts}.md"
                content = "\n".join(blocks)
                st.session_state.generated_files.append((fname, content))
        
        progress_bar.progress(1.0)
        status_text.text("✓ Analysis complete!")
        st.session_state.show_results = True
        st.success(f"✓ Generated {len(st.session_state.generated_files)} files")

# -------------------------
# Results Display
# -------------------------
if st.session_state.show_results and st.session_state.generated_files:
    st.header("Results")
    
    # Create ZIP download button
    zip_data = create_zip_from_files(st.session_state.generated_files)
    ts_zip = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    st.download_button(
        label=f"📦 Download All Results as ZIP ({len(st.session_state.generated_files)} files)",
        data=zip_data,
        file_name=f"censorship_analysis_{ts_zip}.zip",
        mime="application/zip",
        key="download_zip",
        use_container_width=True
    )
    
    st.divider()
    
    # Display preview in expanders
    st.subheader("Preview Generated Files")
    for fname, content in st.session_state.generated_files:
        with st.expander(f"📄 {fname}"):
            st.code(content[:2000] + "\n\n... (truncated for display)" if len(content) > 2000 else content, language="markdown")