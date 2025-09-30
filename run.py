import streamlit as st
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import re
import io
import sqlite3
import hashlib
import secrets
from typing import Optional
from urllib.parse import urlparse

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
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        if not row:
            return False
        if verify_password(row[0], password):
            st.session_state.auth_user = username
            return True
    return False


def log_out():
    st.session_state.auth_user = None


# -------------------------
# Utilities
# -------------------------
def extract_and_clean_chinese(url: str):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
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
    cleaned = re.sub(r'[^\u4e00-\u9fff\w\s\.\,\!\?\(\)\[\]\“\”\《\》]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if len(cleaned) > 4000:
        cleaned = cleaned[:4000] + "..."
    return title, cleaned


def call_provider(api_url: str, api_key: str, models: list[str], context: str, prompt: str, provider_name: str):
    messages = [
        {"role": "system", "content": "你是一位专业的中文分析师。"},
        {"role": "user", "content": f"文章内容：{context}\n\n指令：{prompt}"}
    ]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results = {}
    for m in models:
        payload = {"model": m, "messages": messages, "temperature": 1.0, "max_tokens": 1000}
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content = data['choices'][0]['message']['content'].strip()
        except Exception as e:
            content = f"Error: {e}"
        results[m] = content
    return results


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Censorship Compare", layout="wide")
init_db()
st.title("Censorship Compare - AiHubMix & Hunyuan")
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if st.session_state.get("auth_user"):
    st.success(f"Logged in as {st.session_state.auth_user}")
    if st.button("Log out"):
        log_out()
        st.experimental_rerun()
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sign Up")
        su_user = st.text_input("New Username", key="su_user")
        su_pass = st.text_input("New Password", type="password", key="su_pass")
        if st.button("Create Account"):
            ok = sign_up(su_user, su_pass)
            st.toast("Sign up success" if ok else "Sign up failed: duplicate or invalid")
    with col2:
        st.subheader("Log In")
        li_user = st.text_input("Username", key="li_user")
        li_pass = st.text_input("Password", type="password", key="li_pass")
        if st.button("Log in"):
            ok = log_in(li_user, li_pass)
            if ok:
                st.experimental_rerun()
            else:
                st.error("Login failed")

if not st.session_state.get("auth_user"):
    st.stop()

st.header("Inputs")
col_left, col_right = st.columns(2)

with col_left:
    urls_text = st.text_area("Article URLs (one per line)")
    prompts_text = st.text_area("Prompts (one per line)")

with col_right:
    st.markdown("**AiHubMix**")
    aihubmix_key = st.text_input("AIHUBMIX_API_KEY", type="password")
    aihubmix_models_all = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    aihubmix_models = st.multiselect("Select AiHubMix models", aihubmix_models_all, default=aihubmix_models_all[:2])

    st.markdown("**Hunyuan (Cherry-Studio)**")
    hunyuan_key = st.text_input("CHERRY_API_KEY", type="password")
    hunyuan_models_all = ["hunyuan-pro", "hunyuan-standard", "hunyuan-turbos-latest", "hunyuan-t1-latest"]
    hunyuan_models = st.multiselect("Select Hunyuan models", hunyuan_models_all, default=hunyuan_models_all[:2])

run = st.button("Run Analysis")

generated_files: list[tuple[str, str]] = []  # (filename, content)
if run:
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    if not urls or not prompts:
        st.warning("Please input at least one URL and one prompt.")
    else:
        for url in urls:
            try:
                title, cleaned = extract_and_clean_chinese(url)
            except Exception as e:
                st.error(f"Fetch failed for {url}: {e}")
                continue
            # Compose source name from domain
            try:
                netloc = urlparse(url).netloc
                source = netloc.replace("www.", "").split(":")[0]
            except Exception:
                source = "来源"

            def sanitize_filename(s: str) -> str:
                s = re.sub(r"[\\/:*?\"<>|]", "", s)
                s = re.sub(r"\s+", "", s)
                return s

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Prepare common header sections
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

            # Test A and B data
            tests = [
                ("A", title, cleaned, False),
                ("B", f"{title}_此内容因违规无法查看", cleaned, True),
            ]

            for test_type, test_title, test_context, is_b in tests:
                blocks: list[str] = []
                # Title in content: prepend source
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

                # Provider: aihubmix
                if aihubmix_key and aihubmix_models:
                    blocks.append("## Provider: aihubmix")
                    blocks.append("")
                    for idx, prompt in enumerate(prompts, start=1):
                        blocks.append(f"### 提示语 {idx}: {prompt}")
                        blocks.append("")
                        aihubmix_res = call_provider(
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
                elif aihubmix_models:
                    blocks.append("## Provider: aihubmix")
                    blocks.append("")
                    blocks.append("(skipped: missing key)")
                    blocks.append("")

                # Provider: hunyuan
                if hunyuan_key and hunyuan_models:
                    blocks.append("## Provider: hunyuan")
                    blocks.append("")
                    for idx, prompt in enumerate(prompts, start=1):
                        blocks.append(f"### 提示语 {idx}: {prompt}")
                        blocks.append("")
                        hunyuan_res = call_provider(
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
                elif hunyuan_models:
                    blocks.append("## Provider: hunyuan")
                    blocks.append("")
                    blocks.append("(skipped: missing key)")
                    blocks.append("")

                # Finalize content and filename
                safe_title = sanitize_filename(f"{source}{title}")
                fname = f"{safe_title}_{test_type}_{ts}.md"
                content = "\n".join(blocks)
                generated_files.append((fname, content))

if generated_files:
    st.header("Results (Markdown)")
    for fname, content in generated_files:
        st.download_button(
            label=f"Download {fname}",
            data=content.encode('utf-8'),
            file_name=fname,
            mime="text/markdown",
            key=fname,
        )
        st.code(content, language="markdown")


