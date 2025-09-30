import streamlit as st
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import re
import io

# -------------------------
# Simple in-memory auth demo
# -------------------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None


def sign_up(username: str, password: str) -> bool:
    if not username or not password:
        return False
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = password
    return True


def log_in(username: str, password: str) -> bool:
    if not username or not password:
        return False
    if st.session_state.users.get(username) == password:
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
st.title("Censorship Compare - AiHubMix & Hunyuan")

if st.session_state.auth_user:
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

if not st.session_state.auth_user:
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

results_md = ""
if run:
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
    if not urls or not prompts:
        st.warning("Please input at least one URL and one prompt.")
    else:
        all_blocks = []
        for url in urls:
            try:
                title, cleaned = extract_and_clean_chinese(url)
            except Exception as e:
                st.error(f"Fetch failed for {url}: {e}")
                continue
            # Test A and B contexts
            test_pairs = [("A", title, cleaned), ("B", title + "_此内容因违规无法查看", "此内容因违规无法查看\n\n[此处应有表示审查的图片，但由于文本格式，无法显示]\n\n" + cleaned)]
            for test_type, test_title, test_context in test_pairs:
                blocks = [f"# {test_title}", f"Provider: AiHubMix & Hunyuan", f"处理日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
                if test_type == "B":
                    img_path = Path("D:/My/aihubmix/censorship.png").absolute()
                    if img_path.exists():
                        blocks.append(f"![审查提示图]({img_path})")
                    blocks.append("**此文章因违规已经无法查看，以下内容为屏蔽前手动保存**")
                    blocks.append("---")
                blocks.append("## 审查比较分析")
                blocks.append("- 提示语 1: 判断是否应该删除屏蔽")
                blocks.append("- 提示语 2: 判断已被删除屏蔽的可能性\n")

                for prompt in prompts:
                    blocks.append(f"### 提示语: {prompt}")
                    # AiHubMix
                    if aihubmix_key and aihubmix_models:
                        aihubmix_res = call_provider(
                            api_url="https://api.aihubmix.com/v1/chat/completions",
                            api_key=aihubmix_key,
                            models=aihubmix_models,
                            context=test_context,
                            prompt=prompt,
                            provider_name="aihubmix"
                        )
                        for m, r in aihubmix_res.items():
                            blocks.append(f"#### [AiHubMix] 模型: {m}\n\n{r}\n\n---")
                    else:
                        blocks.append("_AiHubMix skipped (missing key or models)_")
                    # Hunyuan
                    if hunyuan_key and hunyuan_models:
                        hunyuan_res = call_provider(
                            api_url="https://api.hunyuan.cloud.tencent.com/v1/chat/completions",
                            api_key=hunyuan_key,
                            models=hunyuan_models,
                            context=test_context,
                            prompt=prompt,
                            provider_name="hunyuan"
                        )
                        for m, r in hunyuan_res.items():
                            blocks.append(f"#### [Hunyuan] 模型: {m}\n\n{r}\n\n---")
                    else:
                        blocks.append("_Hunyuan skipped (missing key or models)_")

                all_blocks.append("\n".join(blocks))

        results_md = "\n\n\n".join(all_blocks)

if results_md:
    st.header("Results (Markdown)")
    st.download_button(
        label="Download Markdown",
        data=results_md.encode('utf-8'),
        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
    st.code(results_md, language="markdown")

