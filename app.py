import streamlit as st
import pandas as pd
import requests
import os
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from asyncio import Semaphore
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from stqdm import stqdm

# 🔐 Load API key from Streamlit Secrets
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# 🔧 Together AI LLM Call
async def get_anchor_text_async(session, sem, source_excerpt, target_title, target_url):
    prompt = (
        f"You are an SEO agent suggesting internal links. Based on the following source excerpt and the destination page title, suggest:\n\n"
        f"1. A natural anchor text that could be used to link to the destination page from the source context.\n"
        f"2. An improved sentence in which this anchor can be naturally inserted.\n\n"
        f"Source Excerpt:\n{source_excerpt}\n\n"
        f"Destination Title: \"{target_title}\"\n"
        f"Destination URL: {target_url}\n\n"
        f"Respond strictly in this format:\n"
        f"Anchor Text: <text>\n"
        f"Suggested Sentence: <sentence>"
    )

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    json_data = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    async with sem:
        async with session.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=json_data) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

# 🌐 Fetch and extract content from a URL
def extract_content(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string if soup.title else ""
        h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        content = " ".join(paragraphs)
        return title, h1, content
    except Exception as e:
        return None, None, None

# 🚀 Streamlit UI
st.set_page_config(page_title="Internal Linking Agent", layout="wide")
st.title("🔗 Internal Linking Suggestion Agent")

# Step 1: Upload URLs
st.markdown("### 📄 Step 1: Upload or Paste URLs")
input_method = st.radio("Choose input method", ["Paste URLs", "Upload CSV file"])

urls = []
if input_method == "Paste URLs":
    raw_urls = st.text_area("Enter one URL per line")
    if raw_urls:
        urls = [url.strip() for url in raw_urls.splitlines() if url.strip()]
elif input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV containing a column of URLs")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        col_name = st.selectbox("Select URL column", df.columns)
        urls = df[col_name].dropna().tolist()

if urls:
    st.markdown("### 🔍 Step 2: Extracting content from URLs")
    results = []
    failed = []
    for url in stqdm(urls, desc="Fetching URLs"):
        title, h1, content = extract_content(url)
        if content:
            results.append({"url": url, "title": title, "h1": h1, "content": content})
        else:
            failed.append(url)

    if failed:
        st.warning(f"⚠️ Failed to fetch {len(failed)} URLs. See console for details.")

    df = pd.DataFrame(results)

    st.markdown("### 🔗 Step 3: Suggest Internal Links")
    suggestions_per_page = st.slider("How many internal links per page?", min_value=1, max_value=10, value=5)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["content"].tolist(), convert_to_tensor=True)

    suggestions = []

    async def run_llm_tasks():
        sem = Semaphore(3)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, row in df.iterrows():
                source_embedding = embeddings[i]
                scores = util.cos_sim(source_embedding, embeddings)[0].cpu().tolist()
                scored_indices = sorted([(j, s) for j, s in enumerate(scores) if j != i], key=lambda x: -x[1])[:suggestions_per_page]

                for j, score in scored_indices:
                    target_row = df.iloc[j]
                    source_excerpt = row["content"][:600]  # short preview
                    tasks.append((session, sem, source_excerpt, target_row["title"], target_row["url"], row["url"], score))

            results = []
            for i, coro in enumerate(asyncio.as_completed([get_anchor_text_async(s, sem, src, tgt_title, tgt_url) for s, sem, src, tgt_title, tgt_url, _, _ in tasks])):
                try:
                    llm_result = await coro
                    anchor = re.search(r"Anchor Text:\s*(.+)", llm_result)
                    sentence = re.search(r"Suggested Sentence:\s*(.+)", llm_result)
                    meta = tasks[i]
                    results.append({
                        "Source URL": meta[5],
                        "Target URL": meta[4],
                        "Anchor Text": anchor.group(1) if anchor else "",
                        "Suggested Sentence": sentence.group(1) if sentence else "",
                        "Score": meta[6],
                    })
                except Exception as e:
                    continue

            return results

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    suggestions = loop.run_until_complete(run_llm_tasks())

    result_df = pd.DataFrame(suggestions)

    st.markdown("### 📥 Final Suggestions")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "internal_linking_suggestions.csv", "text/csv")
