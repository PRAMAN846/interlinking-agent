import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from asyncio import Semaphore

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Async LLM call with retry and semaphore
async def get_anchor_text_async(session, sem, source_excerpt, target_title, target_url):
    prompt = f"""
You are an SEO agent suggesting internal links. Based on the following source excerpt and the destination page title, suggest:

1. A natural anchor text that could be used to link to the destination page from the source context.
2. An improved sentence in which this anchor can be naturally inserted.

Source Excerpt:
\"\"\"{source_excerpt}\"\"\"

Destination Title: "{target_title}"
Destination URL: {target_url}

Respond strictly in this format:
Anchor Text: <text>
Suggested Sentence: <sentence>
"""
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    async with sem:
        for _ in range(3):  # retry up to 3 times
            try:
                async with session.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, timeout=60) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['choices'][0]['message']['content']
                    else:
                        await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)
        return "⚠️ LLM failed after retries"

def fetch_url_data(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        html = res.text

        soup = BeautifulSoup(html, 'lxml')
        title = soup.title.string.strip() if soup.title else ''
        h1 = soup.find('h1').text.strip() if soup.find('h1') else ''
        readable_article = Document(html).summary()
        readable_text = BeautifulSoup(readable_article, 'lxml').get_text(separator=' ').strip()

        return {"URL": url, "Title": title, "H1": h1, "Content": readable_text}
    except Exception as e:
        return {"URL": url, "Title": f"Error: {e}", "H1": "", "Content": ""}

def extract_representative_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    for s in sentences:
        if 50 < len(s) < 300:
            return s.strip()
    return text[:300]

st.title("🔗 Internal Linking Suggestion Agent (Optimized + Async)")
url_input = st.text_area("Paste URLs (one per line)", height=200)
num_suggestions = st.slider("Suggestions per page", 1, 10, 3)
start_button = st.button("🔍 Run Interlinking Agent")

if start_button and url_input:
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]
    st.info(f"Fetching and processing {len(urls)} URLs...")

    with ThreadPoolExecutor() as executor:
        raw_data = list(executor.map(fetch_url_data, urls))

    df = pd.DataFrame(raw_data)
    df = df[df['Content'] != ""]
    df.fillna("", inplace=True)

    if df.empty:
        st.error("None of the pages could be fetched.")
        st.stop()

    st.success("✅ Content extracted")
    df['text'] = df[['Title', 'H1', 'Content']].agg(' '.join, axis=1)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    st.write("Embedding complete. Generating suggestions in parallel...")
    progress = st.progress(0)

    suggestions = []

    async def run_llm_tasks():
        sem = Semaphore(10)  # limit concurrency to avoid rate limits
        async with aiohttp.ClientSession() as session:
            tasks = []
            total = len(df) * num_suggestions
            counter = 0
            for idx, row in df.iterrows():
                sims = cosine_similarity([embeddings[idx]], embeddings)[0]
                sorted_sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
                count = 0
                for i, score in sorted_sims:
                    if i == idx or count >= num_suggestions:
                        continue
                    source_excerpt = extract_representative_sentence(df.loc[idx, 'Content'])
                    target_title = df.loc[i, 'Title']
                    target_url = df.loc[i, 'URL']
                    source_url = df.loc[idx, 'URL']

                    tasks.append((session, sem, source_excerpt, target_title, target_url, source_url, target_url, round(score, 3)))
                    count += 1

            for i, batch in enumerate(asyncio.as_completed([
                get_anchor_text_async(t[0], t[1], t[2], t[3], t[4]) for t in tasks
            ])):
                llm_result = await batch
                meta = tasks[i]
                anchor = re.search(r"Anchor Text:\s*(.+)", llm_result)
                sentence = re.search(r"Suggested Sentence:\s*(.+)", llm_result)
                suggestions.append({
                    "Source URL": meta[5],
                    "Target URL": meta[6],
                    "Similarity Score": meta[7],
                    "Anchor Text": anchor.group(1).strip() if anchor else "—",
                    "Suggested Sentence": sentence.group(1).strip() if sentence else "—"
                })
                counter += 1
                progress.progress(min(1.0, counter / total))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_llm_tasks())

    result_df = pd.DataFrame(suggestions)
    st.markdown("### 🔗 Internal Linking Suggestions")
    st.dataframe(result_df)
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Suggestions CSV", data=csv, file_name="internal_linking_suggestions.csv", mime='text/csv')
