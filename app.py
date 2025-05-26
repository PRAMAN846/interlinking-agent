import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# 🔐 Load Together AI key from environment
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# 🔧 Call DeepSeek on Together AI to get anchor + sentence
def get_anchor_text_together_ai(source_excerpt, target_title, target_url):
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
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.7
        }
    )
    if response.ok:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"⚠️ LLM failed: {response.status_code} – {response.text}"

# 🌐 Crawl URL and extract main content
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

        return {
            "URL": url,
            "Title": title,
            "H1": h1,
            "Content": readable_text
        }
    except Exception as e:
        return {
            "URL": url,
            "Title": f"Error: {e}",
            "H1": "",
            "Content": ""
        }

# 🔍 Extract one meaningful sentence from body text
def extract_representative_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    for s in sentences:
        if 50 < len(s) < 300:  # Reasonable length
            return s.strip()
    return text[:300]  # fallback

# 🚀 Streamlit App
st.title("🔗 Internal Linking Suggestion Agent (Bulk, Auto, LLM)")

url_input = st.text_area("Paste URLs (one per line)", height=200)
num_suggestions = st.slider("How many suggestions per page?", 1, 10, 3)
start_button = st.button("🔍 Run Interlinking Agent")

if start_button and url_input:
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]
    st.info(f"Fetching and processing {len(urls)} URLs...")

    raw_data = [fetch_url_data(url) for url in urls]
    df = pd.DataFrame(raw_data)
    df = df[df['Content'] != ""]
    df.fillna("", inplace=True)

    if df.empty:
        st.error("None of the pages could be fetched or extracted.")
        st.stop()

    st.success("✅ All pages crawled and content extracted.")
    st.dataframe(df[['URL', 'Title', 'H1']])

    df['text'] = df[['Title', 'H1', 'Content']].agg(' '.join, axis=1)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    all_suggestions = []

    for idx, row in df.iterrows():
        query_embedding = embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(query_embedding, embeddings)[0]
        sim_scores = list(enumerate(sims))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        count = 0
        for i, score in sim_scores:
            if i == idx or count >= num_suggestions:
                continue

            source_excerpt = extract_representative_sentence(df.loc[idx, 'Content'])
            target_title = df.loc[i, 'Title']
            target_url = df.loc[i, 'URL']
            source_url = df.loc[idx, 'URL']

            llm_response = get_anchor_text_together_ai(source_excerpt, target_title, target_url)

            # Parse LLM response (simple version)
            anchor_match = re.search(r"Anchor Text:\s*(.+)", llm_response)
            sentence_match = re.search(r"Suggested Sentence:\s*(.+)", llm_response)

            anchor_text = anchor_match.group(1).strip() if anchor_match else "—"
            suggested_sentence = sentence_match.group(1).strip() if sentence_match else "—"

            all_suggestions.append({
                "Source URL": source_url,
                "Target URL": target_url,
                "Anchor Text": anchor_text,
                "Suggested Sentence": suggested_sentence,
                "Similarity Score": round(score, 3)
            })

            count += 1

    result_df = pd.DataFrame(all_suggestions)
    st.markdown("### 🔗 Internal Linking Suggestions")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Suggestions CSV", data=csv, file_name="internal_linking_suggestions.csv", mime='text/csv')
