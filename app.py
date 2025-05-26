import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 🔐 Load API key from Streamlit Secrets
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# 🔧 Together AI LLM Call
def get_anchor_text_together_ai(source_excerpt, target_title, target_url):
    prompt = f"""
You are an SEO agent. Based on the following source excerpt and the destination title, suggest a natural anchor text and a sentence where this link can be naturally inserted.

Source Content:
\"\"\"{source_excerpt}\"\"\"

Destination Page Title: "{target_title}"
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
            "max_tokens": 200,
            "temperature": 0.7
        }
    )
    if response.ok:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"⚠️ LLM failed: {response.status_code} – {response.text}"

# 🌐 Fetch content from a URL
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

# 🧠 Streamlit Interface
st.title("🔗 Internal Linking Suggestion Agent")
st.write("Paste a list of URLs below. The app will fetch each page, extract content, and suggest internal links using an AI model.")

url_input = st.text_area("Paste URLs (one per line)", height=200)
start_button = st.button("Fetch Page Data")

if start_button and url_input:
    urls = [u.strip() for u in url_input.splitlines() if u.strip()]
    st.info(f"Fetching {len(urls)} URLs. Please wait...")

    data = [fetch_url_data(url) for url in urls]
    df = pd.DataFrame(data)
    df = df[df['Content'] != ""]  # Remove failed URLs

    if df.empty:
        st.error("Could not fetch any content. Check your URLs or try again.")
        st.stop()

    st.success("✅ Fetched & extracted content")
    st.dataframe(df[['URL', 'Title', 'H1']])

    # Compute embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['text'] = df[['Title', 'H1', 'Content']].agg(' '.join, axis=1)
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    # Choose target URL
    url_titles = [f"{row['Title']} ({row['URL']})" for _, row in df.iterrows()]
    selected_title = st.selectbox("Select a URL to get interlinking suggestions:", url_titles)
    selected_index = url_titles.index(selected_title)
    num_suggestions = st.slider("Number of suggestions", 1, 10, 3)

    if st.button("🔍 Get Suggestions"):
        query_embedding = embeddings[selected_index].reshape(1, -1)
        sims = cosine_similarity(query_embedding, embeddings)[0]
        sim_scores = list(enumerate(sims))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        st.markdown(f"### Top {num_suggestions} Suggestions for:")
        st.markdown(f"**{df.loc[selected_index, 'Title']}** — {df.loc[selected_index, 'URL']}")

        results = []
        count = 0

        for i, score in sim_scores:
            if i == selected_index or count >= num_suggestions:
                continue

            source_excerpt = df.loc[selected_index, 'Content'][:700]
            target_title = df.loc[i, 'Title']
            target_url = df.loc[i, 'URL']

            llm_response = get_anchor_text_together_ai(source_excerpt, target_title, target_url)

            results.append({
                "Source URL": df.loc[selected_index, 'URL'],
                "Target URL": target_url,
                "Similarity Score": round(score, 3),
                "LLM Suggestion": llm_response
            })

            count += 1

        results_df = pd.DataFrame(results)
        st.write(results_df)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Suggestions CSV", data=csv, file_name="interlinking_suggestions.csv", mime='text/csv')
