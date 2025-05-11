import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔗 Interlinking Suggestion Agent")
st.write("Upload a CSV with columns: URL, Title, H1, H2, H3")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.fillna("", inplace=True)
    df['text'] = df[['Title', 'H1', 'H2', 'H3']].agg(' '.join, axis=1)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    url_titles = [f"{row['Title']} ({row['URL']})" for _, row in df.iterrows()]
    selected_title = st.selectbox("Select a URL to get interlinking suggestions:", url_titles)
    selected_index = url_titles.index(selected_title)
    num_suggestions = st.slider("Number of suggestions", 1, 20, 5)

    if st.button("🔍 Get Suggestions"):
        query_embedding = embeddings[selected_index].reshape(1, -1)
        sims = cosine_similarity(query_embedding, embeddings)[0]
        sim_scores = list(enumerate(sims))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        st.markdown(f"### Top {num_suggestions} Suggestions for:")
        st.markdown(f"**{df.loc[selected_index, 'Title']}** — {df.loc[selected_index, 'URL']}")

        count = 0
        for i, score in sim_scores:
            if i == selected_index:
                continue
            st.markdown(f"- **[{df.loc[i, 'Title']}]({df.loc[i, 'URL']})** — Similarity: `{round(score, 3)}`")
            count += 1
            if count >= num_suggestions:
                break
