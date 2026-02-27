"""
AI-Based Job Recommendation System — Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from recommender import (
    load_data, build_combined_text, build_tfidf_model,
    build_knn_classifier, recommend_jobs, predict_category
)

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .job-card {
        background: #f8f9ff;
        border-left: 4px solid #1f4e79;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .job-title { font-size: 1.1rem; font-weight: 700; color: #1f4e79; }
    .job-meta { color: #666; font-size: 0.9rem; }
    .score-badge {
        display: inline-block;
        background: #1f4e79;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .category-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1f4e79;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        border: 1px solid #b8d9f5;
    }
    .stTextArea textarea { font-size: 0.95rem; }
    .metric-box {
        background: #eaf4fb;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load models (cached) ──────────────────────────────────
@st.cache_resource
def load_models():
    df = load_data("jobs_dataset.csv")
    df = build_combined_text(df)
    tfidf, tfidf_matrix = build_tfidf_model(df)
    knn, le, acc, report = build_knn_classifier(df, tfidf_matrix)
    return df, tfidf, tfidf_matrix, knn, le, acc

df, tfidf, tfidf_matrix, knn, le, acc = load_models()

# ─── Header ────────────────────────────────────────────────
st.markdown('<div class="main-header">💼 AI-Based Job Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter your skills or paste your resume summary to discover the best-matched jobs</div>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_n = st.slider("Number of recommendations", 3, 15, 5)
    filter_location = st.selectbox("Filter by Location", ["All"] + sorted(df["location"].unique().tolist()))
    filter_category = st.selectbox("Filter by Category", ["All"] + sorted(df["category"].unique().tolist()))

    st.divider()
    st.header("📊 Model Info")
    st.metric("KNN Accuracy", f"{acc*100:.1f}%")
    st.metric("Total Jobs", len(df))
    st.metric("Job Categories", df["category"].nunique())
    st.metric("TF-IDF Features", tfidf_matrix.shape[1])

    st.divider()
    st.caption("Built with Python • Scikit-learn • Streamlit")

# ─── Main Input ────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Your Profile")
    user_input = st.text_area(
        "Enter your skills, experience, or resume summary:",
        placeholder="e.g. Python, machine learning, deep learning, TensorFlow, data analysis, scikit-learn, SQL, statistics...",
        height=150
    )

with col2:
    st.subheader("💡 Quick Profiles")
    st.write("Click to auto-fill:")
    examples = {
        "🔬 Data Scientist": "python machine learning deep learning tensorflow scikit-learn pandas statistics data analysis sql",
        "💻 Full Stack Dev": "javascript react node js html css mongodb rest api sql full stack web development",
        "🔒 Security Analyst": "network security penetration testing ethical hacking kali linux vulnerability assessment",
        "📊 Business Analyst": "sql excel tableau power bi data visualization reporting business analysis statistics",
        "🤖 NLP Engineer": "python nlp bert transformers huggingface text classification sentiment analysis spacy",
    }
    for label, skills in examples.items():
        if st.button(label, use_container_width=True):
            user_input = skills
            st.session_state["user_input"] = skills

if "user_input" in st.session_state:
    user_input = st.session_state["user_input"]

# ─── Search Button ─────────────────────────────────────────
search = st.button("🔍 Find My Jobs", type="primary", use_container_width=True)

if search and user_input.strip():
    st.divider()

    # Filter df if location/category selected
    filtered_df = df.copy()
    if filter_location != "All":
        filtered_df = filtered_df[filtered_df["location"] == filter_location]
    if filter_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == filter_category]

    if len(filtered_df) == 0:
        st.warning("No jobs found with the selected filters. Try 'All'.")
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        from recommender import preprocess_text
        import re, numpy as np

        # Re-run cosine sim on filtered df
        user_clean = preprocess_text(user_input)
        user_vec = tfidf.transform([user_clean])
        filtered_idx = filtered_df.index.tolist()
        sub_matrix = tfidf_matrix[filtered_idx]
        scores = cosine_similarity(user_vec, sub_matrix).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        results = filtered_df.iloc[top_indices].copy()
        results["match_score"] = (scores[top_indices] * 100).round(2)
        results = results[results["match_score"] > 0].reset_index(drop=True)

        # Category prediction
        predicted_cat, top3 = predict_category(user_input, tfidf, knn, le)

        # Metrics row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("🎯 Predicted Category", predicted_cat)
        with col_b:
            st.metric("📋 Results Found", len(results))
        with col_c:
            best_score = results["match_score"].max() if len(results) > 0 else 0
            st.metric("⭐ Best Match Score", f"{best_score}%")

        st.subheader("🏆 Your Top Job Matches")

        if len(results) == 0:
            st.info("No strong matches found. Try adding more skills.")
        else:
            for i, (_, row) in enumerate(results.iterrows()):
                st.markdown(f"""
                <div class="job-card">
                    <span class="job-title">#{i+1} &nbsp; {row['title']}</span>
                    &nbsp;&nbsp;
                    <span class="score-badge">Match: {row['match_score']}%</span>
                    &nbsp;
                    <span class="category-badge">{row['category']}</span>
                    <br>
                    <span class="job-meta">🏢 {row['company']} &nbsp;|&nbsp; 📍 {row['location']}</span>
                    <br>
                    <span class="job-meta">🛠 {row['skills'][:100]}...</span>
                </div>
                """, unsafe_allow_html=True)

        # Category probabilities
        st.subheader("📊 Category Match Analysis")
        cat_df = pd.DataFrame(top3, columns=["Category", "Probability (%)"])
        st.bar_chart(cat_df.set_index("Category"))

elif search and not user_input.strip():
    st.warning("Please enter your skills or resume summary first.")

# ─── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>AI-Based Job Recommendation System | Developed using Python, Scikit-learn & Streamlit</small></center>",
    unsafe_allow_html=True
)
