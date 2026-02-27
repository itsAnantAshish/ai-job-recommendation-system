"""
AI-Based Job Recommendation System
Using TF-IDF Vectorization + Cosine Similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────

def load_data(path="jobs_dataset.csv"):
    df = pd.read_csv(path)
    df.dropna(subset=["title", "skills", "description"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_text(text):
    """Lowercase, remove special characters, normalize whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_combined_text(df):
    """Combine title, skills, and description into one searchable field."""
    df["combined"] = (
        df["title"] + " " +
        df["skills"] + " " +
        df["description"]
    ).apply(preprocess_text)
    return df

# ─────────────────────────────────────────────
# 2. CONTENT-BASED RECOMMENDATION (TF-IDF + Cosine Similarity)
# ─────────────────────────────────────────────

def build_tfidf_model(df):
    """Build and return TF-IDF matrix and vectorizer."""
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1
    )
    tfidf_matrix = tfidf.fit_transform(df["combined"])
    return tfidf, tfidf_matrix

def recommend_jobs(user_input, df, tfidf, tfidf_matrix, top_n=5):
    """
    Given a user's skills/profile text, return top N job matches.
    Returns: DataFrame with matched jobs and similarity scores.
    """
    user_clean = preprocess_text(user_input)
    user_vec = tfidf.transform([user_clean])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    top_indices = scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices][["title", "company", "location", "category", "skills"]].copy()
    results["match_score"] = (scores[top_indices] * 100).round(2)
    results = results[results["match_score"] > 0].reset_index(drop=True)
    results.index += 1  # 1-based ranking
    return results

# ─────────────────────────────────────────────
# 3. KNN CLASSIFIER (Category Prediction)
# ─────────────────────────────────────────────

def build_knn_classifier(df, tfidf_matrix):
    """Train KNN to predict job category from skills text."""
    le = LabelEncoder()
    y = le.fit_transform(df["category"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=0.2, random_state=42, stratify=y
    )
    
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    
    return knn, le, acc, report

def predict_category(user_input, tfidf, knn, le):
    """Predict the most suitable job category for user input."""
    user_clean = preprocess_text(user_input)
    user_vec = tfidf.transform([user_clean])
    pred = knn.predict(user_vec)
    proba = knn.predict_proba(user_vec)[0]
    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [(le.classes_[i], round(proba[i]*100, 1)) for i in top3_idx if proba[i] > 0]
    return le.classes_[pred[0]], top3

# ─────────────────────────────────────────────
# 4. EVALUATION: Precision@K
# ─────────────────────────────────────────────

def precision_at_k(df, tfidf, tfidf_matrix, k=5, n_queries=50):
    """
    Simulate Precision@K: for sampled jobs, check if top-K results
    include jobs from the same category.
    """
    sample = df.sample(n=n_queries, random_state=42)
    precisions = []
    
    for _, row in sample.iterrows():
        user_input = row["skills"]
        user_vec = tfidf.transform([preprocess_text(user_input)])
        scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-(k+1):][::-1]
        
        # Exclude the job itself
        top_indices = [i for i in top_indices if df.iloc[i]["title"] != row["title"]][:k]
        
        relevant = sum(1 for i in top_indices if df.iloc[i]["category"] == row["category"])
        precisions.append(relevant / k)
    
    return round(np.mean(precisions), 4)

# ─────────────────────────────────────────────
# 5. MAIN DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   AI-BASED JOB RECOMMENDATION SYSTEM")
    print("=" * 60)

    # Load & preprocess
    df = load_data()
    df = build_combined_text(df)
    print(f"\n✅ Dataset loaded: {len(df)} jobs, {df['category'].nunique()} categories")

    # Build TF-IDF model
    tfidf, tfidf_matrix = build_tfidf_model(df)
    print(f"✅ TF-IDF model built: {tfidf_matrix.shape[1]} features")

    # Build KNN classifier
    knn, le, acc, report = build_knn_classifier(df, tfidf_matrix)
    print(f"✅ KNN Classifier Accuracy: {acc*100:.2f}%")

    # Precision@K evaluation
    p_at_5 = precision_at_k(df, tfidf, tfidf_matrix, k=5)
    p_at_10 = precision_at_k(df, tfidf, tfidf_matrix, k=10)
    print(f"✅ Precision@5  : {p_at_5}")
    print(f"✅ Precision@10 : {p_at_10}")

    # Demo recommendations
    test_queries = [
        ("python machine learning deep learning tensorflow scikit-learn data analysis",
         "Data Science Profile"),
        ("javascript react node js html css mongodb full stack web development",
         "Full Stack Dev Profile"),
        ("network security penetration testing ethical hacking kali linux",
         "Cybersecurity Profile"),
    ]

    for query, label in test_queries:
        print(f"\n{'─'*60}")
        print(f"👤 {label}")
        print(f"   Skills: {query[:70]}...")
        
        predicted_category, top3 = predict_category(query, tfidf, knn, le)
        print(f"   Predicted Category: {predicted_category}")
        print(f"   Top 3 Categories: {top3}")
        
        results = recommend_jobs(query, df, tfidf, tfidf_matrix, top_n=5)
        print(f"\n   Top 5 Job Recommendations:")
        for idx, row in results.iterrows():
            print(f"   {idx}. {row['title']} @ {row['company']} ({row['location']}) — Score: {row['match_score']}%")

    print(f"\n{'='*60}")
    print("\n📊 Classification Report:")
    print(report)
