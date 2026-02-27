AI-BASED JOB RECOMMENDATION SYSTEM
====================================

FILES:
  generate_dataset.py  → Run first to create jobs_dataset.csv
  recommender.py       → Core ML engine (TF-IDF + Cosine Similarity + KNN)
  app.py               → Streamlit web application

HOW TO RUN:
  Step 1: pip install pandas scikit-learn streamlit matplotlib
  Step 2: python generate_dataset.py
  Step 3: python recommender.py        (to see terminal demo + metrics)
  Step 4: streamlit run app.py         (to launch the web app)

The web app will open at: https://ai-job-recommendation-system1.streamlit.app/
