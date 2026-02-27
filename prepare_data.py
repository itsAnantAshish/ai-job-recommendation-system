import pandas as pd

df = pd.read_csv("combined_jobs_final.csv")

df = df[['Job.ID', 'Title', 'Company', 'City', 'Job.Description']].copy()

df = df.dropna(subset=['Title', 'Job.Description'])

df = df.rename(columns={
    'Job.ID': 'job_id',
    'Title': 'title',
    'Company': 'company',
    'City': 'location',
    'Job.Description': 'description'
})

df['company'] = df['company'].fillna('Unknown Company')
df['location'] = df['location'].fillna('Unknown Location')

def get_category(title):
    title = str(title).lower()
    if any(x in title for x in ['software', 'developer', 'engineer', 'programmer']):
        return 'Software'
    elif any(x in title for x in ['data', 'analyst', 'analytics', 'scientist']):
        return 'Data Science'
    elif any(x in title for x in ['manager', 'director', 'vp', 'president']):
        return 'Management'
    elif any(x in title for x in ['sales', 'account', 'business development']):
        return 'Sales'
    elif any(x in title for x in ['market', 'seo', 'content', 'brand']):
        return 'Marketing'
    elif any(x in title for x in ['design', 'ux', 'ui', 'creative']):
        return 'Design'
    elif any(x in title for x in ['nurse', 'doctor', 'medical', 'health', 'clinical']):
        return 'Healthcare'
    elif any(x in title for x in ['finance', 'accounting', 'financial', 'tax']):
        return 'Finance'
    elif any(x in title for x in ['hr', 'human resource', 'recruiter', 'talent']):
        return 'HR'
    elif any(x in title for x in ['teacher', 'professor', 'instructor', 'tutor']):
        return 'Education'
    else:
        return 'Other'

df['category'] = df['title'].apply(get_category)
df['skills'] = df['description'].str[:200]

print("Shape:", df.shape)
print("\nCategory distribution:")
print(df['category'].value_counts())

df_small = df.sample(n=2000, random_state=42).reset_index(drop=True)
df_small.to_csv("jobs_dataset.csv", index=False)
print("\nSaved jobs_dataset.csv with", len(df_small), "rows!")