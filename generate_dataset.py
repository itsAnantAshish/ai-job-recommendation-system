import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

jobs_data = [
    # Data Science / ML
    {"title": "Data Scientist", "company": "Google", "location": "Bangalore", "category": "Data Science",
     "skills": "python machine learning deep learning tensorflow pandas numpy scikit-learn statistics data analysis",
     "description": "Analyze large datasets, build predictive models, and deploy machine learning solutions. Work with cross-functional teams to drive data-driven decisions."},
    {"title": "Machine Learning Engineer", "company": "Amazon", "location": "Hyderabad", "category": "Data Science",
     "skills": "python pytorch tensorflow deep learning mlops docker kubernetes model deployment",
     "description": "Design and implement ML pipelines, optimize models for production, and collaborate with software engineers to integrate AI features."},
    {"title": "AI Research Scientist", "company": "Microsoft", "location": "Pune", "category": "Data Science",
     "skills": "python research deep learning nlp computer vision publications pytorch",
     "description": "Conduct cutting-edge AI research, publish findings, and develop novel algorithms for real-world applications."},
    {"title": "Data Analyst", "company": "Flipkart", "location": "Bangalore", "category": "Data Science",
     "skills": "sql python excel tableau power bi data visualization statistics reporting",
     "description": "Extract insights from business data, create dashboards, and support decision-making with data-driven reports."},
    {"title": "Business Intelligence Analyst", "company": "Infosys", "location": "Chennai", "category": "Data Science",
     "skills": "sql tableau power bi excel data warehousing etl reporting business analysis",
     "description": "Build BI dashboards, manage data pipelines, and provide actionable insights to stakeholders."},
    {"title": "NLP Engineer", "company": "Sprinklr", "location": "Gurgaon", "category": "Data Science",
     "skills": "python nlp bert transformers huggingface text classification sentiment analysis spacy",
     "description": "Build NLP models for text classification, sentiment analysis, and language understanding tasks."},
    {"title": "Computer Vision Engineer", "company": "Qualcomm", "location": "Hyderabad", "category": "Data Science",
     "skills": "python opencv deep learning convolutional neural networks image processing yolo object detection",
     "description": "Develop computer vision algorithms for object detection, image segmentation, and video analysis."},
    {"title": "Data Engineer", "company": "Uber", "location": "Bangalore", "category": "Data Engineering",
     "skills": "python sql spark hadoop kafka airflow data pipelines etl aws bigquery",
     "description": "Design and build scalable data infrastructure, ETL pipelines, and real-time data processing systems."},
    {"title": "MLOps Engineer", "company": "Razorpay", "location": "Bangalore", "category": "Data Science",
     "skills": "python mlflow docker kubernetes ci cd model monitoring aws sagemaker mlops",
     "description": "Manage ML model lifecycle, deployment pipelines, and monitoring infrastructure."},
    {"title": "Quantitative Analyst", "company": "Goldman Sachs", "location": "Bangalore", "category": "Finance",
     "skills": "python r statistics probability financial modeling risk analysis quantitative research",
     "description": "Develop quantitative models for financial risk, pricing, and algorithmic trading."},

    # Software Development
    {"title": "Software Engineer", "company": "TCS", "location": "Mumbai", "category": "Software",
     "skills": "java python javascript c++ algorithms data structures problem solving oop",
     "description": "Design, develop, and maintain software applications. Collaborate with teams on large-scale projects."},
    {"title": "Full Stack Developer", "company": "Wipro", "location": "Pune", "category": "Software",
     "skills": "javascript react node js html css mongodb sql rest api full stack",
     "description": "Build end-to-end web applications using modern frontend and backend technologies."},
    {"title": "Backend Developer", "company": "Swiggy", "location": "Bangalore", "category": "Software",
     "skills": "java spring boot microservices rest api sql kafka docker kubernetes backend",
     "description": "Develop scalable backend services, APIs, and microservices for high-traffic applications."},
    {"title": "Frontend Developer", "company": "Zomato", "location": "Delhi", "category": "Software",
     "skills": "javascript react html css typescript redux ui ux frontend web development",
     "description": "Create responsive, performant user interfaces using React and modern JavaScript frameworks."},
    {"title": "DevOps Engineer", "company": "Paytm", "location": "Noida", "category": "Software",
     "skills": "docker kubernetes aws ci cd jenkins linux bash automation infrastructure devops",
     "description": "Build and maintain CI/CD pipelines, cloud infrastructure, and deployment automation."},
    {"title": "Cloud Architect", "company": "Accenture", "location": "Hyderabad", "category": "Software",
     "skills": "aws azure gcp cloud architecture terraform kubernetes microservices security",
     "description": "Design cloud-native architectures, oversee cloud migration projects, and ensure scalability."},
    {"title": "Android Developer", "company": "BYJU'S", "location": "Bangalore", "category": "Software",
     "skills": "android java kotlin mobile development firebase rest api xml ui design",
     "description": "Build and maintain Android applications with clean architecture and optimal performance."},
    {"title": "iOS Developer", "company": "PhonePe", "location": "Bangalore", "category": "Software",
     "skills": "swift ios objective c xcode mobile development rest api ui design",
     "description": "Develop iOS applications with smooth user experiences and robust architecture."},

    # Cybersecurity
    {"title": "Cybersecurity Analyst", "company": "IBM", "location": "Bangalore", "category": "Security",
     "skills": "network security ethical hacking penetration testing siem splunk firewalls vulnerability assessment",
     "description": "Monitor and protect systems from cyber threats, conduct security audits, and respond to incidents."},
    {"title": "Ethical Hacker", "company": "HCL", "location": "Chennai", "category": "Security",
     "skills": "penetration testing ethical hacking kali linux metasploit vulnerability scanning network security",
     "description": "Perform security assessments, penetration tests, and identify vulnerabilities in systems."},

    # Finance & HR
    {"title": "Financial Analyst", "company": "HDFC Bank", "location": "Mumbai", "category": "Finance",
     "skills": "financial modeling excel valuation accounting reporting budgeting forecasting analysis",
     "description": "Analyze financial data, prepare reports, and support business planning and investment decisions."},
    {"title": "HR Manager", "company": "Deloitte", "location": "Gurgaon", "category": "HR",
     "skills": "recruitment talent management employee relations hr policies payroll communication leadership",
     "description": "Oversee hiring processes, manage employee relations, and implement HR strategies."},
    {"title": "Product Manager", "company": "Ola", "location": "Bangalore", "category": "Product",
     "skills": "product management roadmap agile stakeholder management analytics user research strategy",
     "description": "Define product vision, manage roadmaps, and work with engineering and design teams to deliver features."},
    {"title": "Digital Marketing Manager", "company": "Nykaa", "location": "Mumbai", "category": "Marketing",
     "skills": "digital marketing seo sem social media google analytics content strategy email marketing",
     "description": "Lead digital marketing campaigns, manage SEO/SEM, and grow online brand presence."},
    {"title": "UX Designer", "company": "Freshworks", "location": "Chennai", "category": "Design",
     "skills": "ux design figma wireframing prototyping user research usability testing ui design",
     "description": "Create user-centered designs, conduct user research, and build intuitive product experiences."},

    # More varied roles
    {"title": "Blockchain Developer", "company": "Polygon", "location": "Mumbai", "category": "Software",
     "skills": "blockchain solidity ethereum smart contracts web3 javascript cryptography",
     "description": "Build decentralized applications and smart contracts on blockchain platforms."},
    {"title": "Robotics Engineer", "company": "Tata Motors", "location": "Pune", "category": "Engineering",
     "skills": "robotics ros python c++ embedded systems control systems automation sensors",
     "description": "Design and develop robotic systems for manufacturing automation and intelligent machines."},
    {"title": "Research Analyst", "company": "KPMG", "location": "Delhi", "category": "Consulting",
     "skills": "research data analysis excel powerpoint report writing market research strategy consulting",
     "description": "Conduct market research, analyze data, and produce strategic reports for clients."},
    {"title": "Database Administrator", "company": "Oracle", "location": "Hyderabad", "category": "Software",
     "skills": "sql oracle mysql postgresql database administration backup performance tuning",
     "description": "Manage, optimize, and secure databases to ensure high availability and performance."},
    {"title": "Technical Writer", "company": "Adobe", "location": "Noida", "category": "Content",
     "skills": "technical writing documentation api docs markdown content management communication",
     "description": "Create clear technical documentation, API references, and user guides for software products."},
]

# Expand dataset to 200 rows by duplicating with slight variations
expanded = []
companies_extra = ["Infosys", "Cognizant", "Tech Mahindra", "Capgemini", "Mindtree", "Mphasis", "Hexaware", "L&T Infotech"]
locations_extra = ["Bangalore", "Hyderabad", "Pune", "Chennai", "Mumbai", "Delhi", "Gurgaon", "Noida", "Kolkata"]

for i, job in enumerate(jobs_data * 7):
    new_job = job.copy()
    if i >= len(jobs_data):
        new_job["company"] = random.choice(companies_extra)
        new_job["location"] = random.choice(locations_extra)
    expanded.append(new_job)

df = pd.DataFrame(expanded[:200])
df["job_id"] = range(1, len(df) + 1)
df = df[["job_id", "title", "company", "location", "category", "skills", "description"]]
df.to_csv("jobs_dataset.csv", index=False)
print(f"Dataset created: {len(df)} jobs across {df['category'].nunique()} categories")
print(df["category"].value_counts())
