from neo4j import GraphDatabase
import pandas as pd
from openai import OpenAI

import streamlit as st
from tqdm import tqdm
import time
import openai


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Neo4j connection details
uri = st.secrets["NEO4J_URI"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]

# OpenAI setup
EMBED_MODEL = "text-embedding-ada-002"  # Or use another OpenAI embedding model

# Cypher query: just questionId and title
query = """
MATCH (q:Question)
RETURN q.uuid AS question_id, q.title AS title
"""

# Step 1: Fetch questions
def fetch_questions(uri, username, password, query):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run(query)
        data = result.data()
        df = pd.DataFrame(data)
    driver.close()
    print(df)
    return df

# Step 2: Get embeddings for a batch of texts using OpenAI SDK >= 1.0.0
def get_embeddings_batch(texts, model=EMBED_MODEL, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = openai.embeddings.create(input=texts, model=model)
            return [record.embedding for record in response.data]
        except Exception as e:
            print(f"Retry {attempt+1}: Error in batch - {e}")
            time.sleep(2 ** attempt)
    return [None] * len(texts)

def embed_dataframe(df, batch_size=512):
    embeddings = []
    num_rows = len(df)
    for start in tqdm(range(0, num_rows, batch_size), desc="Embedding in batches"):
        end = min(start + batch_size, num_rows)
        batch_texts = df["title"].iloc[start:end].tolist()
        batch_embeddings = get_embeddings_batch(batch_texts)
        embeddings.extend(batch_embeddings)
    df["embedding"] = embeddings
    return df

# Main execution
if __name__ == "__main__":
    print("📥 Fetching questions from Neo4j...")
    df = fetch_questions(uri, username, password, query)
    print(f"✅ Retrieved {len(df)} questions.")

    print("🧠 Generating embeddings with OpenAI...")
    df = embed_dataframe(df)

    # Save to CSV
    output_csv_path = "question_embeddings.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved embeddings to {output_csv_path}")
