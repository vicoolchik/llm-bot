from neo4j import GraphDatabase
import streamlit as st  # comment out if not using Streamlit

# Neo4j connection credentials
uri = st.secrets["NEO4J_URI"]
username = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]

# Cypher queries
# You must put this file either in Neo4j's /import folder or use a public HTTP URL
load_embeddings_query = """
LOAD CSV WITH HEADERS
FROM 'https://www.dropbox.com/scl/fi/8alrjz52vxvl94stn21pl/question_embeddings.csv?rlkey=1ne03u4qdb1115yvboxknzkxm&e=1&st=vr5orioz&dl=1' 
AS row
MATCH (q:Question {uuid: toInteger(row.question_id)})
CALL db.create.setVectorProperty(q, 'titleEmbedding', apoc.convert.fromJsonList(row.embedding))
YIELD node
RETURN count(*)
"""

create_vector_index_query = """
CREATE VECTOR INDEX questions IF NOT EXISTS
FOR (q:Question)
ON q.titleEmbedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
"""

def run_query(driver, query: str):
    with driver.session() as session:
        result = session.run(query)
        try:
            return result.data()
        except Exception as e:
            st.error(f"Failed to retrieve results: {e}")
            return []

if __name__ == "__main__":
    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        st.write("Running embedding load query...")
        result1 = run_query(driver, load_embeddings_query)
        st.success(f"✅ Embeddings loaded: {result1}")

        st.write("Creating vector index...")
        result2 = run_query(driver, create_vector_index_query)
        st.success("✅ Vector index created successfully.")
    except Exception as e:
        st.error(f"❌ Error running queries: {e}")
    finally:
        driver.close()
