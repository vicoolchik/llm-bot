import streamlit as st
from llm import llm
from graph import graph
from langchain.schema import StrOutputParser
from operator import itemgetter
from langchain_neo4j import GraphCypherQAChain
from langchain_core.runnables import RunnableLambda, RunnableMap


from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions on stack overflow about  software development.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Few Shot examples
Question: How many questions are there with haskell tag
Output:
    MATCH (q:Question)-[:TAGGED]->(t:Tag name: "haskell")
    RETURN count(q) AS haskell_questions_count
    
Quesion: What are tags for unanswererd questions?    
    MATCH (q:Question)-[:TAGGED]->(t:Tag)
    WHERE NOT t.name IN ['neo4j','cypher']
    AND NOT (q)<-[:ANSWERED]-()
    RETURN t.name as tag, count(q) AS questions
    ORDER BY questions DESC LIMIT 10;

Question: Who are the top users asking questions?
    MATCH (u:User)-[:ASKED]->(q:Question)
    RETURN u.display_name, count(*) AS questions
    ORDER by questions DESC
    LIMIT 10;
Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)



cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,

)

get_stackoverflow = (
    cypher_qa 
    | RunnableLambda(lambda output: "Based on stack overflow stats: " + output.get('result', "None"))

)
