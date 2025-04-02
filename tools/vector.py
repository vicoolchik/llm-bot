import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain.schema import StrOutputParser

from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="questions",
    node_label="Question",
    text_node_property="title",
    embedding_node_property="titleEmbedding",
retrieval_query="""
OPTIONAL MATCH (node)<-[:ASKED]-(user:User)
OPTIONAL MATCH (node)-[:TAGGED]->(tag:Tag)
WITH node, score, user, collect(DISTINCT tag.name) AS tags
RETURN
  node.title + '\\n\\n' + coalesce(node.body_markdown, '') AS text,
  score,
  {
    questionId: node.uuid,
    creationDate: node.creation_date,
    tags: tags,
    user: user.display_name,
    source: 'https://stackoverflow.com/questions/' + node.uuid
  } AS metadata
"""




)


retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def search_similar_question(question):
    """Searching for similar questions"""
    print("Executing embeddings search")
    return (plot_retriever.invoke({"input": question}) )
            # | RunnableLambda(lambda output: print(output.keys()))
            # | RunnableLambda(lambda output: output['result']))
