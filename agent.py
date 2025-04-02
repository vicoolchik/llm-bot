
from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.vector import search_similar_question
from tools.cypher import get_stackoverflow


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a exprienced senior software engineer providing useful information."),
        ("human", "{input}"),
    ]
)

stack_chat = chat_prompt | llm | StrOutputParser()


tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat, not related to software engineering",
        func=stack_chat.invoke,
    ), 
    Tool.from_function(
        name="search similar question",  
        description="Searching for a similar  question on stack overflow about feature/code/method/class/function or anything else related to development and you do not know it in general for example: what are the parameters of numpy.ndarray of plt.imread, what classes Counter class inherit from",
        func=search_similar_question, 
    ),
    Tool.from_function(
        name="Stack overflow information",
        description="Provide information about stack overflow site using Cypher, how many question are there on Stack overflow or how many tag python is seen in questions",
        func = get_stackoverflow.invoke

    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a Stack Overflow agent specializing in programming and software development topics. Provide detailed, helpful answers strictly related to coding, software engineering, debugging, tools, frameworks, libraries, and best programming practices.

Be as helpful as possible, offering comprehensive explanations and relevant code examples.

Do not answer questions unrelated to programming, software engineering, or development tools.

Use only the information provided in the context supplied with each question, avoiding any pre-trained knowledge outside of that context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
handle_parsing_errors=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']