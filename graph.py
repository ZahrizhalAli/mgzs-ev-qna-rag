from typing import Annotated, TypedDict, Literal

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Load the HTML as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)

documents = text_splitter.split_documents(car_docs)

# Store documents in a vector store
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

db = Chroma.from_documents(documents, embeddings_model)

class State(TypedDict):
    messages : str

@tool
def get_car_information(query: State):
    """Call to answer questions related to Car."""
    # Chroma Retriever
    retriever = db.as_retriever()
    # Define RAG Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. Use the following pieces of retrieved 
        context to answer the question. If you don't know the answer, 
        just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        \nQuestion: {question} \nContext: {context} \nAnswer:
        """
    )

    # Setup Chain and model

    llm = ChatOpenAI()

    rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
                 | prompt | llm)

    answer = rag_chain.invoke(query["messages"]).content
    return {"messages": answer}

@tool
def get_weather(location: str):
    """Call to get the current weather. Add advice of driving safety based on condition"""
    # A simplified weather response based on location
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


# List of tools that will be accessible to the graph via the ToolNode
tools = [get_weather, get_car_information]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        # specifically for OpenAI we have to set parallel tool call to false
        # because of st primitively visually rendering the tool results
    ).bind_tools(tools, parallel_tool_calls=False)
    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")

# Compile the state graph into a runnable object
graph_runnable = graph.compile()