import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import os

load_dotenv()

# 1. Load models
llm = ChatNVIDIA(
    model="minimaxai/minimax-m3",
    api_key=os.getenv("NVIDIA_MODELS_KEY"),
    temperature=0.2
)

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=os.getenv("NVIDIA_MODELS_KEY"),
    truncate="NONE"
)

# 2. Connect to the same vector store
chroma = Chroma(
    persist_directory="./chromadb",          # Must match your ingest script
    embedding_function=embeddings,
    collection_name="mira-ai-embeddings",
)

retriever = chroma.as_retriever(search_kwargs={"k": 5})

# 3. System prompt (tells the model how to use knowledge)
SYSTEM_PROMPT = """You are Mira, a helpful assistant that answers questions based only on the provided context.
If the context does not contain the answer, say "I don't have enough information in my documents."
Never use external knowledge – stick strictly to the retrieved passages.
"""

def format_history(history_list):
    """Convert Gradio's list of [user, assistant] to a single string for the prompt."""
    formatted = ""
    for user_msg, bot_msg in history_list:
        formatted += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    return formatted

def send_response(message, history):
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(message)
    knowledge = "\n".join([doc.page_content for doc in docs])

    # 2. Format conversation history (Gradio passes a list of [user, assistant] pairs)
    conversation = ""
    for user_msg, bot_msg in history:
        conversation += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    # 3. Build the prompt
    prompt = f"""{SYSTEM_PROMPT}

Conversation so far:
{conversation}

User's new question: {message}

Context (use this to answer):
{knowledge}

Answer:"""

    # 4. Stream the response – yield only the incremental assistant text
    full_response = ""
    for chunk in llm.stream(prompt):
        # Extract the text content (LangChain returns AIMessageChunk objects)
        content = chunk.content if hasattr(chunk, "content") else chunk
        full_response += content
        yield full_response  # Gradio will update the assistant bubble incrementally

# 4. Launch the interface
chatbot = gr.ChatInterface(
    fn=send_response,
    textbox=gr.Textbox(placeholder="Ask me anything about your documents...",
                       container=False, autoscroll=True, scale=7)
)

chatbot.launch()