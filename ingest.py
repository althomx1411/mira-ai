import os 
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from uuid import uuid4
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = r"files"
#load environment variables
load_dotenv()

#models key
models = os.getenv("NVIDIA_MODELS_KEY")



embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=models,
    truncate="NONE"
)

#database
chroma_db = Chroma(
    persist_directory="./chromadb",
    embedding_function=embeddings,
    collection_name="mira-ai-embeddings",
    )


loader = PyPDFDirectoryLoader(
    path=DATA_PATH
)

raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

chunks = splitter.split_documents(raw_docs)
uuids = [str(uuid4()) for _ in range(len(chunks))]
chroma_db.add_documents(documents=chunks, ids=uuids)

chroma_db.persist()

