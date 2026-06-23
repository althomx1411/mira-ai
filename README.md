
## Mira-AI 
A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, NVIDIA AI endpoints, ChromaDB for vector storage, and Gradio for the user interface.
It ingests PDF documents, splits them into chunks, generates embeddings using NVIDIA's embedding model, and answers questions based on the retrieved context – all while maintaining conversation history.

## System Design

The sequence diagram below illustrates the exact order of operations when a user asks a question through the Gradio interface. It highlights the key components and the asynchronous streaming nature of the response.


``` mermaid

sequenceDiagram
    participant User
    participant UI as Gradio UI (main.py)
    participant Retriever
    participant DB as ChromaDB (./chromadb/)
    participant LLM as NVIDIA LLM (minimax-m3)

    User->>UI: Submits a question
    activate UI
    
    UI->>Retriever: Invoke(query)
    activate Retriever
    
    Retriever->>DB: Query for top-5 similar chunks
    activate DB
    DB-->>Retriever: Returns 5 relevant text chunks
    deactivate DB
    
    Retriever-->>UI: Returns retrieved context
    deactivate Retriever
    
    UI->>UI: Formats conversation history + context + system prompt
    
    UI->>LLM: Stream(prompt)
    activate LLM
    
    loop Stream tokens
        LLM-->>UI: Yields token chunk
        UI-->>User: Updates assistant response incrementally
    end
    
    deactivate LLM
    deactivate UI


```
