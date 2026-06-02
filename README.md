## System Design

This diagram illustrates the Agentic RAG pipeline: a user query first goes to an LLM equipped with tool‑calling capabilities. The LLM autonomously decides whether to retrieve information from a vector database (via cosine similarity), execute a SQL query, or perform a web search. The retrieved results are passed back to the LLM to generate an answer, which is then evaluated for quality. If the answer is insufficient, the system logs feedback and loops back for self‑correction, repeating until a good answer is produced or a maximum number of retries is reached.


``` mermaid

graph LR
    User[User] --> Agent[Agentic LLM<br/>with tool calling]
    
    subgraph Tools
        Vector[Vector DB<br/>Chroma/FAISS]
        SQL[SQL DB<br/>SQLite/Postgres]
        Web[Web Search<br/>Tavily/DDG]
    end
    
    Agent --> Tools
    Tools --> Agent
    
    subgraph Memory
        Conv[Conversation History]
        Eval[Evaluation Log]
    end
    
    Agent --> Conv
    Agent --> Eval
    Eval -->|score < threshold| Agent
    Eval -->|score >= threshold| Output[Final Answer]
    
    User --> Output


```
