graph TD
    %% Styling definitions for a clean, "Google Design Doc" look
    classDef gcpNode fill:#e8f0fe,stroke:#4285f4,stroke-width:2px,color:#1a73e8;
    classDef dataNode fill:#fce8e6,stroke:#ea4335,stroke-width:2px,color:#c5221f;
    classDef processNode fill:#e6f4ea,stroke:#34a853,stroke-width:2px,color:#137333;
    classDef storageNode fill:#fef7e0,stroke:#fbbc04,stroke-width:2px,color:#b06000;

    subgraph Offline_Data_Pipeline ["⚙️ Offline Pipeline (Data Ingestion & Indexing)"]
        direction TB
        RawVid["Raw Videos & Audio (GCS)"] ::: dataNode
        RawDoc["PDFs & Docs (GCS)"] ::: dataNode
        
        VidProcess["FFmpeg (Framer) + Whisper (STT)"] ::: processNode
        TextProcess["Semantic Text Chunker"] ::: processNode
        
        Align["Data Alignment & Metadata Tagging"] ::: processNode
        
        RawVid --> VidProcess
        RawDoc --> TextProcess
        VidProcess -->|"Frames + Timestamps"| Align
        TextProcess -->|"Text Chunks"| Align
        
        Embed1["Vertex AI Multimodal Embeddings"] ::: gcpNode
        Align --> Embed1
        
        DB[/"pgvector / Vertex Vector Search"\] ::: storageNode
        Embed1 -->|"Vectors + Metadata"| DB
    end

    subgraph Online_Serving_Pipeline ["🚀 Online Serving (Retrieval & Generation)"]
        direction TB
        User(["👤 User Query"])
        UI["Streamlit UI"] ::: processNode
        
        Embed2["Vertex AI Multimodal Embeddings"] ::: gcpNode
        
        HybridSearch["Hybrid Search Engine"] ::: processNode
        RRF["Reciprocal Rank Fusion (RRF)"] ::: processNode
        
        PromptBuild["Context Assembly (Prompt Builder)"] ::: processNode
        LLM["Gemini 1.5 Flash/Pro"] ::: gcpNode
        
        User --> UI
        UI -->|"1. User Input"| Embed2
        Embed2 -->|"2. Query Vector"| HybridSearch
        
        HybridSearch -.->|"Dense (Vector)"| DB
        HybridSearch -.->|"Sparse (BM25)"| DB
        DB -.->|"Raw Results"| RRF
        
        RRF -->|"3. Top-K Context (Images & Text)"| PromptBuild
        UI -->|"Raw Query"| PromptBuild
        PromptBuild -->|"4. Packaged Prompt"| LLM
        LLM -->|"5. Grounded Response + Citations"| UI
    end
