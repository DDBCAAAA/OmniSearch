graph TD
    subgraph Offline ["⚙️ Offline Pipeline (Data Ingestion & Indexing)"]
        RawVid[Raw Videos / Audio] --> VidProcess[FFmpeg Framer + Whisper STT]
        RawDoc[PDFs / Docs] --> TextProcess[Semantic Text Chunker]
        VidProcess --> Align[Data Alignment & Metadata Tagging]
        TextProcess --> Align
        Align --> Embed1[Vertex AI Multimodal Embeddings]
        Embed1 --> DB[(pgvector / Vertex Vector Search)]
    end

    subgraph Online ["🚀 Online Serving (Retrieval & Generation)"]
        User((User)) --> UI[Streamlit UI]
        UI --> Embed2[Vertex AI Multimodal Embeddings]
        Embed2 --> HybridSearch[Hybrid Search Engine]
        
        HybridSearch -. Dense & Sparse .-> DB
        DB -. Raw Results .-> RRF[Reciprocal Rank Fusion]
        
        RRF --> PromptBuild[Context Assembly]
        UI --> PromptBuild
        PromptBuild --> LLM[Gemini 1.5 Pro / Flash]
        LLM -->|Grounded Response + Citations| UI
    end
