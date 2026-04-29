🎬 Video Transcript Chatbot
A Streamlit app that lets you load any YouTube video, summarise it, and ask questions about it — powered by LangGraph, LangChain, Groq (LLaMA 3.3), and a Chroma RAG pipeline.

Features

🔒 Password-protected login screen
📹 Load any YouTube video by URL (auto-fetches transcript)
📋 Structured video summary — overview, key points, conclusion
💬 Q&A chat grounded in the transcript via semantic search
🧠 Sliding window conversation memory (last 6 messages)
🔍 RAG pipeline — transcript is chunked, embedded, and stored in Chroma
❌ Friendly error messages for missing captions, private videos, etc.


Architecture
YouTube URL
    │
    ▼
YouTubeTranscriptApi
    │
    ▼
RecursiveCharacterTextSplitter   (500 chars, 50 overlap)
    │
    ▼
HuggingFace Embeddings           (all-MiniLM-L6-v2, runs locally)
    │
    ▼
Chroma Vector DB                 (in-memory, per session)
    │
    ▼
LangGraph
    ├── [summarise node] ──► full transcript ──► LLaMA 3.3 via Groq
    └── [qa node] ──► similarity_search (top-5) ──► LLaMA 3.3 via Groq
                            │
                    InMemoryChatMessageHistory
                    (sliding window — last 6 messages)

Tech Stack
ComponentLibraryUIStreamlitLLMGroq — llama-3.3-70b-versatileOrchestrationLangGraphRAG / chainsLangChainVector DBChroma (in-memory)EmbeddingsHuggingFace all-MiniLM-L6-v2 (local, no API cost)Transcript fetchingyoutube-transcript-api v1.xMemoryInMemoryChatMessageHistory (sliding window)

Project Structure
video_chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API keys and password (never commit this)
├── .env.example        # Template for .env
├── .gitignore          # Excludes .env and cache files
└── README.md           # This file


Known Limitations

Only the first 12,000 characters of the transcript are used for summarisation
Videos without captions cannot be transcribed (Whisper integration is a future enhancement)
Chroma is in-memory — the vector index is rebuilt each time the app restarts or a new video is loaded
Session memory is lost on page refresh
