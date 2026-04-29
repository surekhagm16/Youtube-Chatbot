import streamlit as st
import re
from typing import Annotated, TypedDict, List
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
    RequestBlocked,
    IpBlocked,
    AgeRestricted,
    CouldNotRetrieveTranscript,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os


load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Transcript Chatbot",
    page_icon="🎬",
    layout="wide",
)

if not APP_PASSWORD:
    st.error("**App password not set.**\n\n")
    st.stop()

# ── Password gate ──────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.title("🔒 Video Transcript Chatbot")
    st.markdown("Please enter the password to continue.")
    pwd = st.text_input("Password", type="password", placeholder="Enter password…")
    if st.button("Unlock", use_container_width=False):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K = 5  # chunks retrieved per query
WINDOW_SIZE = 6  # messages kept in sliding window memory (3 exchanges)
CHROMA_COLLECTION = "transcript"


# ── Helpers ────────────────────────────────────────────────────────────────────


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    m = re.search(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def fetch_transcript(video_id: str) -> str:
    """Fetch transcript text from YouTube (v1.x instance-based API)."""
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    try:
        transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        transcript = next(iter(transcript_list))
    fetched = transcript.fetch()
    return " ".join(snippet.text for snippet in fetched)


def build_vectorstore(transcript: str) -> Chroma:
    """
    Split transcript → embed → store in Chroma (in-memory).
    Cached per video_id so rebuilding only happens when the video changes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(texts=[transcript])

    embeddings = HuggingFaceEmbeddings()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def retrieve_context(vectorstore: Chroma, query: str) -> str:
    """Semantic search → return top-k chunks joined as context string."""
    docs = vectorstore.similarity_search(query, k=TOP_K)
    return "\n\n---\n\n".join(d.page_content for d in docs)


def windowed_messages(history: InMemoryChatMessageHistory) -> list:
    """Return the last WINDOW_SIZE messages from memory."""
    return history.messages[-WINDOW_SIZE:]


# ── LangGraph state & graph ─────────────────────────────────────────────────────


class ChatState(TypedDict):
    messages: Annotated[List, add_messages]
    transcript: str
    summary: str
    mode: str  # "summarise" | "qa"
    query: str  # current user question (for retrieval)
    context: str  # retrieved chunks


def build_graph(vectorstore: Chroma, memory: InMemoryChatMessageHistory):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    # ── Node: retrieve ───────────────────────────────────────────────────────
    def retrieve_node(state: ChatState) -> ChatState:
        context = retrieve_context(vectorstore, state["query"])
        return {**state, "context": context}

    # ── Node: summarise ──────────────────────────────────────────────────────
    def summarise_node(state: ChatState) -> ChatState:
        transcript = state["transcript"]
        system = SystemMessage(
            content=(
                "You are an expert video analyst. "
                "When given a transcript, produce a clear, well-structured summary with:\n"
                "1. **Overview** – one-paragraph high-level summary\n"
                "2. **Key Points** – bullet list of the main ideas\n"
                "3. **Conclusion** – what the video concludes or recommends\n"
                "Be concise yet comprehensive."
            )
        )
        # For summarise, use full transcript (chunked context covers everything)
        user = HumanMessage(
            content=f"Summarise this video transcript:\n\n{transcript[:12000]}"
        )
        response = llm.invoke([system, user])
        summary_text = response.content
        memory.add_ai_message(summary_text)
        return {
            **state,
            "summary": summary_text,
            "messages": state["messages"] + [AIMessage(content=summary_text)],
        }

    # ── Node: answer question ────────────────────────────────────────────────
    def qa_node(state: ChatState) -> ChatState:
        context = state.get("context", "").strip()
        summary = state.get("summary", "").strip()
 
        # Build the best context available — retrieved chunks first,
        # fall back to transcript slice so Q&A works before summarise is run
        if context:
            context_section = f"### Retrieved Transcript Chunks\n{context}"
        else:
            context_section = f"### Transcript (first 12 000 chars)\n{state['transcript'][:12000]}"
 
        summary_section = (
            f"### Video Summary\n{summary}"
            if summary else
            "### Video Summary\nNot yet generated — answer from the transcript above."
        )
 
        system = SystemMessage(content=(
            "You are a helpful assistant answering questions about a video.\n"
            "Use the transcript content below to answer accurately. "
            "If the answer genuinely isn't in the provided content, say so politely.\n\n"
            f"{context_section}\n\n{summary_section}"
        ))
 
        history_msgs = windowed_messages(memory)
        messages_for_llm = [system] + history_msgs + [HumanMessage(content=state["query"])]
        response = llm.invoke(messages_for_llm)
        ai_content = response.content
        memory.add_ai_message(ai_content)
        return {**state, "messages": state["messages"] + [AIMessage(content=ai_content)]}
    # ── Router ───────────────────────────────────────────────────────────────
    def router(state: ChatState) -> str:
        return state["mode"]

    # ── Build graph ──────────────────────────────────────────────────────────
    g = StateGraph(ChatState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("summarise", summarise_node)
    g.add_node("qa", qa_node)

    # summarise: skip retrieval (uses full transcript directly)
    # qa: retrieve first, then answer
    g.set_conditional_entry_point(router, {"summarise": "summarise", "qa": "qa"})
    g.add_edge("retrieve", "qa")
    g.add_edge("summarise", END)
    g.add_edge("qa", END)
    return g.compile()


# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "transcript": "",
    "summary": "",
    "chat_history": [],  # list of {"role": "user"|"assistant", "content": str}
    "video_loaded": False,
    "video_id": "",
    "memory": None,  # InMemoryChatMessageHistory
    "vectorstore": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/youtube-play.png", width=64)
    st.title("Video Chatbot")
    st.markdown("Built by SM")
    st.divider()

    st.subheader("📹 Load a Video")
    video_url = st.text_input(
        "YouTube URL", placeholder="https://youtube.com/watch?v=..."
    )

    if st.button("🔄 Load", use_container_width=True):
        if not video_url:
            st.error("Please enter a YouTube URL.")
        else:
            vid_id = extract_video_id(video_url)
            if not vid_id:
                st.error("Could not parse video ID. Check the URL.")
            else:
                with st.spinner("Fetching transcript…"):
                    try:
                        transcript = fetch_transcript(vid_id)
                    except TranscriptsDisabled:
                        st.error("❌ Transcripts are disabled for this video.")
                        transcript = None
                    except NoTranscriptFound:
                        st.error(
                            "❌ No transcript found. This video may not have captions."
                        )
                        transcript = None
                    except VideoUnavailable:
                        st.error(
                            "❌ Video unavailable — private, deleted, or region-locked."
                        )
                        transcript = None
                    except InvalidVideoId:
                        st.error("❌ Invalid video ID. Please double-check the URL.")
                        transcript = None
                    except AgeRestricted:
                        st.error("❌ Age-restricted — transcript cannot be accessed.")
                        transcript = None
                    except (RequestBlocked, IpBlocked):
                        st.error(
                            "❌ YouTube blocked the request. Try again in a few minutes."
                        )
                        transcript = None
                    except CouldNotRetrieveTranscript as e:
                        st.error(f"❌ Could not retrieve transcript: {e}")
                        transcript = None
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                        transcript = None

                if transcript:
                    with st.spinner("Building vector index…"):
                        vectorstore = build_vectorstore(transcript)

                    st.session_state.transcript = transcript
                    st.session_state.video_id = vid_id
                    st.session_state.video_loaded = True
                    st.session_state.summary = ""
                    st.session_state.chat_history = []
                    st.session_state.memory = InMemoryChatMessageHistory()
                    st.session_state.vectorstore = vectorstore
                    chunk_count = vectorstore._collection.count()
                    st.success(
                        f"✅ Ready! {len(transcript):,} chars indexed into {chunk_count} chunks."
                    )

    if st.session_state.video_loaded:
        st.divider()

        if st.button("📋 Summarise Video", use_container_width=True):
            with st.spinner("Summarising…"):
                graph = build_graph(
                    st.session_state.vectorstore, st.session_state.memory
                )
                result = graph.invoke(
                    {
                        "messages": [],
                        "transcript": st.session_state.transcript,
                        "summary": "",
                        "mode": "summarise",
                        "query": "",
                        "context": "",
                    }
                )
                st.session_state.summary = result["summary"]
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"📋 **Video Summary**\n\n{result['summary']}",
                    }
                )
                st.rerun()

        # RAG info
        with st.expander("🔍 RAG & Memory Info", expanded=False):
            st.markdown(f"""
**Splitter:** `RecursiveCharacterTextSplitter`  
**Chunk size:** {CHUNK_SIZE} chars  
**Overlap:** {CHUNK_OVERLAP} chars  
**Embeddings:** `all-MiniLM-L6-v2` (local)  
**Vector DB:** Chroma (in-memory)  
**Top-K retrieval:** {TOP_K} chunks  
**Memory:** Sliding window — last {WINDOW_SIZE} messages  
            """)
            if st.session_state.vectorstore:
                count = st.session_state.vectorstore._collection.count()
                st.metric("Chunks indexed", count)

        with st.expander("📄 Raw Transcript", expanded=False):
            st.text_area(
                "",
                st.session_state.transcript[:3000]
                + ("…" if len(st.session_state.transcript) > 3000 else ""),
                height=200,
                disabled=True,
            )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.memory = InMemoryChatMessageHistory()
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("🎬 Video Transcript Chatbot")

if not st.session_state.video_loaded:
    st.markdown("""
    ### How to get started
    1. **Paste a YouTube URL** in the sidebar (video must have captions)
    2. Click **Load** — the transcript is fetched and indexed into Chroma
    3. Hit **Summarise Video** for a structured overview, or
    4. **Ask any question** — the app retrieves the most relevant chunks and answers
    """)
    c1, c2, c3 = st.columns(3)
    c1.info("**📝 Summarise**\nGet a structured overview in seconds")
    c2.info("**💬 Q&A**\nSemantic search over the full transcript")
    c3.info("**🧠 Memory**\nSliding window keeps conversational context")

else:
    vid_col, _ = st.columns([2, 3])
    with vid_col:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    st.divider()
    st.subheader("💬 Chat")

    if not st.session_state.chat_history:
        st.caption(
            "No messages yet. Ask a question or click **Summarise Video** in the sidebar."
        )
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about the video…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.memory.add_user_message(prompt)

        lc_messages = [HumanMessage(content=prompt)]

        with st.spinner("Retrieving & thinking…"):
            graph = build_graph(st.session_state.vectorstore, st.session_state.memory)
            result = graph.invoke(
                {
                    "messages": lc_messages,
                    "transcript": st.session_state.transcript,
                    "summary": st.session_state.summary,
                    "mode": "qa",
                    "query": prompt,
                    "context": "",
                }
            )

        ai_content = next(
            (
                m.content
                for m in reversed(result["messages"])
                if isinstance(m, AIMessage)
            ),
            "",
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": ai_content}
        )
        st.rerun()
