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
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Transcript Chatbot",
    page_icon="🎬",
    layout="wide",
)


APP_PASSWORD = os.getenv("APP_PASSWORD")

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


# ── Helpers ────────────────────────────────────────────────────────────────────


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


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


# ── LangGraph state & graph ─────────────────────────────────────────────────────


class ChatState(TypedDict):
    messages: Annotated[List, add_messages]
    transcript: str
    summary: str
    mode: str  # "summarise" | "qa"


def build_graph():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

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
        user = HumanMessage(
            content=f"Summarise this video transcript:\n\n{transcript[:12000]}"
        )
        response = llm.invoke([system, user])
        summary_text = response.content
        ai_msg = AIMessage(content=summary_text)
        return {
            **state,
            "summary": summary_text,
            "messages": state["messages"] + [ai_msg],
        }

    # ── Node: answer question ────────────────────────────────────────────────
    def qa_node(state: ChatState) -> ChatState:
        transcript = state["transcript"]
        summary = state.get("summary", "")
        history = state["messages"]

        system = SystemMessage(
            content=(
                "You are a helpful assistant that answers questions about a video. "
                "You have access to the full transcript and a summary below. "
                "Answer questions accurately based on the video content. "
                "If the answer isn't in the transcript, say so politely.\n\n"
                f"### Summary\n{summary}\n\n"
                f"### Full Transcript (first 12 000 chars)\n{transcript[:12000]}"
            )
        )

        messages_for_llm = [system] + [
            m for m in history if isinstance(m, (HumanMessage, AIMessage))
        ]
        response = llm.invoke(messages_for_llm)
        ai_msg = AIMessage(content=response.content)
        return {**state, "messages": state["messages"] + [ai_msg]}

    # ── Router ───────────────────────────────────────────────────────────────
    def router(state: ChatState) -> str:
        return state["mode"]

    # ── Build graph ──────────────────────────────────────────────────────────
    g = StateGraph(ChatState)
    g.add_node("summarise", summarise_node)
    g.add_node("qa", qa_node)
    g.set_conditional_entry_point(router, {"summarise": "summarise", "qa": "qa"})
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

    if st.button("🔄 Load ", use_container_width=True):
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
                        st.session_state.transcript = transcript
                        st.session_state.video_id = vid_id
                        st.session_state.video_loaded = True
                        st.session_state.summary = ""
                        st.session_state.chat_history = []
                        st.success(f"✅ Transcript loaded! ({len(transcript):,} chars)")
                    except TranscriptsDisabled:
                        st.error(
                            "❌ Transcripts are disabled for this video. The creator has turned off captions."
                        )
                    except NoTranscriptFound:
                        st.error(
                            "❌ No transcript found for this video. It may not have captions in any language."
                        )
                    except VideoUnavailable:
                        st.error(
                            "❌ This video is unavailable. It may be private, deleted, or region-locked."
                        )
                    except InvalidVideoId:
                        st.error(
                            "❌ Invalid video ID. Please double-check the YouTube URL."
                        )
                    except AgeRestricted:
                        st.error(
                            "❌ This video is age-restricted and its transcript cannot be accessed."
                        )
                    except (RequestBlocked, IpBlocked):
                        st.error(
                            "❌ YouTube blocked the request. Try again in a few minutes."
                        )
                    except CouldNotRetrieveTranscript as e:
                        st.error(f"❌ Could not retrieve transcript: {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")

    if st.session_state.video_loaded:
        st.divider()
        if st.button("📋 Summarise Video", use_container_width=True):
            with st.spinner("Summarising…"):
                graph = build_graph()
                result = graph.invoke(
                    {
                        "messages": [],
                        "transcript": st.session_state.transcript,
                        "summary": "",
                        "mode": "summarise",
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

        with st.expander("📄 Raw Transcript", expanded=False):
            st.text_area(
                "Transcript",
                st.session_state.transcript[:3000]
                + ("…" if len(st.session_state.transcript) > 3000 else ""),
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("🎬 Video Transcript Chatbot")

if not st.session_state.video_loaded:
    # Landing / onboarding
    st.markdown("""
    ### How to get started


    1. **Paste a YouTube URL** in the sidebar (the video must have captions/subtitles)
    2. Click **Load**
    3. Hit **Summarise Video** for an instant overview, or
    4. **Ask any question** about the video in the chat below
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📝 Summarise**\nGet a structured overview in seconds")
    with col2:
        st.info("**💬 Q&A**\nAsk anything about the video content")
    with col3:
        st.info("**🔍 Deep Dive**\nExtract specific details or insights")

else:
    # Show video embed
    vid_col, _ = st.columns([2, 3])
    with vid_col:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    st.divider()
    st.subheader("💬 Chat")

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.caption(
                "No messages yet. Ask a question or click **Summarise Video** in the sidebar."
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask anything about the video…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Build message history for LangGraph
        lc_messages = []
        for m in st.session_state.chat_history[:-1]:  # exclude latest user msg
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))
            lc_messages.append(HumanMessage(content=prompt))

        with st.spinner("Thinking…"):
            graph = build_graph()
            result = graph.invoke(
                {
                    "messages": lc_messages,
                    "transcript": st.session_state.transcript,
                    "summary": st.session_state.summary,
                    "mode": "qa",
                }
            )

        # Extract last AI message
        ai_content = ""
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage):
                ai_content = m.content
                break

        st.session_state.chat_history.append(
            {"role": "assistant", "content": ai_content}
        )
        st.rerun()
