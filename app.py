import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from modules.video_processor import process_multimodal_video
from modules.vector_brain import VaultBrain

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Page Config
st.set_page_config(page_title="VaultVision AI", page_icon="👁️", layout="wide")
st.title("👁️ VaultVision: Multimodal Enterprise Search")
st.markdown("Upload internal company videos and instantly search through **visuals** and **spoken words**.")

# Initialize the AI Brain in Streamlit's memory so it doesn't erase on refresh
if "brain" not in st.session_state:
    st.session_state.brain = VaultBrain()
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# --- SIDEBAR: INGESTION ZONE ---
with st.sidebar:
    st.header("1. Upload Knowledge")
    uploaded_file = st.file_uploader("Upload Company Video (.mp4)", type=["mp4"])
    
    if uploaded_file is not None and st.button("Process Video into Brain"):
        with st.spinner("Slicing audio and extracting Llama-Vision keyframes... This is heavy lifting..."):
            # Save the file temporarily
            temp_path = f"data/uploads/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.video_path = temp_path
            
            # 1. Extract Multimodal Data
            data = process_multimodal_video(temp_path, extraction_interval=30)
            
            # 2. Feed it to the Vector Brain
            st.session_state.brain.build_multimodal_index(data)
            st.success("✅ Video fully assimilated into VaultVision.")

# --- MAIN WINDOW: SEARCH ZONE ---
st.header("2. Search the Vault")

query = st.text_input("Ask a question about the video (e.g., 'Where is the revenue graph?', 'How do I restart the server?'):")

if query:
    if not st.session_state.video_path:
        st.warning("Please upload and process a video first in the sidebar.")
    else:
        with st.spinner("Searching multimodal vectors..."):
            # 1. Search the FAISS database
            results = st.session_state.brain.search(query, top_k=3)
            
            if not results:
                st.info("No relevant visual or audio data found for that query.")
            else:
                # 2. Let Llama-3 write a conversational answer based on the results
                context_string = "\n".join([f"[{r['type'].upper()} at {int(r['start'])}s]: {r['content']}" for r in results])
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are VaultVision AI. Answer the user's question using ONLY the provided video context. Be concise and mention the timestamp."},
                        {"role": "user", "content": f"Context from video:\n{context_string}\n\nUser Question: {query}"}
                    ],
                    model="llama3-70b-8192",
                    temperature=0.3,
                )
                
                st.markdown("### 🤖 AI Answer:")
                st.write(chat_completion.choices.message.content)
                
                # 3. THE MAGIC: Play the video at the exact timestamp of the best result!
                best_match = results
                jump_time = int(best_match['start'])
                
                st.markdown(f"### 🎬 Jumping to exact moment: **{jump_time} seconds**")
                # Streamlit's st.video has a start_time parameter!
                st.video(st.session_state.video_path, start_time=jump_time)
                
                # Show the backend logic to impress users/recruiters
                with st.expander("🔍 See raw vector search results (Developer View)"):
                    for i, r in enumerate(results):
                        st.write(f"**Match {i+1} [{r['type']}]** at {int(r['start'])}s: {r['content']}")