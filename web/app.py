import streamlit as st
import requests
import os

# CONFIGURATION
# -----------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=" Sentiment AI",
    page_icon="🧠",
    layout="centered"
)

# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# HEADER SECTION
# -----------------------------------------------------------------------------
st.title("🧠 AI Sentiment Analyzer")
st.markdown("""
Welcome! This application uses a **DistilBERT** Transformer model to understand the 
emotion behind your text in real-time.
""")

st.write("---")

# INPUT SECTION
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Enter your text here:", 
        height=150,
        placeholder="Type something like: 'I absolutely loved the movie, it was fantastic!'"
    )

with col2:
    st.info("💡 **Tip:** Try complex sentences with mixed feelings to see how the model reacts.")
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# LOGIC & RESULTS
# -----------------------------------------------------------------------------
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Talking to the AI Brain..."):
            try:
                # 1. Send request to our FastAPI Backend
                response = requests.post(f"{API_URL}/predict", json={"text": user_input})
                
                # 2. Check for success
                if response.status_code == 200:
                    data = response.json()
                    label = data['label']
                    score = data['score']
                    
                    # 3. Display Results
                    st.write("### Analysis Results")
                    
                    # Visual conditioning based on result
                    if label == "POSITIVE":
                        st.markdown(
                            f"""
                            <div class="result-box positive">
                                <h2>😊 Positive Sentiment</h2>
                                <p>The model is <strong>{score:.2%}</strong> confident.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box negative">
                                <h2>😠 Negative Sentiment</h2>
                                <p>The model is <strong>{score:.2%}</strong> confident.</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # 4. Technical Details (Expandable)
                    with st.expander("See Technical Details"):
                        st.json(data)
                        st.code(f"POST {API_URL}/predict\nPayload: {str({'text': user_input})}", language="http")
                        
                else:
                    st.error(f"❌ Server Error: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Failed")
                st.markdown(
                    f"Could not connect to the API at `{API_URL}`. "
                    "Is the backend server running?"
                )