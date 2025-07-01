# policy_explainer.py
import streamlit as st
from utils import generate_output, process_file_upload, load_model
import time

# Configure page
st.set_page_config(
    page_title="Policy Document Explainer", 
    page_icon="📄",
    layout="centered"
)

# Page header
st.title("📄 Policy Document Explainer & Generator")
st.markdown("""
This tool helps you understand complex insurance policy documents by:
- Simplifying technical insurance clauses into plain language
- Creating draft documents based on policy text
""")

# Load model - cache it to avoid reloading on each rerun
@st.cache_resource
def get_model():
    with st.spinner("Loading language model (this may take a moment for first run)..."):
        return load_model("TheBloke/Mistral-7B-Instruct-v0.2-GGUF")

# Model info
with st.sidebar:
    st.info("Model status: 💻 Using Mistral-7B-Instruct")
    st.markdown("### About")
    st.markdown("""
    This app uses the powerful Mistral-7B-Instruct model to process insurance policy text.
    
    **Two modes available:**
    - **Simplify**: Converts complex policy language into simpler terms
    - **Draft**: Creates a structured document based on policy text
    
    Note: First run may take longer as the model is downloaded.
    """)

# Main UI
st.subheader("Input")
st.markdown("Enter policy text using one of these methods:")

# Input options - either text or file upload
input_type = st.radio("Choose input method:", ["Paste Text", "Upload File"])

input_text = ""
if input_type == "Paste Text":
    input_text = st.text_area("Insurance Clause:", height=200, 
                              placeholder="Paste the policy text you'd like to process...")
else:
    uploaded_file = st.file_uploader("Upload Policy Document", type=['txt', 'pdf'])
    if uploaded_file is not None:
        try:
            input_text = process_file_upload(uploaded_file)
            st.success("File uploaded and processed successfully!")
            # Show preview of uploaded text
            with st.expander("Preview uploaded text"):
                st.text(input_text[:500] + ("..." if len(input_text) > 500 else ""))
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Processing options
st.subheader("Processing Options")
col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Choose action:", ["Simplify", "Draft"])
with col2:
    # Sample insurance text for demo
    if st.button("Use Sample Text"):
        input_text = """SECTION III - COVERAGE C - PERSONAL PROPERTY
We cover personal property owned or used by an insured person while it is anywhere in the world. At your request, we will cover personal property owned by others while the property is on the part of the residence premises occupied by an insured person. Also at your request, we will cover personal property owned by a guest or a residence employee while the property is in any residence occupied by an insured person.

Our limit of liability for personal property usually located at an insured person's residence, other than the residence premises, is 10% of Coverage C - Personal Property. This limitation does not apply to personal property maintained in a newly acquired principal residence for 30 days from the time you begin to move the property there."""
        st.rerun()

# Process button
process_button = st.button("Generate Output", type="primary")

# Only load model if needed
if process_button or ('result' in st.session_state and st.session_state.result):
    try:
        # Show progress as model loads
        progress_bar = None
        tokenizer, model = None, None
        
        if process_button:
            if not input_text or input_text.strip() == "":
                st.warning("⚠️ Please enter policy text or upload a file first.")
            else:
                with st.spinner(f"Processing with {mode} mode..."):
                    # Get model on demand
                    tokenizer, model = get_model()
                    
                    # Track progress with a progress bar
                    progress_text = "Generating output..."
                    progress_bar = st.progress(0, text=progress_text)
                    
                    # Simulate progress for better UX
                    for percent_complete in range(0, 101, 10):
                        time.sleep(0.1)  # Small delay for visual feedback
                        progress_bar.progress(percent_complete/100, text=f"{progress_text} {percent_complete}%")
                    
                    # Generate output from model
                    result = generate_output(input_text, mode, tokenizer, model)
                    # Store in session state
                    st.session_state.result = result
                    st.session_state.mode = mode
                    
                    # Complete progress
                    if progress_bar:
                        progress_bar.empty()
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        if progress_bar:
            progress_bar.empty()

# Display result if available
if 'result' in st.session_state and st.session_state.result:
    st.subheader("Output")
    st.success(f"✅ {st.session_state.mode} completed successfully!")
    
    result_container = st.container(border=True)
    with result_container:
        st.markdown("### Processed Result")
        st.markdown(st.session_state.result)
    
    # Download option
    st.download_button(
        label="Download Result",
        data=st.session_state.result,
        file_name=f"policy_{st.session_state.mode.lower()}_result.txt",
        mime="text/plain"
    )
