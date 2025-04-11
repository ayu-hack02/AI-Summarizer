#not good
import streamlit as st
from txtai.pipeline import Summary
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

# Initialize the Summary object only once
@st.cache_resource
def load_summary():
    return Summary()

summary = load_summary()

@st.cache_resource
def summary_text(text):
    result = summary(text)
    return result

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    page = reader.pages[0]
    text = page.extract_text()
    return text

choice = st.sidebar.selectbox("Select", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                result = summary_text(input_text)
                st.markdown("**Your summary**")
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    input_file = st.file_uploader("Upload your document", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Extracted Text**")
                extracted_text = extract_text_from_pdf(input_file)
                st.info(extracted_text)
            with col2:
                st.markdown("**Your summary**")
                summary_result = summary_text(extracted_text)
                st.success(summary_result)
