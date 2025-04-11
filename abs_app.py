import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarization_model()

@st.cache_resource
def summary_text(text):
    result = summarizer(text, max_length=300, min_length=100, do_sample=False)
    return result[0]['summary_text']

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
