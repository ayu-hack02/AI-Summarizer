import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")

# Cache the model and tokenizer to avoid reloading on every run
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "google/pegasus-cnn_dailymail"
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def summarize_text(text):
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=1000, 
        min_length=200, 
        length_penalty=2.0, 
        early_stopping=True
    )
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = summary.replace('<n>', '\n')
    return summary

def extract_text_from_pdf(pdf_file):
    # Open the PDF file using PyPDF2
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("Summarizer")

# Add a file uploader for PDF files
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Add a text area for direct text input
text_input = st.text_area("Or paste your text here")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text:")
        st.write(pdf_text)
        
        if st.button("Summarize PDF Text"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(pdf_text)
                st.write("Summary:")
                st.write(summary)

elif text_input:
    if st.button("Summarize Pasted Text"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(text_input)
            st.write("Summary:")
            st.write(summary)
