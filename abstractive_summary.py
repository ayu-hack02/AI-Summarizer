
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from PyPDF2 import PdfReader
import streamlit as st

st.set_page_config(layout= "wide")

@st.cache_resource
def loading():
    model_name = "google/pegasus-cnn_dailymail"
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = loading()

def summarize_text(text, summary_length):

    if summary_length == "Short":
        max_length = 200
        min_length = 100
    elif summary_length == "Moderate":
        max_length = 450
        min_length = 150
    elif summary_length == "Long":
        max_length = 700
        min_length = 550
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length= max_length, 
        min_length= min_length, 
        num_beams=6, 
        length_penalty=2.0, 
        early_stopping= True,
        no_repeat_ngram_size = 3
    )
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = summary.replace('<n>', "\n")
    return summary

def extract_text_from_pdf(pdf_file):
    # Open the PDF file using PyPDF2
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("Summarizer")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
text_input = st.text_area("Or paste your text here")

summary_length = st.selectbox("Select summary length",("Short", "Moderate", "Long"))

if uploaded_file is not None:
    with st.spinner("Extracting text"):
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text:")
        st.write(pdf_text)

        if st.button("Summarize PDF Text"):
            summary = summarize_text(pdf_text, summary_length)
            st.write("Summary:")
            st.write(summary)

elif text_input:
    if st.button("Summarize Pasted Text"):
        with st.spinner("Summarizing "):
            summary = summarize_text(text_input, summary_length)
            st.write("Summary:")
            st.write(summary)
