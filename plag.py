from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess_text(text):
    # Add your text preprocessing steps here
    return text.lower()

def calculate_similarity(doc1, doc2):
    # Preprocess the text
    doc1 = preprocess_text(doc1)
    doc2 = preprocess_text(doc2)
    
    # Create the Document Term Matrix using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    
    # Compute the cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

if __name__ == "__main__":
    # Sample documents
    document1 = ""
    document2 = "This document is a sample to check plagiarism."

    similarity_score = calculate_similarity(document1, document2)
    print(f"Similarity Score: {similarity_score:.2f}")

    if similarity_score > 0.7:  # Set a threshold
        print("Plagiarism Detected")
    else:
        print("No Plagiarism Detected")
