import streamlit as st
import pytesseract
from PIL import Image
import PyPDF2
import io
import os
import openai
import random
import pandas as pd
from transformers import pipeline

# Set page configuration
st.set_page_config(page_title="StudyAI - Note Summarizer", layout="wide")

# Initialize OpenAI API key from environment variable or user input
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        openai.api_key = openai_api_key

# Initialize Hugging Face summarizer for offline summarization option
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to extract text from image using OCR
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

# Function to generate summary using OpenAI
def generate_summary(text, max_length=150):
    if not openai_api_key:
        # Fallback to Hugging Face if no API key
        summarizer = load_summarizer()
        # Split text into chunks if it's too long
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = []
        for chunk in chunks:
            if len(chunk) > 100:  # Minimum length for the summarizer
                summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        return " ".join(summaries)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic notes concisely."},
                {"role": "user", "content": f"Summarize the following notes in a concise TL;DR format (maximum {max_length} words):\n\n{text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""

# Function to generate flashcards
def generate_flashcards(text, num_cards=5):
    if not openai_api_key:
        st.error("OpenAI API key is required for flashcard generation")
        return []
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates educational flashcards."},
                {"role": "user", "content": f"Create {num_cards} flashcards from the following notes. Format as JSON with 'question' and 'answer' fields:\n\n{text}"}
            ]
        )
        
        # Extract JSON from response
        import json
        content = response.choices[0].message.content
        # Find JSON part in the response
        start_idx = content.find("[")
        end_idx = content.rfind("]") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            try:
                flashcards = json.loads(json_str)
                return flashcards
            except:
                # If JSON parsing fails, return raw text
                st.warning("Could not parse flashcards as JSON. Showing raw output.")
                return content
        else:
            return content
    except Exception as e:
        st.error(f"Error generating flashcards: {e}")
        return []

# Function to generate quiz questions
def generate_quiz(text, num_questions=5):
    if not openai_api_key:
        st.error("OpenAI API key is required for quiz generation")
        return []
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates educational quizzes."},
                {"role": "user", "content": f"Create a {num_questions}-question multiple-choice quiz based on these notes. Format as JSON with 'question', 'options' (array), and 'correct_answer' (index) fields:\n\n{text}"}
            ]
        )
        
        # Extract JSON from response
        import json
        content = response.choices[0].message.content
        # Find JSON part in the response
        start_idx = content.find("[")
        end_idx = content.rfind("]") + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            try:
                quiz = json.loads(json_str)
                return quiz
            except:
                # If JSON parsing fails, return raw text
                st.warning("Could not parse quiz as JSON. Showing raw output.")
                return content
        else:
            return content
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return []

# Main app UI
st.title("📚 StudyAI - Note Summarizer")
st.markdown("Upload your notes and let AI help you study more efficiently!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    summary_length = st.slider("Summary Length (words)", 50, 300, 150)
    num_flashcards = st.slider("Number of Flashcards", 3, 15, 5)
    num_quiz_questions = st.slider("Number of Quiz Questions", 3, 10, 5)
    
    st.header("About")
    st.info("""
    StudyAI helps students process their notes using AI.
    - Upload PDF documents, text files, or images of handwritten notes
    - Get concise summaries
    - Generate flashcards for studying
    - Create quizzes to test your knowledge
    """)

# File upload section
st.header("Upload Your Notes")
file_type = st.radio("Select input type:", ["Text Input", "PDF Document", "Image (Handwritten Notes)"])

extracted_text = ""

if file_type == "Text Input":
    extracted_text = st.text_area("Enter your notes here:", height=200)
    
elif file_type == "PDF Document":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF processed successfully!")
        
elif file_type == "Image (Handwritten Notes)":
    uploaded_file = st.file_uploader("Upload an image of your notes", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Performing OCR on image..."):
            extracted_text = extract_text_from_image(uploaded_file)
        st.success("OCR completed successfully!")

# Display extracted text
if extracted_text:
    with st.expander("View Extracted Text"):
        st.text_area("Extracted Content", extracted_text, height=200, disabled=True)
    
    # Process the text
    tabs = st.tabs(["Summary", "Flashcards", "Quiz"])
    
    with tabs[0]:
        if st.button("Generate Summary", key="gen_summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(extracted_text, max_length=summary_length)
            st.markdown("### TL;DR Summary")
            st.markdown(summary)
            
            # Download option
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
    
    with tabs[1]:
        if st.button("Generate Flashcards", key="gen_flashcards"):
            with st.spinner("Creating flashcards..."):
                flashcards = generate_flashcards(extracted_text, num_cards=num_flashcards)
            
            st.markdown("### Study Flashcards")
            
            if isinstance(flashcards, list):
                # Display flashcards in an interactive way
                for i, card in enumerate(flashcards):
                    if isinstance(card, dict) and 'question' in card and 'answer' in card:
                        with st.expander(f"Card {i+1}: {card['question']}"):
                            st.write(card['answer'])
                
                # Create downloadable CSV
                if flashcards and isinstance(flashcards[0], dict):
                    df = pd.DataFrame(flashcards)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Flashcards (CSV)",
                        data=csv,
                        file_name="flashcards.csv",
                        mime="text/csv"
                    )
            else:
                st.write(flashcards)
    
    with tabs[2]:
        if st.button("Generate Quiz", key="gen_quiz"):
            with st.spinner("Creating quiz questions..."):
                quiz = generate_quiz(extracted_text, num_questions=num_quiz_questions)
            
            st.markdown("### Knowledge Quiz")
            
            if isinstance(quiz, list):
                # Create an interactive quiz
                for i, question in enumerate(quiz):
                    if isinstance(question, dict) and 'question' in question and 'options' in question:
                        st.write(f"**Question {i+1}:** {question['question']}")
                        
                        # Get user answer
                        user_answer = st.radio(
                            f"Select your answer for question {i+1}:",
                            question['options'],
                            key=f"q{i}"
                        )
                        
                        # Check if correct
                        if 'correct_answer' in question:
                            correct_idx = question['correct_answer']
                            correct_answer = question['options'][correct_idx]
                            
                            if user_answer == correct_answer:
                                st.success("Correct! ✅")
                            else:
                                st.error(f"Incorrect. The correct answer is: {correct_answer}")
                        
                        st.markdown("---")
            else:
                st.write(quiz)

else:
    st.info("Please upload a file or enter text to get started.")
