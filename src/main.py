import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import tempfile

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def load_data(uploaded_file) -> pd.DataFrame:
    """Load data from various file formats and convert to pandas DataFrame."""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension == '.csv':
        return pd.read_csv(uploaded_file)
    elif file_extension == '.xlsx':
        return pd.read_excel(uploaded_file)
    elif file_extension == '.json':
        return pd.read_json(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, XLSX, or JSON file.")

def get_basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform basic exploratory data analysis on the dataset."""
    analysis = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': df.columns.tolist(),
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'sample_records': df.head().to_dict('records'),
        'numeric_summary': {}
    }
    
    # Calculate statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        analysis['numeric_summary'][col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    
    return analysis

def generate_content_with_gemini(prompt: str) -> str:
    """Generate content using Gemini 2.0 Flash API directly."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [{
            "parts":[{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        return "No response generated"
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return str(e)

def generate_suggested_questions(df: pd.DataFrame) -> List[str]:
    """Generate relevant questions based on the dataset structure."""
    columns = df.columns.tolist()
    
    prompt = f"Given a dataset with columns: {', '.join(columns)}, "
    prompt += "suggest 3-5 natural language questions that a non-technical person might want to ask about this data. "
    prompt += "Make questions simple and conversational. Return each question on a new line."
    
    try:
        response = generate_content_with_gemini(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:5]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return ["What is the total number of records?",
                "What are the main trends in this data?",
                "Can you show me a summary of the data?"]

def process_question(df: pd.DataFrame, question: str) -> str:
    """Process a question about the data."""
    try:
        # Prepare context about the data
        data_context = f"Given this data with columns {', '.join(df.columns.tolist())}\n"
        data_context += f"First few rows:\n{df.head().to_string()}\n"
        
        # Combine context with the question
        prompt = data_context + f"\nQuestion: {question}\nProvide a clear, conversational answer."
        
        # Get response from Gemini
        response = generate_content_with_gemini(prompt)
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}"

def main():
    st.title("🤖 Interactive Data Analytics Assistant")
    st.write("Upload your data file and start analyzing it instantly!")

    uploaded_file = st.file_uploader("Choose a file (CSV, XLSX, or JSON)", 
                                   type=['csv', 'xlsx', 'json'])

    if uploaded_file is not None:
        try:
            # Load the data
            df = load_data(uploaded_file)
            
            # Store the dataframe in session state
            st.session_state['df'] = df
            
            # Perform basic analysis
            analysis = get_basic_analysis(df)
            
            # Display basic information
            st.subheader("📊 Dataset Overview")
            st.write(f"Number of rows: {analysis['row_count']}")
            st.write(f"Number of columns: {analysis['column_count']}")
            
            # Display sample data
            st.subheader("👀 Sample Data")
            st.dataframe(df.head())
            
            # Display column information
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Data Type': analysis['data_types'],
                'Missing Values': analysis['missing_values']
            })
            st.dataframe(col_info)
            
            # Display numeric summary if available
            if analysis['numeric_summary']:
                st.subheader("📈 Numeric Column Statistics")
                for col, stats in analysis['numeric_summary'].items():
                    st.write(f"**{col}**")
                    st.write(f"Mean: {stats['mean']:.2f}")
                    st.write(f"Median: {stats['median']:.2f}")
                    st.write(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")
            
            # Generate and display suggested questions
            st.subheader("❓ Suggested Questions")
            if 'suggested_questions' not in st.session_state:
                st.session_state['suggested_questions'] = generate_suggested_questions(df)
            
            # Display questions as buttons
            for i, question in enumerate(st.session_state['suggested_questions'], 1):
                if st.button(f"Q{i}: {question}", key=f"q_{i}"):
                    with st.spinner('Analyzing your question...'):
                        response = process_question(df, question)
                        st.write("Answer:", response)
            
            # Custom question input
            st.subheader("🔍 Ask Your Own Question")
            custom_question = st.text_input("What would you like to know about your data?")
            
            if custom_question:
                with st.spinner('Analyzing your question...'):
                    response = process_question(df, custom_question)
                    st.write("Answer:", response)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()