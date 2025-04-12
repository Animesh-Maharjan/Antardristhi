import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
import tempfile

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

def generate_suggested_questions(df: pd.DataFrame) -> List[str]:
    """Generate relevant questions based on the dataset structure."""
    questions = []
    columns = df.columns.tolist()
    
    # Create a prompt for the LLM to generate questions
    llm = Ollama(model="llama3.2")
    context = f"Given a dataset with columns: {', '.join(columns)}, "
    # context += "suggest 3-5 natural language questions that a non-technical person might want to ask about this data. "
    context += "Make questions simple and conversational."
    
    response = llm.predict(context)
    questions = [q.strip() for q in response.split('\n') if q.strip()]
    return questions[:5]  # Return up to 5 questions

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
            suggested_questions = generate_suggested_questions(df)
            
            for i, question in enumerate(suggested_questions, 1):
                if st.button(f"Q{i}: {question}"):
                    # Create Pandas DataFrame Agent
                    agent = create_pandas_dataframe_agent(
                        llm=Ollama(model="llama2"),
                        df=df,
                        verbose=True,
                        allow_dangerous_code=True
                    )
                    response = agent.run(question)
                    st.write("Answer:", response)
            
            # Custom question input
            st.subheader("🔍 Ask Your Own Question")
            custom_question = st.text_input("What would you like to know about your data?")
            
            if custom_question:
                agent = create_pandas_dataframe_agent(
                    llm=Ollama(model="llama2"),
                    df=df,
                    verbose=True,
                    allow_dangerous_code=True
                )
                response = agent.run(custom_question)
                st.write("Answer:", response)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()