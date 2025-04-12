# DataChat: Conversational Data Analysis with Ollama
DataChat is an interactive web application that lets you analyze and explore your datasets using natural language. Simply upload your CSV or Excel file, and start asking questions about your data in plain English. DataChat leverages the power of Ollama (gemma:2b) for language understanding and LangChain for seamless integration with data analysis tools.

[![Ollama](https://img.shields.io/badge/Ollama-gemma%3A2b-blueviolet)](https://ollama.ai/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-blue)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-green)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

*   **Intuitive Chat Interface:** Ask questions about your data in a conversational manner.
*   **Support for CSV and Excel Files:** Easily upload your data from common file formats.
*   **Powerful Data Analysis:**  Get answers to questions about trends, statistics, distributions, and relationships within your data.
*   **Seamless Integration:** Leverages LangChain to connect Ollama with data analysis tools like Pandas.

## Getting Started

1.  **Prerequisites:**
    *   Python 3.12
    *   Ollama installed and running (see: [https://ollama.ai/](https://ollama.ai/))

2.  **Installation:**

    ```bash
    git clone [https://github.com/WizKnight/DataChat]
    cd DataChat
    pip install -r requirements.txt
    ```

3.  **Running the App:**

    ```bash
    streamlit run src/main.py
    ```

    This will open DataChat in your web browser.

## Usage

1.  **Upload Data:** Click "Select a file..." and choose your CSV or Excel file.
2.  **Start Chatting:** Type your questions about the data in the chat input box.
3.  **Get Answers:** DataChat will process your question and display the answer in the chat.

## Examples

*   "What is the average age of customers?"
*   "How many sales were made in each region?"
*   "Which product category has the highest revenue?"
*   "Show me a histogram of customer ages."

## Customization

*   **LLM:**  Experiment with different Ollama models or versions for improved performance.
*   **Data Analysis Tools:**  Extend DataChat by adding custom LangChain tools to integrate with other analysis libraries or APIs.
*   **UI/UX:** Customize the Streamlit interface to match your preferences.

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgements

*   This project is inspired by the growing potential of large language models and their application to data analysis.
*   I thank the developers of Ollama, LangChain, and Streamlit for their excellent tools and resources.
