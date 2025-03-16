# LangChain Chatbot with Groq + BERT Embeddings 🤖📄

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DocQuery** is an interactive chatbot that lets you ask questions about a document. It combines LangChain, Groq's language model, Hugging Face's BERT embeddings, and FAISS for efficient document retrieval—all wrapped in a sleek Gradio interface. 🚀

---

## Table of Contents 📚

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [License](#license)
- [Contact](#contact)

---

## Overview

**DocQuery** allows users to interactively query the content of a text document. It works by:

- **Processing the Document:** Loads and splits a text file into manageable chunks. ✂️
- **Embedding & Indexing:** Converts text chunks into numerical embeddings using a BERT model and indexes them with FAISS for similarity search. 🔍
- **Retrieval & Response Generation:** Retrieves the most relevant chunks for a query and generates answers using a Groq-based language model. 💡
- **Interactive Interface:** Provides an intuitive chat interface with Gradio. 💬

---

## Features

- **Document Processing:** Efficiently loads and splits documents for improved query handling. 📄
- **Advanced Embeddings:** Uses Hugging Face’s pre-trained BERT model for high-quality text representation. 🧠
- **Efficient Search:** Implements FAISS for rapid similarity search over document embeddings. ⚡
- **Smart Answering:** Leverages a Groq-based LLM to generate insightful responses. 🤖
- **User-Friendly UI:** Offers an engaging chat experience via a modern Gradio interface. 👩‍💻
- **Robust Error Handling:** Gracefully manages errors during query processing. 🚫

---

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.7+** 🐍
- **Groq API Key:** Set as an environment variable.
- **Required Python Packages:**
  - `gradio`
  - `langchain`
  - `langchain_groq`
  - `langchain_huggingface`
  - `langchain_community`
  - `faiss-cpu` (or `faiss-gpu` if applicable)

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. **Install Dependencies:**
   ```bash
   pip install gradio langchain langchain_groq langchain_huggingface langchain_community faiss-cpu

3. **Set Your Groq API Key:**

   Export your Groq API key in the environment:
  ```bash
  export GROQ_API_KEY="your_groq_api_key"
  ```

4. **Prepare Your Document:**

     Place your text document (e.g., sample.txt) in the project root directory. 📁



**Usage**

    Launch the Gradio interface by running the main script:
  ```bash
  python your_script.py
  ```
    Then, open the provided local URL in your browser to interact with the chatbot and ask questions          about   your document. 💬
