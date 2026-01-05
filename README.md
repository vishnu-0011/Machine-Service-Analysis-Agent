# 🤖 Machine Service Analysis Agent

Welcome! This tool allows you to **chat with your machine service records**. 
Instead of searching through thousands of rows in a CSV file, you can simply ask questions in plain English, and the AI will find the answer for you.

It uses two smart methods to help you:
1.  **Search (RAG):** Finds specific details about machines, problems, and dates.
2.  **Analyze (Data Analyst):** Calculates totals, averages, and trends (like "How much did we spend in 2023?").

---

## ✨ Features
- **Instant Answers:** Just ask questions like "Which machine had the most problems?"
- **Smart Analysis:** Calculates costs, hours, and counts automatically.
- **Large Data Support:** Handles thousands of records easily.
- **Privacy First:** Runs **entirely locally** on your computer using Ollama (no data leaves your machine).

---

## 🛠️ Prerequisites
Before running the agent, make sure you have:

1.  **Python 3.9+** installed.
2.  **Ollama** installed and running.
    *   [Download Ollama here](https://ollama.com/download).
    *   This powers the AI brains of the project.

---

## 🚀 Installation & Setup

### 1. Set up the AI Models
Open your terminal or command prompt and run these commands to download the necessary AI models:
```bash
ollama serve                    # Start the Ollama server (in a separate terminal)
ollama pull mxbai-embed-large   # For understanding your data
ollama pull qwen3:4b            # For answering your questions
```

### 2. Install Python Packages
Install the required libraries to let Python talk to the AI:
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
Make sure your data file `machine_service.csv` is in the same folder as the script.

---

## 💡 How to Use

1.  **Start the Agent:**
    Run the main script in your terminal:
    ```bash
    python main.py
    ```

2.  **Ask Questions:**
    The system will load your data (this might take a moment the first time). Once ready, just type your question!

    **Examples:**
    *   "How many status updates were 'Completed'?"
    *   "What is the total cost for all repairs?"
    *   "Tell me about the problems with Machine ID 102."
    *   "Which problem type is the most frequent?"

3.  **Quit:**
    Type `q`, `quit`, or `exit` to stop the program.

---

## 🔧 Troubleshooting
*   **"Connection refused" error?** Make sure Ollama is running (`ollama serve`).
*   **First run is slow?** The first time you run it, the system builds a "search index" of your data. This can take a few minutes for large files, but subsequent runs will be instant!
*   **Want to reset?** If you change your CSV data, delete the `chrome_langchain_db` folder to let the system rebuild the index.

---

**Enjoy analyzing your data!**
