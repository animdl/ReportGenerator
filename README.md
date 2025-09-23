
# ReportGenerator
Agentic RAG Prototype

## Setup Instructions
### Clone/Download Project
``` bash
git clone <repo-url>
cd <repo-name>
```
### Python Environment Setup
``` bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt 
```
### Ollama Setup
``` bash
# Install required models
ollama pull llama3.2:3b
ollama pull mxbai-embed-large:335m

# Start Ollama server (keep this running in a separate terminal)
ollama serve
```
### Environment Config
Create a '.env' file in the project root.

## Usage
### Populate Vector Database
First, process the PDFs and create the chroma vector database
``` bash
python populate_db.py
```
### Query the Agent
Basic Query:
``` bash
python query.py "input-query-here"
```
Advanced Query:
``` bash
# Retrieve more chunks for complex questions
python query.py "input-query-here" --k 10

# Use similarity threshold filtering
python query.py "input-query-here" --score-threshold 0.5
```
