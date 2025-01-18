# Knowledge Agent Demo

A Python-based knowledge agent that can answer questions using web-based documentation. This implementation uses:
- Groq for LLM inference
- Cohere for embeddings
- Chroma for vector storage
- LangChain for the agent framework

## Features

- Fetches and processes content from web URLs
- Creates embeddings using Cohere's embed-english-v3.0 model
- Uses Groq's Mixtral-8x7b model for question answering
- Handles rate limiting and batch processing
- Tracks and displays source URLs for answers

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd knowledge_agent_demo
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

4. Update `sample_knowledge.txt` with your URLs:
```
https://url1.com
https://url2.com
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Extract URLs from sample_knowledge.txt
2. Fetch and process content from each URL
3. Create embeddings and build a knowledge base
4. Answer the example question using the knowledge base

## Configuration

- Adjust chunk sizes in `create_knowledge_base()` function
- Modify batch size and rate limiting in the same function
- Change the number of chunks per URL (currently limited to 10)
- Update the example question in `main()`
