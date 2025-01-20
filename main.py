import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import logging
import requests
import re
import time
from bs4 import BeautifulSoup
import gradio as gr

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_urls_from_file(file_path: str) -> list:
    """Extract URLs from a text file."""
    with open(file_path, 'r') as f:
        content = f.read()
    # Find URLs using regex pattern
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, content)


def extract_urls_from_text(text: str) -> list:
    """Extract URLs from text input."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, text)


def fetch_url_content(url: str) -> str:
    """Fetch content from a URL and extract main text."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML and extract main text
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip()
                  for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {str(e)}")
        return ""


def create_knowledge_base(urls: list):
    """Create a vector store from URL contents using Cohere embeddings."""
    # Initialize the embedding model
    embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="embed-english-v3.0"
    )

    # Text splitter for smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=50
    )

    all_texts = []

    # Fetch and process content from URLs
    for url in urls:
        content = fetch_url_content(url)
        if content:
            logger.info(f"Successfully fetched content from {url}")
            # Split into smaller chunks
            chunks = text_splitter.split_text(content)
            # Take only first 10 chunks per URL to avoid rate limits
            chunks = chunks[:10]
            all_texts.extend(chunks)
            # Add source metadata
            all_texts[-1] = f"Source: {url}\n{all_texts[-1]}"

    # Create vector store with rate limiting
    texts_with_metadata = [
        {"text": text, "metadata": {"source": urls[i % len(urls)]}}
        for i, text in enumerate(all_texts)
    ]

    # Process in smaller batches
    batch_size = 5
    all_embeddings = []
    for i in range(0, len(texts_with_metadata), batch_size):
        batch = texts_with_metadata[i:i + batch_size]
        try:
            # Create vector store
            if i == 0:
                vectorstore = Chroma.from_texts(
                    texts=[d["text"] for d in batch],
                    embedding=embeddings,
                    metadatas=[d["metadata"] for d in batch],
                    collection_name="demo_knowledge",
                    persist_directory=".praison"
                )
            else:
                vectorstore.add_texts(
                    texts=[d["text"] for d in batch],
                    metadatas=[d["metadata"] for d in batch]
                )
            logger.info(
                f"Processed batch {i//batch_size + 1}/{(len(texts_with_metadata) + batch_size - 1)//batch_size}")
            # Rate limiting
            time.sleep(2)
        except Exception as e:
            logger.error(
                f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue

    return vectorstore


def create_qa_chain(vectorstore):
    """Create a conversational QA chain using Groq."""
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"
    )

    # Create the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True
    )

    return qa_chain


def process_query(urls_text: str, question: str):
    """Process URLs and question through the knowledge agent."""
    try:
        # Extract URLs from the text input
        urls = extract_urls_from_text(urls_text)
        if not urls:
            return "Please provide at least one valid URL."

        # Create the knowledge base from URLs
        vectorstore = create_knowledge_base(urls)
        logger.info("Successfully created knowledge base from URLs")

        # Create the QA chain
        qa_chain = create_qa_chain(vectorstore)
        logger.info("Successfully created QA chain")

        # Get the answer
        chat_history = []
        result = qa_chain({"question": question, "chat_history": chat_history})
        answer = result["answer"]

        # Format sources
        sources = "\n\nSources used:"
        for doc in result["source_documents"]:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources += f"\n- {doc.metadata['source']}"

        return answer + sources

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"An error occurred: {str(e)}"


def main():
    # Create Gradio interface
    demo = gr.Interface(
        fn=process_query,
        inputs=[
            gr.Textbox(
                label="Enter URLs (one per line)",
                placeholder="https://example1.com\nhttps://example2.com",
                lines=5
            ),
            gr.Textbox(
                label="Enter your question",
                placeholder="What would you like to know?"
            )
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Knowledge Agent Demo",
        description="Add URLs to create a knowledge base, then ask questions about the content.",
        examples=[
            [
                "https://python.langchain.com/docs/langgraph\nhttps://python.langchain.com/docs/expression_language/get_started",
                "What is LangGraph and what are its core benefits compared to other LLM frameworks?"
            ]
        ]
    )

    # Launch the interface
    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
