import requests
from bs4 import BeautifulSoup, Comment
import logging
import ollama
from typing import Optional, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from sentence_transformers import SentenceTransformer
import subprocess
from pydantic import Field
from langchain.docstore.document import Document


# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Create/Download Ollama Model
# (Adjust model & parameters to your hardware needs)
# -----------------------------
ollama.create(
    model="wikillama1B",
    from_="llama3.2:1B",
    parameters={
        "num_predict": 128,   # Fewer tokens to reduce load
        "temperature": 0.1,
        "repeat_last_n": 3
    },
    stream=False
)


# -----------------------------
# Custom Llama LLM Wrapper
# -----------------------------
class LlamaLLM(LLM):
    """
    Custom wrapper for the Llama model using Ollama.
    """
    model_name: str = Field(default="wikillama1B")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Sends the prompt to the Ollama CLI and retrieves the model's response.
        """
        try:
            # Use the prompt as a positional argument
            command = ['ollama', 'run', self.model_name, prompt]
            # Run the command with UTF-8 encoding for the output
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True, 
                encoding='utf-8'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama failed: {e.stderr}")
            return "Error processing the request."

    @property
    def _llm_type(self) -> str:
        """
        Identifies the type of LLM being used.
        """
        return "ollama"


# -----------------------------
# Local Embeddings with SentenceTransformers
# -----------------------------
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# -----------------------------
# Wikipedia HTML Retrieval and Cleaning
# -----------------------------
def get_wikipedia_html(search_query: str, language: str = 'en') -> Optional[str]:
    """
    Fetch Wikipedia page HTML for the given search query.
    """
    search_url = f"https://{language}.wikipedia.org/w/index.php"
    params = {'search': search_query, 'title': 'Special:Search', 'fulltext': '1'}

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve search results: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    first_result = soup.find('div', {'class': 'mw-search-result-heading'})
    if not first_result:
        logging.warning("No search results found.")
        return None

    page_url = f"https://{language}.wikipedia.org" + first_result.find('a')['href']
    try:
        page_response = requests.get(page_url, timeout=10)
        page_response.raise_for_status()
        return clean_html(page_response.text)
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve Wikipedia page: {e}")
        return None


def clean_html(html_content: str) -> str:
    """
    Clean Wikipedia HTML content by removing scripts, styles, and comments.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in ['script', 'style', 'header', 'footer']:
        for element in soup.find_all(tag):
            element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup.get_text(separator=' ', strip=True)


# -----------------------------
# LangChain Setup for RAG
# -----------------------------
def setup_langchain(text: str) -> RetrievalQA:
    """
    Create a Retrieval-Augmented Generation pipeline using LangChain.
    """
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Create Documents with metadata for source-tracking
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={"source": f"Chunk-{i}"}))

    # Initialize embeddings
    embeddings = LocalSentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    # Create a vector store
    vector_store = FAISS.from_documents(docs, embeddings)

    # Set up the LLM
    llm = LlamaLLM(model_name="wikillama1B")

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain


# -----------------------------
# Main Function for Wikipedia Queries
# -----------------------------
def main():
    search_query = input("Enter your Wikipedia search query: ").strip()
    if not search_query:
        print("No query entered. Exiting.")
        return

    # Retrieve and clean the Wikipedia content
    html_content = get_wikipedia_html(search_query, language='en')
    if not html_content:
        print("Failed to retrieve Wikipedia content.")
        return

    # Set up LangChain for RAG
    qa_chain = setup_langchain(html_content)

    # Query loop
    print("Ready to answer queries about:", search_query)
    print("Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ").strip()
        if query.lower() == 'exit':
            break

        try:
            result = qa_chain.invoke({"query": query})
            answer = result.get("result", "No answer generated.")
            source_docs = result.get("source_documents", [])

            print("\nAnswer:", answer)
            print("\nRelevant Sources:")
            for doc in source_docs:
                print(f"- {doc.metadata.get('source', 'Unknown source')}")
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            print("Unable to generate an answer at this time.")


if __name__ == "__main__":
    main()
