# wiki_backend.py

import requests
from bs4 import BeautifulSoup, Comment
import logging
import ollama
import subprocess
from typing import Optional, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from pydantic import Field
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Ollama Model Initialization
# -----------------------------
ollama.create(
    model="wikillama1B",
    from_="llama3.2:1B",
    parameters={"num_predict": 256, "temperature": 0.1, "repeat_last_n": 3},
    stream=False
)

# -----------------------------
# Custom Ollama LLM
# -----------------------------
class LlamaLLM(LLM):
    model_name: str = Field(default="wikillama1B")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            command = ['ollama', 'run', self.model_name, prompt]
            result = subprocess.run(
                command, capture_output=True, text=True, check=True, encoding='utf-8'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama failed: {e.stderr}")
            return "Error processing the request."

    @property
    def _llm_type(self) -> str:
        return "ollama"


# -----------------------------
# Local Embeddings
# -----------------------------
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# -----------------------------
# Utility Functions
# -----------------------------
def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in ['script', 'style', 'header', 'footer']:
        for element in soup.find_all(tag):
            element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup.get_text(separator=' ', strip=True)


def get_top_wikipedia_results(search_query: str, language: str = 'en', num_results: int = 5):
    """
    Returns the top N (num_results) search results from Wikipedia for a given query.
    Each item is a dict with 'title' and 'url'.
    """
    base_url = f"https://{language}.wikipedia.org"
    search_url = f"{base_url}/w/index.php"
    params = {
        'search': search_query,
        'title': 'Special:Search',
        'fulltext': '1'
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve search results: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('div', class_='mw-search-result-heading', limit=num_results)
    output = []
    for r in results:
        a_tag = r.find('a')
        if a_tag:
            title = a_tag.get_text(strip=True)
            url = base_url + a_tag['href']
            output.append({'title': title, 'url': url})
    return output


def get_wikipedia_html_by_url(page_url: str) -> Optional[str]:
    """
    Fetch raw HTML from a Wikipedia page URL.
    """
    try:
        page_response = requests.get(page_url, timeout=10)
        page_response.raise_for_status()
        return clean_html(page_response.text)
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve Wikipedia page: {e}")
        return None


# -----------------------------
# RAG Setup
# -----------------------------
def setup_langchain(text: str) -> RetrievalQA:
    """
    Create a Retrieval-Augmented Generation pipeline using LangChain,
    returning a RetrievalQA chain object.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(page_content=chunk, metadata={"source": f"Chunk-{i}"}))

    embeddings = LocalSentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)

    llm = LlamaLLM(model_name="wikillama")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
