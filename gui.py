import tkinter as tk
from tkinter import ttk, messagebox
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup, Comment
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom LLM Wrapper
class LlamaLLM:
    def __init__(self, model_name="llama3.1:8B"):
        self.model_name = model_name

    def call(self, prompt):
        try:
            command = ['ollama', 'run', self.model_name, prompt, '--max_tokens', '512']
            logging.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logging.error(f"Ollama failed: {e.stderr}")
            return "Error processing the request."

# Custom Embeddings Class
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Wikipedia Retrieval Functions
def get_wikipedia_html(search_query, language="en"):
    search_url = f"https://{language}.wikipedia.org/w/index.php"
    params = {'search': search_query, 'title': 'Special:Search', 'fulltext': '1'}
    response = requests.get(search_url, params=params)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        first_result = soup.find('div', {'class': 'mw-search-result-heading'})
        if first_result:
            page_url = f"https://{language}.wikipedia.org" + first_result.find('a')['href']
            page_response = requests.get(page_url)
            return clean_html(page_response.text)
        else:
            return "No results found on Wikipedia."
    return "Failed to retrieve Wikipedia page."

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in ['script', 'style', 'header', 'footer']:
        for element in soup.find_all(tag):
            element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup.get_text(separator=' ', strip=True)

# Setup LangChain
def setup_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = LocalSentenceTransformerEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    llm = LlamaLLM()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# GUI Application
def run_gui_app():
    def search_wikipedia():
        query = search_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a search query.")
            return

        status_label.config(text="Searching Wikipedia...")
        root.update()
        html_content = get_wikipedia_html(query)

        if "No results" in html_content or "Failed to retrieve" in html_content:
            messagebox.showerror("Error", html_content)
            status_label.config(text="Ready")
            return

        status_label.config(text="Setting up QA system...")
        qa_chain = setup_langchain(html_content)

        def ask_question():
            question = question_entry.get()
            if not question:
                messagebox.showerror("Error", "Please enter a question.")
                return

            status_label.config(text="Processing question...")
            root.update()
            try:
                result = qa_chain.invoke({"query": question})
                answer_text.set(result.get("result", "No answer generated."))
                sources_text.set("\n".join([doc.metadata.get('source', 'Unknown source') for doc in result.get("source_documents", [])]))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process question: {e}")
            finally:
                status_label.config(text="Ready")

        # Update UI for QA input
        ttk.Label(root, text="Enter your question:").pack(pady=5)
        question_entry = ttk.Entry(root, width=50)
        question_entry.pack(pady=5)
        ttk.Button(root, text="Ask", command=ask_question).pack(pady=5)
        ttk.Label(root, text="Answer:").pack(pady=5)
        ttk.Label(root, textvariable=answer_text, wraplength=400).pack(pady=5)
        ttk.Label(root, text="Sources:").pack(pady=5)
        ttk.Label(root, textvariable=sources_text, wraplength=400).pack(pady=5)

        status_label.config(text="Ready")

    root = tk.Tk()
    root.title("Wikipedia QA App")
    root.geometry("600x600")

    # Search UI
    ttk.Label(root, text="Enter Wikipedia search query:").pack(pady=10)
    search_entry = ttk.Entry(root, width=50)
    search_entry.pack(pady=10)
    ttk.Button(root, text="Search", command=search_wikipedia).pack(pady=10)

    # Status
    status_label = ttk.Label(root, text="Ready")
    status_label.pack(pady=10)

    # Variables for answer and sources
    answer_text = tk.StringVar()
    sources_text = tk.StringVar()

    root.mainloop()

if __name__ == "__main__":
    run_gui_app()
