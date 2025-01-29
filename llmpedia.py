import requests
from bs4 import BeautifulSoup, Comment
import logging
import ollama
import subprocess
from typing import Optional, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from pydantic import Field
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

import tkinter as tk
from tkinter import ttk, messagebox
import threading

##############################################################################
#                           BACKEND (Integrated)                             #
##############################################################################

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# -----------------------------
# Ollama Utility Functions
# -----------------------------
def get_ollama_models() -> List[str]:
    """
    Get a list of available Ollama models.
    """
    try:
        command = ['ollama', 'list']
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return [name.split(" ")[0] for name in result.stdout.strip().split("\n")]
    except subprocess.CalledProcessError as e:
        logging.error(f"Ollama failed: {e.stderr}")
        return []

# -----------------------------
# Custom Ollama LLM
# -----------------------------
class LlamaLLM(LLM):
    model_name: str = Field(default="wikillama")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Process a prompt using the specified Ollama model.
        """
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
        """
        Initialize the sentence transformer embeddings model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        """
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query text.
        """
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# -----------------------------
# Wikipedia Utilities
# -----------------------------
def clean_html(html_content: str) -> str:
    """
    Clean raw HTML by removing scripts, styles, headers, footers, and comments.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in ['script', 'style', 'header', 'footer']:
        for element in soup.find_all(tag):
            element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    return soup.get_text(separator=' ', strip=True)

def extract_sources(html_content: str) -> List[str]:
    """
    Extract external sources from the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    sources = soup.find_all('a', {'class': 'external text'})
    logging.info(f"Found {len(sources)} sources.")
    return [source['href'] for source in sources if source.get('href')]

def save_html_with_images(html_content: str, file_name: str) -> None:
    """
    Save raw HTML content with images for later rendering in the GUI.
    """
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(html_content)

def get_top_wikipedia_results(search_query: str, language: str = 'en', num_results: int = 5) -> List[Dict[str, str]]:
    """
    Get top search results from Wikipedia for a query.
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

def get_wikipedia_html_by_url(page_url: str, save_file: str = "output_with_images.html") -> Optional[str]:
    """
    Fetch raw HTML from a Wikipedia page URL, save to 'output_with_images.html',
    and return cleaned text for retrieval QA.
    """
    try:
        page_response = requests.get(page_url, timeout=10)
        page_response.raise_for_status()
        # Save the full HTML to 'output_with_images.html'
        save_html_with_images(page_response.text, save_file)
        # Return the cleaned text for embeddings
        return clean_html(page_response.text)
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve Wikipedia page: {e}")
        return None

# -----------------------------
# RAG Setup
# -----------------------------
def setup_langchain(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    num_predict: int,
    base_llm_name: str
) -> RetrievalQA:
    """
    Create a Retrieval-Augmented Generation pipeline using LangChain.
    """
    ollama.create(
        model="wikillama",
        from_=base_llm_name,
        parameters={"num_predict": num_predict, "temperature": 0.1, "repeat_last_n": 2},
        stream=True
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    docs = [Document(page_content=chunk, metadata={"source": f"Chunk-{i}"}) for i, chunk in enumerate(chunks)]

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


##############################################################################
#              FRONTEND (Removed Preview, Q&A Panel on the Right)            #
##############################################################################

PERFORMANCE_SETTINGS = {
    "fast": {
        "chunk_size": 300,
        "chunk_overlap": 20,
        "num_predict": 128,
    },
    "balanced": {
        "chunk_size": 600,
        "chunk_overlap": 50,
        "num_predict": 256,
    },
    "accurate": {
        "chunk_size": 800,
        "chunk_overlap": 80,
        "num_predict": 4096,
    },
}

class WikipediaRAGApp:
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the main application window and layout.
        """
        self.root = root
        self.root.title("Wikipedia RAG Demo")

        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Frames for top bar, left side (search results), right side (Q&A)
        self.setup_top_frame()
        self.results_frame = ttk.Frame(self.main_frame, padding="10")
        self.results_frame.pack(side="left", fill="y", expand=False)

        # Right side used for Q&A
        self.answer_frame = ttk.Frame(self.main_frame, padding="10")
        self.answer_frame.pack(side="right", fill="both", expand=True)

        # Keep references for dynamic updates
        self.results_listbox = None
        self.progress_bar = ttk.Progressbar(self.main_frame, mode="indeterminate")
        self.chain = None

        # We'll create Q&A widgets later
        self.setup_qa_widgets()

    def setup_top_frame(self) -> None:
        """
        Setup the top frame with search options and controls.
        """
        self.top_frame = ttk.Frame(self.main_frame, padding="5")
        self.top_frame.pack(fill="x", expand=False, side="top")

        # Language
        ttk.Label(self.top_frame, text="Language:").pack(anchor="w")
        self.language_var = tk.StringVar(value="en")
        self.language_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.language_var, state="readonly", width=5
        )
        self.language_dropdown["values"] = ("en", "es", "fr", "de", "it", "pt")
        self.language_dropdown.pack(padx=(0, 0), pady=(0, 10))

        # Performance
        ttk.Label(self.top_frame, text="Performance:").pack(anchor="w")
        self.performance_var = tk.StringVar(value="balanced")
        self.performance_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.performance_var, state="readonly", width=10
        )
        self.performance_dropdown["values"] = ("fast", "balanced", "accurate")
        self.performance_dropdown.pack(padx=(0, 0), pady=(0, 10))

        # LLM model
        ttk.Label(self.top_frame, text="LLM Model:").pack(anchor="w")
        self.llm_model_var = tk.StringVar(value="default")
        self.llm_model_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.llm_model_var, state="readonly", width=20
        )
        self.llm_model_dropdown["values"] = get_ollama_models()
        self.llm_model_dropdown.pack(padx=(0, 0), pady=(0, 10))

        # Query
        ttk.Label(self.top_frame, text="Enter your query:").pack(anchor="w")
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(self.top_frame, textvariable=self.query_var, width=40)
        self.query_entry.pack(fill="x", pady=(0, 10))

        # Search button
        self.search_button = ttk.Button(
            self.top_frame, text="Search", command=self.on_search
        )
        self.search_button.pack()

    def setup_qa_widgets(self) -> None:
        """
        Create the Q&A interface on the right side.
        """
        # Label prompting the user
        ttk.Label(self.answer_frame, text="Ask a question about the chosen page:").pack(anchor="w")

        # User question input
        self.user_query_var = tk.StringVar()
        user_query_entry = ttk.Entry(self.answer_frame, textvariable=self.user_query_var, width=50)
        user_query_entry.pack(side="left", padx=(0, 10))

        # "Ask" button
        ask_button = ttk.Button(self.answer_frame, text="Ask", command=self.on_ask_question)
        ask_button.pack(side="left")

        # Frame to display the answer
        self.answer_display_frame = ttk.Frame(self.answer_frame, padding="10")
        self.answer_display_frame.pack(fill="both", expand=True)

    def on_search(self) -> None:
        """
        Called when the user clicks 'Search'.
        """
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query.")
            return

        self.clear_frame(self.results_frame)
        self.results_frame.pack(side="left", fill="y", expand=False)

        # Clear previous Q&A answer
        self.clear_frame(self.answer_display_frame)

        self.progress_bar.pack(fill="x", padx=10, pady=(10, 0))
        self.progress_bar.start()

        thread = threading.Thread(target=self.fetch_search_results, args=(query,))
        thread.start()

    def fetch_search_results(self, query: str) -> None:
        """
        Retrieve top Wikipedia results in a thread, then update the GUI.
        """
        results = get_top_wikipedia_results(
            search_query=query,
            language=self.language_var.get(),
            num_results=10
        )
        self.root.after(0, lambda: self.show_search_results(results))

    def show_search_results(self, results: List[dict]) -> None:
        """
        Display top search results in the left-side Listbox.
        """
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        if not results:
            ttk.Label(self.results_frame, text="No results found.").pack()
            return

        ttk.Label(self.results_frame, text="Select the correct Wikipedia page:").pack(anchor="w")

        self.results_listbox = tk.Listbox(
            self.results_frame, height=8, width=50, selectmode=tk.SINGLE
        )
        self.results_listbox.pack(pady=(5, 5))

        self.search_results = results
        for i, r in enumerate(results):
            self.results_listbox.insert(i, r["title"])

        select_button = ttk.Button(
            self.results_frame, text="Confirm Selection", command=self.on_select_page
        )
        select_button.pack(pady=(5, 0))

    def on_select_page(self) -> None:
        """
        Fetch the selected Wikipedia page and build the QA chain.
        """
        if not self.results_listbox:
            return

        selection = self.results_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a page from the list.")
            return

        index = selection[0]
        chosen_result = self.search_results[index]
        chosen_url = chosen_result["url"]

        self.progress_bar.pack(fill="x", padx=10, pady=(10, 0))
        self.progress_bar.start()

        thread = threading.Thread(target=self.build_rag_pipeline, args=(chosen_url,))
        thread.start()

    def build_rag_pipeline(self, url: str) -> None:
        """
        Fetch the page content, build the retrieval QA chain.
        """
        cleaned_text = get_wikipedia_html_by_url(url, save_file="output_with_images.html")
        if not cleaned_text:
            self.root.after(
                0, lambda: messagebox.showerror("Error", "Failed to retrieve Wikipedia page.")
            )
            self.root.after(0, self.stop_progress_bar)
            return

        # Build the retrieval QA chain from the cleaned text
        perf_setting = PERFORMANCE_SETTINGS[self.performance_var.get()]
        chunk_size = perf_setting["chunk_size"]
        chunk_overlap = perf_setting["chunk_overlap"]
        num_predict = perf_setting["num_predict"]

        self.chain = setup_langchain(
            base_llm_name=self.llm_model_var.get(),
            text=cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_predict=num_predict
        )

        self.root.after(0, self.stop_progress_bar)

    def on_ask_question(self) -> None:
        """
        Retrieve an answer to the user's question from the QA chain.
        """
        if not self.chain:
            messagebox.showerror("Error", "QA chain not initialized. Select a page first.")
            return

        question = self.user_query_var.get().strip()
        if not question:
            return

        self.progress_bar.pack(fill="x", padx=10, pady=(10, 0))
        self.progress_bar.start()

        thread = threading.Thread(target=self.run_rag_query, args=(question,))
        thread.start()

    def run_rag_query(self, question: str) -> None:
        """
        Run the QA query on the chain.
        """
        try:
            result = self.chain({"query": question})
            answer = result.get("result", "No answer generated.")
            sources = result.get("source_documents", [])
        except Exception as e:
            answer = f"Error: {e}"
            sources = []

        self.root.after(0, lambda: self.show_answer(answer, sources))

    def show_answer(self, answer: str, sources: List[Document]) -> None:
        """
        Display the answer in the right-side frame.
        """
        self.stop_progress_bar()
        self.clear_frame(self.answer_display_frame)

        ttk.Label(self.answer_display_frame, text="Answer:").pack(anchor="w")
        answer_text = tk.Text(self.answer_display_frame, wrap="word", height=10, width=70)
        answer_text.insert("end", answer)
        answer_text.config(state="disabled")
        answer_text.pack(pady=(5, 5))

        copy_button = ttk.Button(
            self.answer_display_frame,
            text="Copy Answer",
            command=lambda: self.copy_to_clipboard(answer)
        )
        copy_button.pack(anchor="w", pady=(0, 5))

        if sources:
            ttk.Label(self.answer_display_frame, text="Sources:").pack(anchor="w")
            for doc in sources:
                doc_src = doc.metadata.get("source", "Unknown")
                ttk.Label(self.answer_display_frame, text=f"\u2022 {doc_src}").pack(anchor="w")

    def copy_to_clipboard(self, text: str) -> None:
        """
        Copy the text to the clipboard.
        """
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "Answer copied to clipboard!")

    def stop_progress_bar(self) -> None:
        """
        Stop and hide the progress bar.
        """
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

    @staticmethod
    def clear_frame(frame: ttk.Frame) -> None:
        """
        Clear all widgets from a frame.
        """
        for widget in frame.winfo_children():
            widget.destroy()

def main() -> None:
    """
    Main entry point for the Wikipedia RAG GUI application.
    """
    root = tk.Tk()
    app = WikipediaRAGApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
