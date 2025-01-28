import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import List

# Import your backend module
import wiki_backend

# --------------------------------------------------------------------------
# Additional dictionary for performance/accuracy tradeoff
# --------------------------------------------------------------------------
PERFORMANCE_SETTINGS = {
    "fast": {
        "chunk_size": 400,
        "chunk_overlap": 30,
        "num_predict": 128,  # Shorter generation window for speed
    },
    "balanced": {
        "chunk_size": 600,
        "chunk_overlap": 50,
        "num_predict": 256,  # Medium generation length
    },
    "accurate": {
        "chunk_size": 800,
        "chunk_overlap": 80,
        "num_predict": 512,  # Longer generation window for higher accuracy
    },
}


class WikipediaRAGApp:
    """
    A tkinter-based GUI for demonstrating
    Wikipedia Retrieval-Augmented Generation (RAG).
    """
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the main application window and layout.
        """
        self.root = root
        self.root.title("Wikipedia RAG Demo")

        # Main container frame (single column layout).
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Frame for search options and results on top
        self.top_frame = ttk.Frame(self.main_frame, padding="5")
        self.top_frame.pack(fill="x", expand=True, side="top")

        # Frame for user Q&A
        self.qa_frame = ttk.Frame(self.main_frame, padding="10")
        self.chain = None  # Will hold the RetrievalQA chain once built

        # ------------------------------
        # Search Options
        # ------------------------------
        ttk.Label(self.top_frame, text="Language:").pack(anchor="w")
        self.language_var = tk.StringVar(value="en")
        self.language_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.language_var, state="readonly", width=5
        )
        self.language_dropdown["values"] = ("en", "es", "fr", "de", "it", "pt")
        self.language_dropdown.pack(padx=(0, 0), pady=(0, 10))

        ttk.Label(self.top_frame, text="Performance:").pack(anchor="w")
        self.performance_var = tk.StringVar(value="balanced")
        self.performance_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.performance_var, state="readonly", width=10
        )
        self.performance_dropdown["values"] = ("fast", "balanced", "accurate")
        self.performance_dropdown.pack(padx=(0, 0), pady=(0, 10))

        ttk.Label(self.top_frame, text="LLM Model:").pack(anchor="w")
        self.llm_model_var = tk.StringVar(value="default")
        self.llm_model_dropdown = ttk.Combobox(
            self.top_frame, textvariable=self.llm_model_var, state="readonly", width=20
        )
        self.llm_model_dropdown["values"] = wiki_backend.get_ollama_models()
        self.llm_model_dropdown.pack(padx=(0, 0), pady=(0, 10))

        ttk.Label(self.top_frame, text="Enter your query:").pack(anchor="w")
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(self.top_frame, textvariable=self.query_var, width=40)
        self.query_entry.pack(fill="x", pady=(0, 10))

        # Search Button
        self.search_button = ttk.Button(
            self.top_frame, text="Search", command=self.on_search
        )
        self.search_button.pack()

        # Loading bar (indeterminate progress)
        self.progress_bar = ttk.Progressbar(self.main_frame, mode="indeterminate")

        # Frame for listing top results
        self.results_frame = ttk.Frame(self.main_frame, padding="10")
        self.results_listbox: tk.Listbox | None = None

    def on_search(self) -> None:
        """
        Called when the user clicks the 'Search' button.
        Retrieves the top Wikipedia results in a separate thread.
        """
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query.")
            return

        # Remove old results UI
        self.clear_frame(self.results_frame)
        self.results_frame.pack_forget()

        # Remove any old QA widgets
        self.clear_frame(self.qa_frame)
        self.qa_frame.pack_forget()

        # Start progress bar
        self.progress_bar.pack(fill="x", padx=10, pady=(10, 0))
        self.progress_bar.start()

        # Run search in a separate thread
        thread = threading.Thread(target=self.fetch_search_results, args=(query,))
        thread.start()

    def fetch_search_results(self, query: str) -> None:
        """
        Worker thread to retrieve top Wikipedia results
        and then update the GUI on the main thread.
        """
        results = wiki_backend.get_top_wikipedia_results(
            query, num_results=10, language=self.language_var.get()
        )
        self.root.after(0, self.show_search_results, results)

    def show_search_results(self, results: List[dict]) -> None:
        """
        Displays the top results in a Listbox, allowing the user
        to confirm which one is correct.
        """
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        self.results_frame.pack(fill="x", expand=True)

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
        User selects the page from the list, fetches HTML,
        and prepares the RAG pipeline.
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

        # Show progress bar while building pipeline
        self.progress_bar.pack(fill="x", padx=10, pady=(10, 0))
        self.progress_bar.start()

        thread = threading.Thread(target=self.build_rag_pipeline, args=(chosen_url,))
        thread.start()

    def build_rag_pipeline(self, url: str) -> None:
        """
        Worker thread to fetch the chosen Wikipedia page and
        build the QA chain with chosen performance settings.
        """
        html_content = wiki_backend.get_wikipedia_html_by_url(url)
        if not html_content:
            self.root.after(
                0, lambda: messagebox.showerror("Error", "Failed to retrieve Wikipedia page.")
            )
            self.root.after(0, self.stop_progress_bar)
            return

        # Gather the performance settings
        perf_setting = PERFORMANCE_SETTINGS[self.performance_var.get()]
        chunk_size = perf_setting["chunk_size"]
        chunk_overlap = perf_setting["chunk_overlap"]
        num_predict = perf_setting["num_predict"]

        # Build QA chain
        self.chain = wiki_backend.setup_langchain(
            base_llm_name=self.llm_model_var.get(),
            text=html_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_predict=num_predict
        )

        self.root.after(0, self.show_qa_interface)

    def show_qa_interface(self) -> None:
        """
        Once the pipeline is ready, display the Q&A interface.
        """
        self.stop_progress_bar()

        self.qa_frame.pack(fill="both", expand=True)

        ttk.Label(self.qa_frame, text="Ask a question about the chosen page:").pack(anchor="w")
        self.user_query_var = tk.StringVar()
        user_query_entry = ttk.Entry(self.qa_frame, textvariable=self.user_query_var, width=50)
        user_query_entry.pack(side="left", padx=(0, 10))

        ask_button = ttk.Button(self.qa_frame, text="Ask", command=self.on_ask_question)
        ask_button.pack(side="left")

        # Frame to display the answer
        self.answer_frame = ttk.Frame(self.qa_frame, padding="10")
        self.answer_frame.pack(fill="both", expand=True)

    def on_ask_question(self) -> None:
        """
        Send the user’s question to the RAG pipeline for an answer.
        """
        if not self.chain:
            messagebox.showerror("Error", "QA chain not initialized.")
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
        Worker thread: run the RAG query and collect answer & sources.
        """
        try:
            result = self.chain({"query": question})
            answer = result.get("result", "No answer generated.")
            sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        except Exception as e:
            answer = f"Error: {e}"
            sources = []

        # Show the result on the main thread
        self.root.after(0, lambda: self.show_answer(answer, sources))

    def show_answer(self, answer: str, sources: List[str]) -> None:
        """
        Display the final answer and allow copying. Also show chunk-based sources.
        """
        self.stop_progress_bar()

        # Clear old answer widgets
        self.clear_frame(self.answer_frame)

        ttk.Label(self.answer_frame, text="Answer:").pack(anchor="w")
        answer_label = ttk.Label(self.answer_frame, text=answer, wraplength=600, justify="left")
        answer_label.pack(anchor="w", pady=(0, 5))

        # "Copy Answer" button
        copy_button = ttk.Button(
            self.answer_frame,
            text="Copy Answer",
            command=lambda: self.copy_to_clipboard(answer)
        )
        copy_button.pack(anchor="w", pady=(0, 5))

        if sources:
            ttk.Label(self.answer_frame, text="Sources:").pack(anchor="w")
            for src in sources:
                ttk.Label(self.answer_frame, text=f"• {src}").pack(anchor="w")

    def copy_to_clipboard(self, text: str) -> None:
        """
        Copy the given text to the system clipboard.
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
        Helper method to remove all children from a given frame.
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
