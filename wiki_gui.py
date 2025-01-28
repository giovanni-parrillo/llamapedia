# wiki_gui.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import List

# Import your backend module
import wiki_backend

class WikipediaRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wikipedia RAG Demo")

        # Search frame
        self.search_frame = ttk.Frame(self.root, padding="10")
        self.search_frame.pack(fill="x", expand=True)

        ttk.Label(self.search_frame, text="Enter your Wikipedia search query:").pack(anchor="w")
        self.language_var = tk.StringVar(value="en")
        self.language_dropdown = ttk.Combobox(self.search_frame, textvariable=self.language_var, state="readonly")
        self.language_dropdown['values'] = ("en", "es", "fr", "de", "it", "pt")
        self.language_dropdown.pack(side="left", padx=(0, 10))
        self.query_var = tk.StringVar()
        self.query_entry = ttk.Entry(self.search_frame, textvariable=self.query_var, width=40)
        self.query_entry.pack(side="left", padx=(0, 10))
        
        self.search_button = ttk.Button(self.search_frame, text="Search", command=self.on_search)
        self.search_button.pack(side="left")

        # Loading bar (indeterminate progress)
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")
        # We'll pack/start/stop this as needed.

        # Frame for listing top results
        self.results_frame = ttk.Frame(self.root, padding="10")
        self.results_listbox = None

        # Frame for user Q&A
        self.qa_frame = ttk.Frame(self.root, padding="10")
        self.chain = None  # Will hold the RetrievalQA chain once built

    def on_search(self):
        """
        Triggered when user clicks the 'Search' button. Fetches top 5 Wikipedia results in a thread.
        """
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query.")
            return
        
        # Remove any old widgets from the results/QA frames
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.results_frame.pack_forget()

        for widget in self.qa_frame.winfo_children():
            widget.destroy()
        self.qa_frame.pack_forget()

        # Start progress bar
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.start()

        # Run search in a separate thread
        thread = threading.Thread(target=self.fetch_search_results, args=(query,))
        thread.start()

    def fetch_search_results(self, query: str):
        """
        Worker thread to retrieve top 5 Wikipedia results, then populate a Listbox in the main thread.
        """
        results = wiki_backend.get_top_wikipedia_results(query, num_results=10, language=self.language_var.get())

        # Update the UI on the main thread
        self.root.after(0, self.show_search_results, results)

    def show_search_results(self, results):
        """
        Display the top 5 results in a Listbox for user selection.
        """
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        self.results_frame.pack(fill="both", expand=True)

        if not results:
            ttk.Label(self.results_frame, text="No results found.").pack()
            return

        ttk.Label(self.results_frame, text="Select the correct Wikipedia page:").pack(anchor="w")

        self.results_listbox = tk.Listbox(self.results_frame, height=5, width=60, selectmode=tk.SINGLE)
        self.results_listbox.pack(pady=(5, 5))

        self.search_results = results  # store for reference
        for i, r in enumerate(results):
            self.results_listbox.insert(i, r["title"])

        self.select_button = ttk.Button(self.results_frame, text="Confirm Selection", command=self.on_select_page)
        self.select_button.pack(pady=(5, 0))

    def on_select_page(self):
        """
        When user confirms which Wikipedia page is correct.
        """
        selection = self.results_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a page from the list.")
            return
        
        index = selection[0]
        chosen_result = self.search_results[index]
        chosen_url = chosen_result["url"]

        # Start progress bar while we fetch page and build QA
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.start()

        # Run page setup in a separate thread
        thread = threading.Thread(target=self.build_rag_pipeline, args=(chosen_url,))
        thread.start()

    def build_rag_pipeline(self, url: str):
        """
        Fetch chosen Wikipedia page, clean it, then initialize the QA chain.
        """
        html_content = wiki_backend.get_wikipedia_html_by_url(url)
        if not html_content:
            # Update UI on main thread
            self.root.after(0, lambda: messagebox.showerror("Error", "Failed to retrieve Wikipedia page."))
            self.root.after(0, self.stop_progress_bar)
            return

        # Build QA chain
        self.chain = wiki_backend.setup_langchain(html_content)

        # Once done, show the Q&A interface on the main thread
        self.root.after(0, self.show_qa_interface)

    def show_qa_interface(self):
        """
        Displays the question entry and output area.
        """
        self.stop_progress_bar()

        self.qa_frame.pack(fill="both", expand=True)

        ttk.Label(self.qa_frame, text="Ask a question about the chosen page:").pack(anchor="w")
        self.user_query_var = tk.StringVar()
        self.user_query_entry = ttk.Entry(self.qa_frame, textvariable=self.user_query_var, width=50)
        self.user_query_entry.pack(side="left", padx=(0, 10))

        self.ask_button = ttk.Button(self.qa_frame, text="Ask", command=self.on_ask_question)
        self.ask_button.pack(side="left")

        # Frame to display the answer
        self.answer_frame = ttk.Frame(self.qa_frame, padding="10")
        self.answer_frame.pack(fill="x", expand=True)

    def on_ask_question(self):
        """
        Trigger the RAG pipeline to answer the user's query.
        """
        if not self.chain:
            messagebox.showerror("Error", "QA chain not initialized.")
            return

        question = self.user_query_var.get().strip()
        if not question:
            return

        # Show loading bar while generating answer
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.start()

        thread = threading.Thread(target=self.run_rag_query, args=(question,))
        thread.start()

    def run_rag_query(self, question: str):
        """
        Worker thread to run the RAG query and return the result.
        """
        try:
            result = self.chain({"query": question})
            answer = result.get("result", "No answer generated.")
            sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        except Exception as e:
            answer = f"Error: {e}"
            sources = []

        # Show result on main thread
        self.root.after(0, lambda: self.show_answer(answer, sources))

    def show_answer(self, answer: str, sources: List[str]):
        self.stop_progress_bar()

        # Clear old answer widgets
        for widget in self.answer_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.answer_frame, text="Answer:").pack(anchor="w")
        answer_label = ttk.Label(self.answer_frame, text=answer, wraplength=500, justify="left")
        answer_label.pack(anchor="w", pady=(0, 5))

        if sources:
            ttk.Label(self.answer_frame, text="Sources:").pack(anchor="w")
            for src in sources:
                ttk.Label(self.answer_frame, text=f"• {src}").pack(anchor="w")

    def stop_progress_bar(self):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()


def main():
    root = tk.Tk()
    app = WikipediaRAGApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
