from flask import Flask, request, render_template
from prova import get_wikipedia_html, setup_langchain

app = Flask(__name__)

# Initialize global variables for the QA chain and text
global_qa_chain = None
wikipedia_text = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    global wikipedia_text, global_qa_chain

    query = request.form.get('search_query')
    if not query:
        return render_template('index.html', error="Please enter a search query.")

    try:
        # Retrieve Wikipedia content
        wikipedia_text = get_wikipedia_html(query)
        if "No results" in wikipedia_text or "Failed to retrieve" in wikipedia_text:
            return render_template('index.html', error=wikipedia_text)

        # Set up LangChain QA system
        global_qa_chain = setup_langchain(wikipedia_text)
        return render_template('index.html', success="Wikipedia page found and QA system ready!", wikipedia_text=wikipedia_text)
    except Exception as e:
        return render_template('index.html', error=f"Error: {e}")

@app.route('/ask', methods=['POST'])
def ask():
    global global_qa_chain

    if not global_qa_chain:
        return render_template('index.html', error="Please search for a Wikipedia page first.")

    question = request.form.get('question')
    if not question:
        return render_template('index.html', error="Please enter a question.")

    try:
        # Query the QA system
        result = global_qa_chain.invoke({"query": question})
        answer = result.get("result", "No answer generated.")
        sources = "\n".join([doc.metadata.get('source', 'Unknown source') for doc in result.get("source_documents", [])])
        return render_template('index.html', question=question, answer=answer, sources=sources, wikipedia_text=wikipedia_text)
    except Exception as e:
        return render_template('index.html', error=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
