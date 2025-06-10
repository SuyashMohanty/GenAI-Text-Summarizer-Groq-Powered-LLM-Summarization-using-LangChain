# 🧠 GenAI Text Summarizer — Groq-Powered LLM Summarization using LangChain

A modular, beginner-friendly Generative AI project for **text summarization** using **Groq’s ultra-fast LLMs** with **LangChain**. This repo demonstrates and compares three summarization strategies — **Stuff**, **Map-Reduce**, and **Refine** — all using blazing-fast inference from Groq-backed LLMs like **Mixtral**, **Gemma**, and **LLaMA**.

---

## 📌 Project Overview

Text summarization helps convert lengthy documents into digestible summaries. With Groq's low-latency LLMs and LangChain’s structured chains, this project showcases different summarization pipelines for scalable, efficient, and high-quality text summarization:

- **Stuff**: Concatenate all content, summarize once.
- **Map-Reduce**: Summarize chunks, then summarize the summaries.
- **Refine**: Incrementally update the summary with more content.

---

## 🚀 Features

- ⚡ Uses Groq API for low-latency inference
- 📄 Summarize plain text, PDFs, or other document types
- 🔁 Modular chain types: Stuff, Map-Reduce, Refine
- 🧠 Built with LangChain and Groq LLM support
- ✅ Notebook-based for ease of use and experimentation

---

## 🏗️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/genai-text-summarizer-groq.git
   cd genai-text-summarizer-groq
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq API key to a `.env` file:

   ```env
   GROQ_API_KEY=your-groq-api-key
   ```

---

## 📚 Summarization Techniques Explained

### 📦 1. Stuff Chain

**Description**: The simplest approach — all document content is sent in a single prompt.

**How It Works**:
- All text is concatenated into one large input.
- LLM generates a summary in one go.

**Pros**:
- Very fast.
- Ideal for short content.

**Cons**:
- Hits token limits for long documents.
- No internal optimization or chunking.

```python
chain = load_summarize_chain(llm, chain_type="stuff")
```

---

### 🗺️ 2. Map-Reduce Chain

**Description**: Based on distributed computing; each chunk is summarized individually, then those are combined and summarized again.

**How It Works**:
- **Map**: Each chunk gets a local summary.
- **Reduce**: All summaries are merged and summarized again.

**Pros**:
- Scalable for long documents.
- Parallelizable.

**Cons**:
- Slightly more compute-intensive.
- Output quality depends on chunk summaries.

```python
chain = load_summarize_chain(llm, chain_type="map_reduce")
```

---

### 🛠️ 3. Refine Chain

**Description**: Incrementally builds the summary by refining it with each new document chunk.

**How It Works**:
- Starts with a base summary.
- Each new chunk refines the existing summary.

**Pros**:
- Context-aware.
- Great for detailed or evolving narratives.

**Cons**:
- More steps = slower than Stuff.
- Risk of verbosity or drift over time.

```python
chain = load_summarize_chain(llm, chain_type="refine")
```

---

## 🧪 Example Usage with Groq LLMs

```python
from langchain.chat_models import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader

# Load your document
loader = TextLoader("path/to/your/document.txt")
docs = loader.load()

# Instantiate Groq LLM
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")  # other options: llama3-70b, gemma-7b

# Choose chain type: stuff, map_reduce, or refine
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(docs)

print(summary)
```

---

## ⚙️ Technology Stack

- 🐍 Python
- 🔗 LangChain
- ⚡ Groq LLMs (via `langchain.chat_models.ChatGroq`)
- 📄 PyMuPDF, Unstructured, or TextLoader
- 🧪 Jupyter Notebooks

---

## 📈 Future Enhancements

- 🌐 Add web page or YouTube transcript summarization
- 🧰 Streamlit or Gradio UI for interaction
- 📊 Summary evaluation with ROUGE or BERTScore
- 🧠 Add vector DB for retrieval-based summarization

---