## 🏏 IPL 2022 RAG Saga: Post-Knowledge-Cutoff Cricket Wisdom

*"Because even LLMs deserve to enjoy the IPL drama after their training ends!"* 😎 — A wise AI wrangler


## 🤔 What Is This?

This is not your average data project. This is **RAG** 🧠 + **IPL 2022** 🎉 = 🔥 **Cricket Time Travel** for language models!

I have built a RAG pipeline so your favorite LLMs (whose memory tragically ends before IPL 2022) can finally *know what Jos Buttler did last summer*.

### TL;DR:

📄 **Raw CSVs** → 🧠 **Human-ish Summaries** → 💾 **FAISS Vector Store** → 🧙‍♂️ **LLM Sorcery** → 🎤 **Post-match insights with sass**


## 🎬 The IPL Magic: Code Breakdown

| Stage | Description |
|---|---|
| 🛠️ `preprocessing_data.py` | Converts raw stats into player summaries so LLMs can *feel the cricket fever*. |
| 📦 `create_vector_store.py` | Embeds and stores those spicy summaries into a FAISS vector store. |
| 🔮 `rag_pipeline.py` | Retrieves relevant player intel and generates smart answers with LLMs. |


## 🧙‍♂️ Prerequisites (a.k.a. Ritual Requirements)

Before you summon the cricket gods, make sure your spellbook (read: system) contains:

* 🐍 Python **3.11.9**
* 🦙 Ollama **0.5.1**
* ⚡ Internet to pull models (unless you're Gandalf and already have them)
* ❤️ Love for cricket (not required, but highly recommended for full enlightenment)


## 🧪 Installation: Light the Fire

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ipl-2022-rag.git
    cd ipl-2022-rag
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **Note:** You'll need to create a `requirements.txt` file in the root directory of your project if it doesn't exist. It should list all the Python libraries your project depends on (e.g., `pandas`, `langchain`, `faiss-cpu`, `ollama`).


## 🛠️ How to Cast the Spell (a.k.a. Run the Code)

### 🔹 Step 1: Preprocess the CSV

```bash
python rag_pipeline/preprocessing_data.py --input_csv_file_path ./data/IPL_2022.csv --output_text_file_path ./data/IPL_2022_summary.txt
```

🎤 This step whispers the truth of IPL 2022 into a language LLMs understand.

### 🔹 Step 2: Create the Vector Store

```bash
python rag_pipeline/create_vector_store.py --input_summary_file_path ./data/IPL_2022_summary.txt --embedding_model granite-embedding:30m --output_vector_store_path ./vector_store --output_vector_store_index_name ipl_2022
```

🧠 Embeds cricket knowledge into the RAG engine.

### 🔹 Step 3: Ask LLMs the Unanswerable (Until Now)

```bash
python rag_pipeline/rag_pipeline.py --user_query "how many runs did Buttler score?" --embedding_model granite-embedding:30m --chat_model gemma3:4b --output_vector_store_path ./vector_store --output_vector_store_index_name ipl_2022
```

🔍 Retrieves summaries + 🧙‍♂️ channels LLM wisdom = 💬 answers.


## 💡 Example Queries

* "How many Six Quinton De Kock scored in IPL 2022?"
* "How did Dhoni perform in 2022?"
* "Give me Buttler’s stats like I’m five."
* "Highest score of Hardik Pandya?"


## 🧪 Project Structure (a.k.a. Code Playground)

```
ipl-2022-rag/
├── data/
│   ├── IPL_2022.csv                # Your raw player stats
│   └── IPL_2022_summary.txt        # LLM-readable summaries
├── fast-api/                       # On the way (future feature)
├── rag_pipeline/
│   ├── vector_store/               # Saved FAISS vector index
│   ├── preprocessing_data.py       # CSV → Text
│   ├── create_vector_store.py      # Text → FAISS
│   └── rag_pipeline.py             # Ask LLMs like a boss
└── README.md                       # You’re reading it, legend
```


## 🤖 Models Used

* `granite-embedding:30m` for embeddings
* `gemma3:4b` for generating answers (but feel free to swap in your favorite Ollama model!)

**⚠️ Make sure these are available in your Ollama instance.**


## 🤝 Contributions

PRs are welcome, just like unexpected no-balls in the final over.

Raise an issue if:

* You find bugs.
* You want to suggest a cool new feature.
* You’re emotionally moved by Buttler’s batting and want to talk.


## 📜 License

MIT License. Because cricket insights should be free like Dhoni's helicopter shots. 🛫


## 🏁 Final Words

This project exists because LLMs deserve cricket updates too. And now, thanks to you, they’ll finally know who ruled IPL 2022.

Now go ahead, run it, ask questions, and let the magic unfold! 🧙‍♂️🏏🔥

"The only thing cooler than an LLM answering cricket queries is an LLM answering correct cricket queries."
