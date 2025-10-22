### **RAG using Generative AI service and Oracle 23 ai Vector Search**

## **OCI Generative AI Integration** 

### **1\. Overview**

* OCI Generative AI can integrate with **open-source frameworks** and **OCI services** to build powerful LLM-based applications.

* A key integration is with **LangChain**, a framework designed to build **context-aware**, **language-model-powered** applications.

---

### **2\. LangChain Overview**

* **LangChain** simplifies developing LLM applications by providing interchangeable components.

* These include:

  * **LLMs & Chat Models**

  * **Prompts**

  * **Memory**

  * **Chains**

  * **Vector Stores**

  * **Document Loaders**

  * **Text Splitters**

* Components are **modular** — you can switch between models or tools with minimal code changes.

---

### **3\. Model Types in LangChain**

| Type | Description | Input | Output |
| ----- | ----- | ----- | ----- |
| **LLM** | Text completion model | String prompt | String completion |
| **Chat Model** | Conversational model backed by an LLM | List of chat messages | AI message |

---

### **4\. Prompting in LangChain**

Two main types of prompts:

1. **Prompt Template**

   * Created from a formatted Python string combining fixed text and placeholders.

   * Commonly used with **generation models**, but can also work with chat models.

2. **Chat Prompt Template**

   * A list of **chat messages**, each with a **role** (system, user, assistant) and **content**.

   * Used specifically with **chat models**.

---

### **5\. Chains in LangChain**

* Chains allow combining multiple components (LLMs, prompts, etc.) in a workflow.

* Two ways to create chains:

  1. **LangChain Expression Language (LCEL):** Declarative and preferred method.

  2. **Python Classes (e.g., LLMChain):** Programmatic approach.

---

### **6\. How Prompt, LLM, and Chain Work Together**

1. The **user query** is received.

2. Additional **context or instructions** are added using **prompts**.

3. The **prompt value** is passed to the **LLM**.

4. The **LLM** generates a **response**.

5. **Chains** automate this sequence — from user input to final LLM response.

---

### **7\. Using Memory in LangChain**

* **Memory** stores the conversation history between the user and chatbot.

* When a new query arrives:

  * The chain retrieves prior messages from **memory**.

  * Passes the full conversation and the new query to the **LLM**.

  * Writes the new query and response back to **memory**.

* Types of memory vary depending on what’s stored or returned:

  * Full conversation contents

  * Summaries of conversations

  * Extracted entities (e.g., names)

---

### **8\. Oracle 23ai Integration**

* **Oracle 23ai** can act as a **vector store** in LangChain.

* LangChain provides **Python classes** to store and search embeddings in Oracle 23ai.

---

### **9\. OCI Generative AI with Oracle 23ai**

OCI Generative AI integrates with Oracle 23ai in multiple ways:

1. **Embedding Generation:**

   * Generate embeddings outside the database using OCI Generative AI via **DB Utils** or **REST APIs**.

2. **Natural Language SQL Generation:**

   * **SELECT AI** in Oracle 23ai uses OCI Generative AI to generate SQL queries from **natural language** input.

3. **LangChain Integration:**

   * Use OCI Generative AI directly through **LangChain Python classes**.

4. **Application Development:**

   * Build applications combining OCI Generative AI and Oracle 23ai vector store using **OCI Python SDK**.

---

### **10\. Summary**

* **LangChain \+ OCI Generative AI** enables modular, context-aware, conversational applications.

* **Memory and Chains** manage dialogue flow and structure.

* **Oracle 23ai** acts as a vector store and can execute **AI-assisted SQL** queries.

* Together, they form a robust ecosystem for building intelligent, enterprise-grade AI applications on OCI.

**Retrieval-Augmented Generation (RAG)**

**What RAG solves**

* Traditional LLMs rely only on training data and can be outdated or carry biases/errors.

* RAG supplements an LLM with *retrieved, up-to-date* external content so responses are more accurate, grounded, and contextually specific.

* Helps bypass token/scale limits by feeding only top-K relevant chunks instead of entire corpora.

**High-level benefits**

* Reduces hallucination by grounding answers in retrieved sources.

* Mitigates training-data bias by consulting multiple external documents.

* Extends knowledge without retraining the base model.

* Efficient: keeps prompts small by selecting only the most relevant chunks.

**Core pipeline (three phases)**

1. **Ingestion**

   * Load documents (corpora, manuals, web pages, transcripts).

   * Split documents into chunks (paragraphs, sliding windows) to focus context.

   * Convert each chunk to an embedding (semantic vector).

   * Index embeddings in a vector store for fast nearest-neighbor search.

2. **Retrieval**

   * User issues a query.

   * Query is embedded and used to search the index.

   * Retrieve and rank candidate chunks; select top-K most relevant results.

   * Optionally apply filters (date, source, trust score) or rerankers.

3. **Generation**

   * Assemble prompt: user question \+ retrieved top-K chunks (and instructions/system prompt).

   * Send prompt to the generative model (LLM) to produce a grounded response.

   * Optionally run an attribution step to produce citations/extract supporting snippets.

**Practical considerations**

* Chunk size and overlap affect retrieval quality and hallucination risk.

* Top-K and similarity threshold tune precision vs. recall.

* Vector store choice (latency, scalability, persistence) matters for production.

* Source provenance and citation improve trust and enable verification.

* Retrieval quality is often the single biggest factor in final answer quality.

**When to use RAG**

* Question answering over private or frequently updated corpora (docs, FAQs, legal text).

* Domain-specific assistants where base LLM lacks coverage.

* Any application requiring up-to-date facts or verifiable citations.

**Summary**

* RAG \= embed → index → retrieve top-K → augment prompt → generate.

* It combines the broad language ability of LLMs with precise, current knowledge from external sources to produce more accurate and reliable results.

# **Process Documents**

* **Goal:** turn raw documents into semantically useful chunks that can be embedded and indexed for fast retrieval.

* **Input types:** PDFs, CSV, HTML, JSON, plain text, transcripts, etc. Loaders (e.g., LangChain loaders) typically support single files or entire directories.

## **Splitting into chunks**

* **Why split:** LLMs have context-window limits; chunks let you fit relevant context into prompts.

* **Chunk size:** trade-off

  * Too small → lose semantic meaning (no useful context).

  * Too large → may exceed model context window or mix unrelated concepts.

* **Chunk overlap:** include part of the previous chunk in the next to preserve continuity and avoid boundary information loss.

* **Semantic-aware splitting:** prefer splitting at paragraph/sentence boundaries rather than arbitrary character counts.

  * Text splitters try a hierarchy: preserve paragraphs → sentences → words to reach the desired chunk size while keeping meaning.

## **Typical ingestion steps**

1. **Load documents** with an appropriate loader (PDF loader, CSV loader, HTML loader, etc.).

2. **Extract raw text** from each document.

3. **Instantiate a text splitter** with `chunk_size` and `chunk_overlap`.

4. **Split text** into chunks (or split documents directly).

5. **Convert chunks to embeddings** and index them in a vector store.

## **Practical example (Python / LangChain-style)**

\# example using LangChain-like APIs

from langchain.document\_loaders import PyPDFLoader

from langchain.text\_splitter import RecursiveCharacterTextSplitter

\# 1\) load PDF

loader \= PyPDFLoader("my\_doc.pdf")

docs \= loader.load()  \# returns Document objects with page/text metadata

\# 2\) configure splitter

splitter \= RecursiveCharacterTextSplitter(chunk\_size=1000, chunk\_overlap=200)

\# 3\) split into semantically meaningful chunks

chunks \= splitter.split\_documents(docs)  \# list of Documents/chunks ready for embedding

## **Key practical tips**

* **Choose chunk size relative to your model’s context window.** For a 4k-token model, chunks of 500–1,000 tokens with overlap often work well; tune for your data.

* **Use overlap** (e.g., 10–30%) to maintain context across boundaries.

* **Prefer semantic boundaries** (paragraphs/sentences) over pure character counts.

* **Test and iterate:** inspect chunks for meaningfulness and adjust splitter strategy.

* **Automate loaders** for heterogeneous sources (bulk ingestion) and keep provenance metadata (source, page, offset) for citations.

**Embed and Store Documents**

* **Embeddings Concept**

  * Embeddings represent words, sentences, or documents as numerical vectors in a multidimensional space.

  * Semantically similar items are located close together in this space.

  * Trained embedding models learn these representations, allowing machines to measure similarity.

* **Example**

  * Words like *tiger* and *lion* (animals) have embeddings close to each other, while *apple* or *Paris* are farther apart.

  * Embeddings can be created not only for words but also for sentences, paragraphs, or full documents.

* **Creating Embeddings**

  * Oracle 23ai supports embedding models both **inside** and **outside** the database.

  * External embeddings can be generated using third-party models.

  * Internal embeddings can be created by importing **ONNX-format** embedding models directly into Oracle 23ai.

* **Storing Embeddings**

  * Oracle 23ai introduces a **vector data type** to store embeddings.

  * A table can include columns of this vector type alongside standard data types.

  * Insert or update statements can be used to store or modify vector data.

* **Process: Embedding and Storing Chunks**

  * Establish a database connection using username, password, and data source name.

  * Convert text chunks into document objects with metadata (ID, link, text, page content).

  * Create embeddings using the **OCI Generative AI** embedding model (parameters: model name, endpoint, compartment, authtype).

  * Use **Oracle VS (Vector Store) class** and `from_documents()` method to store embeddings.

    * Inputs: documents, embedding model, database connection, table name, distance strategy.

* **Result**

  * The vector store can now be queried to search for documents semantically matching user queries.

**Retrieval and Generation**

* **Query Retrieval Overview**

  1. User query is first encoded using the **same embedding model** used for chunks.

  2. Vector search is performed to find chunks **semantically similar** to the query.

  3. Multiple chunks may be returned; the top relevant chunks provide context to the LLM.

* **Similarity Measures**

  1. **Dot Product**: Measures similarity considering both magnitude and angle of vectors.

  2. **Cosine Similarity**: Measures similarity considering only the angle, ignoring magnitude.

  3. In NLP, higher magnitude may indicate richer content, smaller angle indicates higher similarity.

* **Efficient Search for Large Data**

  1. As chunks grow, naive comparison becomes slow; **indexes** improve performance.

  2. Indexes are specialized data structures for similarity search.

  3. Techniques include clustering, partitioning, and neighbor graphs.

* **Types of Vector Indexes**

  1. **HNSW (Hierarchical Navigable Small-World Graph)**: In-memory neighbor graph, efficient for approximate similarity search.

  2. **IVF (Inverted File Flat)**: Partition-based index, narrows search space using clusters/neighbor partitions.

* **LLM Contextual Response**

  1. Retrieved chunks are sent as context to the LLM along with the query.

  2. LLM generates responses using both query and relevant context.

* **Retrieval Code Workflow**

  1. Import necessary classes: `RetrievalQA`, `ChatOCIGenAI`, `OracleVS`.

  2. Create vector store with `OracleVS`: provide embedding model, DB connection, table name, distance strategy.

  3. Create retriever: set `search_type='similarity'` and `k=3` to get top 3 relevant chunks.

  4. Initialize LLM with `ChatOCIGenAI`: provide model ID, endpoint, compartment, auth type.

  5. Create RetrievalQA chain: pass LLM and retriever; set `return_source_documents=True`.

  6. Invoke chain with user query to get response along with source documents.

**Demo: LangChain Basics**

* **Models**

  1. `ChatOCIGenAI` class represents OCI Generative AI service in LangChain.

  2. LLM object is created with parameters: model name, service endpoint, compartment ID, and max tokens (limits output length).

  3. `invoke` method executes the LLM with a user query; `temperature` parameter controls output creativity.

* **Prompts**

  1. **Prompt Template**: Python string template combining fixed text and runtime variables.

  2. Input variables are defined (e.g., `user_input`, `city`) and passed at runtime.

  3. `invoke` method generates the full prompt by filling in variables.

  4. **Chat Prompt Template**: Uses a list of messages instead of a single string; invoked similarly.

* **Chains**

  1. Prompt and LLM can be chained: output of the prompt becomes input to the LLM.

  2. Use the `invoke` method on the chain with necessary inputs to get LLM response.

* **Memory**

  1. **Conversation Buffer Memory**: Tracks previous interactions in a conversation.

  2. **Conversation Chain**: Combines LLM and memory for context-aware responses.

  3. LLM can remember earlier inputs in a conversation and provide responses based on prior context.

  4. Example:

     * First input: “Hello, my name is Hemant.” → stored in memory.

     * Second input: “Can you tell me my name?” → LLM responds using memory: “Certainly, Hemant.”

* **Summary of Workflow**

  1. Initialize LLM with model parameters.

  2. Create prompt templates with fixed and variable text.

  3. Chain prompt and LLM to execute end-to-end.

  4. Use conversation memory to maintain context across multiple queries.

**Conversational RAG**

* **RAG (Retrieval-Augmented Generation) Overview**

  * Enhances LLM responses by retrieving **relevant documents** from a corpus.

  * Helps provide **more accurate and context-specific answers**.

  * Commonly used in **chatbots**.

* **Chat Flow**

  * Sequence of questions and answers.

  * Each user question can depend on **previous interactions**.

* **Memory in RAG-based Chat**

  * Stores **previous questions and answers**.

  * Memory is updated with each new interaction and passed to the LLM.

  * LLM uses both **retrieved documents** and **conversation history** to generate responses.

  * LangChain provides **memory** and **chain classes** to implement context-aware conversational systems.

* **Example**

  * Q1: “Tell me about Las Vegas.”

  * Q2: “Tell me about its typical temperature throughout the year.”

  * LLM uses context from Q1 to understand that “its” refers to Las Vegas.

**Demo: RAG with Oracle Database 23ai**

* **Autonomous Database Setup**

  * Create Autonomous Database in Oracle console: provide display name, DB name, compartment, workload type (Data Warehouse), and deployment type (Serverless).

  * Configure admin credentials, allowed IPs, and secure access.

  * Copy **connection string** for use with `oracledb` Python library.

* **Document Ingestion**

  * Use `PdfReader` to read PDF pages and extract text.

  * Use **TextCharacterSplitter** to split text into chunks:

    * `separator`: defines split points (e.g., full stop).

    * `chunk_overlap`: maintains context between chunks.

  * Convert chunks into **document objects** using metadata (page number, text).

* **Embedding and Storing in Oracle Vector Store**

  * Create **embedding model** using `OCIGenerativeAIEmbeddings` (provide model ID, service endpoint, compartment).

  * Use `OracleVS.from_documents()` to embed documents and store them in a database table:

    * Specify table name and distance strategy.

    * Stored table includes columns: primary key, text, metadata, embedding.

* **Query and Retrieval Workflow**

  * Connect to database via `oracledb`.

  * Create `ChatOCIGenAI` object with model parameters and `OCIGenAIEmbeddings` for embedding.

  * Create a **prompt template** combining retrieved documents and user question.

  * Initialize **OracleVS vector store** with embedding function, connection, table name, and distance strategy.

  * Create a **retriever** with `search_type="similarity"` and top-k documents (e.g., k=3).

  * Create a **chain** passing retriever as context; user question is processed via the prompt.

  * Invoke the chain to get LLM response based on retrieved documents.

* **Example Execution**

  * User question: “Tell us about Module 4 of AI Foundation Certification Course.”

  * LLM response: “According to the provided context, Module 4 of the AI Foundation Certification Course is about Generative AI and LLMs.”

* **Summary**

  * PDF documents → split into chunks → converted to documents → embedded → stored in Oracle Vector Store.

  * Queries → retrieved relevant chunks → LLM generates answers using context.

  * Demonstrates **RAG pipeline**: document ingestion, embedding, storage, retrieval, and context-aware response generation.

