### **OCI Generative AI Series**

**Lesson: OCI Generative AI Service**

* **Overview:**  
   OCI Generative AI Service is a **fully managed, serverless platform** that offers customizable **Large Language Models (LLMs)** via a **single API**, allowing easy switching between foundational models with minimal code changes.

---

### **1\. Key Characteristics**

1. **Pre-Trained Foundational Models** (from Meta & Cohere)

2. **Flexible Fine-Tuning** (custom models with own data)

3. **Dedicated AI Clusters** (GPU-based compute for training & inference)

---

### **2\. How It Works**

* User provides **text input (prompt)** → service processes it → returns a generated response.

* Supports **natural language queries**, **document inputs**, and **semantic reasoning**.

* Common use cases: **Chatbots**, **Text generation**, **Information retrieval**, **Semantic search**.

---

### **3\. Pre-Trained Foundational Models**

#### **A. Chat Models**

* **Cohere Models:**

  * `command-r` and `command-r-plus`

  * `llama3-70b-instruct` (Meta)

* **Differences:**

  * *Command-R-Plus*: More powerful, supports larger prompts (128k tokens), higher cost.

  * *Command-R*: More affordable, supports up to 16k tokens.

* **Capabilities:**

  * Maintain conversational context (multi-turn dialogue).

  * *Instruction-tuned*: Follow natural language instructions (e.g., summarization, email generation).

#### **B. Embedding Models**

* **Types:**

  * `embed-english`

  * `embed-multilingual` (supports 100+ languages)

* **Purpose:**

  * Convert text into numerical vectors to represent semantic meaning.

  * Used in **semantic search**, **recommendation systems**, and **clustering**.

* **Features:**

  * Multilingual model enables **cross-language retrieval** (e.g., Chinese query → French document).

---

### **4\. Fine-Tuning**

* **Definition:**  
   Customizing a pre-trained model using **domain-specific data** to improve accuracy and efficiency.

* **Benefits:**

  * Enhances task-specific performance.

  * Reduces inference time and cost.

* **OCI Fine-Tuning Method:**

  * **T-Few Fine-Tuning:** Inserts new layers and updates only a fraction of model weights → faster and cheaper than full fine-tuning.

---

### **5\. Dedicated AI Clusters**

* **Purpose:**  
   Host fine-tuning and inference workloads using **isolated GPU resources**.

* **Features:**

  * Dedicated GPUs with **RDMA cluster networking** (ultra-low latency).

  * Secure, isolated GPU pools for each customer’s workload.

---

### **Summary**

OCI Generative AI Service provides:

* **Choice of top foundational models** (Meta, Cohere)

* **Efficient fine-tuning** via T-Few method

* **Secure, high-performance AI clusters** for customization and deployment

It enables building scalable, secure, and domain-adapted **generative AI applications** with minimal infrastructure management.

**OCI Generative AI Service – Demo Notes**

* **Accessing the Service:**

  * Available in select OCI regions (e.g., Germany Central – Frankfurt).

  * Access via OCI Console → Analytics & AI → Generative AI.

* **Dashboard Overview:**

  * **Playground:** Visual interface to explore pre-trained and custom models without code.

  * **Dedicated AI Clusters:** GPU-based resources for fine-tuning and hosting models.

  * **Custom Models:** Fine-tuned models created from base models.

  * **Endpoints:** Used to host fine-tuned models for inference.

* **Playground – Chat Models:**

  * Models: *Command-R*, *Command-R-Plus*, *Meta Llama 3 (70B Instruct)*.

  * *Command-R-Plus*: More powerful, up to **128k tokens** per prompt.

  * *Command-R*: Affordable, **16k tokens** limit.

  * *Llama 3*: **8k tokens** limit.

  * Available in limited regions.

  * Keeps chat context for multi-turn conversations.

  * “View Code” option provides ready-to-use **Python** or **Java** API code for integration.

  * Parameters adjustable:

    * **Preamble Override** – changes tone/behavior (e.g., “Answer like a pirate”).

    * **Temperature** – controls randomness of responses.

* **Playground – Embedding Models:**

  * Models: *Embed English* and *Embed Multilingual* (100+ languages).

  * Converts text to numerical vectors for **semantic search** (meaning-based search).

  * Embeddings close in vector space \= semantically similar text.

  * Visual example shows clustering of similar HR articles (skills, leaves, etc.).

  * Code for embedding generation and integration available in the console.

* **Dedicated AI Clusters:**

  * GPU-based resources for **fine-tuning** and **inference hosting**.

  * Created by selecting purpose (Hosting / Fine-tuning) and model type.

  * Provides isolated, secure compute environments.

* **Fine-Tuning Custom Models:**

  * Create new model → Select base model → Choose fine-tune method → Assign GPU cluster.

  * Fine-tuned models improve performance for domain-specific tasks.

* **Endpoints:**

  * Created to serve inference requests from fine-tuned models.

  * Requires model name, configuration, and linked AI cluster.

**Summary:**  
 OCI Generative AI Service allows users to explore, fine-tune, and deploy large language and embedding models via a no-code console playground and API access. Key components include pre-trained models, fine-tuning options, GPU clusters, and endpoint management for serving AI applications.

**Chat Models**

### **1\. Understanding Tokens**

* LLMs process **tokens** instead of characters or words.

* A **token** can be:

  * An entire word (e.g., "apple")

  * Part of a word (e.g., "friendship" → "friend" \+ "ship")

  * Punctuation marks (e.g., comma, period)

* Token count depends on text complexity:

  * Simple text ≈ 1 token per word

  * Complex/uncommon text ≈ 2–3 tokens per word

**Example:**  
 Sentence: “Many words map to one token, but some don't, indivisible.”

* Words: 10

* Tokens after tokenization: 15

* Less common words like “indivisible” → 2 tokens (“indiv”, “isible”)

---

### **2\. Pre-trained Chat Models in OCI**

* **Command-R-Plus**

  * Instruction-following conversational model

  * User prompt: up to **128,000 tokens**

  * Response: up to **4,000 tokens**

  * Use cases: Q\&A, information retrieval, sentiment analysis, chat

* **Command-R-16k**

  * Smaller, faster version of Command-R-Plus

  * User prompt: up to **16,000 tokens**

  * Response: up to **4,000 tokens**

  * Optimized for **speed and cost**

* **Meta Llama 3.1 Family**

  * Sizes: **70B** and **400B** parameters

  * Prompt \+ response: up to **128,000 tokens**

  * 400B model: largest public LLM, suited for **complex enterprise applications**

---

### **3\. Key Parameters to Control Output**

1. **Maximum Output Tokens**

   * Max number of tokens generated per response

2. **Preamble Override**

   * Initial guideline/context for the chat

   * Can change **tone, style, behavior**

   * Example: Default vs. pirate tone

3. **Temperature**

   * Controls randomness of output

   * Temperature \= 0 → deterministic (always pick highest probability token)

   * Higher temperature → more varied outputs (select lower probability tokens)

4. **Top-k**

   * Limits next token selection to top **k** probable tokens

   * Example: Top-k \= 3 → only top 3 tokens considered

5. **Top-p (Nucleus Sampling)**

   * Limits token selection to **tokens whose cumulative probability ≤ p**

   * Example: Top-p \= 0.15 → only tokens contributing to 15% cumulative probability considered

6. **Frequency Penalty**

   * Penalizes tokens that appear **frequently** in preceding text

   * Reduces repetition

7. **Presence Penalty**

   * Penalizes tokens that have appeared **at least once**

   * Promotes **diverse output**

---

### **4\. Summary**

* OCI chat models support **instruction-following, multi-turn conversations**.

* Users can control **behavior, randomness, and diversity** through parameters: preamble, temperature, top-k, top-p, frequency and presence penalties.

* Choosing the right model and tuning parameters is key for desired output in applications like chatbots, Q\&A, and content generation.

**Demo Chat Models**

### **1\. Accessing Chat Models**

* Log in to **OCI Console** → select **region** (e.g., Frankfurt).

* Navigate: **Burger Menu → Analytics & AI → Generative AI** → **Playground** → **Chat Models**.

* Available models:

  * **Cohere Command R-Plus** (larger, longer context)

  * **Cohere Command R-16k** (faster, cost-effective)

  * **Meta Llama 3.1 Family** (70B & 405B parameters)

* “View More Details” allows seeing **token limits, model size, use cases**.

---

### **2\. Typical Chat Scenario**

* Example: Generating a course outline for **OCI Generative AI Service**.

  * Command R-Plus used for **longer context and detailed response**.

  * Can **expand individual modules** for more detail.

* The chat is **multi-turn**: model remembers context for follow-ups.

* Output token limits apply (e.g., 600 tokens in demo).

---

### **3\. Controlling Output with Parameters**

* **Temperature**: randomness of the output

  * 0 → deterministic

  * 1 → more random/probabilistic output

* **Preamble Override**: change **tone or style** (e.g., pirate tone)

* **Top-k / Top-p / Frequency / Presence penalties** can also be adjusted to affect output behavior.

* Demo showed **poem generation** with different temperatures and preamble changes.

---

### **4\. Example Use Cases with Chat Models**

1. **Text Generation / Chat**

   * Generate course outlines, expand modules, interact conversationally.

2. **Data Extraction**

   * Example: Extract entities from Wikipedia text about NVIDIA.

   * Useful for **summarizing or processing long documents**.

3. **Text Classification / Sentiment Analysis**

   * Example: Sentiment analysis for customer messages.

   * Can classify messages as positive, negative, or aspect-based sentiment.

---

### **5\. Key Takeaways**

* OCI chat models support **multi-turn conversation** and **instruction-following**.

* Parameter tuning allows **customizing tone, randomness, and behavior**.

* Practical applications include:

  * Chat and dialogue systems

  * Text generation

  * Information extraction

  * Sentiment analysis and text classification

**Demo Generative AI Inference API**

### **1\. Overview**

* Previous demos showed **chat with models via OCI Console Playground**.

* This demo focuses on **programmatic access using the Inference API**.

* Goal: Run the same chat models **via Python code / Jupyter Notebook** instead of the console.

---

### **2\. Setting Up the Environment**

* **Jupyter Notebooks**: Browser-based interactive Python environment.

  * Can install via `pip install jupyter` or use **Anaconda Navigator**.

* **OCI Python SDK**: Required to interact with OCI services programmatically.

  * Installed via pip: `pip install oci`.

---

### **3\. Authentication & Configuration**

* **Compartment ID**: OCI compartment where the Generative AI service runs.

* **Config Profile**: Stored in OCI config file; includes API keys, tenancy info, etc.

* **Service Endpoint**: Region-specific endpoint (e.g., Frankfurt) for Inference API.

import oci

config \= oci.config.from\_file("\~/.oci/config", "DEFAULT")

compartment\_id \= "ocid1.compartment.oc1..."

---

### **4\. Creating the Inference Client**

* **Generative AI Inference Client** object connects the script to the API.

* Key parameters:

  * `service_endpoint`

  * `retry` (e.g., none)

  * `timeout` (e.g., 240 seconds)

client \= oci.ai\_language.AIServiceLanguageClient(config, service\_endpoint=endpoint)

---

### **5\. Setting Up the Chat Request**

* Create **chat details object** and **chat request object**.

* Parameters include:

  * Prompt (user message)

  * Max tokens (e.g., 600\)

  * Temperature (controls randomness)

  * Model ID (specific chat model OCID)

  * Serving mode (on-demand)

chat\_request \= {

    "messages": \[{"role": "user", "content": "Generate a job description for a Data Visualization Expert"}\],

    "max\_tokens": 600,

    "temperature": 0

}

---

### **6\. Executing the Request**

* Attach **chat request** to **chat detail object** along with compartment ID.

* Send request to OCI Generative AI service:

chat\_response \= client.create\_chat\_completion(

    create\_chat\_completion\_details=chat\_request

)

* **Jupyter Notebook execution**: `Shift + Enter` → shows star while running.

* Response stored in **chat\_response object**.

---

### **7\. Understanding the Response**

* **Status**: 200 → successful.

* **Chat History**: Shows all previous messages.

  * Role `user` → user input

  * Role `assistant/chatbot` → model response

* **Text Output**: Current response from model.

* **Other Info**: Model ID, model version, finish reason, etc.

---

### **8\. Key Takeaways**

* Inference API allows **programmatic access to OCI Generative AI models**.

* Can be run in **Jupyter Notebooks** or embedded in **applications**.

* Maintains **multi-turn chat history**.

* Parameters like **temperature, max tokens, and model selection** are fully configurable.

**Demo Config setup for Generative AI Inference API**

### **1\. Scenario**

* Previously, **API invocation worked successfully** (200 status code) using the private key file.

* Now, we **delete the content of the private key file** in the local config.

  ---

  ### **2\. API Invocation Failure**

* Re-running the Jupyter Notebook after deleting the private key content **causes the request to fail**.

* **Error Message**:  Provided key is not a private key or the provided passphrase is incorrect  
* Reason: The SDK uses the private key to **sign requests**; without the valid key, authentication fails.

  ---

  ### **3\. Fix: Regenerate API Key in OCI Console**

1. Navigate to **OCI Console → Profile → API Keys**.

2. Delete the old/invalid key.

3. Click **Add API Key**:

   * Can **generate a new key pair** (public \+ private key) in PEM format.

   * Download the **private key file**.

   * Add the **public key** to your OCI account.

4. Copy all necessary details into your **config file**:

   * `user OCID`

   * `fingerprint` (new one generated)

   * `tenancy OCID`

   * `region`

   * Path to the **new private key file**

5. Save the config file.

   ---

   ### **4\. Retry API Invocation**

* Run the Jupyter Notebook again with **correct config and private key**.

* Result: **Successful invocation** (200 status code), same as before.

* The Inference API now correctly authenticates and returns responses.

  ---

  ### **5\. Key Takeaways**

* The **private key is crucial** for authentication in OCI SDK calls.

* Deleting or altering the key **will break API requests**.

* Always ensure the **config file matches the OCI console API key details**.

* Regenerating the key and updating the config **restores functionality**.

  ---

This demo demonstrates how **authentication errors can be diagnosed and fixed** when using OCI Generative AI programmatically.

**Embedding Models**

## **1\. What Are Embeddings?**

* Embeddings are **numerical representations of text** (words, phrases, sentences, paragraphs, or documents) as vectors.

* Purpose: Helps computers **understand relationships and meaning** between pieces of text.

---

## **2\. How Embeddings Work**

* **Encoders** convert text into vectors.

* Example: Sentence `"They sent me a"` → Encoder → Vector for each word \+ sentence vector.

* **Vector representation** enables tasks like:

  * Translation (sequence-to-sequence)

  * Semantic similarity comparisons

  * Retrieval-augmented generation (RAG)

---

## **3\. Word Embeddings**

* Words like `"kitten"`, `"dog"`, `"lion"` can be mapped to vectors based on properties like size, age, and others.

* Embeddings with **similar meaning** are **numerically similar**.

* Similarity measures:

  * **Cosine similarity**

  * **Dot product similarity**

---

## **4\. Sentence & Phrase Embeddings**

* Sentence embeddings assign **vectors to entire sentences or phrases**.

* Similar sentences → similar vectors.

* Can compare **words vs. sentences, sentences vs. paragraphs**, etc.

* Enables clustering and semantic similarity detection.

---

## **5\. Use Case: Retrieval-Augmented Generation (RAG)**

1. Break documents into **chunks/paragraphs**.

2. Generate embeddings for each chunk and store in a **vector database**.

3. Encode user query as a vector.

4. Retrieve **nearest-matching document embeddings**.

5. Insert retrieved content into prompt → feed to LLM for informed responses.

* **Embeddings are crucial** for high-quality retrieval and accurate RAG performance.

---

## **6\. OCI Embedding Models**

* **Cohere.embed-English**: Converts English text to 1,024-dimensional vectors, max 512 tokens.

* **Cohere.embed-English-lite**: Smaller, faster version.

* **Cohere.embed-Multilingual**: Supports 100+ languages, enables cross-language search.

* **Embed v3 models**:

  * Evaluate **query-document topic match**.

  * Rank **high-quality content** higher → better for noisy data.

* **Light versions of v3**:

  * Smaller vectors (384 dimensions), faster execution.

* **Previous generation models**: 1,024-dimensional vectors, max 512 tokens.

* **Input limits**: Max 96 inputs per run, each input ≤ 512 tokens.

---

## **7\. Applications of Embeddings**

* **Semantic search**

* **Text classification**

* **Text clustering**

* **Retrieval-augmented generation (RAG)**

---

**Key Takeaways**

* Embeddings convert text into vectors for **numerical and semantic similarity**.

* Sentence, phrase, or document embeddings enable **intelligent search, classification, and clustering**.

* OCI Generative AI supports **English, multilingual, v3, and light models** with high-dimensional embeddings.

* Maximum token limits and input limits should be considered for embedding tasks.

**Demo: Embedding Models**

## **1\. Accessing Embedding Models**

* Open the **OCI Generative AI Playground**.

* From the **Model dropdown**, select embedding models:

  * **Cohere Embed English v3.0**

  * Cohere Embed English v2.0

  * Multilingual Embed v3.0

* Input can be **individual sentences** or **batch from a file**.

---

## **2\. How Embeddings Work in the Demo**

* Embeddings are **numerical representations of text** as vectors (1,024 dimensions in v3 models).

* Each vector captures **context and semantic meaning** of a sentence or phrase.

* Example: Questions about capitals or smallest/largest states are converted into vectors.

---

## **3\. Visualizing Embeddings**

* Embeddings can be **projected into 2D** (or 3D) for visualization.

  * Vectors that are **numerically similar** are **close together** in the plot.

  * Example:

    * Questions about capitals cluster together.

    * Question about the smallest US state appears as an outlier initially.

    * Adding similar questions (“smallest state in India”) shows related questions clustering together.

* Note: Dimensionality reduction (1,024 → 2D) **loses some information** but is useful for intuition.

---

## **4\. Observing 1,024-Dimensional Vectors**

* Each embedding is a **floating-point vector of 1,024 dimensions**.

* Example workflow:

  1. Call embedding model API from **Python or Java** (playground → View Code).

  2. Retrieve embeddings as arrays of numbers.

  3. Inspect in **Jupyter Notebook** or **VS Code**.

* Each line of the vector corresponds to one dimension of the embedding for the sentence.

---

## **5\. Key Concepts Demonstrated**

* **Numerical similarity → Semantic similarity**

  * Sentences with similar meaning have vectors that are close in the embedding space.

* **Clustering of embeddings** reflects semantic relationships between sentences.

* Embeddings allow **semantic search, retrieval, and analysis** even without exact keyword matches.

---

**Key Takeaways**

* Embeddings transform sentences into **high-dimensional vectors** capturing meaning.

* Similar sentences cluster together; dissimilar sentences are far apart.

* You can **visualize embeddings in 2D/3D**, but full vectors are 1,024 dimensions (for v3 models).

* Embeddings are foundational for **semantic search, classification, clustering, and RAG** workflows.

**Prompt Engineering** 

---

## **1\. Prompt and Prompt Engineering**

* **Prompt:** Input text provided to a Large Language Model (LLM).

* **Prompt Engineering:** Iteratively refining a prompt to elicit a **specific response style or type** from an LLM.

* LLMs are **next-word predictors**, generating sequences based on the input.

* Example: Prompt “Four score and seven years ago, our …” → LLM completes the text like Lincoln’s Gettysburg Address.

---

## **2\. Why Prompt Engineering is Needed**

* Base LLMs predict next tokens; they aren’t inherently instruction-following.

* Direct instructions may not work well without fine-tuning.

* **LLM Alignment:** Models are fine-tuned (e.g., Llama 2 chat) using **Reinforcement Learning from Human Feedback (RLHF)**.

  * Human annotators rate outputs and train a **reward model**.

  * This aligns model behavior with human preferences.

---

## **3\. In-Context Learning & Few-Shot Prompting**

* **In-Context Learning:** Providing task demonstrations in the prompt without changing model parameters.

* **k-Shot Prompting:** Explicitly providing **k examples** of the task.

  * Example: Translating English → French using 3 examples → **Three-shot prompting**.

* **Zero-Shot Prompting:** Only task description is given; no examples.

* Few-shot prompting usually yields **better performance than zero-shot**.

---

## **4\. Prompt Format Matters**

* LLMs are trained with specific **prompt formats**; deviating can reduce output quality.

* **Dialogue Management (Llama 2 Example):**

  * Use **instruction tags** to mark beginning/end of instructions.

  * System prompt guides model behavior; user messages are formatted properly.

---

## **5\. Advanced Prompting Strategies**

1. **Chain-of-Thought Prompting (CoT):**

   * Provide reasoning steps in the prompt.

   * Helps model **break complex tasks into smaller steps**.

   * Example: Solve a math problem by verbalizing intermediate steps.

2. **Zero-Shot Chain-of-Thought:**

   * No examples given; prompt includes “**let’s think step by step**”.

   * Model generates intermediate reasoning steps automatically.

---

## **6\. Key Takeaways**

* Prompt engineering is **critical for extracting desired outputs** from LLMs.

* Effective prompting strategies include:

  * **Few-shot prompting**

  * **Chain-of-thought prompting**

  * Proper **prompt formatting**

  * Using **system prompts** to guide behavior

* RLHF enables modern LLMs to **follow instructions broadly**.

**Customize LLMs with your data**

## **1\. Prompt & Prompt Engineering**

* **Prompt:** Initial text provided to an LLM.

* **Prompt Engineering:** Iteratively refining a prompt to elicit **desired responses** or a particular style from an LLM.

* LLMs are **next-word predictors**: they generate the most likely next sequence of words based on the input.

* Example: Prompting with “Four score and seven years ago…” → LLM completes Lincoln’s Gettysburg Address.

---

## **2\. Why Prompt Engineering is Needed**

* Base LLMs are trained to predict next tokens, **not follow instructions naturally**.

* To get desired outputs, input must be crafted such that **the natural continuation matches the expected output**.

* **Fine-tuned models (e.g., Llama 2 chat)** can follow instructions using **RLHF (Reinforcement Learning from Human Feedback)**:

  * Human annotators provide feedback on outputs.

  * A **reward model** is trained to align LLM behavior with human preferences.

---

## **3\. In-Context Learning & Few-Shot Prompting**

* **In-Context Learning:** LLM is conditioned with **examples of the task** in the prompt; model parameters do not change.

* **k-Shot Prompting:** Providing **k examples** of the task in the prompt (e.g., 3-shot prompting).

* **Zero-Shot Prompting:** Only task description is given; no examples.

* Few-shot prompting generally improves results compared to zero-shot.

---

## **4\. Prompt Format**

* Proper **prompt formatting is critical**; different formats may lead to suboptimal results.

* Example (Llama 2 dialogue):

  * Use tags: **beginning of sequence, beginning of instruction, end of instruction**.

  * System prompts can be modified to inject context guiding model outputs.

---

## **5\. Advanced Prompting Strategies**

1. **Chain-of-Thought Prompting (CoT):**

   * Provide examples including **reasoning steps**.

   * Break complex tasks into smaller intermediate steps.

   * Useful for tasks requiring **intermediate reasoning** (e.g., multi-step math problems).

2. **Zero-Shot Chain-of-Thought:**

   * No example needed; prompt includes a phrase like: **“Let’s think step by step.”**

   * Model automatically breaks problem into steps and solves it sequentially.

---

## **6\. Key Takeaways**

* Prompt engineering is **essential for extracting optimal responses** from LLMs.

* Strategies include:

  * Few-shot & zero-shot prompting

  * Chain-of-thought prompting

  * Correct prompt formatting

  * System prompts to guide model behavior

* Modern LLMs can follow instructions because of **fine-tuning \+ RLHF**.

  **Fine Tuning and Inference in OCI Generative AI**

  ## **1\. Key Concepts**

* **Fine-tuning:** Adapting a pre-trained foundational model to a **specific task or dataset** using additional training.

* **Inference:** Using a trained (or fine-tuned) model to generate outputs based on new input.

  * In LLMs, this means **receiving input text** and generating output text.

  ---

  ## **2\. Custom Models**

* **Custom Model:** A model derived from a pre-trained model, fine-tuned with your own dataset.

* **Workflow for creating a custom model:**

  1. Create a **dedicated AI cluster** (fine-tuning cluster).

  2. Gather and prepare **training data**.

  3. Kickstart the **fine-tuning process**.

  4. Obtain the **fine-tuned custom model**.

  ---

  ## **3\. Inference Workflow**

* **Model Endpoint:** A point on a dedicated AI cluster where the model **accepts requests and returns responses**.

* Steps:

  1. Create a **hosting cluster**.

  2. Create the **endpoint**.

  3. Serve requests from users (production load).

  ---

  ## **4\. Dedicated AI Clusters**

* **Single-tenant GPU deployment**: GPUs are dedicated to your custom models, ensuring consistent throughput.

* **Cluster types:**

  * **Fine-tuning cluster:** Used for training/fine-tuning.

  * **Hosting cluster:** Used for inference/model endpoints.

  ---

  ## **5\. Fine-Tuning Techniques**

  ### **5.1 Vanilla Fine-Tuning**

* Updates all or most layers of the model.

* **Cons:** High training time and serving cost.

  ### **5.2 T-Few Fine-Tuning**

* Parameter-efficient technique: updates only **0.01% of model weights** by adding **T-Few layers**.

* Advantages:

  1. Faster and cheaper than Vanilla fine-tuning.

  2. Retains most base model knowledge.

  3. Generates **supplementary weights** that are applied to specific transformer layers.

* **Workflow:**

  1. Start with **base model weights**.

  2. Use **annotated training dataset** (input/output pairs).

  3. Generate **T-Few weights** (\~0.01% of model size).

  4. Apply weights to **T-Few transformer layers** only.

  ---

  ## **6\. Inference Optimization**

* **Hosting multiple models on same GPU:**

  * Base model \+ multiple fine-tuned models can **share GPU resources** (multi-tenancy).

  * Minimal overhead due to **parameter sharing** (most weights are common).

* **Memory management:**

  * T-Few fine-tuned models only differ slightly from base model.

  * Reduces **GPU memory overhead** when switching between models.

  * Efficient deployment allows multiple models to serve **concurrent requests**.

  ---

  ## **7\. Benefits of T-Few Fine-Tuning**

* Reduced training time and cost.

* Retains general knowledge from base model.

* Supports multiple **custom models on same cluster** efficiently.

* Lowers inference cost and memory overhead.

**Dedicated AI Cluster Sizing and Pricing**

## **1\. Cluster Unit Types**

OCI Generative AI service offers **four types of dedicated AI cluster units**:

| Cluster Type | Purpose | Supported Models | Fine-Tuning? |
| ----- | ----- | ----- | ----- |
| **Large Cohere Dedicated** | Fine-tuning & hosting | Cohere Command R family | Yes (limited models) |
| **Small Cohere Dedicated** | Fine-tuning & hosting | Cohere Command R models | Yes |
| **Embed Cohere Dedicated** | Hosting embeddings only | Cohere English & multilingual embedding models | ❌ No fine-tuning |
| **Large Meta Dedicated** | Fine-tuning & hosting | Meta Llama 3.3, 3.1 (various parameter sizes) | Yes |

**Key points:**

* Cohere models → use Large/Small Cohere units.

* Embedding models → use Embed Cohere units.

* Llama models → use Large Meta units.

* You cannot mix cluster types for different models.

---

## **2\. Service Limits**

* Each cluster unit type has a **service limit**, e.g., `dedicated-unit-large-cohere-count`.

* **Default:** 0 units; must request **service limit increase** to provision clusters.

---

## **3\. Sizing Clusters for Models**

| Model | Fine-Tuning Units Required | Hosting Units Required | Notes |
| ----- | ----- | ----- | ----- |
| Cohere Command R-plus 08-2024 | ❌ Not supported | 2 large Cohere units | Hosting only |
| Cohere Command R 08-2024 | 8 small Cohere units | 1 small Cohere unit | Fine-tuning \+ hosting |
| Meta Llama 3.3 / 3.1 | 4 large Meta units | 1 large Meta unit | Fine-tuning \+ hosting |
| Embedding Models | ❌ Not supported | 1 embed Cohere unit | Hosting only |

**Example:**

* Fine-tune Cohere Command R 08-2024 → 8 small units.

* Host the model → 1 small unit.

* **Total units needed:** 9 small Cohere dedicated units.

---

## **4\. Pricing Example**

**Scenario:** Bob fine-tunes and hosts Cohere Command R 08-2024.

**Fine-Tuning:**

* 8 units × 5 hours per session × 4 weeks \= **160 unit-hours/month**

* Minimum billing: 1 hour per fine-tuning session.

**Hosting:**

* Runs full month → 744 unit-hours

* Can host multiple models on the same cluster.

**Total Unit-Hours:** 160 \+ 744 \= 904 unit-hours

**Unit Price Example:** $6.50/unit-hour

* **Monthly cost:** 904 × $6.50 ≈ $5,900

**Notes:**

* Fine-tuning cost depends on **training duration**.

* Hosting cost depends on **full month usage** (minimum commitment).

* Always verify **latest pricing** on OCI documentation.

**Demo: Dedicated AI Clusters**

## **1\. Accessing Dedicated AI Clusters**

* Go to **OCI Console → Generative AI → Dedicated AI Clusters**.

* Ensure your account has the **service limits enabled**:

  * Navigate: **Governance & Administration → Tenancy Management → Limits, Quotas, and Usage → Generative AI**.

  * Check your available cluster units (e.g., Small Cohere, Large Cohere, Llama2-70, Embed Cohere).

  * Only the units with limits enabled in your account can be provisioned.

---

## **2\. Cluster Types and Requirements**

| Cluster Type | Usage | Model Constraints | Unit Requirements | Min Commitment |
| ----- | ----- | ----- | ----- | ----- |
| **Fine-Tuning Cluster** | Fine-tune models | Must match base model to account-enabled cluster units | 2 small Cohere units (for demo) | 1 unit-hour minimum |
| **Hosting Cluster** | Host custom/fine-tuned models | Base model must match cluster units | 1 small Cohere unit (for demo) | 744 unit-hours (full month) |

**Notes:**

* Fine-tuning cluster can be used to create custom models.

* Hosting cluster can host **up to 50 models** if using **T-Few fine-tuning**.

---

## **3\. Creating a Fine-Tuning Cluster**

1. Click **Create Dedicated AI Cluster → Choose Compartment → Name it**.

2. Select **Cluster Type: Fine Tuning**.

3. Select **Base Model:** must match enabled cluster units (e.g., Coher Command Light).

4. Confirm **unit provisioning**: 2 units for demo.

5. Check the **commitment box**: minimum 1 unit-hour.

6. Click **Create** → cluster will be provisioned in a few minutes.

---

## **4\. Creating a Hosting Cluster**

1. Click **Create Dedicated AI Cluster → Choose Compartment → Name it**.

2. Select **Cluster Type: Hosting**.

3. Select **Base Model:** must match enabled cluster units (e.g., Command Light).

4. Confirm **unit provisioning**: 1 unit for demo.

5. Minimum commitment: **744 unit-hours** (full month).

6. Click **Create** → cluster will be provisioned in a few minutes.

---

## **5\. Cluster Details**

* **Fine-Tuning Cluster:**

  * Active status

  * Unit size: Small Cohere

  * Units allocated: 2

  * Can create fine-tuned models → click **Create Model**

* **Hosting Cluster:**

  * Endpoint capacity: 50 models (T-Few method)

  * Unit size: Small Cohere

  * Units allocated: 1

  * Can host models by creating endpoints once models are fine-tuned

---

## **6\. Next Steps**

* Kick off a **fine-tuning job** on the fine-tuning cluster.

* Once a **custom model** is ready, create an **endpoint** to host it on the hosting cluster.

**Fine-tuning configuration**

## **1\. Fine-Tuning Methods**

OCI Generative AI supports two main **Parameter Efficient Fine-Tuning (PEFT)** methods:

| Method | Description | Analogy |
| ----- | ----- | ----- |
| **T-Few** | Adds small helper layers to tweak the model without updating all weights | Like adding small helper parts to a machine |
| **LoRA (Low-Rank Adaptation)** | Adjusts certain weights in a low-rank manner, leaving main model unchanged | Like adding special gears to adjust a machine |

**Key idea:** Both methods allow a model to learn a new task without modifying the entire model, making fine-tuning faster and cheaper.

---

## **2\. Fine-Tuning Hyperparameters**

These control how the model is fine-tuned:

| Hyperparameter | Meaning (Analogy) |
| ----- | ----- |
| **Total Training Epochs** | How many times the model “studies” the dataset; more epochs \= more study |
| **Training Batch Size** | Number of examples processed at once; larger batch \= faster learning, smaller batch \= more detailed insights |
| **Learning Rate** | Speed at which model adjusts weights; higher \= faster, lower \= careful updates |
| **Early Stopping Threshold** | Minimum improvement required to continue training |
| **Early Stopping Patience** | How long the model waits before stopping if no improvement |
| **Log Model Metrics Interval** | How often progress is recorded during training |

⚠️ Always check current OCI documentation, as valid ranges and defaults may change.

---

## **3\. Evaluating Fine-Tuning Results**

Two main metrics are used:

### **Accuracy**

* Measures **percentage of correct tokens** predicted versus annotated tokens (ground truth).

* Example:

  * Ground truth: `the cat sat on the mat`

  * Model prediction: `the cat slept on the rug`

  * Correct tokens: 4/6 → Accuracy \= 67%

* Limitation: Even if the meaning is correct but wording differs, accuracy may appear low.

### **Loss**

* Measures **how wrong the model outputs are**, using probability distribution differences between predicted and true tokens.

* Example 1 (minor mistakes):

  * Model: `the cat slept on the rug` → low loss (context preserved)

* Example 2 (irrelevant output):

  * Model: `the airplane flew at midnight` → high loss (context lost)

* **Advantage:** Loss is a better metric than accuracy for generative AI, as it accounts for contextual similarity rather than exact token match.

---

## **4\. Key Takeaways**

1. **T-Few and LoRA** enable efficient fine-tuning without retraining the entire model.

2. **Hyperparameters** control learning behavior and efficiency; understanding their impact is crucial.

3. **Accuracy vs Loss:**

   * Accuracy \= token match percentage

   * Loss \= measures contextual correctness and severity of mistakes

   * Loss is generally preferred for evaluating generative AI outputs.

**Demo: Fine-tuning and Custom Models**

## **1\. Objective**

* Fine-tune a pre-trained base model to **rephrase human requests into AI virtual assistant responses**.

* Dataset source: *Sound Control Natural Rephrasing in Dialog Systems*.

* Focus only on two columns:

  * `human request` (input)

  * `virtual assistant utterance` (desired output)

  ---

  ## **2\. Data Preparation**

* OCI requires **JSON Lines (JSONL) format** for fine-tuning.

  * Each line \= a separate JSON object.

  * Two required properties per line:

    1. **prompt** → the input (human request)

    2. **completion** → the desired output (virtual assistant utterance)

  * Must be **UTF-8 encoded**

* Example line:

* {"prompt": "ask my aunt if she can go to the JDRF walk with me on October 6",   
*  "completion": "can you go to the JDRF walk with me on October 6?"}  
    
* Dataset size in this demo: \~2,000 examples.

  ---

  ## **3\. Model & Cluster Selection**

* Base model: **Cohere Command Light**

  * Must match the **cluster unit type**; Command Light → small Cohere unit

* Fine-tuning method: **T-Few** (parameter-efficient, ideal for smaller datasets)

* Dedicated AI cluster: **Custom Fine-Tuning Cluster** (created previously)

* Hyperparameters: default values used initially; can be adjusted based on accuracy/loss later.

  ---

  ## **4\. Upload & Configure Training File**

1. Upload JSONL dataset to an OCI **bucket**.

2. Ensure proper **IAM policies** for the service to access the file.

3. Preview first few lines to confirm format.

4. Assign file as **training data** in the fine-tuning workflow.

   ---

   ## **5\. Fine-Tuning Process**

* Steps followed:

  1. Gathered dataset and formatted in JSONL.

  2. Created dedicated fine-tuning cluster.

  3. Uploaded training data.

  4. Kicked off **T-Few fine-tuning** process.

* Outcome: once complete, a **custom model** is created.

* Performance metrics to monitor: **accuracy** and **loss**.

  ---

  ## **6\. Key Notes**

* Matching **base model → cluster type** is essential; mismatch prevents fine-tuning.

* **T-Few** is efficient for small datasets; vanilla fine-tuning is heavier.

* JSONL formatting is strict; any deviation may cause workflow failure.

* After fine-tuning, the custom model can be **hosted** and used for inference.

**Demo: Inference using Endpoint**

## **1\. Objective**

* Demonstrate how to **create and test an endpoint** for a previously fine-tuned custom model in OCI Generative AI.

* Use the endpoint to compare **base model vs custom model performance**.

  ---

  ## **2\. Reviewing Model Performance**

* Model metrics after fine-tuning:

  * **Accuracy:** 0.98 → 98% of output tokens match training data tokens (excellent performance).

  * **Loss:** trending toward **0**, meaning minimal randomness and strong prediction accuracy.

* Accuracy \= % of correctly matched tokens.

* Loss \= measure of how wrong model predictions are.

  ---

  ## **3\. Creating an Endpoint**

**Steps:**

1. Go to **OCI Generative AI Console → Endpoints → Create Endpoint**.

2. Provide:

   * Optional name & description

   * **Model Name:** select the **custom model** created earlier (visible with version info)

   * **Cluster:** choose the dedicated AI cluster

3. Check capacity:

   * Using **Preview Fine-Tuning**, up to **50 endpoints** can be hosted (each for one custom model).

4. **Content Moderation:** optional feature to filter toxic or biased output.

   * In this demo: turned **off**, as dataset is known and safe.

5. Click **Create Endpoint** → status becomes *Active* after a few minutes.

   ---

   ## **4\. Testing the Endpoint**

* Open **Playground** in OCI console.

* Under *Model* dropdown, select the new **custom model endpoint**.

* Test with prompts from the **test dataset** (data unseen during training).

**Example prompt:**

“Turn this message to a virtual assistant into the correct action.”

**Results:**

* **Custom model output:** “Do you want to go hiking in Yellowstone with me from 8 until 11th.”

  * Consistent across temperature settings (0 → deterministic, 5 → creative).

* **Base model output (Cohere Command Light):** “Hello Elon, would you be interested in joining me?”

  * Irrelevant response, shows weaker alignment.

  ---

  ## **5\. Key Insights**

* **Endpoints** make the custom model accessible for **real-time inference/testing**.

* **Custom models** outperform base models for domain-specific or task-specific rephrasing.

* **Temperature variation test** proves **output stability** and model robustness.

* Model generalizes well — accurate predictions on unseen test data.

  ---

  ## **6\. Summary**

* **High accuracy, low loss → strong fine-tuning quality.**

* **Endpoint creation** allows deploying and testing the model easily.

* **Custom models** deliver superior contextual performance over base models for targeted applications.

  ## **OCI Generative AI Security**

  ### **1\. Core Principle**

* **Security and privacy** are foundational design principles in OCI Generative AI.

* The service ensures **complete isolation** of customer data, models, and compute resources.

  ---

  ### **2\. Dedicated AI Infrastructure**

* **Dedicated GPU Clusters**:

  * GPUs for each customer are **not shared** with others.

  * Operate on a **dedicated RDMA network** ensuring high-speed, secure communication.

* Each cluster runs only the customer’s **base models** and **fine-tuned models**.

* This setup guarantees **model isolation** and **data isolation** between tenants.

  ---

  ### **3\. Tenant-Level Data Access Control**

* **Customer data remains within the customer’s tenancy**.

* Applications belonging to one customer **cannot access** another customer’s models or data.

* Example:

  * Customer 1’s models and endpoints are only accessible by Customer 1’s applications.

  * Customer 2’s applications cannot access Customer 1’s resources.

  ---

  ### **4\. Integration with OCI Security Services**

OCI Generative AI leverages core OCI security features for layered protection:

1. **OCI Identity and Access Management (IAM):**

   * Handles **authentication and authorization**.

   * Controls which applications or users can access which models.

   * Example:

     * Application X → authorized for Custom Model X

     * Application Y → authorized for Base Model

2. **OCI Key Management Service (KMS):**

   * Securely stores **encryption keys** for both base and fine-tuned models.

   * Ensures data encryption at rest and in transit.

3. **OCI Object Storage:**

   * Stores **model weights** (base and custom).

   * All stored data is **encrypted by default** and secured via KMS-managed keys.

   ---

   ### **5\. Summary**

* **Dedicated resources:** GPUs and RDMA network ensure computational isolation.

* **Tenant separation:** Strict access boundaries prevent data/model leakage.

* **Integrated security:** IAM for access control, KMS for encryption, and Object Storage for secure data persistence.

* OCI Generative AI builds on existing OCI security layers to provide a **secure, private, and isolated AI environment**.

