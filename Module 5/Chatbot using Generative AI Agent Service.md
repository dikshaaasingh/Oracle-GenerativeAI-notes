### **Chatbot using Generative AI Agent Service**

**Oracle Generative AI Agents** 

* **Overview**

  * Fully managed OCI service combining **LLMs \+ retrieval systems** to create context-aware, actionable responses.

  * Agents interpret user intent, retrieve relevant data, and execute actions (e.g., booking flight \+ hotel).

  * Ready-to-use, validated, and scalable for enterprise applications.

  * Interaction via **chat interface** or **API**.

* **Architecture**

  * **Interface:** Chatbot, web app, or voice interface for user interaction.

  * **Inputs to LLM:**

    * *Short/long-term memory* – maintains conversation context.

    * *Tools* – integrates APIs, databases, or external systems.

    * *Prompt* – user’s query/task instruction.

  * **Core LLM Operations:**

    * *Reasoning* – generates logical responses.

    * *Acting* – triggers actions (e.g., API/database calls).

    * *Persona* – maintains tone/style consistency.

    * *Planning* – organizes multi-step workflows.

  * **External Knowledge Access:** RAG enables responses beyond trained data.

  * **Feedback Loop:** Output updates short-term memory for continuity.

* **Core Concepts**

  * **Generative AI Model:** LLM trained on large datasets to generate coherent text.

  * **Agent:** Autonomous system built on LLM for natural interactions.

  * **RAG Agents:** Retrieve external data for grounded, accurate answers.

    * *Answerability* – relevant responses to user queries.

    * *Groundedness* – responses traceable to data sources.

* **Data Hierarchy**

  * **Data Store:** Where data resides (object storage, DB).

  * **Data Source:** Connection details for accessing the store.

  * **Knowledge Base:** Vector storage system that organizes data for efficient retrieval.

* **Supported Data Options**

  * **Object Storage:** Upload PDF/text files (≤100 MB each; images ≤8 MB).

  * **OpenSearch:** Use indexed data from OCI Search with OpenSearch.

  * **Oracle Database Vector Store:** Use vector embeddings from Oracle 23ai DB or Autonomous DB.

* **Object Storage Guidelines**

  * One bucket per data source.

  * Supports `.pdf` and `.txt` formats only.

  * 2D charts must have labeled axes; tables interpretable directly.

  * Hyperlinks extracted as clickable links in chat.

  * Can pre-create empty folder for later ingestion.

* **Oracle Database Guidelines**

  * Agent doesn’t manage DB; user must configure connection.

  * Create table with fields: `DOCID`, `BODY`, `VECTOR` (optional: `CHUNKID`, `URL`, `TITLE`, `PAGE_NO`).

  * Define function (e.g., `retrieval_func_ai(p_query, top_k)`) to return vector search results.

  * Use same **embedding model** for query and stored embeddings (e.g., `Cohere embed multilingual v3`).

  * Function returns `SYS_REFCURSOR` with fields: `DOCID`, `BODY`, `SCORE`.

  * Uses cosine or Euclidean distance to rank top k results by similarity.

* **Workflow of Agent Creation**

  * **Create Knowledge Base**

    * Choose store type (Object Storage or Oracle 23ai Vector Store).

    * Enable hybrid search (lexical \+ semantic).

    * Specify data source and ingestion job.

  * **Create Agent**

    * Add details, welcome message, and RAG instructions.

    * Link to created knowledge base.

  * **Create Endpoint**

    * Defines communication channel for external systems.

    * Configure moderation, trace, and citation options.

  * **Chat with Agent**

    * Use endpoint to query and observe citations (sources) and trace (chat history).

* **Key Features**

  * **Session:** Maintains context across interactions.

  * **Trace:** Logs conversation history (prompts \+ responses).

  * **Citation:** Provides source details (title, path, doc ID, pages).

  * **Content Moderation:** Filters hate, self-harm, ideological harm, exploitation.

    * Can apply to prompts, responses, or both.

* **Default Resource Limits**

  * Defined per account; can request increases if needed.

* **Summary**

  * Oracle Generative AI Agents \= LLM-powered, retrieval-augmented, enterprise-ready systems.

  * Enable intelligent, contextual, and traceable automation using object storage or Oracle 23ai vector databases.

**Chatbot Demo using Object Storage** 

* **Overview**

  * Objective: Create a Generative AI Agent using **Object Storage** as the data store.

  * Process includes three steps:

    1. Create Knowledge Base

    2. Create Agent

    3. Test and Chat with Agent

* **Accessing the Service**

  * Log in via **cloud.oracle.com**.

  * Ensure service availability in your selected **region** (e.g., Frankfurt).

  * Navigate to: **Navigation Menu → Analytics & AI → Generative Agents**.

  * Opens **Generative AI Agents Overview Page**.

### **1\. Create Knowledge Base**

* Go to **Knowledge Bases → Create Knowledge Base**.

* Enter a **name** (e.g., `demo-kb`) and select the **compartment** with proper permissions.

* **Optional:** Add description.

* Choose **Object Storage** as **Data Store Type**.

* Enable **Hybrid Search** (combines lexical \+ semantic search).

  * *Lexical:* Finds exact keyword matches.

  * *Semantic:* Finds context-based matches using vectors.

  * *Hybrid:* Uses both (typical in RAG systems).

#### **Data Source Configuration**

* One data source allowed per knowledge base.

* Define **Data Source Name** (e.g., `demo-data-source`).

* Option to **enable multimodal parsing** (includes charts and graphs).

* Select a **bucket** (e.g., `Gen AI Agents`).

  * Files used:

    * `faq.txt` (Oracle FAQs)

    * `oci-ai-foundations.pdf` (course transcript)

* Supported formats: `.txt` and `.pdf` (max 100 MB each; images ≤ 8 MB).

#### **Ingestion Job**

* Check the option **Start Ingestion Job** during creation.

* After ingestion:

  * Verify **status logs** for successful ingestion.

  * If failures occur (e.g., file too large), fix and restart the job.

  * Restarting skips already ingested files and reprocesses failed ones.

* Once complete, **Knowledge Base status \= Active**.

#### **Knowledge Base Management**

* One data source per KB.

* You can **update** existing sources (no need to delete).

* To delete a KB:

  * Delete data sources first.

  * Delete agents using that KB.

* Deleted resources **cannot be restored**.

* Agents with deleted data sources continue running but lose related data access.

### **2\. Create Agent**

* Create an agent linked to the knowledge base.

* Once active, open **Agent → Endpoints (under Resources)**.

* If auto-created, verify endpoint settings:

  * **Trace**, **Citation**, **Session** \= Enabled by default.

  * **Content Moderation** \= Disabled by default.

#### **Create or Edit Endpoint**

* If no endpoint, click **Create Endpoint**.

  * Provide a name.

  * **Enable Session:** Keeps chat context; define **Idle Timeout** (default 3600s \= 1 hr, up to 7 days).

  * **Enable Content Moderation:** On input, output, or both.

  * **Enable Trace:** Logs prompts and responses.

  * **Enable Citation:** Displays information sources for answers.

* Can **edit** endpoint later to modify moderation, trace, citation, or timeout.

* If session not enabled initially, timeout cannot be edited later.

### **3\. Chat with the Agent**

* Go to **Chat** under Generative AI Agents.

* Select the **Agent** (e.g., `demo-agent`) and **Endpoint** created earlier.

* **Welcome Message Example:**  
   “Hi user, I am an AI Assistant. I can help you with answering questions and providing information based on AI Foundations course and Oracle FAQs.”

#### **Example Interactions**

1. **Query:** “Please tell me about Oracle Free Tier.”

   * Response generated with citations.

   * *Citations* show document title, path, doc ID, and extracted text.

   * *Trace* shows query, retrieved sources, and generated answer.

2. **Query:** “How many modules are there in Oracle AI Foundations course?”

   * Response: “The course is split into six modules.”

   * Citations and traces viewable.

3. **Query:** “Who are the instructors for this course?”

   * Returns instructor names with supporting citations.

   * Demonstrates **session memory**—understands context without repeating “AI Foundations course.”

### **4\. Key Learnings from Demo**

* Successfully created:

  * A **Knowledge Base** (using Object Storage).

  * An **Agent** (linked to KB).

  * An **Endpoint** (for chat interface).

* Demonstrated features:

  * **Hybrid Search**, **Session Context**, **Citations**, **Trace**, **Content Moderation**.

* Showcased contextual understanding, traceable responses, and real data grounding via RAG.

**Summary**  
 This demo illustrated how to:

1. Create a **Knowledge Base** using Object Storage.

2. Build a **Generative AI Agent** connected to it.

3. Configure an **Endpoint** for chat.

4. Interact with the agent and verify **citations**, **trace**, and **session-based memory** for context-aware answers.

**Chatbot Demo using Oracle 23ai** 

**Objective**

Demonstrate creation of a Generative AI Agent using **Oracle Database 23ai** as the data store instead of Object Storage.

### **1\. Prerequisites**

* Necessary **policies, permissions, VCN**, and **security rules** already configured.

* A **Vault** with **encryption key** created for storing database secrets.

* Reference setup documentation: \[docs.oracle.com → Generative AI Agents → Getting Access & Database Guidelines\].

### **2\. Create Autonomous Database (ADB)**

1. Navigate to **Oracle Database → Autonomous Database → Create Autonomous Database**.

2. **Details:**

   * Name: `demoagent` (for both display and database).

   * **Workload type:** Data Warehouse.

   * **Deployment:** Serverless.

   * **Version:** 23ai.

   * **Network access:** Private endpoint only.

   * **VCN/Subnet:** Select pre-created ones.

   * **TLS authentication:** Keep unchecked.

   * **Email:** Provide valid ID.

3. Click **Create Autonomous Database**.

4. Wait until **Lifecycle State \= Available**.

5. Copy **Connection String** and **Private Endpoint IP** from Database Connection and Network sections for later use.

### **3\. Create Database Tools Connection**

1. Go to **Developer Services → Database Tools → Connections → Create Connection**.

2. Provide:

   * Name: `demoagent`.

   * Compartment info.

   * **Database Cloud Service:** Oracle Autonomous Database.

   * **Select Database:** newly created ADB.

   * **Username:** `admin`.

3. **Create Password Secret:**

   * Name: `demoagent`.

   * Select **Vault** and **Encryption Key** created earlier.

   * Use same password as during ADB creation.

4. Use this secret as database password.

5. Modify connection string:

   * Change `retry_count` from 20 → 3\.

   * Replace **host** with copied **private IP**.

6. **Private endpoint:** select existing one.

7. **SSL details:** Wallet \= None.

8. Click **Create → Validate** → should show *Connection successful*.

### **4\. Load Vector Data in Autonomous Database**

1. Go to **SQL Worksheet** for the database.

2. Execute SQL code blocks sequentially:

   * **Access Control List:** allow outbound connections for embedding model access.

   * **Credentials Setup:**

     * Create DBMS credentials for Generative AI Service using:

       * `user_ocid`, `tenancy_ocid`, `fingerprint`, and `private_key`.

   * **Embedding Test:**

     * Verify embedding generation for text “Hello”.

     * Uses `cohere.embed-multilingual-v3.0`.

     * Confirms embedding setup works.

### **5\. Prepare Data for Vectorization**

1. In **Object Storage**, locate bucket (e.g., `GenAI-Agents`) containing `faq.txt`.

2. Create **Preauthenticated Request (PAR)** link for the file and copy it.

3. Back in SQL Worksheet:

   * Run code to **chunk** data from `faq.txt` using `utl_to_chunks`.

   * Creates table **AI\_EXTRACTED\_DATA** with fields:

     * `chunk_id`, `chunk_offset`, `chunk_length`, `chunk_data`.

   * Confirm table creation and view sample rows.

4. Run SQL to create **AI\_EXTRACTED\_DATA\_VECTOR** table.

5. Insert vector embeddings using `cohere.embed-multilingual-v3`.

6. Execute and verify **retrieval\_func\_ai** creation (vector search function).

Test with query:

 SELECT \* FROM TABLE(retrieval\_func\_ai('Tell me about Oracle Free Tier Account', 10));

7.  Confirms retrieval works (shows 10 results).

### **6\. Verify Data in Tables**

* **AI\_EXTRACTED\_DATA:** stores text chunks.

* **AI\_EXTRACTED\_DATA\_VECTOR:** stores embeddings and corresponding text.

* Confirms vectorized data successfully loaded and retrievable.

### **7\. Create Knowledge Base (KB)**

1. Go to **Analytics & AI → Generative AI Agents → Knowledge Bases → Create Knowledge Base**.

2. Provide:

   * Name: `demo-knowledge-base`.

   * **Data Store Type:** Oracle AI Vector Search.

   * **Database Tool Connection:** `demoagent`.

   * **Test Connection:** should succeed.

   * **Vector Search Function:** `RETRIEVAL_FUNCTION_AI`.

3. Click **Create** and wait until KB status \= *Active*.

### **8\. Create Agent and Endpoint**

1. Under **Agents**, click **Create Agent**.

   * Name: `demo-agent`.

   * **Welcome Message:**  
      “Hi user, I am a chatbot for Oracle FAQs. Please ask me related questions.”

   * Select the knowledge base created above.

   * Enable **Auto-create Endpoint**.

2. Accept **License Agreement** and **Use Policy** → Submit.

3. Once agent status \= *Active*, proceed to Chat.

### **9\. Chat and Validate**

* Go to **Chat → Select Agent: demo-agent** and corresponding **Endpoint**.

* Interact via sample query:

  * **Q:** “Tell me about Oracle Free Tier.”

  * **A:** Agent responds with relevant answer plus **citations** and **trace logs**.

    * *Trace:* shows input and output.

    * *Citations:* show referenced document sources.

### **10\. Summary**

* Created and configured:

  * **Autonomous Database (23ai)**

  * **Database Tools Connection**

  * **Vectorized Data Tables**

  * **Knowledge Base** (Oracle AI Vector Search)

  * **Generative AI Agent \+ Endpoint**

* Demonstrated working **chatbot** integrated with Oracle Database vector search.

* SQL details are secondary; focus is on **OCI Generative AI Agent pipeline** and integration workflow.

* Recommended follow-up: **Oracle Database 23ai course** for deeper SQL understanding.

