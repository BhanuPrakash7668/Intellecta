# Research Intelligence System

An advanced multi-agent system designed to automate and enhance the research process. This system leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to retrieve, analyze, and synthesize information from multiple sources, producing structured research reports.

## ✨ Features

-   **Multi-Agent Workflow:** Orchestrates a pipeline of specialized AI agents (e.g., Literature Retriever, Analysis Agent) using LangGraph for complex, stateful workflows.
-   **Conditional Routing:** Dynamically routes tasks between agents based on the current state and findings.
-   **RAG-Powered Analysis:** Utilizes a RAG pipeline with ChromaDB for deep, context-aware analysis of retrieved documents.
-   **Human-in-the-Loop:** Includes checkpoints for human review and intervention, ensuring accuracy and control over the research process.
-   **Stateful & Resilient:** Employs SQLite for workflow checkpointing, allowing for error recovery and resumption of long-running tasks.
-   **Structured Output:** Synthesizes findings from all agents into a coherent, structured final report.
-   **Containerized & Scalable:** Packaged with Docker and Docker Compose for easy setup and deployment.

## 🛠️ Tech Stack

-   **Backend:** FastAPI
-   **AI/LLM Orchestration:** LangChain, LangGraph
-   **LLM Provider:** OpenAI (configurable)
-   **Vector Store:** ChromaDB
-   **Database:** PostgreSQL
-   **Containerization:** Docker, Docker Compose
-   **Monitoring:** LangSmith (for tracing)

## 🏛️ System Architecture

The system operates as a stateful graph where each node represents an agent or a specific tool.

1.  **Initiation:** A user submits a research topic or question via the API.
2.  **Literature Retrieval:** The `Literature Retriever` agent is triggered. It queries various sources to gather relevant articles, papers, and documents.
3.  **Analysis:** The retrieved documents are passed to the `Analysis Agent`. This agent uses a RAG pipeline to chunk, embed, and store the documents in ChromaDB. It then performs in-depth analysis to extract key insights, arguments, and data based on the initial query.
4.  **Conditional Logic:** The workflow graph includes conditional edges. For example, if the analysis is deemed insufficient, the graph can route back to the retrieval agent to find more sources.
5.  **Human-in-the-Loop:** At critical junctures, the workflow can pause and wait for human input. This allows a user to validate findings, adjust the research direction, or approve the intermediate results before proceeding.
6.  **Report Synthesis:** Once the analysis is complete and approved, a final agent synthesizes all the gathered information into a structured, comprehensive report.
7.  **Persistence:** The entire state of the workflow is checkpointed using a SQLite backend, ensuring that long-running research tasks can be recovered in case of failure.

## 🚀 Getting Started

Follow these instructions to get the project running locally.

### Prerequisites

-   Docker
-   Docker Compose
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd research-intelligence-system
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the root of the project by copying the contents below. You will need to add your OpenAI API key.

    ```env
    # API Configuration
    API_PORT=8000

    # PostgreSQL Configuration
    POSTGRES_DB=ris_db
    POSTGRES_USER=ris_user
    POSTGRES_PASSWORD=your_strong_password
    POSTGRES_PORT=5432

    # ChromaDB Configuration
    CHROMA_PORT=8001

    # OpenAI API Key
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # LangSmith (Optional, for tracing and debugging)
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_API_KEY=ls__xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # LANGCHAIN_PROJECT=research-intelligence-system
    ```

3.  **Build and Run with Docker Compose:**
    This command will build the Docker images and start the API, PostgreSQL, and ChromaDB services.

    ```bash
    docker-compose up --build -d
    ```

    The `-d` flag runs the containers in detached mode.

## Usage

Once the services are running, the FastAPI application will be accessible.

-   **API Docs:** http://localhost:8000/docs

You can use the interactive Swagger UI to explore and test the API endpoints for initiating research tasks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request.

1.  Fork the repository .
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

