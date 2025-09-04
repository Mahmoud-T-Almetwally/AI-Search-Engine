# Multi-Modal AI Search Engine

This project is a sophisticated, multi-modal search engine built with Python and Django. It goes beyond simple keyword matching by leveraging modern AI models to understand the semantic meaning of queries across text, image, and audio data. The entire application is containerized with Docker for easy setup and deployment.

## Core Features

- **Multi-Modal Search:** Perform searches using text, images, or audio to find relevant content of any type.
- **Dual Search Paradigms:**
  - **AI-Powered Semantic Search:** Understands the *meaning* behind a query (e.g., searching for "car" can find images of an "automobile").
  - **Blazing-Fast Keyword Search:** Utilizes PostgreSQL's Full-Text Search for instant, precise keyword matching across all indexed text, including image alt-text and audio transcripts.
- **Asynchronous Data Processing:** A robust, Celery-based task queue handles the slow processes of web crawling and data indexing in the background, ensuring the API remains fast and responsive.
- **Scalable Architecture:** Built on a modular, containerized architecture using Django, Celery, PostgreSQL, and Redis, demonstrating professional development practices.

## Tech Stack & Design

### Backend & Frameworks
- **Python** & **Django**: The core framework for the web application and APIs.
- **Django Rest Framework (DRF)**: For building clean and powerful RESTful APIs.
- **PostgreSQL**: The primary relational database.
- **pgvector**: PostgreSQL extension for efficient vector similarity search.
- **Redis**: The message broker for the Celery task queue.
- **Celery**: A distributed task queue for asynchronous crawling and indexing.
- **Gunicorn**: Production-ready WSGI web server.

### AI & Data Processing
- **Sentence-Transformers**: For generating high-quality text embeddings (`all-MiniLM-L6-v2`).
- **OpenAI CLIP**: For creating joint text and image embeddings, enabling text-to-image and image-to-image search (`clip-vit-base-patch32`).
- **LAION CLAP**: For creating joint text and audio embeddings, enabling text-to-audio and audio-to-audio search (`clap-htsat-unfused`).
- **OpenAI Whisper**: For Automatic Speech Recognition (ASR) to generate transcripts from audio files for keyword search (`whisper-base`).
- **Beautiful Soup**: For parsing HTML during the web crawling process.

### DevOps
- **Docker & Docker Compose**: For containerizing the entire application stack, ensuring a consistent and reproducible development environment.

## How to Run the Project

This project is fully containerized, so all you need is **Docker** and **Docker Compose** installed on your machine.

### 1. Initial Setup

**Clone the repository:**
```bash
git clone https://github.com/Mahmoud-T-Almetwally/AI-Search-Engine.git
cd AI-Search-Engine/ai_search_engine_project
```
**Create the environment file:**
Copy the example environment file and fill in your desired database credentials.
```bash
cp .env.example .env
```
*(You can edit the `.env` file with your preferred username, password, and database name.)*

### 2. Build and Run the Containers

This single command will build the Django image and start all the services (web app, database, Redis, and Celery worker) in the background.

```bash
docker-compose up --build -d
```

### 3. Test Everything is Working

To test the application is running and functional, run the following command.

```bash
# Apply migrations (creates the tables and vector extension)
docker-compose exec app python manage.py test
```

### 4. Start Crawling and Indexing

Use the `run_crawler` management command to start populating your search engine. The crawler will find content and dispatch indexing tasks to your Celery worker.

```bash
# Example: Crawl up to 20 pages starting from a seed URL
docker-compose exec app python manage.py run_crawler http://books.toscrape.com --limit=20
```
You can monitor the progress of your Celery worker with:
```bash
docker-compose logs -f celery_worker
```

### 5. Use the API

The search API will be available at `http://localhost:8000/api/`.

**AI-Powered Semantic Search:**
```bash
# Text-to-Text Search
curl "http://localhost:8000/api/search/ai/?type=text&q=a+story+about+love"

# Text-to-Image Search
curl "http://localhost:8000/api/search/ai/?type=image&q=a+girl+with+a+red+hat"
```

**Fast Keyword Search:**
```bash
# Search for the exact word "mystery" across all content types
curl "http://localhost:8000/api/search/keyword/?q=mystery"
```

### Stopping the Application

To stop all running containers:
```bash
docker-compose down
```
To stop and remove the database volume (deletes all data):
```bash
docker-compose down -v
```
