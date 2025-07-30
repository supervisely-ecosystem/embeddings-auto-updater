<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/embeddings-auto-updater/releases/download/v0.1.0/poster.jpg" alt="Embeddings Auto-Updater Poster"/>

# Embeddings Auto-Updater

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#deployment">Deployment</a> •
  <a href="#monitoring">Monitoring</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/embeddings-auto-updater)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)

</div>

# Overview

🧩 **Embeddings Auto-Updater** is a **system-level microservice** designed to enhance the **AI Search** feature in Supervisely by automatically generating and updating embeddings for project images.

Key features:

-   **Instance-level service**: Runs as a system container for the entire Supervisely instance.
-   **Automatic embeddings generation**: For all projects with AI Search enabled.
-   **Scheduled updates**: Configurable intervals for embedding updates.
-   **Integration with Qdrant**: Efficient vector database for embedding storage.
-   **CLIP Service integration**: High-quality image embeddings.
-   **Smart update logic**: Processes only new or modified images.
-   **Zero-downtime operation**: Runs continuously in the background.

The service ensures synchronization between Supervisely and Qdrant by:

1. Generating embeddings for new images.
2. Updating embeddings for modified images.
3. Removing embeddings for deleted images.

## Architecture

The application uses a scheduler-based architecture to periodically check for projects requiring embedding updates:

-   **Containerized Service**: Runs as a Docker container at the instance level.
-   **Scheduler**: Uses `AsyncIOScheduler` for configurable update intervals.
-   **CLIP Service**: Generates high-quality embeddings using CLIP models.
-   **Qdrant Integration**: Efficiently stores and manages vector embeddings.
-   **Smart Filtering**: Processes only images needing updates based on timestamps.
-   **Multi-project Support**: Handles multiple projects concurrently.

## Deployment

### Prerequisites

-   Supervisely instance with admin access.
-   Docker environment for container deployment.
-   Running CLIP as Service instance (task ID or service endpoint).
-   Qdrant vector database instance (URL).

### Environment Variables

Configure the service using the environment variables in `docker-compose.yml`.

### Configuration

-   **Qdrant DB**: Full URL including protocol (https/http) and port (e.g., `https://192.168.1.1:6333`).
-   **CLIP Service**: Task ID for CLIP as Service session or its host including port (e.g., `1234` or `https://192.168.1.1:51000`).
-   **Update Interval**: Set the interval in minutes for updating embeddings (1-1440 minutes, default: 10).

The service starts automatically on instance startup and continuously monitors all projects with AI Search enabled.

## Monitoring

The service provides robust logging and monitoring capabilities:

-   **Health checks**: Built-in health check endpoint.
-   **Detailed logging**: Configurable log levels for debugging and monitoring.
-   **Error handling**: Robust error handling with retry mechanisms.

### API Endpoints

The service exposes several API endpoints for monitoring and management:

#### GET /health

Health check endpoint that verifies the status of all connected services (Qdrant, CLIP, Generator).

#### POST /stop-embeddings-update/{project_id}

Stop embeddings update task for a specific project only if it's currently the active task.

**Parameters:**

-   `project_id` (path parameter): ID of the project to stop

**Response when project is current task:**

```json
{
	"project_id": 12345,
	"is_current_task": true,
	"stopped": true,
	"message": "Current auto update task stopped successfully.",
	"details": {
		"cancel_task": true,
		"clear_current_task": true,
		"clear_in_progress_flag": true,
		"clear_update_flag": true,
		"clear_autorestart": true
	}
}
```

**Response when project is not current task:**

```json
{
	"project_id": 12345,
	"is_current_task": false,
	"stopped": false,
	"message": "[Project: 12345] Is not the current active task in the auto-updater.",
	"details": {}
}
```

---

For technical support and questions, join our [Supervisely Ecosystem Slack community](https://supervisely.com/slack).
