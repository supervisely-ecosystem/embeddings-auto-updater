<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/embeddings-auto-updater/releases/download/v0.1.0/poster.jpg"/>

# Embeddings Auto-Updater

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#Configuration">Configuration</a> •
  <a href="#Features">Features</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/embeddings-auto-updater)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/embeddings-auto-updater)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/embeddings-auto-updater.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/embeddings-auto-updater.png)](https://supervisely.com)

</div>

# Overview

🧩 This application is a part of the **AI Search** feature in Supervisely and is designed to enhance the capabilities of the **Embeddings Generator** app.

**Embeddings Auto-Updater** is a microservice that automatically generates and updates embeddings for project images in Supervisely projects with AI Search enabled. This application ensures that your image embeddings are always up-to-date, enabling efficient similarity search and AI-powered image discovery.

Application key points:

- **Automatic embeddings generation** for projects with AI Search enabled
- **Scheduled updates** with configurable intervals
- **Integration with Qdrant** vector database for embedding storage
- **CLIP Service integration** for high-quality image embeddings
- **Smart update logic** - only processes new or modified images

The service continuously monitors projects with AI Search enabled and automatically:
1. Generates embeddings for new images
2. Updates embeddings for modified images  
3. Removes embeddings for deleted images
4. Maintains synchronization between Supervisely and Qdrant

## Architecture

The application uses a scheduler-based architecture that periodically checks for projects requiring embedding updates:

- **Scheduler**: Uses AsyncIOScheduler to run updates at configurable intervals
- **CLIP Service**: Generates high-quality image embeddings using CLIP models
- **Qdrant Integration**: Stores and manages vector embeddings efficiently
- **Smart Filtering**: Only processes images that need updates based on timestamps

## How To Run

**Prerequisites:**

- Supervisely instance with admin access
- Running CLIP as Service instance (task ID)
- Qdrant vector database instance (URL)

When launching the service, configure these settings in the modal dialog:

1. **Qdrant DB**: Full URL including protocol (https/http) and port (e.g., `https://192.168.1.1:6333`).
2. **CLIP Service**: Task ID for CLIP as Service session or its host including port (e.g., `1234` or `https://192.168.1.1:51000`).
3. **Update Interval**: Set the interval in minutes for updating embeddings (1-1440 minutes, default: 10)

After configuration, click "Run" to deploy the service. The application will start in headless mode and will automatically start monitoring and updating embeddings for all projects with AI Search enabled.

![configuration](https://github.com/supervisely-ecosystem/embeddings-auto-updater/releases/download/v0.1.0/how_to_run.jpg)

---

For technical support and questions, please join our [Supervisely Ecosystem Slack community](https://supervisely.com/slack).
