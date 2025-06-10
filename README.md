<div align="center" markdown>
<img src=""/>

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

# How To Run

**Step 1.** Deploy the application from the Ecosystem and configure the required services.

**Step 2.** Fill in the configuration parameters:

- **Qdrant DB**: Enter the host of your Qdrant database including port (e.g., `https://192.168.1.1:6333`)
- **CLIP Service**: Enter Task ID of the CLIP Service or its host including port (e.g., `1234` or `https://192.168.1.1:51000`)
- **Update Interval**: Set the interval in minutes for updating embeddings (1-1440 minutes, default: 10)

![configuration]()

**Step 3.** Press the `Run` button to start the service.

**Step 4.** The service will automatically start monitoring and updating embeddings for all projects with AI Search enabled.

![deployed]()

# Configuration

## Environment Variables

The application can be configured using the following environment variables:

- `QDRANT_HOST`: Qdrant database host URL
- `CAS_HOST`: CLIP Service host URL or Task ID  
- `UPDATE_INTERVAL`: Update interval in minutes

## Modal Configuration

When running the application, you can configure:

- **Qdrant DB Host**: Full URL including protocol and port
- **CLIP Service**: Either a Task ID for running CLIP service or direct host URL
- **Update Interval**: How frequently to check for embedding updates (1-1440 minutes)

# Features

## Automatic Project Discovery
- Scans for projects with AI Search enabled
- Monitors project updates and new image additions

## Smart Update Logic
- **Incremental updates**: Only processes new or modified images
- **Timestamp-based filtering**: Uses project and image timestamps to determine what needs updating
- **Force update option**: Can regenerate all embeddings when needed
- **Cleanup handling**: Removes embeddings for deleted images

## Integration Features
- **Qdrant Vector Database**: Efficient storage and retrieval of image embeddings
- **CLIP Service**: High-quality image encoding using state-of-the-art models
- **Batch Processing**: Processes images in batches for optimal performance
- **Error Handling**: Robust retry mechanisms and error recovery

## Performance Optimizations
- **Async Processing**: Non-blocking operations for better performance
- **Batch Operations**: Groups operations for efficiency
- **Smart Scheduling**: Prevents overlapping jobs and manages resource usage
- **Progress Tracking**: Detailed logging and progress monitoring

## Plan Management
- **Free Plan Detection**: Automatically disables embeddings for free plan teams
- **Usage Monitoring**: Respects plan limitations and quotas
- **Graceful Degradation**: Handles plan changes smoothly

The service runs continuously in the background, ensuring your AI Search capabilities are always powered by up-to-date embeddings without manual intervention.