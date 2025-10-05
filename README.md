This module demonstrates how to package and deploy AI model APIs using Docker containers, building on the FastAPI server from Module 3.

## Container Architecture

```
Docker Container
├── Python 3.11-slim base image
├── FastAPI application (server.py)
├── AI Model (ResNet-18, loaded on startup)
└── SQLite Database (mounted from host)
```

The container exposes port 8000 and uses a mounted volume for database persistence across container restarts.

## Quick Start

### Option 1: Pull from Docker Hub

I pushed a pre-built container onto Docker Hub, so you can simply pull it:

```bash
docker pull yanlincs/image-classify-api:v1.0

# Run with port and volume mapping
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --name ai-server \
  yanlincs/image-classify-api:v1.0
```

Note that this image only supports `linux/amd64` architecture, which might be not compatible with the computer you have. In that case, you will have to build the container yourself.

### Option 2: Build Locally

```bash
# Build the image
docker build -t image-classify-api:v1.0 .

# Run the container
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --name ai-server \
  image-classify-api:v1.0
```

## Running the Container

### Basic Run (No Persistence)

```bash
docker run -p 8000:8000 image-classify-api:v1.0
```

This runs the container with port mapping but without database persistence. The database is lost when the container stops.

### Production Run (With Persistence)

```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --name ai-server \
  image-classify-api:v1.0
```

**Flags explained:**
- `-d`: Run in detached mode (background)
- `-p 8000:8000`: Map host port 8000 to container port 8000
- `-v $(pwd)/data:/app/data`: Mount local `data/` directory to container's `/app/data`
- `--name ai-server`: Name the container for easy management

## Container Management

### View logs
```bash
docker logs ai-server
docker logs -f ai-server  # Follow logs in real-time
```

### Stop the container
```bash
docker stop ai-server
```

### Start the container
```bash
docker start ai-server
```

### Remove the container
```bash
docker rm -f ai-server
```

## Testing the API

Once the container is running, the API is accessible at `http://localhost:8000`.

You can use the `client.py` from Module 3 (which I also include in this repository) to interact with the containerized server:

```bash
python client.py meal.png
```

## Pushing to Docker Hub

If you want to share your own version:

```bash
# Tag the image
docker tag image-classify-api:v1.0 YOUR_USERNAME/image-classify-api:v1.0

# Login to Docker Hub
docker login

# Push the image
docker push YOUR_USERNAME/image-classify-api:v1.0
```
