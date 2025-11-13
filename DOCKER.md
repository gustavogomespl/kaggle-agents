# Docker Guide for Kaggle Agents

Quick start guide for running Kaggle Agents with Docker.

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)
- At least 8GB RAM available
- 10GB free disk space

## Quick Start

### 1. Set Up Environment

Create a `.env` file with your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional - for Kaggle submissions
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
KAGGLE_AUTO_SUBMIT=false

# Optional - configuration
LLM_MODEL=gpt-4o-mini
MAX_ITERATIONS=3
TARGET_PERCENTILE=20.0
```

### 2. Build the Image

```bash
docker-compose build
```

This creates an optimized Docker image (~5-6GB).

### 3. Run a Competition

```bash
docker-compose run --rm kaggle-agents \
  kaggle-agents start titanic \
  --problem-type binary_classification \
  --metric accuracy \
  --max-iterations 3
```

### 4. Interactive Shell

For interactive usage:

```bash
docker-compose run --rm kaggle-agents /bin/bash
```

Then inside the container:

```bash
# Show configuration
kaggle-agents config

# Start a competition
kaggle-agents start house-prices-advanced-regression-techniques \
  --problem-type regression \
  --metric rmse
```

## CLI Commands

### Start Competition

```bash
kaggle-agents start <competition-name> [OPTIONS]
```

Options:
- `--description, -d`: Competition description
- `--problem-type, -p`: Problem type (binary_classification, multiclass_classification, regression)
- `--metric, -m`: Evaluation metric
- `--max-iterations, -i`: Maximum iterations (default: 3)

Example:

```bash
kaggle-agents start digit-recognizer \
  --problem-type multiclass_classification \
  --metric accuracy \
  --max-iterations 5
```

### Show Configuration

```bash
kaggle-agents config
```

Displays current settings and environment variables.

## Data Persistence

Docker volumes are automatically created for:

- `./data` - Competition data and working files
- `./models` - Trained models
- `./logs` - Execution logs
- `./submissions` - Generated submission files
- `./.cache` - Embeddings and vector database cache

These directories persist between container runs.

## Advanced Usage

### Custom Resource Limits

Edit `docker-compose.yml` to adjust CPU and memory limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Increase for more CPU
      memory: 16G    # Increase for more RAM
```

### Using GPU

For GPU support (NVIDIA only):

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml`:

```yaml
services:
  kaggle-agents:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Rebuild and run:

```bash
docker-compose build
docker-compose run --rm kaggle-agents kaggle-agents start <competition>
```

### Development Mode

Mount source code for live development:

```bash
docker-compose run --rm \
  -v $(pwd)/kaggle_agents:/app/kaggle_agents \
  kaggle-agents /bin/bash
```

## Troubleshooting

### Build Fails

**Problem**: Docker build fails with "no space left on device"

**Solution**: Clean up Docker system:

```bash
docker system prune -a
docker volume prune
```

### Out of Memory

**Problem**: Container killed due to OOM

**Solution**: Increase memory limit in docker-compose.yml or system Docker settings.

### API Keys Not Working

**Problem**: API keys not recognized inside container

**Solution**:
1. Ensure `.env` file exists in project root
2. Check file is not empty
3. Verify no extra quotes around values

### Slow Performance

**Problem**: Workflow runs slowly

**Solution**:
1. Increase CPU/memory limits
2. Use SSD for volume mounts
3. Enable GPU support if available

## Cleaning Up

Remove all data:

```bash
docker-compose down -v
rm -rf data/ models/ logs/ submissions/ .cache/
```

Remove images:

```bash
docker-compose down --rmi all
```

## Production Deployment

For production use:

1. Use specific version tags instead of `latest`
2. Set up proper logging (mount `/app/logs`)
3. Configure restart policies
4. Use secrets management for API keys
5. Set resource limits appropriately
6. Monitor container health

Example production `docker-compose.yml`:

```yaml
services:
  kaggle-agents:
    image: kaggle-agents:1.0.0
    restart: always
    env_file: .env
    volumes:
      - /mnt/data:/app/data
      - /mnt/logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 16G
    healthcheck:
      interval: 60s
      timeout: 30s
      retries: 3
```

## Security Notes

- Never commit `.env` file to version control
- Use read-only mounts when possible
- Run container as non-root user in production
- Regularly update base images
- Scan images for vulnerabilities

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- GitHub Issues: https://github.com/yourusername/kaggle-agents/issues
- Documentation: See README.md
