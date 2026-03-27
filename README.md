# Green volume estimation pipeline for TLS data 

## Docker & Automation

The pipeline includes Docker support for containerized execution and automated workflows.

### Docker Setup

Build the Docker image:

```bash
sudo docker build -t green-volume-pipeline -f docker/Dockerfile .
```

Run the pipeline in Docker:

```bash
./run_docker.sh                    
```

Edit the path variables at the top of each script to match your data.