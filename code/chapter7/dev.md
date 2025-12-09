# SGLang Development Guide

## Dev Container

SGLang has a native dev container with a comprehensive set of useful tools installed. 

The dev container Dockerfile is available at: https://github.com/sgl-project/sglang/blob/main/.devcontainer/Dockerfile

### Starting the Dev Container

#### 1. Pull the Docker Container

```bash
docker pull lmsysorg/sglang:dev
```

#### 2. Start the Container

**Important:** Mount essential Docker volumes for models and data.

```bash
docker run -itd \
  --gpus "device=0,1,2,3,4,5,6,7" \
  --shm-size 10g \
  -v /raid/data/models:/models \
  --ulimit nofile=65535:65535 \
  --network host \
  --name sglang-dev \
 	lmsysorg/sglang:dev 
```

### Attaching VS Code to the Container

After starting the Docker container, follow the VS Code guide to attach to the dev container:

📖 https://code.visualstudio.com/docs/devcontainers/attach-container

**Workspace folder:** `/sgl-workspace/sglang`

### Installing SGLang from Source

Once attached to the dev container, install SGLang from source with editable mode:

📖 https://docs.sglang.ai/get_started/install.html#method-2-from-source

**Note:** The container has oh-my-zsh pre-installed. Use `zsh` instead of `bash` for better terminal experience.

```bash
# Switch to zsh
zsh

# Navigate to workspace
cd /sgl-workspace/sglang

# Checkout desired version
git checkout v0.4.2.post1  # or whatever desired version

# Upgrade pip
pip install --upgrade pip

# Install SGLang in editable mode
pip install -e "python"
```

## Code Formatting

After debugging or development, use SGLang's official pre-commit hooks to ensure all files comply with formatting standards:
 
```bash
pre-commit run --all-files
```

## VS Code Launch Configuration

Create a `launch.json` file under `.vscode/launch.json` to enable VS Code Python Debugger.

### Example Configuration

```json
{
	"version": "0.2.0",
	"configurations": [
    	{
        	"name": "Launch SGLang Server - Llama-3.1-8B-Instruct",
        	"type": "debugpy",
        	"request": "launch",
        	"module": "sglang.launch_server",
        	"justMyCode": false,
        	"env": {
            	"PYTHONPATH": "${workspaceFolder}",
                          	"CUDA_VISIBLE_DEVICES": "5,6",
        "HF_TOKEN": "your_huggingface_token_here"
            },
        	"args": [
            	"--model-path",
            	"/models/Llama-3.1-8B-Instruct",
            	"--port",
            	"8081",
            	"--host",
            	"0.0.0.0",
            	"--log-level",
            	"info",
            	"--tp",
            	"1",
        "--enable-metrics"
        // Uncomment as needed:
            	// "--log-requests",
        // "--is-embedding"
        	]
    	}
    	]
}
```

### Launch Configuration Parameters

- `--model-path`: Path to the model directory
- `--port`: Server port (default: 30000)
- `--host`: Host address (0.0.0.0 for all interfaces)
- `--log-level`: Logging level (debug, info, warning, error)
- `--tp`: Tensor parallelism size
- `--enable-metrics`: Enable Prometheus metrics
- `--log-requests`: Log all incoming requests (optional)
- `--is-embedding`: Run as embedding model (optional)

## Quick Reference

### Common Commands

```bash
# Start server manually
python -m sglang.launch_server --model-path /models/model-name

# Run tests
pytest tests/

# Check formatting
pre-commit run --all-files

# View logs
tail -f logs/sglang.log
```

### Useful Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control which GPUs to use
- `HF_TOKEN`: Hugging Face authentication token
- `PYTHONPATH`: Python module search path
- `SGLANG_LOG_LEVEL`: Override default log level
