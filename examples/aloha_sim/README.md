# Run Aloha Sim

This example demonstrates running π₀ and π₀.₅ models on the ALOHA simulator for cube transfer tasks.

## Available Configurations

- **pi0_aloha_sim**: π₀ model fine-tuned for ALOHA simulator
- **pi05_aloha_sim**: π₀.₅ model fine-tuned for ALOHA simulator (with improved generalization)

For detailed information about the model architectures, see [Model Architecture Documentation](../../docs/model_architecture.md).

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server with π₀ model
uv run scripts/serve_policy.py --env ALOHA_SIM

# Or run with π₀.₅ model
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_aloha_sim --policy.dir=<checkpoint_path>
```

## Training

To fine-tune the π₀.₅ model on ALOHA simulator data:

```bash
# Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_sim

# Run training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_sim --exp-name=my_experiment --overwrite
```
