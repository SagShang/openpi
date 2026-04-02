# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

Before running the commands below, install Docker and the NVIDIA container
toolkit as described in `docs/docker.md`. On Ubuntu 22.04, you can use
`scripts/docker/install_docker_ubuntu22.sh` and
`scripts/docker/install_nvidia_container_toolkit.sh`.

```bash
# If you are behind an HTTP proxy, export it first. For example:
# proxy 7890
#
# Point Docker at the current X11 auth cookie. This works for both local
# desktop sessions and SSH X11 forwarding.
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx instead (use this if you have egl errors):
MUJOCO_GL=glx PYOPENGL_PLATFORM=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

If you are on a local desktop X11 session and prefer using `xhost`, install it on
Ubuntu with `sudo apt install x11-xserver-utils` and run `xhost +local:docker`
without `sudo`. For SSH-forwarded X11 sessions such as `DISPLAY=localhost:10.0`,
`xhost +local:docker` is not the right mechanism; use the `XAUTHORITY` setup above.

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
