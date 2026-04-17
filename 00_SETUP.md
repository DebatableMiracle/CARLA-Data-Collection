# CARLA + OpenVLA Data Collection Setup
## Ubuntu 24.04 | Dual RTX 6000 (using GPU 0) | Single-GPU Mode

---

## STEP 1: System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    wget curl git unzip \
    libvulkan1 vulkan-tools \
    libomp5 \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    libpng16-16 libjpeg8 libtiff5 \
    libsdl2-2.0-0 libsdl2-image-2.0-0 \
    xserver-xorg mesa-utils \
    ffmpeg

# Pin GPU 0 for CARLA (ignore GPU 1 for now)
export CUDA_VISIBLE_DEVICES=0
```

---

## STEP 2: Install CARLA 0.9.15 (Latest Stable)

```bash
mkdir -p ~/carla && cd ~/carla

# Download CARLA prebuilt binary
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
tar -xvzf CARLA_0.9.15.tar.gz

# Also grab additional maps (Towns 01-07 + more)
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.15.tar.gz
tar -xvzf AdditionalMaps_0.9.15.tar.gz -C ~/carla/

# Confirm structure
ls ~/carla/
# Should see: CarlaUE4.sh, PythonAPI/, Import/, Engine/, etc.
```

---

## STEP 3: Python Environment

```bash
python3.10 -m venv ~/venvs/carla_vla
source ~/venvs/carla_vla/bin/activate

# CARLA Python API (must match server version 0.9.15)
pip install carla==0.9.15

# Data collection deps
pip install numpy opencv-python-headless Pillow tqdm \
            transforms3d scipy h5py pyarrow \
            torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## STEP 4: Launch CARLA Server (Headless, GPU 0 Only)

```bash
# Terminal 1 — Start CARLA server
cd ~/carla

# Headless (no display) with GPU 0, fixed timestep
CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh \
    -RenderOffScreen \
    -quality-level=Epic \
    -fps=20 \
    -carla-server \
    -benchmark \
    2>&1 | tee carla_server.log &

# Wait ~15 seconds for it to boot, then verify
sleep 15
python3 -c "import carla; c = carla.Client('localhost', 2000); c.set_timeout(10); print('CARLA version:', c.get_server_version())"
```

> ⚠️ If you see "Vulkan" errors, add `-opengl` flag after `-RenderOffScreen`

---

## STEP 5: Project Structure

```
carla_vla/
├── collect_data.py          ← Main data collection script
├── sensor_config.py         ← Sensor suite definition
├── autopilot_controller.py  ← Autopilot + perturbation logic
├── data_writer.py           ← HDF5/parquet writer for OpenVLA format
├── run_collection.sh        ← Launch script
└── data/
    └── episodes/            ← Collected episodes go here
```

---

## STEP 6: Verify GPU usage

```bash
# While CARLA is running:
nvidia-smi -l 2   # Should show GPU 0 at high utilization, GPU 1 idle
```

Now proceed to run `collect_data.py` (see the Python scripts).
