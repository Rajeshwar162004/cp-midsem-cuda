cd gpu-memory-toolkit

# Optional: confirm CUDA tools and your GPU are visible
nvidia-smi
nvcc --version
ncu --version


# Global-memory access: bad vs good
sudo -E env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 profiler/profile.py benchmarks/global_access/bad_global.cu --arch sm_89
sudo chown -R "$USER:$USER" results/bad_global

sudo -E env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 profiler/profile.py benchmarks/global_access/good_global.cu --arch sm_89
sudo chown -R "$USER:$USER" results/good_global

# Shared-memory bank conflicts: bad vs good
sudo -E env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 profiler/profile.py benchmarks/bank_conflict/bad_bank.cu --arch sm_89
sudo chown -R "$USER:$USER" results/bad_bank

sudo -E env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 profiler/profile.py benchmarks/bank_conflict/good_bank.cu --arch sm_89
sudo chown -R "$USER:$USER" results/good_bank

# Register spilling: bad vs good
sudo -E env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 profiler/profile.py benchmarks/register_spill/bad_spill.cu --arch sm_89
sudo chown -R "$USER:$USER" results/bad_spill

sudo -E env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 profiler/profile.py benchmarks/register_spill/good_spill.cu --arch sm_89
sudo chown -R "$USER:$USER" results/good_spill