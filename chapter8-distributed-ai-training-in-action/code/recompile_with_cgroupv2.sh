#!/bin/bash
# Recompile Slurm with cgroup v2 support (even without BPF tokens)
# This will build cgroup v2 plugin with limited device constraint functionality

set -e

cd /home/fuhwu/workspace/distributedai/resources/slurm

echo "=========================================="
echo "Recompiling Slurm with cgroup v2 support"
echo "=========================================="
echo ""
echo "Requirements:"
echo "  - Your kernel (5.15) uses cgroup v2 ✅"
echo "  - BPF tokens: NOT available (requires kernel 6.9+)"
echo "    → Device constraints will be limited"
echo "  - dbus-1-dev: Required for cgroup v2 plugin"
echo ""

# Check for dbus
echo "=== Step 0: Checking dependencies ==="
if ! pkg-config --exists dbus-1 2>/dev/null; then
    echo "❌ dbus-1 development headers not found"
    echo ""
    echo "Please install:"
    echo "  sudo apt-get install libdbus-1-dev"
    echo ""
    read -p "Install now? (requires sudo) [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get install -y libdbus-1-dev
    else
        echo "Exiting. Please install libdbus-1-dev and run again."
        exit 1
    fi
else
    echo "✅ dbus-1 found: $(pkg-config --modversion dbus-1)"
fi

# Clean previous build
echo ""
echo "=== Step 1: Cleaning previous build ==="
make clean 2>/dev/null || true
echo "✅ Cleaned"

# Reconfigure with cgroup v2 explicitly enabled
echo ""
echo "=== Step 2: Configuring with cgroup v2 ==="
./configure \
  --prefix=/home/fuhwu/slurm \
  --sysconfdir=/home/fuhwu/slurm/etc \
  --with-munge \
  --enable-multiple-slurmd \
  --enable-cgroupv2

echo ""
echo "=== Step 3: Checking cgroup v2 configuration ==="
if grep -q "WITH_CGROUPV2" config.h 2>/dev/null; then
    echo "✅ cgroup v2 is configured"
    grep "WITH_CGROUPV2\|HAVE_BPF_TOKENS" config.h | head -5
else
    echo "⚠️  cgroup v2 might not be enabled, but continuing..."
fi

# Build
echo ""
echo "=== Step 4: Building Slurm (this may take 10-15 minutes) ==="
make -j$(nproc)

# Install
echo ""
echo "=== Step 5: Installing ==="
make install

# Check if cgroup v2 plugin was built
echo ""
echo "=== Step 6: Verifying cgroup v2 plugin ==="
if [ -f /home/fuhwu/slurm/lib/slurm/cgroup_v2.so ]; then
    echo "✅ cgroup v2 plugin built successfully!"
    ls -lh /home/fuhwu/slurm/lib/slurm/cgroup_v2.so
else
    echo "❌ cgroup v2 plugin not found"
    echo "   Available cgroup plugins:"
    ls -la /home/fuhwu/slurm/lib/slurm/cgroup*.so 2>/dev/null || echo "   (none found)"
fi

echo ""
echo "=========================================="
echo "Recompile complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update cgroup.conf to use cgroup/v2:"
echo "   echo 'CgroupVersion=2' >> /home/fuhwu/slurm/etc/cgroup.conf"
echo ""
echo "2. Restart Slurm daemons:"
echo "   bash /home/fuhwu/workspace/distributedai/code/chapter8/slurm_setup.sh"

