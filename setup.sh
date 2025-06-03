#!/bin/bash
# setup.sh: Setup Sigma Security for bug-catching in Termux or VM
# Squashes setup bugs with sigma discipline

# Update Termux and install basics
pkg update && pkg upgrade -y
pkg install python docker git -y

# Fix Termux storage issues
termux-setup-storage
mkdir -p /data/data/com.termux/files/home/storage
ln -sf /data/data/com.termux/files/home/storage ~/storage

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# Setup Docker
systemctl start docker || true  # Skip if not VM
docker build -t sigma-security .

echo "Sigma Security setup complete. Run 'python src/main.py' to catch bugs!"
