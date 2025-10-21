#!/bin/bash
# Copy best model to VM
# Usage: ./copy_model_to_vm.sh [vm_host]

VM_HOST=${1:-"root@serv-3307"}
MODEL_FILE="data/ball_action/experiments/ball_finetune_long_004/fold_5/model-006-0.901643.pth"
CONFIG_FILE="data/ball_action/experiments/ball_finetune_long_004/config.json"
SOURCE_FILE="data/ball_action/experiments/ball_finetune_long_004/source.py"

echo "============================================================"
echo "Copying best model to VM: $VM_HOST"
echo "============================================================"
echo ""

# Create directory on VM
echo "Creating directory structure on VM..."
ssh "$VM_HOST" "mkdir -p /workspace/ball-action-spotting/data/ball_action/experiments/ball_finetune_long_004/fold_5"

# Copy model file
echo ""
echo "Copying model file (53MB)..."
scp "$MODEL_FILE" "$VM_HOST:/workspace/ball-action-spotting/$MODEL_FILE"

# Copy config and source
echo ""
echo "Copying config files..."
scp "$CONFIG_FILE" "$VM_HOST:/workspace/ball-action-spotting/$CONFIG_FILE"
scp "$SOURCE_FILE" "$VM_HOST:/workspace/ball-action-spotting/$SOURCE_FILE"

echo ""
echo "============================================================"
echo "✅ Model copied successfully!"
echo "============================================================"
echo ""
echo "On VM, verify with:"
echo "  ls -lh /workspace/ball-action-spotting/$MODEL_FILE"
echo ""
echo "Then run:"
echo "  python mvp/run_mvp.py video.mp4 19043 $MODEL_FILE"
echo ""
