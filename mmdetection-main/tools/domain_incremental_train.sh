#!/bin/bash
# Domain Incremental Training Script for LDB Deformable DETR
# Training flow: VOC -> Clipart -> Watercolor -> Comic

set -e

# Change to mmdetection directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "LDB Domain Incremental Training"
echo "Training sequence: VOC -> Clipart -> Watercolor -> Comic"
echo "=============================================="

# Step 1: Train on VOC (Base domain)
echo ""
echo "[Step 1/4] Training on VOC (Base domain)..."
echo "=============================================="
python tools/train.py configs/ldb/ldb_deformable_detr_voc.py

# Step 2: Incremental training on Clipart
echo ""
echo "[Step 2/4] Incremental training on Clipart..."
echo "=============================================="
python tools/train.py configs/ldb/ldb_deformable_detr_clipart.py

# Step 3: Incremental training on Watercolor
echo ""
echo "[Step 3/4] Incremental training on Watercolor..."
echo "=============================================="
python tools/train.py configs/ldb/ldb_deformable_detr_watercolor.py

# Step 4: Incremental training on Comic
echo ""
echo "[Step 4/4] Incremental training on Comic..."
echo "=============================================="
python tools/train.py configs/ldb/ldb_deformable_detr_comic.py

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
echo ""

# Evaluation on all domains
echo "Starting evaluation on all domains..."
echo ""

echo "[Eval 1/4] Evaluating on VOC..."
python tools/test.py configs/ldb/ldb_deformable_detr_voc.py \
    work_dirs/ldb_deformable_detr_comic/epoch_20.pth \
    --work-dir work_dirs/eval_voc

echo "[Eval 2/4] Evaluating on Clipart..."
python tools/test.py configs/ldb/ldb_deformable_detr_clipart.py \
    work_dirs/ldb_deformable_detr_comic/epoch_20.pth \
    --work-dir work_dirs/eval_clipart

echo "[Eval 3/4] Evaluating on Watercolor..."
python tools/test.py configs/ldb/ldb_deformable_detr_watercolor.py \
    work_dirs/ldb_deformable_detr_comic/epoch_20.pth \
    --work-dir work_dirs/eval_watercolor

echo "[Eval 4/4] Evaluating on Comic..."
python tools/test.py configs/ldb/ldb_deformable_detr_comic.py \
    work_dirs/ldb_deformable_detr_comic/epoch_20.pth \
    --work-dir work_dirs/eval_comic

echo ""
echo "=============================================="
echo "All evaluations completed!"
echo "=============================================="
