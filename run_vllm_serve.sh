#!/bin/bash

# Configuration
MODEL_PATH="/run/user/1014/Qwen3-30B-A3B"
# MODEL_PATH="/dev/shm/deepseek-moe-16b-base"
MAX_SEQS=1
GPU_UTIL="0.8"
# PORT=8080
PORT=8088

# GPUs 2,3 are usually taken by another job.
#
# Prefer 1 over 0 if it is free: measured pinned H2D is ~17.7 GiB/s on GPU 0
# versus ~20-24 on GPUs 1-3, and that gap survives correct NUMA binding on both
# (node 0 and node 1 pinning give GPU 0 the same ~17.6). GPU 0 sits alone on
# node 0 and reaches everything else over SYS, so this is the link, not page
# placement. Since staging is PCIe-bound end to end, it is a ~25% decode
# difference for free.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Bind to the GPU's own NUMA node. The offloaded expert weights are pinned host
# memory, so every H2D copy is a DMA read from wherever those pages landed: on
# the wrong socket the transfer crosses UPI and drops from ~24 to ~17 GiB/s, and
# because pages follow whichever thread called pin_memory(), unbound runs scatter
# layers across both nodes and the throughput swings run to run.
# Here GPU 0 is on node 0 and GPUs 1-3 on node 1, hence the lookup rather than a
# constant. PCI_BUS_ID makes CUDA_VISIBLE_DEVICES agree with nvidia-smi's -i.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_BDF=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader \
    -i "${CUDA_VISIBLE_DEVICES%%,*}" | tr 'A-Z' 'a-z' | sed 's/^0000//')
NUMA_NODE=$(cat "/sys/bus/pci/devices/${GPU_BDF}/numa_node" 2>/dev/null || echo -1)

if [[ "$NUMA_NODE" -ge 0 ]]; then
    NUMACTL=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE")
    echo "Binding to NUMA node $NUMA_NODE (GPU $CUDA_VISIBLE_DEVICES @ $GPU_BDF)"
else
    NUMACTL=()
    echo "WARNING: no NUMA affinity for GPU $CUDA_VISIBLE_DEVICES; H2D will be slow"
fi

# Expert-cache offload: routed expert weights stay in pinned CPU memory and are
# staged into a ping-pong GPU cache a layer at a time. Without this the 57 GiB
# bf16 model does not fit on one 40 GiB A100 at all.
#
# NUM_CACHE_SLOTS: experts per ping/pong buffer. 0 = one slot per expert (128),
#   i.e. a buffer can hold a whole layer -- no prediction can ever miss. Start
#   here to check correctness, then lower it to squeeze GPU memory.
# PREDICTOR_DIR:  empty disables prediction; experts are then fetched on demand,
#   which is correct but synchronous. This is the baseline prediction is
#   measured against.
NUM_CACHE_SLOTS="${NUM_CACHE_SLOTS:-0}"
PREDICTOR_DIR="${PREDICTOR_DIR:-$(dirname "$0")/vllm/model_executor/layers/expert_prefetch/checkpoints/qwen3_moe/final}"

# Stage the experts over PCIe quantized instead of bf16. The bus is the whole
# bottleneck here -- a layer of 32 experts is ~288 MiB of bf16 and this link
# tops out near 20 GiB/s -- so the bit width is the main lever on decode speed.
#
# EXPERT_QUANT_BITS: 0 (bf16, the baseline), or 2 / 4 / 8. Measured per-layer
#   staging cost for 32 experts on this box: bf16 ~14 ms, int8 6.4, int4 3.6,
#   int2 2.1. The GPU cost is fixed either way (the packed bytes land in a
#   small shared ring, not a buffer per cache slot).
#   This is a model-quality change: RTN is uncalibrated, ~11% relative weight
#   error at int4 and far worse at int2, where a group has four levels. Run an
#   eval before trusting any of these -- start at 8, which is nearly lossless.
# EXPERT_QUANT_GROUP_SIZE: scale group along the reduction dim, or -1 for one
#   scale per row. Both reduction dims here (hidden 2048, moe_intermediate 768)
#   divide by 128 at tp=1. The latter is sharded by TP, so if you add
#   --tensor-parallel-size and it no longer divides, use -1.
#
# Note this costs startup time: the int store is built on the GPU at load, over
# every expert of every layer, before memory profiling.
EXPERT_QUANT_BITS="${EXPERT_QUANT_BITS:-4}"
EXPERT_QUANT_GROUP_SIZE="${EXPERT_QUANT_GROUP_SIZE:-128}"

PYTHON_EXEC="/run/user/1014/miniconda3/envs/vllm_hpclab/bin/python"

# Run the server
"${NUMACTL[@]}" $PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --override-generation-config '{"temperature": 0.0}' \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --enforce-eager \
    --offload-backend expert_cache \
    --num-cache-slots "$NUM_CACHE_SLOTS" \
    --expert-quant-bits "$EXPERT_QUANT_BITS" \
    --expert-quant-group-size "$EXPERT_QUANT_GROUP_SIZE" \
    --expert-predictor-dir "$PREDICTOR_DIR"
