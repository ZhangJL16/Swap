#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-hmappo_cbf_flow_$(date +%m%d_%H%M%S)}"
MODEL_ROOT="${MODEL_ROOT:-./model_runs_hmappo_cbf_flow}"
LOG_ROOT="${LOG_ROOT:-logs/uav_energy_delivery_hmappo_cbf_flow}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

N_AGENTS="${N_AGENTS:-4}"
EPISODE_LIMIT="${EPISODE_LIMIT:-400}"
MAX_ACTIVE_ORDERS="${MAX_ACTIVE_ORDERS:-8}"
N_STEPS_LOW="${N_STEPS_LOW:-150000}"
N_STEPS_FLOW="${N_STEPS_FLOW:-80000}"
N_STEPS_DISTILL="${N_STEPS_DISTILL:-80000}"
N_STEPS_ENERGY="${N_STEPS_ENERGY:-80000}"
N_STEPS_HIGH_Q="${N_STEPS_HIGH_Q:-100000}"
EVALUATE_CYCLE="${EVALUATE_CYCLE:-5000}"
EVALUATE_EPOCH="${EVALUATE_EPOCH:-20}"
CUDA="${CUDA:-True}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-123}"

RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
MODEL_DIR="${MODEL_ROOT}/${RUN_NAME}"
ARTIFACT_DIR="${MODEL_DIR}/hmappo_cbf_flow/UAVEnergyDeliveryHierarchical/cbf_flow"
mkdir -p "${RUN_DIR}"

run_phase() {
  local phase="$1"
  local steps="$2"
  local log_file="${RUN_DIR}/${phase}.log"
  echo "START phase=${phase} steps=${steps} log=${log_file}"
  "${PYTHON_BIN}" main_level.py \
    --alg hmappo_cbf_flow \
    --map UAVEnergyDeliveryHierarchical \
    --training_phase "${phase}" \
    --uav_n_agents "${N_AGENTS}" \
    --episode_limit "${EPISODE_LIMIT}" \
    --uav_total_orders "${MAX_ACTIVE_ORDERS}" \
    --uav_max_active_orders "${MAX_ACTIVE_ORDERS}" \
    --agent_entry_interval 1 \
    --order_max_duration 120 \
    --n_steps "${steps}" \
    --evaluate_cycle "${EVALUATE_CYCLE}" \
    --evaluate_epoch "${EVALUATE_EPOCH}" \
    --cuda "${CUDA}" \
    --gpu_id "${GPU_ID}" \
    --seed "${SEED}" \
    --eval_seed "$((SEED + 100000))" \
    --replay_dir "" \
    --model_dir "${MODEL_DIR}" \
    --cbf_flow_artifact_dir "${ARTIFACT_DIR}" \
    --cbf_flow_load_artifact_dir "${ARTIFACT_DIR}" \
    --actor_correct_align_coef 0.1 \
    > "${log_file}" 2>&1
  echo "DONE phase=${phase}"
}

run_phase low_cbf "${N_STEPS_LOW}"
run_phase flow "${N_STEPS_FLOW}"
run_phase distill "${N_STEPS_DISTILL}"
run_phase energy "${N_STEPS_ENERGY}"
run_phase high_q "${N_STEPS_HIGH_Q}"

echo "DONE all phases"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "ARTIFACT_DIR=${ARTIFACT_DIR}"
