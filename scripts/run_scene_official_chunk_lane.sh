#!/usr/bin/env bash
set -uo pipefail

method="${1:-}"
task_id="${2:-}"

if [[ -z "${method}" || -z "${task_id}" ]]; then
  echo "usage: $0 <action_nce|sigreg> <task_id>" >&2
  exit 2
fi

case "${method}" in
  action_nce)
    policy="scene/lewm_masked_action_nce_e10_s3072"
    method_slug="action_nce"
    ;;
  sigreg)
    policy="scene/lewm_sigreg_e10_s3072"
    method_slug="sigreg"
    ;;
  *)
    echo "unknown method: ${method}" >&2
    exit 2
    ;;
esac

mkdir -p /tmp/scene_official_chunks

for start in 0 10 20 30 40; do
  end=$((start + 9))
  filename="scene_official_${method_slug}_s3072_task${task_id}_eps${start}_${end}_n10.txt"
  volume_path="/scene/${filename}"
  verified=0

  for attempt in 1 2 3; do
    log="/tmp/scene_official_${method_slug}_s3072_task${task_id}_eps${start}_${end}_attempt${attempt}.log"
    overrides="eval.num_eval=10 eval.episode_start=${start} eval.task_ids=[${task_id}] eval.env_batch_size=1 output.filename=${filename} output.save_video=false"

    echo "[$(date)] launching ${method_slug} task=${task_id} eps=${start}-${end} attempt=${attempt}" >&2
    nohup .venv/bin/modal run --detach modal_app.py::evaluate \
      --config-name scene_official \
      --policy "${policy}" \
      --overrides "${overrides}" \
      > "${log}" 2>&1 &
    pid=$!
    wait "${pid}"
    rc=$?
    echo "[$(date)] modal command exited rc=${rc}; log=${log}" >&2

    rm -f "/tmp/scene_official_chunks/${filename}"
    if .venv/bin/modal volume get multi-future-lewm-cache "${volume_path}" /tmp/scene_official_chunks/ >/tmp/scene_official_volume_get.log 2>&1; then
      echo "[$(date)] verified ${volume_path}" >&2
      verified=1
      break
    fi

    echo "[$(date)] missing ${volume_path}; retrying after short backoff" >&2
    tail -40 "${log}" >&2 || true
    cat /tmp/scene_official_volume_get.log >&2 || true
    sleep 60
  done

  if [[ "${verified}" -ne 1 ]]; then
    echo "failed to produce ${volume_path} after 3 attempts" >&2
    exit 1
  fi
done
