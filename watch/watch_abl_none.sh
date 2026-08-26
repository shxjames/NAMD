#!/bin/bash
# Watch the NONE-arm (0/0 VAE) NAMD ablation training job.
#   JOBID=27526313  logs NAMD-abl-none.{out,err} in NAMD-abl0/
# Emits: queue->run transition, checkpoint-load report, epoch progress on a
# stride, any error signature, and the terminal SLURM state. Exits when the job
# leaves the queue so the watch does not dangle.
JOBID=27526313
DIR=/scratch/liyues_root/liyues/shxjames/NAMD-abl0
OUT=$DIR/NAMD-abl-none.out
ERR=$DIR/NAMD-abl-none.err
STRIDE=10

started=0
declare -A seen
last_epoch=-1

emit() { echo "[abl-none $1] $2"; }

while true; do
  st=$(squeue -j "$JOBID" -h -o '%t' 2>/dev/null)

  if [ -n "$st" ] && [ "$st" = "R" ] && [ $started -eq 0 ]; then
    started=1
    emit RUN "job $JOBID started on $(squeue -j "$JOBID" -h -o '%N' 2>/dev/null)"
  fi

  # --- startup load report: confirms the uncond ckpt actually loaded ---
  if [ -f "$OUT" ] && [ -z "${seen[load]}" ]; then
    line=$(grep -m1 "Restored from" "$OUT" 2>/dev/null)
    if [ -n "$line" ]; then
      seen[load]=1
      emit LOAD "$line"
      drop=$(grep -c "Deleting key" "$OUT" 2>/dev/null || echo 0)
      emit LOAD "dropped-by-shape keys: $drop (expect ~33: input_blocks.0.0 + 32 attn2.to_k/to_v)"
    fi
  fi

  # --- epoch progress on a stride, so a 70-epoch run does not fire 70 events ---
  if [ -f "$ERR" ]; then
    e=$(grep -oE "Epoch [0-9]+" "$ERR" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    if [ -n "$e" ] && [ "$e" -ne "$last_epoch" ] 2>/dev/null; then
      if [ $((e % STRIDE)) -eq 0 ] && [ -z "${seen[ep$e]}" ]; then
        seen[ep$e]=1
        loss=$(grep -oE "val/loss[^ ,]*[= ][0-9.]+" "$ERR" 2>/dev/null | tail -1)
        emit EPOCH "epoch $e/70 ${loss:+($loss)}"
      fi
      last_epoch=$e
    fi
  fi

  # --- failure signatures, unthrottled ---
  for f in "$OUT" "$ERR"; do
    [ -f "$f" ] || continue
    hit=$(grep -nE "Traceback|CUDA out of memory|RuntimeError|FATAL|Killed|srun: error|AssertionError|No such file or directory" "$f" 2>/dev/null | tail -1)
    if [ -n "$hit" ]; then
      key="err_$(basename "$f")_$(echo "$hit" | cut -d: -f1)"
      if [ -z "${seen[$key]}" ]; then
        seen[$key]=1
        emit ERROR "$(basename "$f"): $(echo "$hit" | cut -c1-300)"
      fi
    fi
  done

  # --- terminal state: load-bearing, a job can FAIL with nothing in the logs ---
  if [ -z "$st" ]; then
    sleep 10
    final=$(sacct -j "$JOBID" -n -o State%20 2>/dev/null | head -1 | tr -d ' ')
    emit DONE "job $JOBID left the queue; final state: ${final:-UNKNOWN}"
    if [ -f "$OUT" ]; then
      tail -5 "$OUT" 2>/dev/null | sed 's/^/[abl-none TAIL] /'
    fi
    exit 0
  fi

  sleep 180
done
