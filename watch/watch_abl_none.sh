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
  # Read the run's own one-line-per-epoch summary in .out rather than scraping
  # the progress bar in .err: the bar's trailing "val/loss=0.322." picked up a
  # stray '.' and carried no train loss. This line has both, already formatted.
  if [ -f "$OUT" ]; then
    summary=$(tr '\r' '\n' < "$OUT" 2>/dev/null | grep -E "^\[Epoch +[0-9]+\]" | tail -1)
    e=$(echo "$summary" | grep -oE "^\[Epoch +[0-9]+\]" | grep -oE "[0-9]+")
    if [ -n "$e" ] && [ "$e" -ne "$last_epoch" ] 2>/dev/null; then
      if [ $((e % STRIDE)) -eq 0 ] && [ -z "${seen[ep$e]}" ]; then
        seen[ep$e]=1
        emit EPOCH "$(echo "$summary" | tr -s ' ')  [$e/70]"
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
    # Summarise, never raw-tail. A plain `tail` here dumps Lightning progress-bar
    # lines that each carry a few hundred per-tensor grad-norm fields, which the
    # notification then truncates mid-token -- pages of noise, no signal. The
    # run's own one-line-per-epoch summary is the thing worth reporting.
    # The per-epoch summary is written to stdout (.out); .err carries only the
    # Lightning progress bar. Verified against the epoch-4 log on 2026-08-27.
    if [ -f "$OUT" ]; then
      tr '\r' '\n' < "$OUT" 2>/dev/null \
        | grep -E "^\[Epoch +[0-9]+\]" | tail -3 \
        | sed 's/^/[abl-none LAST] /'
    fi
    # On a non-clean exit, surface the real error instead of the epoch summary.
    case "${final:-}" in
      COMPLETED|CANCELLED*) ;;
      *)
        for f in "$ERR" "$OUT"; do
          [ -f "$f" ] || continue
          grep -E "Traceback|Error|CUDA out of memory|Killed" "$f" 2>/dev/null \
            | tail -3 | cut -c1-240 | sed "s|^|[abl-none FAIL $(basename "$f")] |"
        done
        ;;
    esac
    exit 0
  fi

  sleep 180
done
