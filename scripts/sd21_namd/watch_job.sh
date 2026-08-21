#!/bin/bash
# Reusable SLURM watch for a training/eval job. One stdout line = one event.
#
#   scripts/sd21_namd/watch_job.sh JOBID LOGSTEM LABEL [STRIDE]
#
# LOGSTEM is the sbatch --job-name, since the launchers use --output=%x.out.
# STRIDE throttles epoch progress (default 10): a 100-epoch run would otherwise
# fire 100 events and the harness auto-stops monitors that flood. Errors and
# terminal states are never throttled.
#
# Coverage is the point: silence must not look like success. This emits on every
# terminal SLURM state, not just COMPLETED, because a job can report FAILED with
# nothing in the log matching a pattern filter.
JOBID=$1
STEM=$2
LABEL=$3
STRIDE=${4:-10}

OUT="${STEM}.out"
ERR="${STEM}.err"
LAST_EPOCH=-1
STARTED=0

size_of() { [ -f "$1" ] && stat -c %s "$1" || echo 0; }

# Start past whatever is already in the log. The launchers write --output=%x.out,
# keyed on job NAME, so a resubmit reuses the same path: starting at 0 would
# replay the previous run's traceback as if it were this one's. (It did.)
OFF_OUT=$(size_of "$OUT")
OFF_ERR=$(size_of "$ERR")

# Read bytes appended since the last poll, with \r split so progress-bar
# redraws become separate lines instead of one giant one.
#
# A shrunk file means SLURM truncated it as the new job started writing, so the
# saved offset now points past the end and every future read would come back
# empty. Rewind to 0 and take the new run from the top.
since() {
  local f=$1 off=$2 sz
  sz=$(size_of "$f")
  [ "$sz" -lt "$off" ] && off=0
  [ "$sz" -le "$off" ] && return 0
  tail -c +$((off + 1)) "$f" 2>/dev/null | tr '\r' '\n'
}

echo "[$LABEL] watching job $JOBID (stride ${STRIDE} epochs)"

while true; do
  STATE=$(squeue -h -j "$JOBID" -o "%T" 2>/dev/null)

  if [ "$STARTED" = "0" ] && [ "$STATE" = "RUNNING" ]; then
    STARTED=1
    NODE=$(squeue -h -j "$JOBID" -o "%R" 2>/dev/null)
    echo "[$LABEL] RUNNING on ${NODE}"
  fi

  NEW_OUT=$(since "$OUT" "$OFF_OUT")
  NEW_ERR=$(since "$ERR" "$OFF_ERR")
  OFF_OUT=$(size_of "$OUT")
  OFF_ERR=$(size_of "$ERR")
  BOTH="${NEW_OUT}
${NEW_ERR}"

  # Startup milestones: the derived LR schedule and what the ckpt load dropped.
  echo "$BOTH" | grep -aE '^steps/epoch=|Restored from .* with .* missing|^Applied config overrides' \
    | head -4 | while read -r l; do [ -n "$l" ] && echo "[$LABEL] $l"; done

  # Failure signatures. Widened deliberately -- extra noise beats a missed crash.
  echo "$BOTH" | grep -aE 'Traceback|FATAL|CUDA out of memory|CUDA error|RuntimeError|AssertionError|OutOfMemoryError|Killed|Segmentation fault|srun: error' \
    | head -3 | while read -r l; do [ -n "$l" ] && echo "[$LABEL] ERROR: $l"; done

  # Epoch progress, throttled to the stride.
  EP=$(echo "$NEW_ERR" | grep -aoE 'Epoch [0-9]+' | tail -1 | grep -oE '[0-9]+')
  if [ -n "$EP" ] && [ "$EP" -ge $((LAST_EPOCH + STRIDE)) ]; then
    LAST_EPOCH=$EP
    CK=$(ls -1t checkpoints/"${STEM}"*/ 2>/dev/null | grep -m1 'epoch=' || true)
    echo "[$LABEL] epoch ${EP}${CK:+  latest ckpt: $CK}"
  fi

  # Left the queue -> report the terminal state and stop the watch.
  if [ -z "$STATE" ]; then
    sleep 20   # let sacct catch up
    FINAL=$(sacct -j "$JOBID" --format=State,Elapsed,MaxRSS -n -P 2>/dev/null | head -1)
    echo "[$LABEL] TERMINAL: ${FINAL:-unknown (job gone from squeue)}"
    exit 0
  fi

  sleep 180
done
