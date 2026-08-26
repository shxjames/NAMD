#!/bin/bash
# Watch the NONE-arm scale-factor cross-check job (27526314).
# The one line that matters is "Recommended scale_factor:" -- compare it against
# the 0.395414 pinned in run_abl_none_llm.sh.
JOBID=27526314
DIR=/scratch/liyues_root/liyues/shxjames/NAMD-abl0
OUT=$DIR/abl-none-sf.out
ERR=$DIR/abl-none-sf.err

started=0
declare -A seen

emit() { echo "[abl-sf $1] $2"; }

while true; do
  st=$(squeue -j "$JOBID" -h -o '%t' 2>/dev/null)

  if [ -n "$st" ] && [ "$st" = "R" ] && [ $started -eq 0 ]; then
    started=1
    emit RUN "scale-factor check started"
  fi

  if [ -f "$OUT" ] && [ -z "${seen[sf]}" ]; then
    line=$(grep -m1 "Recommended scale_factor" "$OUT" 2>/dev/null)
    if [ -n "$line" ]; then
      seen[sf]=1
      emit RESULT "$line   (training pinned 0.395414)"
      val=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+")
      ok=$(awk -v v="$val" 'BEGIN{d=v-0.395414; if(d<0)d=-d; print (d<0.01)?"MATCH":"MISMATCH"}')
      emit VERDICT "$ok (delta vs 0.395414)"
    fi
  fi

  for f in "$OUT" "$ERR"; do
    [ -f "$f" ] || continue
    hit=$(grep -nE "Traceback|CUDA out of memory|RuntimeError|FATAL|Killed|srun: error|AssertionError" "$f" 2>/dev/null | tail -1)
    if [ -n "$hit" ]; then
      key="err_$(basename "$f")_$(echo "$hit" | cut -d: -f1)"
      if [ -z "${seen[$key]}" ]; then
        seen[$key]=1
        emit ERROR "$(basename "$f"): $(echo "$hit" | cut -c1-300)"
      fi
    fi
  done

  if [ -z "$st" ]; then
    sleep 10
    final=$(sacct -j "$JOBID" -n -o State%20 2>/dev/null | head -1 | tr -d ' ')
    emit DONE "final state: ${final:-UNKNOWN}"
    grep -m1 "Recommended scale_factor" "$OUT" 2>/dev/null | sed 's/^/[abl-sf TAIL] /'
    exit 0
  fi

  sleep 180
done
