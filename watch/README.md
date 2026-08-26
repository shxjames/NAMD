# Watchers for the NONE-arm ablation run

Poll-and-emit scripts for the two jobs launched by `run_abl_none_llm.sh` and
`check_abl_none_scalefactor.sh`. One stdout line per event; both exit once their
job leaves the queue, so they never dangle.

| script | job | emits |
|---|---|---|
| `watch_abl_none.sh` | 27526313 `NAMD-abl-none` | queue→run, checkpoint-load report + dropped-key count, epoch progress on a stride of 10, error signatures, terminal SLURM state |
| `watch_abl_sf.sh` | 27526314 `abl-none-sf` | the `Recommended scale_factor` line and a MATCH/MISMATCH verdict against the 0.395414 pinned in `run_abl_none_llm.sh` |

Run either as `bash watch/watch_abl_none.sh`; they take no arguments (job IDs are
pinned at the top — edit those if the jobs are ever resubmitted).

## Why these live here and not in a job tmp dir

They were originally written to `$CLAUDE_JOB_DIR/tmp`, which is deleted when the
owning session's job is deleted — i.e. they would disappear at exactly the moment
another session needed to pick up reporting. Committed here so the handoff
survives the session that armed them.

Terminal-state checking is load-bearing, not belt-and-braces: a SLURM job can go
`FAILED` with nothing in its logs matching any error pattern, so a watcher that
only greps the log will sit silent through it. Both scripts check `sacct` after
the job leaves the queue for exactly that reason.
