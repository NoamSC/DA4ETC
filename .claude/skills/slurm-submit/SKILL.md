---
name: slurm-submit
description: Scaffold and submit a SLURM job for the da4etc repo using the repo's standard template. Use whenever the user wants to run training/inference/analysis on the cluster, "submit a slurm job", "sbatch", "run on GPU", "run week N training", or any job that takes more than a few minutes (which per repo policy MUST go through SLURM, never a login-node background process).
---

# slurm-submit

Generate a SLURM script under `slurm_files/` that matches this repo's conventions, then submit it with `sbatch`. Long jobs MUST run via SLURM (login shell is `tcsh`; never background a long job on the login node).

## Hard conventions (copy exactly)

- **Partition:** `killable`. Always include `#SBATCH --signal=USR1@120` (scripts checkpoint on USR1 and auto-resume).
- **Python:** `/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python` (assign to `$PYTHON`). Do NOT rely on `conda activate`.
- **Run from repo root:** `cd /home/anatbr/students/noamshakedc/da4etc`. All entrypoints are `scripts/<stage>/<script>.py` and add the repo root to `sys.path` themselves.
- **Logs:** write to `logs/`. Use the timestamped-rename pattern below so logs sort chronologically (`logs/slurm-%j.out.tmp` → `logs/<TIMESTAMP>_<EXPNAME>_<JOBID>.out`).
- **GPU jobs:** `--gres=gpu:1`, typically `--mem=16G`, `--cpus-per-task=4` (raise to 8 for sharded/array work).
- **Datasets:** TLS root is `/home/anatbr/dataset/CESNET-TLS-Year22_v2` (current) — confirm with the user if a job needs the older `/home/anatbr/dataset/CESNET-TLS-Year22`.

## Steps

1. Ask only what's missing: which script/stage, key args (week, exp_name, method, lambdas, data fracs), wall time, and whether it's an **array** job (sharded `--num-shards/--shard-id` or per-week 0–52).
2. Copy `single.slurm.tmpl` (or `array.slurm.tmpl`) from this skill folder into `slurm_files/run_<name>.slurm`, filling the `#SBATCH --job-name`, `--time`, the `EXP_NAME_FULL`/`EXPNAME`, and the `$PYTHON scripts/...` invocation.
3. Show the user the final script. Submit with `sbatch slurm_files/run_<name>.slurm` only after they confirm (or if they already said "submit it").
4. Report the returned job id and remind them to monitor with `squeue -u $USER` / tail the timestamped file in `logs/`.

## Notes

- Disk quota on the netapp tree has been hit before — per-epoch checkpoints under `exps/` are the bloat. For long training, remind the user the trainer keeps `best_model.pth` + recent epochs.
- Source weeks for the paper: TLS source is **WEEK-2022-01** (not 00); benchmark also uses **Week-16**. Don't infer the source week from peak accuracy. (See repo CLAUDE.md "Critical facts".)
