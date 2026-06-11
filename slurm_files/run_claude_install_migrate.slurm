#!/bin/bash
#SBATCH --job-name=claude_install_migrate
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=00:30:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Change to project directory (repo convention)
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Timestamped, chronologically-sortable log files
EXPNAME=claude_install_migrate
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# ----------------------------------------------------------------------------
# Task: finish relocating the Claude CLI off the (full) home quota.
#   - Binary already installed at $NETAPP_BIN (done on login node).
#   - Here: migrate runtime config ~/.claude -> netapp, and wire PATH +
#     CLAUDE_CONFIG_DIR into both tcsh (login shell) and zsh rc files.
#   - We do NOT delete the old ~/.claude: a live session may still use it.
#     Reclaim those ~247MB manually once you've confirmed the new config works.
# ----------------------------------------------------------------------------
set -uo pipefail

NETAPP_ROOT=/home/anatbr/students/noamshakedc/env/claude
NETAPP_BIN="$NETAPP_ROOT/.local/bin"
CFG="$NETAPP_ROOT/config"
MARK="# >>> claude code (netapp) >>>"
ENDMARK="# <<< claude code (netapp) <<<"

echo "== Job start: $(date) on $(hostname) =="
echo "HOME=$HOME"
echo "NETAPP_ROOT=$NETAPP_ROOT"

# 1) Migrate config dir (read home, write netapp -> does not touch home quota).
mkdir -p "$CFG"
echo "== rsync ~/.claude -> $CFG =="
rsync -a --info=stats1 "$HOME/.claude/" "$CFG/"
rc=$?
# rsync exit 24 = "some source files vanished" (live session appending) -> benign.
if [ $rc -ne 0 ] && [ $rc -ne 24 ]; then
    echo "ERROR: rsync failed with code $rc" >&2
    exit $rc
fi
echo "config migrated: $(du -sh "$CFG" | cut -f1)"
ls "$CFG/.credentials.json" >/dev/null 2>&1 && echo "credentials.json present (auth carried over)" || echo "WARN: no .credentials.json found"

# 2) Idempotently wire up zsh (~/.zshrc).
ZRC="$HOME/.zshrc"
if grep -qF "$MARK" "$ZRC" 2>/dev/null; then
    echo "zsh: claude block already present in $ZRC (skip)"
else
    {
        echo ""
        echo "$MARK"
        echo "export CLAUDE_CONFIG_DIR=$CFG"
        echo "export PATH=$NETAPP_BIN:\$PATH"
        echo "$ENDMARK"
    } >> "$ZRC"
    echo "zsh: appended claude block to $ZRC"
fi

# 3) Idempotently wire up tcsh login shell (~/.tcshrc; falls back to ~/.cshrc).
if [ -e "$HOME/.tcshrc" ]; then TRC="$HOME/.tcshrc"; else TRC="$HOME/.cshrc"; fi
if grep -qF "$MARK" "$TRC" 2>/dev/null; then
    echo "tcsh: claude block already present in $TRC (skip)"
else
    {
        echo ""
        echo "$MARK"
        echo "setenv CLAUDE_CONFIG_DIR $CFG"
        echo "setenv PATH $NETAPP_BIN:\$PATH"
        echo "$ENDMARK"
    } >> "$TRC"
    echo "tcsh: appended claude block to $TRC"
fi

# 4) Verify the relocated binary runs against the relocated config.
echo "== verify =="
CLAUDE_BIN="$NETAPP_BIN/claude"
echo "claude symlink -> $(readlink -f "$CLAUDE_BIN")"
CLAUDE_CONFIG_DIR="$CFG" "$CLAUDE_BIN" --version 2>&1 | sed 's/^/  /' || echo "WARN: --version returned nonzero"

echo ""
echo "== SUMMARY =="
echo "  binary : $CLAUDE_BIN"
echo "  config : $CFG ($(du -sh "$CFG" 2>/dev/null | cut -f1))"
echo "  PATH/env wired into: $ZRC and $TRC"
echo "  Old home config still at ~/.claude ($(du -sh "$HOME/.claude" 2>/dev/null | cut -f1)) -- delete it"
echo "  manually to reclaim home quota once you've confirmed the new setup."
echo "== Job done: $(date) =="
