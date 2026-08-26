#!/bin/sh
# ===========================================================================
# ASL_FastDenoising — clone / update the repo on Tianhe (天河).
#
# Run on the LOGIN node (compute nodes usually have no internet).
#
#   sh env/hpc/sync_repo.sh          # from an existing clone (updates in place), OR
#   sh sync_repo.sh                  # copied anywhere (first-time clone into $REPO)
#
# The repo is PUBLIC, so the default HTTPS URL needs no credential at all — no
# deploy key, no token. Transport is chosen from the URL:
#   https://...  -> anonymous, nothing to configure          (default)
#   git@...      -> SSH, requires the private key at $SSH_KEY and its .pub
#                   registered under the repo's Settings -> Deploy keys
# So `GIT_URL=git@github.com:Jian-uon/ASL_FastDenoising.git sh env/hpc/sync_repo.sh`
# still works if the repo is ever made private again.
#
# ⚠ This must be ITS OWN clone, separate from ASL_dmvae: the two repos host
# different paper lines and must not be mixed (CLAUDE.md §8).
#
# Override any path with an env var, e.g.:  REPO=/path/to/elsewhere sh sync_repo.sh
# ===========================================================================
set -eu

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_FastDenoising}   # local clone dir
GIT_URL=${GIT_URL:-https://github.com/Jian-uon/ASL_FastDenoising.git}
SSH_KEY=${SSH_KEY:-/fs1/home/duancaohui/jian/ssh/id_ed25519}         # only used for git@ URLs
BRANCH=${BRANCH:-master}

case "$GIT_URL" in
  git@*|ssh://*)
    if [ ! -f "$SSH_KEY" ]; then
      echo "ERROR: SSH URL given ($GIT_URL) but no private key at $SSH_KEY"
      echo "       (the private key is that path WITHOUT the .pub suffix)"
      echo "       For the public repo just drop GIT_URL and use the HTTPS default."
      exit 1
    fi
    # Use this key for all git transport here; accept GitHub's host key on first
    # use (non-interactive). IdentitiesOnly stops ssh from trying other agent keys.
    export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    echo "verifying GitHub SSH auth ..."
    ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | head -1 || true
    ;;
  *)
    echo "using anonymous HTTPS ($GIT_URL) — no credential needed"
    ;;
esac

if [ -d "$REPO/.git" ]; then
  echo "updating existing clone at $REPO (branch $BRANCH) ..."
  cd "$REPO"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  echo "cloning $GIT_URL -> $REPO ..."
  mkdir -p "$(dirname "$REPO")"
  git clone --branch "$BRANCH" "$GIT_URL" "$REPO"
  cd "$REPO"
fi

echo
echo "HEAD now at:"
git --no-pager log -1 --oneline
echo
echo "DONE. Next:"
echo "  1) build the env (once):  sh env/hpc/install_env.sh   (on a gpu node)"
echo "  2) check data path:       grep root_path env/hpc/configs/server_v35_joint.yml"
echo "  3) submit (A0 baseline):  yhbatch env/hpc/slurm/submit_v35_joint.sh"
echo "     A1 window fusion:      WIN_LEVELS=2 WIN_K=t1  yhbatch env/hpc/slurm/submit_v35_joint.sh"
echo "     A3 T1-free control:    WIN_LEVELS=2 WIN_K=asl yhbatch env/hpc/slurm/submit_v35_joint.sh"
