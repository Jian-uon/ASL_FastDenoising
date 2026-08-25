#!/bin/sh
# ===========================================================================
# CIG-Net v2 — clone / update the repo on Tianhe (天河) via SSH deploy key.
#
# Run on the LOGIN node (compute nodes usually have no internet). It uses the
# SSH key at $SSH_KEY to reach GitHub, then clones $REPO if missing or `git pull`
# if it already exists.
#
#   sh env/hpc/sync_repo.sh          # from an existing clone (updates in place), OR
#   sh sync_repo.sh                  # copied anywhere (first-time clone into $REPO)
#
# Override any path with an env var, e.g.:  REPO=/path/to/ASL_dmvae sh sync_repo.sh
# ===========================================================================
set -eu

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_dmvae}                 # local clone dir
SSH_KEY=${SSH_KEY:-/fs1/home/duancaohui/jian/ssh/id_ed25519}      # PRIVATE key (the .pub sits next to it)
GIT_URL=${GIT_URL:-git@github.com:Jian-uon/ASL_dmvae.git}
BRANCH=${BRANCH:-master}

if [ ! -f "$SSH_KEY" ]; then
  echo "ERROR: SSH private key not found at $SSH_KEY"
  echo "       (you gave the public key .../id_ed25519.pub — the private key is the same path WITHOUT .pub)"
  exit 1
fi

# Use this key for all git transport here; accept GitHub's host key on first use
# (non-interactive). IdentitiesOnly stops ssh from trying other agent keys.
export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

echo "verifying GitHub SSH auth ..."
ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | head -1 || true

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
echo "  2) check data path:       grep root_path env/hpc/configs/server_v37.yml"
echo "  3) submit:                sh env/hpc/slurm/submit_all.sh"
