git init
git add .
git commit -m "Add credit-risk pipeline, models, and dashboards"
git branch -M main
git remote add origin https://github.com/P07awan/Loan-Default-Credit-Risk-Dashboard.git
git pull origin main --allow-unrelated-histories -X ours --no-edit
git push -u origin main#!/usr/bin/env bash
# Run from inside this project folder.  Needs git + GitHub login (HTTPS token or SSH).
set -e
REPO=https://github.com/P07awan/Loan-Default-Credit-Risk-Dashboard.git
[ -d .git ] || git init
git add .
git commit -m "Add credit-risk pipeline, models, and dashboards" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO"
# If the remote already has your README-only "first commit", merge it, keeping our README:
git fetch origin main 2>/dev/null && git merge origin/main --allow-unrelated-histories -X ours -m "Merge remote main" || true
git push -u origin main
