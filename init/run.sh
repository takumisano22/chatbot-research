#!/usr/bin/env sh
# Docker ワンショット「init」: Alembic でスキーマ適用（RDS やローカルでも同じコマンドを想定）。
set -eu
cd /app
echo "[init] alembic upgrade head"
exec alembic -c alembic.ini upgrade head
