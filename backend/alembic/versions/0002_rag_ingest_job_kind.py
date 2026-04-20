"""rag_ingest_jobs に job_kind を追加（キュー種別のディスパッチ用）

Revision ID: 0002_job_kind
Revises: 0001_initial
Create Date: 2026-04-13

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_job_kind"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "rag_ingest_jobs",
        sa.Column(
            "job_kind",
            sa.Text(),
            server_default=sa.text("'pdf_upload'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("rag_ingest_jobs", "job_kind")
