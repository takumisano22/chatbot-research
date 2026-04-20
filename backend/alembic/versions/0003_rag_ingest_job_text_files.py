"""rag_ingest_job_text_files（txt/md キュー用子テーブル）

Revision ID: 0003_text_files
Revises: 0002_job_kind
Create Date: 2026-04-13

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_text_files"
down_revision: str | None = "0002_job_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rag_ingest_job_text_files",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column(
            "sort_order",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("file_bytes", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["rag_ingest_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "idx_rag_ingest_job_text_files_job_sort",
        "rag_ingest_job_text_files",
        ["job_id", "sort_order"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_rag_ingest_job_text_files_job_sort",
        table_name="rag_ingest_job_text_files",
    )
    op.drop_table("rag_ingest_job_text_files")
