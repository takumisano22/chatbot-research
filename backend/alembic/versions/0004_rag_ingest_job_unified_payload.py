"""rag_ingest_jobs.payload_items に集約し子テーブルを廃止

Revision ID: 0004_unified_payload
Revises: 0003_text_files
Create Date: 2026-04-13

"""

from __future__ import annotations

import base64
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_unified_payload"
down_revision: str | None = "0003_text_files"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "rag_ingest_jobs",
        sa.Column("payload_items", sa.JSON(), nullable=True),
    )
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    meta = sa.MetaData()
    only: list[str] = ["rag_ingest_jobs"]
    if inspector.has_table("rag_ingest_job_pdfs"):
        only.append("rag_ingest_job_pdfs")
    if inspector.has_table("rag_ingest_job_text_files"):
        only.append("rag_ingest_job_text_files")
    meta.reflect(bind=conn, only=only)
    jobs_t = meta.tables["rag_ingest_jobs"]

    if "rag_ingest_job_pdfs" in meta.tables:
        pdfs_t = meta.tables["rag_ingest_job_pdfs"]
        for row in conn.execute(sa.select(pdfs_t.c.job_id).distinct()).all():
            jid = row[0]
            rows = conn.execute(
                sa.select(pdfs_t.c.filename, pdfs_t.c.pdf_bytes)
                .where(pdfs_t.c.job_id == jid)
                .order_by(pdfs_t.c.sort_order)
            ).all()
            items: list[dict[str, str]] = []
            for fn, blob in rows:
                raw = bytes(blob) if blob is not None else b""
                items.append({"filename": str(fn), "body_b64": base64.b64encode(raw).decode("ascii")})
            conn.execute(sa.update(jobs_t).where(jobs_t.c.id == jid).values(payload_items=items))

    if "rag_ingest_job_text_files" in meta.tables:
        txt_t = meta.tables["rag_ingest_job_text_files"]
        for row in conn.execute(sa.select(txt_t.c.job_id).distinct()).all():
            jid = row[0]
            rows = conn.execute(
                sa.select(txt_t.c.filename, txt_t.c.file_bytes)
                .where(txt_t.c.job_id == jid)
                .order_by(txt_t.c.sort_order)
            ).all()
            items = []
            for fn, blob in rows:
                raw = bytes(blob) if blob is not None else b""
                items.append({"filename": str(fn), "body_b64": base64.b64encode(raw).decode("ascii")})
            conn.execute(sa.update(jobs_t).where(jobs_t.c.id == jid).values(payload_items=items))

    if inspector.has_table("rag_ingest_job_pdfs"):
        op.drop_index("idx_rag_ingest_job_pdfs_job_sort", table_name="rag_ingest_job_pdfs")
        op.drop_table("rag_ingest_job_pdfs")
    if inspector.has_table("rag_ingest_job_text_files"):
        op.drop_index("idx_rag_ingest_job_text_files_job_sort", table_name="rag_ingest_job_text_files")
        op.drop_table("rag_ingest_job_text_files")


def downgrade() -> None:
    op.create_table(
        "rag_ingest_job_pdfs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column(
            "sort_order",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("pdf_bytes", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["rag_ingest_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_rag_ingest_job_pdfs_job_sort",
        "rag_ingest_job_pdfs",
        ["job_id", "sort_order"],
        unique=False,
    )
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
    )
    op.create_index(
        "idx_rag_ingest_job_text_files_job_sort",
        "rag_ingest_job_text_files",
        ["job_id", "sort_order"],
        unique=False,
    )
    op.drop_column("rag_ingest_jobs", "payload_items")
