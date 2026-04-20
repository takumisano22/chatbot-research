from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    JSON,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# -----------------------------------------------------------------------------
# 役割: rag_ingest_jobs テーブルのキュー行（取り込みジョブ）を表す ORM。
# 主な呼び出し元: ingestion_job_repository、ingest ワーカー、ingest API の状態参照。
# 流れ: キュー時は payload_items にペイロード → 成功時 result_json、payload_items は NULL。
# -----------------------------------------------------------------------------


class RagIngestJobRow(Base):
    __tablename__ = "rag_ingest_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_rag_ingest_jobs_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    # ワーカーが ingest_pipeline.jobs のどのハンドラを使うか決める（ingest_pipeline.enums の job_kind と一致させる）。
    job_kind: Mapped[str] = mapped_column(Text, nullable=False, server_default="rag_upload")
    # ジョブ単位の短い説明（バッチ件数など）。成功時は result_json でファイル単位の結果を参照。
    payload_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # キュー待ち〜実行中のみ: [{"filename": str, "body_b64": str}, ...]。完了時は NULL。
    payload_items: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
