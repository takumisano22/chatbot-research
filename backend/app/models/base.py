from sqlalchemy.orm import DeclarativeBase

# -----------------------------------------------------------------------------
# 役割: SQLAlchemy Declarative の共通 Base（メタデータの集約先）。
# 主な呼び出し元: 各 ORM モデル、Alembic、テストの create_all。
# 流れ: 各モデルが Base を継承し、同一 MetaData 下にテーブル定義を登録する。
# -----------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass
