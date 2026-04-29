from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Final, Protocol

# -----------------------------------------------------------------------------
# 役割: research_pair の ingest_pipeline_id を解決し、ingest_pipeline/library 配下の
# モジュールを安全に load する（既存ロジック経路は壊さない）。
# モジュール契約:
#   - 定数 SUPERSEDES: tuple[str, ...]   "convert" / "normalize" / "chunking" の部分集合。
#                                        含まれるステージは batch 側で skip される。
#   - 関数 ingest(settings, session, *, filename, data, source) -> int
# -----------------------------------------------------------------------------

# 代替可能なステージ。ここに無いステージ名を SUPERSEDES に書いた場合は弾く。
PipelineStage = str
_VALID_STAGES: Final[frozenset[str]] = frozenset({"convert", "normalize", "chunking"})

# ingest 関数のキーワード引数名（位置と一致を強制）。
_EXPECTED_INGEST_PARAMS: Final[tuple[str, ...]] = (
    "settings",
    "session",
    "filename",
    "data",
    "source",
)


class IngestFn(Protocol):
    def __call__(
        self,
        settings: object,
        session: object,
        *,
        filename: str,
        data: bytes,
        source: str,
    ) -> int: ...


@dataclass(frozen=True)
class IngestPipelineModule:
    name: str
    superseded: frozenset[str]
    ingest: IngestFn


def is_superseded(superseded: frozenset[str], stage: PipelineStage) -> bool:
    return stage in superseded


def load_ingest_pipeline(pipeline_id: str) -> IngestPipelineModule:
    pid = (pipeline_id or "").strip()
    if not pid:
        raise ValueError("ingest_pipeline_id が空です")

    fullname = f"app.rag.ingest_pipeline.library.{pid}"
    mod = importlib.import_module(fullname)

    superseded = _validate_superseded(mod)
    fn = _validate_ingest_callable(mod)
    return IngestPipelineModule(name=pid, superseded=superseded, ingest=fn)


# -----------------------------------------------------------------------------
# 検査ヘルパ
# -----------------------------------------------------------------------------


def _validate_superseded(mod: ModuleType) -> frozenset[str]:
    raw = getattr(mod, "SUPERSEDES", None)
    if raw is None:
        # 何も上書きしないライブラリも許容（KISS）。
        return frozenset()
    if not isinstance(raw, (tuple, list, set, frozenset)):
        raise TypeError(
            f"{mod.__name__}.SUPERSEDES は tuple/list/set である必要があります（現値: {raw!r}）"
        )
    items = tuple(raw)
    invalid = [s for s in items if s not in _VALID_STAGES]
    if invalid:
        raise ValueError(
            f"{mod.__name__}.SUPERSEDES に無効なステージ名: {invalid!r}"
            f"（許可: {sorted(_VALID_STAGES)})"
        )
    return frozenset(items)


def _validate_ingest_callable(mod: ModuleType) -> IngestFn:
    fn: Callable[..., object] | None = getattr(mod, "ingest", None)
    if not callable(fn):
        raise TypeError(f"{mod.__name__}.ingest が定義されていないか callable ではありません")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    names = tuple(p.name for p in params)
    if names != _EXPECTED_INGEST_PARAMS:
        raise TypeError(
            f"{mod.__name__}.ingest の引数名は {_EXPECTED_INGEST_PARAMS} の順で必要です"
            f"（現値: {names!r}）"
        )

    # filename / data / source はキーワード専用にする（順序事故防止）。
    for name in ("filename", "data", "source"):
        p = sig.parameters[name]
        if p.kind not in (inspect.Parameter.KEYWORD_ONLY,):
            raise TypeError(
                f"{mod.__name__}.ingest の {name!r} はキーワード専用引数である必要があります"
            )

    # 戻り値アノテーションが int であることを確認（実行時の型は呼び出し側で再確認する）。
    if sig.return_annotation is inspect.Signature.empty or sig.return_annotation is not int:
        raise TypeError(
            f"{mod.__name__}.ingest の戻り値アノテーションは int である必要があります"
            f"（現値: {sig.return_annotation!r}）"
        )

    return fn  # type: ignore[return-value]
