from __future__ import annotations

from collections import defaultdict

from app.core.config import Settings
from app.rag.schemas import RetrievedChunk

# -----------------------------------------------------------------------------
# 役割: RERANKING logic_03 — 構造化チャンキング (chunking_logic_06) 由来の階層
# metadata を活かし、距離検索で取得した数十件の候補を LLM 向けに圧縮する。
#
# reranking_logic_02 との違い:
#   子→親への集約を削除。孫→子への集約は維持する。
#   親チャンクは子が存在する場合でも統合されず、スコア閾値を超えれば独立して残る。
#   これにより長文の親チャンクが LLM コンテキストを圧迫することを防ぐ。
#
# 主な処理:
#   1. 元データ件数が下限 (_MIN_TOP_K) 未満ならフィルタせずそのまま返す。
#   2. top_k を 1/6 と 5 の大きい方で再構築する。
#   3. metadata の chunk_role / parent_chunk_id / child_chunk_id を用いて重複・冗長
#      チャンクを段階的に削除し、孫→子へ代表スコアを引き継ぐ。
#   4. 最高スコアの _SCORE_RETAIN_RATIO 倍未満を除外し、上位 new_top_k 件を返す。
#
# 互換性: 必要キーが metadata に無い候補（非構造化チャンクなど）はグループ化
# 対象から外し、削除や引き継ぎの影響を受けない設計とする。
#
# 子省略（merge_child）ケースの特記:
#   chunking_logic_06 は親章内の子(条)が1件のみの場合、子チャンクを emit せず
#   親チャンクが子の metadata を兼ねる（child_chunk_id = 親自身の chunk_id）。
#   一方その子配下の孫は独立 emit されるため、孫の child_chunk_id が指す ID は
#   ベクトル DB に存在しない。これを「孤立孫」として検出し、step 4b で親へ
#   引き継いで削除することで親・孫の重複出力を防ぐ。
# -----------------------------------------------------------------------------


_MIN_TOP_K = 5
_TOP_K_DIVISOR = 6
# ベストスコアに対する保持下限の比率。小さいほど多く残り、大きいほど絞られる。
_SCORE_RETAIN_RATIO = 0.6


def rerank(
    settings: Settings,
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
) -> tuple[list[RetrievedChunk], int]:
    _ = (settings, query)

    if not chunks:
        return [], 0

    # 元データ件数が下限未満ならフィルタは適用せずそのまま返す。
    # 少数データに階層集約や平均カットを効かせると、LLM 文脈が極端に痩せるため。
    if len(chunks) < _MIN_TOP_K:
        return list(chunks), len(chunks)

    new_top_k = max(_MIN_TOP_K, top_k // _TOP_K_DIVISOR)

    # final_score を集約用スコアとして使う（vector hit では vector_score_norm と同値）。
    items = [_Item(chunk=c, score=float(c.final_score)) for c in chunks]

    # 1. 同一 chunk_id（論理チャンクID）は最高スコアのみへ集約
    items = _dedupe_by_chunk_id(items)

    # 2. 同一 parent_chunk_id 内に parent がいれば、低スコアの child/grandchild を削除
    items = _drop_lower_descendants(
        items,
        group_key="parent_chunk_id",
        keep_role="parent",
        drop_roles={"child", "grandchild"},
    )

    # 3. 同一 child_chunk_id 内に child がいれば、低スコアの grandchild を削除
    items = _drop_lower_descendants(
        items,
        group_key="child_chunk_id",
        keep_role="child",
        drop_roles={"grandchild"},
    )

    # 4. 残った grandchild の最高スコアを同 child_chunk_id の代表 child へ引き継ぎ、
    #    同 child_chunk_id の grandchild は全削除。同 child_chunk_id の child が複数
    #    あった場合は元スコア最高の child のみが受け取る（他の child は維持）。
    items = _promote_score(
        items,
        group_key="child_chunk_id",
        from_role="grandchild",
        to_role="child",
        require_min_from_count=1,
    )

    # 4b. 子省略（merge_child）ケース: step 4 で対応 child が見つからず残った孤立孫を、
    #     parent_chunk_id をキーに親チャンクへ引き継いで削除する。
    #     （孤立孫とは child_chunk_id に対応する chunk が存在しない grandchild のこと）
    items = _promote_orphan_grandchildren(items)

    # 5. 最高スコアの _SCORE_RETAIN_RATIO 倍を閾値とし、それ未満を除外する。
    max_score = max((it.score for it in items), default=0.0)
    threshold = max_score * _SCORE_RETAIN_RATIO
    items = [it for it in items if it.score >= threshold]

    # 6. 残数 < new_top_k なら topK を残数に修正、そうでなければ上位 new_top_k を返す。
    items.sort(key=lambda it: it.score, reverse=True)
    if len(items) < new_top_k:
        new_top_k = len(items)
    else:
        items = items[:new_top_k]

    return [it.to_chunk() for it in items], new_top_k


# -----------------------------------------------------------------------------
# 内部処理
# -----------------------------------------------------------------------------


class _Item:
    """rerank ワークセットの軽量ラッパ。引き継ぎでスコアを上書きするため、
    score を chunk から独立して保持する。
    """

    __slots__ = ("chunk", "score")

    def __init__(self, chunk: RetrievedChunk, score: float) -> None:
        self.chunk = chunk
        self.score = score

    @property
    def role(self) -> str:
        meta = self.chunk.metadata or {}
        return str(meta.get("chunk_role", ""))

    def meta(self, key: str) -> str | None:
        meta = self.chunk.metadata or {}
        v = meta.get(key)
        return None if v is None else str(v)

    def to_chunk(self) -> RetrievedChunk:
        # 引き継ぎ後のスコアを final_score として返却用 chunk に反映する。
        return self.chunk.model_copy(update={"final_score": self.score})


def _dedupe_by_chunk_id(items: list[_Item]) -> list[_Item]:
    """同一 chunk_id を最高スコアのみに集約する。chunk_id が空の項目は素通し。"""
    by_id: dict[str, _Item] = {}
    extras: list[_Item] = []
    for it in items:
        cid = it.chunk.chunk_id
        if not cid:
            extras.append(it)
            continue
        existing = by_id.get(cid)
        if existing is None or it.score > existing.score:
            by_id[cid] = it
    return list(by_id.values()) + extras


def _drop_lower_descendants(
    items: list[_Item],
    *,
    group_key: str,
    keep_role: str,
    drop_roles: set[str],
) -> list[_Item]:
    """同一 group_key グループ内で keep_role 最高スコア未満の drop_roles を削除する。"""
    groups: dict[str, list[_Item]] = defaultdict(list)
    others: list[_Item] = []
    for it in items:
        gk = it.meta(group_key)
        if gk is None:
            # 必要 metadata が無い候補はグループ化対象外（互換性維持のため素通し）。
            others.append(it)
            continue
        groups[gk].append(it)

    survivors: list[_Item] = list(others)
    for members in groups.values():
        anchor = max(
            (m.score for m in members if m.role == keep_role),
            default=None,
        )
        if anchor is None:
            survivors.extend(members)
            continue
        for m in members:
            if m.role in drop_roles and m.score < anchor:
                continue
            survivors.append(m)
    return survivors


def _promote_orphan_grandchildren(items: list[_Item]) -> list[_Item]:
    """step 4 を通過して残った grandchild を親チャンクへ引き継いで削除する。

    step 4 通過後に残る grandchild は、対応する child が存在しない（子省略または
    step 2 で削除済み）ため step 4 でスルーされた「孤立孫」である。

    注意: RetrievedChunk.chunk_id は chunker.py が振る連番 "{doc_id}:{i}" 形式だが、
    parent_chunk_id / child_chunk_id は chunking_logic_06 が生成する
    "chunk_sec_{xxx}" 形式で別の名前空間。よって chunk_id_set による照合は使えない。
    親チャンクは自身の meta("parent_chunk_id") が自身の section-based base_id と
    同値になる性質を利用し、同じ値をキーに parent_by_pk でルックアップする。
    """
    grandchildren = [it for it in items if it.role == "grandchild"]
    if not grandchildren:
        return items

    # parent_chunk_id（section-based ID）でグループ化する。
    groups: dict[str, list[_Item]] = defaultdict(list)
    for it in grandchildren:
        pk = it.meta("parent_chunk_id")
        if pk:
            groups[pk].append(it)

    if not groups:
        return items

    # 親チャンクを meta("parent_chunk_id") で逆引きする。
    # 親チャンクは meta("parent_chunk_id") == 自身の section-based ID（自己参照）。
    parent_by_pk: dict[str, _Item] = {}
    for it in items:
        if it.role == "parent":
            pk = it.meta("parent_chunk_id")
            if pk is None:
                continue
            existing = parent_by_pk.get(pk)
            if existing is None or it.score > existing.score:
                parent_by_pk[pk] = it

    orphan_set = set(grandchildren)
    survivors = [it for it in items if it not in orphan_set]

    for pk_val, gcs in groups.items():
        parent_item = parent_by_pk.get(pk_val)
        if parent_item is None:
            # 対応する親チャンクが取得されていなければ grandchild を維持する。
            survivors.extend(gcs)
            continue
        donor_score = max(it.score for it in gcs)
        if donor_score > parent_item.score:
            parent_item.score = donor_score
        # grandchild は削除（survivors には追加しない）。

    return survivors


def _promote_score(
    items: list[_Item],
    *,
    group_key: str,
    from_role: str,
    to_role: str,
    require_min_from_count: int,
) -> list[_Item]:
    """同一 group_key 内の from_role 最高スコアを to_role 代表に引き継ぎ、from_role
    を全削除する。to_role が複数あれば「元スコアが最も高いもの」が代表となり、他は維持。

    require_min_from_count: from_role がこの件数未満ならグループに対して何もしない。
    """
    groups: dict[str, list[_Item]] = defaultdict(list)
    others: list[_Item] = []
    for it in items:
        gk = it.meta(group_key)
        if gk is None:
            others.append(it)
            continue
        groups[gk].append(it)

    survivors: list[_Item] = list(others)
    for members in groups.values():
        from_items = [m for m in members if m.role == from_role]
        to_items = [m for m in members if m.role == to_role]
        if not to_items or len(from_items) < require_min_from_count:
            survivors.extend(members)
            continue
        donor_score = max(m.score for m in from_items)
        # 元スコア最高の to_role 代表が引き継ぎを受ける。既存スコアを下回る引き継ぎは
        # 行わない（上位ロールが既に強いとき、下位スコアで上書きすると関連性評価が劣化）。
        recipient = max(to_items, key=lambda m: m.score)
        if donor_score > recipient.score:
            recipient.score = donor_score
        for m in members:
            if m.role == from_role:
                continue
            survivors.append(m)
    return survivors
