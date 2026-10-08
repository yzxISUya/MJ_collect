# -*- coding: utf-8 -*-
"""本地 JSON 状态：读写 / 已见集合 / 消费台账 / 池子库存 / 选品清单。"""

from __future__ import annotations

import json
from pathlib import Path

from collector.schema import MANUAL_ENTRY_TEMPLATE


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: Path):
    """读 JSON，失败返回 None。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _iter_pool_files(d: Path):
    if d.is_dir():
        for f in sorted(d.glob("*.json")):  # 文件名含时间戳，字典序=时间序
            yield f


def load_seen_ids(*dirs: Path) -> set:
    """已见集合：历史 pool/shortlist 里出现过的全部 id（联网增量去重用）。"""
    seen: set = set()
    for d in dirs:
        for f in _iter_pool_files(d):
            data = read_json(f)
            if not isinstance(data, dict):
                continue
            for e in data.get("entries") or []:
                if isinstance(e, dict) and e.get("id"):
                    seen.add(e["id"])
            for cid in data.get("_consumed_ids") or []:
                if cid:
                    seen.add(cid)
    return seen


def load_consumed(shortlist_dir: Path) -> set:
    """已消费台账：进过任何清单的 id（entries 现存 + _consumed_ids 历史）。
    删条目不清台账——保证『删掉=彻底放弃』不会被库存回补破坏。"""
    consumed: set = set()
    for f in _iter_pool_files(shortlist_dir):
        data = read_json(f)
        if not isinstance(data, dict):
            continue
        for e in data.get("entries") or []:
            if isinstance(e, dict) and e.get("id"):
                consumed.add(e["id"])
        for cid in data.get("_consumed_ids") or []:
            if cid:
                consumed.add(cid)
    return consumed


def load_inventory(pool_dir: Path, consumed: set) -> list:
    """池子库存：全部达线入库过的作品，去掉被任何清单消费过的。
    按池文件名升序 + 文件内原序 = 入库先后（FIFO，约等于热榜顺序）。"""
    inventory: list = []
    seen_ids: set = set()
    for f in _iter_pool_files(pool_dir):
        data = read_json(f)
        if not isinstance(data, dict):
            continue
        for e in data.get("entries") or []:
            if not isinstance(e, dict):
                continue
            eid = e.get("id")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            if eid not in consumed:
                inventory.append(e)
    return inventory


def load_shortlist(path: Path):
    """读选品清单，返回 (entries, consumed_ids)；文件不存在=空；解析失败=None。"""
    if not path.exists():
        return [], []
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    return data.get("entries") or [], list(data.get("_consumed_ids") or [])


def save_shortlist(path: Path, entries: list, consumed_ids: list, readme: dict) -> None:
    write_json(path, {
        "type": "shortlist",
        "_readme": readme,
        "entries": entries,
        "_consumed_ids": consumed_ids,
        "_manual_entry_template": MANUAL_ENTRY_TEMPLATE,
    })
