# -*- coding: utf-8 -*-
"""本地 JSON 状态：原子读写 / 消费台账（兼已见集合）/ 选品清单。

历史注记：曾有一层『采集池 data/pool』存达线溢出作品作库存（第 0 页）。
2026-10-08 体检移除：v3 全扫模式下溢出条目只要还在 MJ 的 2 天可见窗口内，
下次运行自然重新入池竞争，库存唯一独特价值（超窗续命）用不上，机制却牵连
整个已见/去重层。移除后『已见』与『消费台账』合一，删除语义不变。
"""

from __future__ import annotations

import json
from pathlib import Path

from collector.schema import MANUAL_ENTRY_TEMPLATE


def write_json(path: Path, obj) -> None:
    """原子写入：先写临时文件再替换，崩溃也不留半截 JSON（清单是手改文件，安全第一）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def read_json(path: Path):
    """读 JSON，失败返回 None。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_consumed(shortlist_dir: Path) -> set:
    """已消费台账 = 已见集合：进过任何清单的 id（entries 现存 + _consumed_ids 历史）。
    双重职责：联网去重跳过这些 id；删条目不清台账保证『删掉=彻底放弃』不复活。"""
    consumed: set = set()
    if not shortlist_dir.is_dir():
        return consumed
    for f in sorted(shortlist_dir.glob("*.json")):
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
