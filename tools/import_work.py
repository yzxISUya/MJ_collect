#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动选品导入：给 MJ 作品链接（或 job id），自动拉元数据+下图+并入选品清单。
与 collect_mj.py 的清单同一套 schema/台账——导入后联网去重自动认账。

eg:
  python tools/import_work.py https://www.midjourney.com/jobs/3b793474-4d69-4614-860f-562588fc3885?index=1
  python tools/import_work.py <url1> <url2> --name vol1
  python tools/import_work.py <url> --no-images

链接带 ?index=N 表示选批次里的第 N 格图（2×2 网格 0~3）；不带则取第 0 格。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collector import log, now_iso  # noqa: E402
from collector.config import ROOT, DEFAULT_CONFIG, load_config  # noqa: E402
from collector.fetch import http_get_retry  # noqa: E402
from collector.schema import JOB_PAGE, CDN_IMAGE, build_readme  # noqa: E402
from collector.store import write_json, load_shortlist, save_shortlist  # noqa: E402

JOB_STATUS_API = "https://www.midjourney.com/api/job-status"
UUID_RE = re.compile(r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")


def parse_work_ref(ref: str) -> tuple[str, int]:
    """'链接?index=1' 或裸 uuid → (job_id, 格子序号)。"""
    m = UUID_RE.search(ref)
    if not m:
        raise ValueError(f"识别不出 job id: {ref}")
    idx = 0
    mi = re.search(r"[?&]index=(\d+)", ref)
    if mi:
        idx = int(mi.group(1))
    return m.group(1), idx


def http_post_json(url: str, payload: dict, timeout: float, retries: int = 2):
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, method="POST", headers={
                "x-csrf-protection": "1",
                "Content-Type": "application/json",
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp), None
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    return None, str(last)


def parse_command(full_command: str) -> tuple[str, dict]:
    """'prompt 文本 --ar 2:3 --v 8.2 …' → (prompt 文本, params 字典)。
    参数旗标从 ' --' 处切开；原始完整命令另存 entry.command，一字不丢。"""
    parts = (full_command or "").split(" --")
    text = parts[0].strip()
    params = {
        "version": None, "ar": None, "stylize": None, "chaos": None, "weird": None,
        "seed": None, "raw": False, "no": [], "sref": [], "personalize": None,
    }
    for seg in parts[1:]:
        toks = seg.split()
        if not toks:
            continue
        flag = toks[0].lower()
        val = " ".join(toks[1:]).strip()
        if flag == "ar":
            params["ar"] = val
        elif flag in ("v", "version"):
            params["version"] = val
        elif flag == "raw":
            params["raw"] = True
        elif flag == "profile":
            params["personalize"] = val or None
        elif flag == "sref":
            params["sref"] = val.split()
        elif flag == "no":
            params["no"] = [val] if val else []
        elif flag in ("stylize", "chaos", "weird", "seed"):
            try:
                params[flag] = int(val) if flag == "seed" else float(val)
            except ValueError:
                params[flag] = val
        # 其余旗标（--hd 等）保留在 command 里，不进 params
    return text, params


def image_candidates(job_id: str, cell: int) -> list:
    """图片质量降级链：原图 PNG → 2048 PNG → 640 webp 预览。
    原图档（0_{cell}.png）即网页『下载』按钮给的 ~6MB 版本，免登录可取。"""
    base = f"https://cdn.midjourney.com/{job_id}/0_{cell}"
    return [
        (f"{base}.png", ".png"),
        (f"{base}_2048_N.png", ".png"),
        (f"{base}_640_N.webp", ".webp"),
    ]


def entry_from_job_status(raw: dict, index: int, fetched_at: str) -> dict:
    uuid = raw.get("id", "")
    cmd = raw.get("full_command") or ""
    text, params = parse_command(cmd)
    et = raw.get("enqueue_time")
    publish = None
    if et:
        try:
            publish = datetime.fromisoformat(et).astimezone().strftime("%Y-%m-%d %H:%M")
        except ValueError:
            publish = et
    return {
        "id": uuid if not index else f"{uuid}-{index}",
        "source": "midjourney",
        "author": raw.get("display_name") or raw.get("username_v2"),
        "prompt_text": text,
        "command": cmd,  # 原始完整命令（含全部参数）一字不动
        "params": params,
        "size": {"width": raw.get("width"), "height": raw.get("height")},
        "job_type": raw.get("job_type"),
        "publish_time": publish,
        "feed_rank": None,
        "source_url": JOB_PAGE.format(job_id=uuid) + (f"?index={index}" if index else ""),
        "image": {
            "url": CDN_IMAGE.format(job_id=uuid, grid=index, width="640"),
            "local": None,
        },
        "fetched_at": fetched_at,
        "notes": "手动导入",
        "match": None,
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    p = argparse.ArgumentParser(description="手动选品导入：MJ 链接 → 元数据+图片+选品清单")
    p.add_argument("works", nargs="+", help="MJ 作品链接或 job id（可多个）")
    p.add_argument("--name", default="vol1", help="选品清单文件名（不含扩展名）［默认 vol1］")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    p.add_argument("--no-images", action="store_true", help="不下载图片")
    p.add_argument("--timeout", type=float, default=20.0, help="请求超时秒［默认 20］")
    args = p.parse_args()

    cfg = load_config(args.config)
    img_cfg = cfg["图片"]

    # 1) 解析作品引用
    refs: list = []  # [(uuid, index)]
    for w in args.works:
        try:
            refs.append(parse_work_ref(w))
        except ValueError as e:
            log(f"[!] {e}")
            return 1

    # 2) 拉元数据（批量一次请求）
    log(f"拉取 {len(refs)} 个作品的元数据…")
    items, err = http_post_json(JOB_STATUS_API, {
        "jobIds": [u for u, _ in refs],
        "_frontend_source": "useJobSubmitter_fetchJobStatus",
    }, timeout=args.timeout)
    if items is None:
        log(f"[!] 元数据接口失败: {err}")
        return 1
    by_id = {it.get("id"): it for it in items}

    # 3) 并入选品清单（已存在的跳过，台账同步记账）
    fetched_at = now_iso()
    shortlist_path = ROOT / cfg["输出"]["选品目录"] / f"{args.name}.json"
    loaded = load_shortlist(shortlist_path)
    if loaded is None:
        log(f"[!] 旧清单无法解析，拒绝合并: {shortlist_path.relative_to(ROOT)}")
        return 1
    existing, file_consumed = loaded
    for e in existing:
        if isinstance(e, dict) and e.get("id") and e["id"] not in file_consumed:
            file_consumed.append(e["id"])

    new_entries: list = []
    for uuid, index in refs:
        raw = by_id.get(uuid)
        if raw is None:
            log(f"[!] 接口未返回该作品: {uuid}")
            continue
        entry = entry_from_job_status(raw, index, fetched_at)
        if entry["id"] in file_consumed:
            log(f"  跳过（清单里已有）: {entry['id']}")
            continue
        if raw.get("current_status") not in (None, "completed"):
            log(f"  跳过（状态 {raw.get('current_status')}）: {entry['id']}")
            continue
        new_entries.append(entry)
        file_consumed.append(entry["id"])
        log(f"  + {entry['id']} | {entry['author']} | {entry['prompt_text'][:40]!r}")

    if not new_entries:
        log("没有可导入的新作品。")
        return 0

    existing.extend(new_entries)
    fetch_info = {
        "时间": fetched_at,
        "来源": "手动导入",
        "导入条数": len(new_entries),
        "清单总条数": len(existing),
    }
    save_shortlist(shortlist_path, existing, file_consumed, build_readme(fetch_info, cfg["筛选"], img_cfg))
    log(f"选品清单 -> {shortlist_path.relative_to(ROOT)}（本次导入 {len(new_entries)} 条，共 {len(existing)} 条）")

    # 4) 下图（原图优先，按格子取图，文件名用条目 id）
    if not args.no_images and img_cfg.get("下载图片", True):
        day_dir = ROOT / cfg["输出"]["图片目录"] / datetime.now().strftime("%Y-%m-%d")
        for entry in new_entries:
            uuid = entry["id"].rsplit("-", 1)[0] if entry["id"].count("-") > 4 else entry["id"]
            # id 形如 {uuid}-{cell} 或 {uuid}；从 source_url 取格子更稳
            mi = re.search(r"[?&]index=(\d+)", entry["source_url"] or "")
            cell = int(mi.group(1)) if mi else 0
            saved = False
            for url, ext in image_candidates(uuid, cell):
                body, derr = http_get_retry(url, timeout=args.timeout, image=True)
                if body is None:
                    continue
                path = day_dir / f"{entry['id']}{ext}"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(body)
                entry["image"]["url"] = url
                entry["image"]["local"] = str(path.relative_to(ROOT)).replace("\\", "/")
                log(f"  图片 -> {entry['image']['local']}（{len(body)/1024:.0f} KB）")
                saved = True
                break
            if not saved:
                log(f"  [!] 图片失败 {entry['id']}: {derr}")
        save_shortlist(shortlist_path, existing, file_consumed,
                       build_readme(fetch_info, cfg["筛选"], img_cfg))

    log("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
