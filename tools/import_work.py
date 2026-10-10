#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动选品导入：MJ 作品链接 → 元数据 + 原图 + 并入【本期文件夹】。
布局（posts/<期>/，与自动采集的草稿区分开）：
  posts/vol2/vol2.txt     链接清单（# 注释，行序=导入序）
  posts/vol2/vol2.json    作品条目（本工具写入）
  posts/vol2/vol2_pic/    作品原图（本工具写入）
  posts/vol2/vol2.md      最终稿件（写作环节产出）
消费台账跨树互认——手动导入的作品不会被采集器重复捞来。

eg:
  python tools/import_work.py --file posts/vol2/vol2.txt     # 批量：清单文件（主入口）
  python tools/import_work.py <url> [url2 ...]               # 散手：单条/多条命令行
  python tools/import_work.py <url> --name vol2 --no-images
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
from collector.store import (  # noqa: E402
    load_consumed, load_shortlist, save_shortlist,
)

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


def extract_refs_from_file(path: Path) -> list:
    """清单文件 → [(uuid, index)]，按出现顺序。
    宽容解析：逐行去 # 注释，空白分词后凡含 uuid 的片段都算一条——txt/md 通吃。"""
    refs: list = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("#", 1)[0]
        for tok in code.split():
            if not UUID_RE.search(tok):
                continue
            try:
                refs.append(parse_work_ref(tok))
            except ValueError as e:
                log(f"  [!] 第 {line_no} 行: {e}")
    return refs


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

    p = argparse.ArgumentParser(
        description="手动选品导入：MJ 链接（或链接清单文件）→ 元数据+原图+手动清单",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "清单文件：每行 1 个链接（可带 ?index=N），# 注释，行序=导入序。\n"
            "文件名即期号，按期文件夹组织：posts/<期>/<期>.txt → 同目录 <期>.json + <期>_pic/\n"
            "示例:\n"
            "  python tools/import_work.py --file posts/vol2/vol2.txt\n"
            "  python tools/import_work.py https://www.midjourney.com/jobs/xxx?index=1\n"
        ),
    )
    p.add_argument("works", nargs="*", help="MJ 作品链接或 job id（可多个）")
    p.add_argument("--file", type=Path, help="链接清单文件（文件名即期号）")
    p.add_argument("--name", default=None,
                   help="期号（默认=清单文件名，或 vol1）→ 输出到 posts/<期>/ 文件夹")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    p.add_argument("--no-images", action="store_true", help="不下载图片")
    p.add_argument("--timeout", type=float, default=20.0, help="请求超时秒［默认 20］")
    args = p.parse_args()

    cfg = load_config(args.config)
    img_cfg = cfg["图片"]

    # 1) 收集作品引用：清单文件（主入口）+ 命令行散手
    refs: list = []
    if args.file:
        if not args.file.exists():
            log(f"[!] 清单文件不存在: {args.file}")
            return 1
        file_refs = extract_refs_from_file(args.file)
        log(f"清单 {args.file.name}: {len(file_refs)} 个链接")
        refs.extend(file_refs)
    for w in args.works:
        try:
            refs.append(parse_work_ref(w))
        except ValueError as e:
            log(f"[!] {e}")
            return 1
    if not refs:
        log("没有可导入的链接。")
        return 1

    name = args.name or (args.file.stem if args.file else "vol1")
    # 期目录布局（posts/<期>/）：<期>.txt 清单、<期>.json 作品、<期>_pic/ 图片、<期>.md 稿件
    if args.file:
        fpath = args.file if args.file.is_absolute() else ROOT / args.file
        issue_dir = fpath.parent if fpath.parent.name == name else fpath.parent / name
    else:
        issue_dir = ROOT / "posts" / name
    issue_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = issue_dir / f"{name}.json"
    pic_dir = issue_dir / f"{name}_pic"

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

    # 3) 并入本期清单（跨树查重：手动/自动清单里出现过的一律跳过）
    fetched_at = now_iso()
    global_consumed = load_consumed(shortlist_path.parent)
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
        if entry["id"] in global_consumed or entry["id"] in file_consumed:
            log(f"  跳过（已入库）: {entry['id']}")
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
    log(f"手动清单 -> {shortlist_path.relative_to(ROOT)}（本次导入 {len(new_entries)} 条，共 {len(existing)} 条）")

    # 4) 下图（原图优先，按格子取图，文件名用条目 id）→ <期>_pic/
    if not args.no_images and img_cfg.get("下载图片", True):
        pic_dir.mkdir(parents=True, exist_ok=True)
        for entry in new_entries:
            mi = re.search(r"[?&]index=(\d+)", entry["source_url"] or "")
            cell = int(mi.group(1)) if mi else 0
            saved = False
            for url, ext in image_candidates(entry["id"].rsplit("-", 1)[0] if cell else entry["id"], cell):
                body, derr = http_get_retry(url, timeout=args.timeout, image=True)
                if body is None:
                    continue
                path = pic_dir / f"{entry['id']}{ext}"
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
