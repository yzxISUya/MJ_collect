#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MJ 采集器 — Midjourney Explore 公开接口作品抓取（免登录）

从 https://www.midjourney.com/api/explore 拉取热榜作品，产出三种文件：

  1. data/pool/mj_<时间戳>.json        本次抓取全量存档（选品原料，自动管理）
  2. data/shortlist/<名字>.json       按筛选条件过滤后的选品清单（人工可删添）
  3. images/<日期>/<job_id>.webp      选品条目作品图（可关）

可调参数见 config/collector.json；命令行参数可临时覆盖配置文件。

用法示例:
  python tools/collect_mj.py
  python tools/collect_mj.py --total 50 --no-images
  python tools/collect_mj.py --keyword watercolor aquarelle --name vol1
  python tools/collect_mj.py --exclude portrait realistic --ar 9:16 3:4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- 常量

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "collector.json"

EXPLORE_API = "https://www.midjourney.com/api/explore"
JOB_PAGE = "https://www.midjourney.com/jobs/{job_id}"
CDN_IMAGE = "https://cdn.midjourney.com/{job_id}/0_{grid}_{width}_N.webp"

# MJ 站点对非浏览器请求较敏感，统一用常见浏览器 UA
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

PER_PAGE = 50  # MJ explore 接口每页固定 50 条

# ---------------------------------------------------------------- 基础工具


def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def http_get(url: str, timeout: float, image: bool = False) -> bytes:
    """GET 请求。image=True 时带上图床防盗链头。"""
    headers = {
        "User-Agent": USER_AGENT,
        "x-csrf-protection": "1",
    }
    if image:
        headers["Referer"] = "https://www.midjourney.com/"
        headers["Sec-Fetch-Dest"] = "image"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_get_retry(url: str, timeout: float, image: bool = False, retries: int = 2):
    """带重试的 GET，返回 (bytes|None, 错误信息|None)。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return http_get(url, timeout=timeout, image=image), None
        except Exception as e:  # noqa: BLE001 — 网络层错误统一收集
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    return None, str(last_err)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- 配置


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_cli_config(cfg: dict, args: argparse.Namespace) -> dict:
    """命令行参数覆盖配置文件（None 表示未指定）。"""
    grab, filt, img = cfg["抓取"], cfg["筛选"], cfg["图片"]
    if args.total is not None:
        grab["总条数"] = args.total
    if args.start_page is not None:
        grab["起始页"] = args.start_page
    if args.delay is not None:
        grab["请求间隔秒"] = args.delay
    if args.timeout is not None:
        grab["请求超时秒"] = args.timeout
    if args.keyword is not None:
        filt["包含关键词"] = args.keyword
    if args.exclude is not None:
        filt["排除关键词"] = args.exclude
    if args.ar is not None:
        filt["画幅"] = args.ar
    if args.min_len is not None:
        filt["prompt字数下限"] = args.min_len
    if args.max_len is not None:
        filt["prompt字数上限"] = args.max_len
    if args.no_images:
        img["下载图片"] = False
    if args.image_width is not None:
        img["宽度档"] = args.image_width
    if args.grid is not None:
        img["格子"] = args.grid
    return cfg


# ---------------------------------------------------------------- 抓取


def fetch_pool(total: int, start_page: int, delay: float, timeout: float):
    """按页抓取直到凑够 total 条或接口给空。返回 (原始条目列表, 错误列表)。"""
    raw_items: list = []
    errors: list = []
    page = start_page
    while len(raw_items) < total:
        url = f"{EXPLORE_API}?page={page}&feed=top&_ql=explore"
        body, err = http_get_retry(url, timeout=timeout)
        if body is None:
            errors.append({"环节": "抓取列表", "页码": page, "错误": err})
            log(f"  [!] 第 {page} 页抓取失败: {err}")
            break
        try:
            items = json.loads(body)
        except json.JSONDecodeError as e:
            errors.append({"环节": "解析列表", "页码": page, "错误": str(e)})
            break
        if not items:
            log(f"  第 {page} 页为空，可见数据已抓完")
            break
        raw_items.extend(items)
        log(f"  第 {page} 页: +{len(items)} 条（累计 {len(raw_items)}）")
        page += 1
        time.sleep(delay)
    return raw_items[:total], errors


# ---------------------------------------------------------------- 解析


def parse_entry(raw: dict, fetched_at: str) -> dict:
    """把 MJ 原始条目整理成可读的选品条目结构。"""
    prompt = raw.get("prompt") or {}
    segments = prompt.get("decodedPrompt") or []
    prompt_text = " ".join(seg.get("content", "") for seg in segments).strip()

    ar = prompt.get("ar")
    ar_str = f"{ar['w']}:{ar['h']}" if ar else None

    srefs = prompt.get("styleRef") or []
    personalize = prompt.get("personalize") or []

    job_id = raw.get("id", "")
    enqueue_ms = raw.get("enqueue_time")
    enqueue_str = (
        datetime.fromtimestamp(enqueue_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M")
        if enqueue_ms
        else None
    )

    return {
        "id": job_id,
        "source": "midjourney",
        "author": raw.get("display_name") or raw.get("username_v2"),
        "prompt_text": prompt_text,
        "params": {
            "version": prompt.get("version"),
            "ar": ar_str,
            "stylize": prompt.get("stylize"),
            "chaos": prompt.get("chaos"),
            "weird": prompt.get("weird"),
            "seed": prompt.get("seed"),
            "raw": bool(prompt.get("styleRaw")),
            "no": prompt.get("no") or [],
            "sref": [s.get("content") for s in srefs],
            "personalize": personalize[0].get("content") if personalize else None,
        },
        "size": {"width": raw.get("width"), "height": raw.get("height")},
        "job_type": raw.get("job_type"),
        "publish_time": enqueue_str,
        "source_url": JOB_PAGE.format(job_id=job_id),
        "image": {
            "url": CDN_IMAGE.format(
                job_id=job_id, grid="0", width="640"  # 占位，下载时按配置改写
            ),
            "local": None,
        },
        "fetched_at": fetched_at,
        "notes": "",
    }


# ---------------------------------------------------------------- 筛选


def match_filters(entry: dict, filt: dict) -> bool:
    text = entry["prompt_text"].lower()

    includes = [k.lower() for k in (filt.get("包含关键词") or [])]
    if includes and not any(k in text for k in includes):
        return False

    excludes = [k.lower() for k in (filt.get("排除关键词") or [])]
    if excludes and any(k in text for k in excludes):
        return False

    ars = filt.get("画幅") or []
    if ars and entry["params"]["ar"] not in ars:
        return False

    n = len(entry["prompt_text"])
    min_len = filt.get("prompt字数下限") or 0
    max_len = filt.get("prompt字数上限")
    if n < min_len:
        return False
    if max_len is not None and n > max_len:
        return False

    return True


# ---------------------------------------------------------------- 图片


def download_images(entries: list, img_cfg: dict, images_root: Path,
                    delay: float, timeout: float, errors: list) -> None:
    width = img_cfg["宽度档"]
    grid = img_cfg["格子"]
    day_dir = images_root / datetime.now().strftime("%Y-%m-%d")
    total = len(entries)
    for i, entry in enumerate(entries, 1):
        url = CDN_IMAGE.format(job_id=entry["id"], grid=grid, width=width)
        entry["image"]["url"] = url
        body, err = http_get_retry(url, timeout=timeout, image=True)
        if body is None:
            errors.append({"环节": "下载图片", "id": entry["id"], "错误": err})
            log(f"  [!] 图片失败 {entry['id'][:8]}: {err}")
        else:
            path = day_dir / f"{entry['id']}.webp"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(body)
            entry["image"]["local"] = str(path.relative_to(ROOT)).replace("\\", "/")
            log(f"  [{i}/{total}] {entry['id'][:8]} -> {entry['image']['local']}")
        time.sleep(delay)


# ---------------------------------------------------------------- 文档头


def build_readme(fetch_info: dict, filt: dict, img_cfg: dict) -> dict:
    return {
        "怎么删": "删掉 entries 数组里不要的整个 { } 块（注意保留上下条目的逗号）。",
        "怎么加": (
            "把文件末尾的 _manual_entry_template 整个复制进 entries 数组，"
            "改好内容；手动条目 source 写 'manual'（或其他平台名），id 自拟唯一即可。"
        ),
        "注意": [
            "author（作者署名）与 source_url（源链接）是合规红线，必留。",
            "entries 里的条目顺序即排版候选顺序，可自由挪动。",
            "多余字段可自行添加，后续流水线会忽略不认识的字段。",
        ],
        "字段说明": {
            "id": "MJ job UUID；手动条目自拟唯一 id",
            "source": "来源：midjourney / manual / 其他平台名",
            "author": "作者署名（必填）",
            "prompt_text": "prompt 原文（一字不动）",
            "params": "MJ 参数（解析自原文）：版本/画幅ar/stylize/chaos/weird/seed/raw/no排除词/sref风格参考/personalize",
            "size": "图像素尺寸",
            "job_type": "MJ 任务类型",
            "publish_time": "作品发布（入队）时间",
            "source_url": "作品源链接（必填）",
            "image.url": "图片直链（384/640 档）",
            "image.local": "下载后的本地路径，没下图则为 null",
            "fetched_at": "采集时间",
            "notes": "人工备注位（自由使用）",
        },
        "本次运行": fetch_info,
        "筛选条件": filt,
        "图片设置": img_cfg,
    }


MANUAL_ENTRY_TEMPLATE = {
    "id": "manual-YYYYMMDD-01",
    "source": "manual",
    "author": "原作者署名（必填）",
    "prompt_text": "prompt 原文，没有就留空字符串",
    "params": {
        "version": None,
        "ar": None,
        "stylize": None,
        "chaos": None,
        "weird": None,
        "seed": None,
        "raw": False,
        "no": [],
        "sref": [],
        "personalize": None,
    },
    "size": {"width": None, "height": None},
    "job_type": None,
    "publish_time": None,
    "source_url": "作品源链接（必填）",
    "image": {"url": None, "local": None},
    "fetched_at": "录入时间",
    "notes": "",
}


# ---------------------------------------------------------------- 主流程


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MJ 采集器：抓取 Midjourney 热榜作品，产出可编辑的选品 JSON。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python tools/collect_mj.py\n"
            "  python tools/collect_mj.py --total 50 --no-images\n"
            "  python tools/collect_mj.py --keyword watercolor aquarelle --name vol1\n"
        ),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    p.add_argument("--total", type=int, help="抓取总条数（默认 150，免登录上限约 150）")
    p.add_argument("--start-page", type=int, help="起始页码（默认 1）")
    p.add_argument("--delay", type=float, help="请求间隔秒（默认 1.5，勿低于 1）")
    p.add_argument("--timeout", type=float, help="单请求超时秒（默认 20）")
    p.add_argument("--keyword", nargs="+", help="包含关键词（命中任一即收）")
    p.add_argument("--exclude", nargs="+", help="排除关键词（命中任一即弃）")
    p.add_argument("--ar", nargs="+", help="画幅筛选，如 9:16 3:4")
    p.add_argument("--min-len", type=int, help="prompt 字数下限")
    p.add_argument("--max-len", type=int, help="prompt 字数上限")
    p.add_argument("--no-images", action="store_true", help="不下载图片")
    p.add_argument("--image-width", type=int, choices=[384, 640], help="图片宽度档")
    p.add_argument("--grid", type=int, help="图片格子序号（2×2 网格，默认 0）")
    p.add_argument("--name", help="选品清单文件名（不含扩展名，默认今天日期）")
    return p


def main() -> int:
    # Windows 控制台编码兜底
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = build_parser().parse_args()
    cfg = merge_cli_config(load_config(args.config), args)
    grab, filt, img, out = cfg["抓取"], cfg["筛选"], cfg["图片"], cfg["输出"]

    fetched_at = now_iso()
    log(f"MJ 采集器 | 目标 {grab['总条数']} 条 | 间隔 {grab['请求间隔秒']}s")
    log(f"筛选: 包含={filt['包含关键词'] or '—'}  排除={filt['排除关键词'] or '—'}"
        f"  画幅={filt['画幅'] or '—'}  字数={filt['prompt字数下限']}~{filt['prompt字数上限'] or '∞'}")

    # 1) 抓取全量
    log("\n[1/4] 抓取热榜…")
    raw_items, errors = fetch_pool(
        total=grab["总条数"],
        start_page=grab["起始页"],
        delay=grab["请求间隔秒"],
        timeout=grab["请求超时秒"],
    )
    if not raw_items:
        log("没有抓到任何数据，退出。")
        return 1

    entries = [parse_entry(r, fetched_at) for r in raw_items]

    # 2) 全量存档（采集池）
    stamp = now_stamp()
    pool_path = ROOT / out["采集池目录"] / f"mj_{stamp}.json"
    write_json(pool_path, {
        "type": "pool",
        "fetched_at": fetched_at,
        "count": len(entries),
        "entries": entries,
    })
    log(f"[2/4] 采集池 -> {pool_path.relative_to(ROOT)}（{len(entries)} 条全量）")

    # 3) 筛选 → 选品清单
    matched = [e for e in entries if match_filters(e, filt)]
    name = args.name or datetime.now().strftime("%Y-%m-%d")
    shortlist_path = ROOT / out["选品目录"] / f"{name}.json"
    if shortlist_path.exists():
        log(f"[!] 选品清单已存在，拒绝覆盖: {shortlist_path.relative_to(ROOT)}")
        log(f"    请换名字运行（--name vol1），或自行删除/改名旧文件。")
        return 1
    fetch_info = {
        "时间": fetched_at,
        "抓取条数": len(entries),
        "筛中条数": len(matched),
        "报错数": len(errors),
    }
    write_json(shortlist_path, {
        "type": "shortlist",
        "_readme": build_readme(fetch_info, filt, img),
        "entries": matched,
        "_manual_entry_template": MANUAL_ENTRY_TEMPLATE,
    })
    log(f"[3/4] 选品清单 -> {shortlist_path.relative_to(ROOT)}（筛中 {len(matched)}/{len(entries)} 条）")

    # 4) 下载筛中条目的图片
    if img["下载图片"] and matched:
        log("[4/4] 下载图片…")
        download_images(
            matched, img, ROOT / out["图片目录"],
            delay=grab["请求间隔秒"], timeout=grab["请求超时秒"], errors=errors,
        )
        # 图片本地路径回写进选品清单
        write_json(shortlist_path, {
            "type": "shortlist",
            "_readme": build_readme(fetch_info, filt, img),
            "entries": matched,
            "_manual_entry_template": MANUAL_ENTRY_TEMPLATE,
        })
    else:
        log("[4/4] 跳过图片下载")

    if errors:
        err_path = ROOT / out["采集池目录"] / f"mj_{stamp}_errors.json"
        write_json(err_path, errors)
        log(f"\n[!] {len(errors)} 个错误，详见 {err_path.relative_to(ROOT)}")

    log("\n完成。下一步：打开选品清单，删掉不要的条目、按需手动补条目。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
