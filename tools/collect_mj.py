#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行参数可临时覆盖配置文件
eg:
  python tools/collect_mj.py
  python tools/collect_mj.py --target 15 --no-images
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

# 常量

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "collector.json"

EXPLORE_API = "https://www.midjourney.com/api/explore"
JOB_PAGE = "https://www.midjourney.com/jobs/{job_id}"
CDN_IMAGE = "https://cdn.midjourney.com/{job_id}/0_{grid}_{width}_N.webp"

# MJ站点对非浏览器请求较敏感，统一用常见浏览器 UA
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

PER_PAGE = 50  # MJ explore 接口每页固定 50 条

# 基础工具


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


def load_seen_ids(*dirs: Path) -> set:
    """汇总历史 pool/shortlist 文件里出现过的 job_id（跨次增量去重的『已见集合』）。"""
    seen: set = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
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
    if not shortlist_dir.is_dir():
        return consumed
    for f in sorted(shortlist_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
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
    """池子库存：全部筛中过的作品，去掉被任何清单消费过的。
    按池文件名（含时间戳）升序 + 文件内原序 = 入库先后（FIFO，约等于热榜顺序）。"""
    inventory: list = []
    seen_ids: set = set()
    if not pool_dir.is_dir():
        return inventory
    for f in sorted(pool_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
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


# 配置


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_cli_config(cfg: dict, args: argparse.Namespace) -> dict:
    """命令行参数覆盖配置文件（None 表示未指定）。"""
    grab, filt, img = cfg["抓取"], cfg["筛选"], cfg["图片"]
    if args.target is not None:
        filt["目标条数"] = args.target
    if args.include_seen:
        filt["去重"] = False
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


# 抓取


def fetch_and_match(target, filt: dict, start_page: int, delay: float,
                    timeout: float, fetched_at: str, seen: set):
    """逐页抓取并筛选（去重时跳过已见、只处理新作品），筛中凑够 target 条即停止翻页。
    target 为 null/0 表示不设上限。未命中条目不落盘（窗口内改筛选条件可回捞）。
    返回 (新抓条目, 全部筛中条目, 错误列表, 跳过已见条数)。
    """
    dedup = bool(filt.get("去重", True))
    entries: list = []
    matched: list = []
    errors: list = []
    skipped = 0
    page = start_page
    while target is None or target <= 0 or len(matched) < target:
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
        new_count = 0
        for raw in items:
            job_id = raw.get("id", "")
            if dedup:
                if job_id in seen:
                    skipped += 1
                    continue
                seen.add(job_id)  # 本次运行内也不重复
            entry = parse_entry(raw, fetched_at)
            entries.append(entry)
            new_count += 1
            if match_filters(entry, filt):
                matched.append(entry)
        log(f"  第 {page} 页: +{len(items)} 条（新 {new_count}，已见跳过 {len(items) - new_count}，筛中 {len(matched)}）")
        if target and len(matched) >= target:
            log(f"  筛中已达目标 {target} 条，停止翻页")
            break
        page += 1
        time.sleep(delay)
    return entries, matched, errors, skipped


# 解析


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


# 筛选


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


# 图片


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


# 文档头


def build_readme(fetch_info: dict, filt: dict, img_cfg: dict) -> dict:
    return {
        "怎么删": "删掉 entries 数组里不要的整个 { } 块（注意保留上下条目的逗号）。删除后不会被重新送上（_consumed_ids 台账仍记着它）；反悔就去 data/pool 复制回来。",
        "怎么加": (
            "把文件末尾的 _manual_entry_template 整个复制进 entries 数组，"
            "改好内容；手动条目 source 写 'manual'（或其他平台名），id 自拟唯一即可。"
        ),
        "注意": [
            "author（作者署名）与 source_url（源链接）是合规红线，必留。",
            "entries 里的条目顺序即排版候选顺序，可自由挪动。",
            "同名再运行=新条目追加到 entries 末尾（优先从池子库存取），已有条目（含手改/手补）一字不动。",
            "删掉的条目不会再被自动抓回（_consumed_ids 台账记着）；要找回就去 data/pool 对应文件复制。",
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
            "_consumed_ids": "消费台账（工具维护）：进过本清单的全部 id，删条目不清账",
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


# 主流程


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MJ 采集器：抓取 Midjourney 热榜作品，产出可编辑的选品 JSON。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python tools/collect_mj.py\n"
            "  python tools/collect_mj.py --target 15 --no-images\n"
            "  python tools/collect_mj.py --keyword watercolor aquarelle --name vol1\n"
        ),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    p.add_argument("--target", type=int, help="目标筛中条数，凑够即停（0=不限，默认 15）")
    p.add_argument("--include-seen", action="store_true",
                   help="关闭去重，包含历史上已抓过的作品（重看当前热榜）")
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
    p.add_argument("--name", help="选品清单文件名（不含扩展名，默认今天日期；同名自动追加合并）")
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
    target = filt.get("目标条数")
    dedup = bool(filt.get("去重", True))
    stamp = now_stamp()
    log(f"MJ 采集器 | 目标筛中 {target if target else '不限'} 条 | 间隔 {grab['请求间隔秒']}s")
    log(f"筛选: 包含={filt['包含关键词'] or '—'}  排除={filt['排除关键词'] or '—'}"
        f"  画幅={filt['画幅'] or '—'}  字数={filt['prompt字数下限']}~{filt['prompt字数上限'] or '∞'}")

    # 0) 库存与台账（联网去重的已见集合 + 池子库存 + 已消费台账）
    pool_dir = ROOT / out["采集池目录"]
    shortlist_dir = ROOT / out["选品目录"]
    consumed: set = load_consumed(shortlist_dir)
    seen: set = load_seen_ids(pool_dir, shortlist_dir)
    if dedup:
        log(f"去重：开，历史已见 {len(seen)} 条（联网只捞新作品）")
    else:
        log(f"去重：关（--include-seen，联网可含已见 {len(seen)} 条）")

    # 1) 选品取货：池子库存优先（过当前筛选），不够才联网补足
    log("\n[1/4] 选品取货（库存优先）…")
    inventory = [e for e in load_inventory(pool_dir, consumed) if match_filters(e, filt)]
    picked: list = []
    picked_ids: set = set()
    for e in inventory:
        if target and target > 0 and len(picked) >= target:
            break
        picked.append(e)
        picked_ids.add(e["id"])
    inv_count = len(picked)
    if inv_count:
        log(f"  池子库存取 {inv_count} 条（当前筛选下库存余 {len(inventory) - inv_count} 条）")
    need = None if not target or target <= 0 else max(target - inv_count, 0)
    entries: list = []
    matched: list = []
    errors: list = []
    skipped = 0
    if need is None or need > 0:
        if need is None:
            log("  目标不限：联网全量补足…")
        else:
            log(f"  还差 {need} 条，联网补足…")
        entries, matched, errors, skipped = fetch_and_match(
            target=need,
            filt=filt,
            start_page=grab["起始页"],
            delay=grab["请求间隔秒"],
            timeout=grab["请求超时秒"],
            fetched_at=fetched_at,
            seen=seen,
        )
    else:
        log("  库存已够，无需联网")

    if not picked and not entries and not matched:
        if skipped:
            log(f"没有新作品：{skipped} 条全部已见，本次无入库。")
        else:
            log("没有抓到任何数据，退出。")
        if errors:
            err_path = pool_dir / f"mj_{stamp}_errors.json"
            write_json(err_path, errors)
            log(f"[!] {len(errors)} 个错误，详见 {err_path.relative_to(ROOT)}")
        return 0 if skipped else 1

    # 联网新命中的也按 FIFO 进取货单（库存已够时上方循环已满员，这里自然跳过）
    for e in matched:
        if target and target > 0 and len(picked) >= target:
            break
        if e["id"] in picked_ids or e["id"] in consumed:
            continue
        picked.append(e)
        picked_ids.add(e["id"])

    # 2) 口味库存（采集池=联网新筛中，含清单未收的余量；--include-seen 时不重复入库）
    pool_entries = matched if dedup else [e for e in matched if e["id"] not in seen]
    if pool_entries:
        pool_path = pool_dir / f"mj_{stamp}.json"
        write_json(pool_path, {
            "type": "pool",
            "fetched_at": fetched_at,
            "count": len(pool_entries),
            "entries": pool_entries,
        })
        log(f"[2/4] 采集池 -> {pool_path.relative_to(ROOT)}（筛中入库 {len(pool_entries)} 条）")
    else:
        log("[2/4] 无新筛中作品入库")

    # 3) 选品清单（同名追加合并 + 消费台账；已有条目一字不动）
    name = args.name or datetime.now().strftime("%Y-%m-%d")
    shortlist_path = shortlist_dir / f"{name}.json"
    existing: list = []
    file_consumed: list = []
    if shortlist_path.exists():
        try:
            data = json.loads(shortlist_path.read_text(encoding="utf-8"))
            existing = data.get("entries") or []
            file_consumed = list(data.get("_consumed_ids") or [])
        except (json.JSONDecodeError, OSError):
            log(f"[!] 旧清单无法解析，拒绝合并: {shortlist_path.relative_to(ROOT)}")
            return 1
    # 台账回填：兼容没有 _consumed_ids 的旧文件（现存 entries 视为已消费）
    for e in existing:
        if isinstance(e, dict) and e.get("id") and e["id"] not in file_consumed:
            file_consumed.append(e["id"])
    existing.extend(picked)
    file_consumed.extend(p["id"] for p in picked)
    fetch_info = {
        "时间": fetched_at,
        "目标条数": target if target else "不限",
        "去重": "开" if dedup else "关",
        "库存取": inv_count,
        "联网取": len(picked) - inv_count,
        "联网新抓": len(entries),
        "跳过已见": skipped,
        "联网筛中": len(matched),
        "清单新增": len(picked),
        "清单总条数": len(existing),
        "报错数": len(errors),
    }
    if existing:
        write_json(shortlist_path, {
            "type": "shortlist",
            "_readme": build_readme(fetch_info, filt, img),
            "entries": existing,
            "_consumed_ids": file_consumed,
            "_manual_entry_template": MANUAL_ENTRY_TEMPLATE,
        })
        log(f"[3/4] 选品清单 -> {shortlist_path.relative_to(ROOT)}（本次新增 {len(picked)} 条，共 {len(existing)} 条）")
    else:
        log("[3/4] 无筛中条目，选品清单未创建/未变更")

    # 4) 只给本次新进清单的条目下图
    if img["下载图片"] and picked:
        log("[4/4] 下载图片…")
        download_images(
            picked, img, ROOT / out["图片目录"],
            delay=grab["请求间隔秒"], timeout=grab["请求超时秒"], errors=errors,
        )
        # 图片本地路径回写进选品清单
        write_json(shortlist_path, {
            "type": "shortlist",
            "_readme": build_readme(fetch_info, filt, img),
            "entries": existing,
            "_consumed_ids": file_consumed,
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
