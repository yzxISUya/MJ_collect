#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MJ 采集器入口（命令行参数可临时覆盖 config/collector.json）
eg:
  python tools/collect_mj.py
  python tools/collect_mj.py --target 8 --no-images
  python tools/collect_mj.py --keyword girl portrait realistic --name vol1
  python tools/collect_mj.py --keyword girl portrait abstract --min-score 125
  python tools/collect_mj.py --include-seen
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collector import log, now_iso, now_stamp  # noqa: E402
from collector.config import ROOT, DEFAULT_CONFIG, load_config, merge_cli_config  # noqa: E402
from collector.fetch import fetch_page, download_images  # noqa: E402
from collector.filter import (  # noqa: E402
    build_scorer, passes_hard, resolve_gate, hit_count, above_line, pick_top,
)
from collector.schema import parse_entry, build_readme  # noqa: E402
from collector.store import (  # noqa: E402
    write_json, load_seen_ids, load_consumed, load_inventory,
    load_shortlist, save_shortlist,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MJ 采集器：抓取 Midjourney 热榜作品，产出可编辑的选品 JSON（分数排序择优）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python tools/collect_mj.py\n"
            "  python tools/collect_mj.py --target 8 --no-images\n"
            "  python tools/collect_mj.py --keyword girl portrait realistic --name vol1\n"
        ),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径")
    p.add_argument("--target", type=int, help="清单新增条数，凑够即停（0=不限）")
    p.add_argument("--include-seen", action="store_true",
                   help="关闭去重，联网可含历史上已抓过的作品（重看当前热榜）")
    p.add_argument("--start-page", type=int, help="起始页码（默认 1）")
    p.add_argument("--delay", type=float, help="请求间隔秒（默认 1.5，勿低于 1）")
    p.add_argument("--timeout", type=float, help="单请求超时秒（默认 20）")
    p.add_argument("--keyword", nargs="+", help="包含关键词（越靠前权重越高）")
    p.add_argument("--weight", nargs="+", type=float,
                   help="关键词权重，与关键词同序（默认 100 50 25 10…）")
    p.add_argument("--min-score", type=float, help="最低分门槛（默认自动：w1+w最小）")
    p.add_argument("--decay", type=float, help="翻页降分比例（默认 0.5）")
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
    stamp = now_stamp()
    target = filt.get("目标条数")
    dedup = bool(filt.get("去重", True))
    score_fn, weights = build_scorer(filt)
    min_score, decay_step, floor = resolve_gate(filt, weights)

    log(f"MJ 采集器 | 清单新增 {target if target else '不限'} 条 | 间隔 {grab['请求间隔秒']}s")
    log(f"筛选: 包含={filt['包含关键词'] or '—'}  权重={weights or '—'}  排除={filt['排除关键词'] or '—'}"
        f"  画幅={filt['画幅'] or '—'}  字数={filt['prompt字数下限']}~{filt['prompt字数上限'] or '∞'}")
    log(f"门槛: 最低分 {min_score:g}，每翻一页降 {decay_step:g}，下限 {floor:g}"
        + ("（无关键词：先到先得）" if not weights else "（主词必中+至少再中一个）"))

    # 0) 库存与台账
    pool_dir = ROOT / out["采集池目录"]
    shortlist_dir = ROOT / out["选品目录"]
    consumed = load_consumed(shortlist_dir)
    seen = load_seen_ids(pool_dir, shortlist_dir)
    seen0 = set(seen)  # 本次运行前的已见快照（新入库判定基准）
    if dedup:
        log(f"去重：开，历史已见 {len(seen)} 条（联网只捞新作品）")
    else:
        log(f"去重：关（--include-seen，联网可含已见 {len(seen)} 条）")

    # 1) 候选池：库存=第 0 页，联网翻页补足；门槛只管停止与录取线
    log("\n[1/4] 选品取货（分数排序，衰减门槛）…")
    candidates: list = []  # [(score, entry)]，扫描序 = 库存 FIFO + 热榜序
    cand_ids: set = set()
    inv_count = 0
    for e in load_inventory(pool_dir, consumed):
        if not passes_hard(e, filt):
            continue
        inv_count += 1
        cand_ids.add(e["id"])
        candidates.append((score_fn(e), e))
    log(f"  库存候选 {inv_count} 条（过硬条件）")

    threshold = min_score
    errors: list = []
    skipped = 0
    fetched_new = 0
    page = grab["起始页"]
    exhausted = False
    while True:
        if target and target > 0 and hit_count(candidates, threshold) >= target:
            log(f"  达线 {hit_count(candidates, threshold)} 条 ≥ 门槛 {threshold:g}，停止翻页")
            break
        items, err = fetch_page(page, grab["请求超时秒"])
        if items is None:
            errors.append({"环节": "抓取列表", "页码": page, "错误": err})
            log(f"  [!] 第 {page} 页抓取失败: {err}")
            exhausted = True
            break
        if not items:
            log(f"  第 {page} 页为空，可见数据已抓完")
            exhausted = True
            break
        new_count = 0
        for raw in items:
            job_id = raw.get("id", "")
            if dedup and job_id in seen:
                skipped += 1
                continue
            if dedup:
                seen.add(job_id)  # 仅本次运行内防重；未入库的下次仍可回捞
            if job_id in cand_ids:
                continue  # --include-seen 时防库存/页间重复
            entry = parse_entry(raw, fetched_at)
            new_count += 1
            fetched_new += 1
            if not passes_hard(entry, filt):
                continue
            cand_ids.add(job_id)
            candidates.append((score_fn(entry), entry))
        log(f"  第 {page} 页: 新 {new_count} 跳过 {len(items) - new_count}"
            f" | 累计候选 {len(candidates)} | 达线 {hit_count(candidates, threshold)}（门槛 {threshold:g}）")
        page += 1
        if target and target > 0:
            threshold = max(threshold - decay_step, floor)
        time.sleep(grab["请求间隔秒"])

    # 录取线：目标凑满=当前门槛；搜到尽头=降到底线（保证有产出）
    final_line = floor if exhausted else threshold
    admitted = above_line(candidates, final_line)
    picked = pick_top(admitted, target)
    if admitted:
        scores = ", ".join(f"{s:g}" for s, _ in sorted(admitted, key=lambda c: -c[0])[:len(picked)])
        log(f"  录取线 {final_line:g}：达线 {len(admitted)} 条，取 top {len(picked)}（分数 {scores}）")
    else:
        log(f"  录取线 {final_line:g} 之上无候选"
            + (f"（候选 {len(candidates)} 条都不达线，可调低最低分）" if candidates else ""))

    if not admitted:
        if skipped and not fetched_new:
            log(f"没有新作品：{skipped} 条全部已见，本次无入库。")
        if errors:
            err_path = pool_dir / f"mj_{stamp}_errors.json"
            write_json(err_path, errors)
            log(f"[!] {len(errors)} 个错误，详见 {err_path.relative_to(ROOT)}")
        return 0 if (candidates or skipped) else 1

    # 2) 口味库存：达线者（含溢出），只入本次新见
    pool_entries = [e for _, e in admitted if e["id"] not in seen0]
    if pool_entries:
        pool_path = pool_dir / f"mj_{stamp}.json"
        write_json(pool_path, {
            "type": "pool",
            "fetched_at": fetched_at,
            "count": len(pool_entries),
            "entries": pool_entries,
        })
        log(f"[2/4] 采集池 -> {pool_path.relative_to(ROOT)}（达线入库 {len(pool_entries)} 条）")
    else:
        log("[2/4] 无新达线作品入库")

    # 3) 选品清单（同名追加合并 + 消费台账；已有条目一字不动）
    name = args.name or datetime.now().strftime("%Y-%m-%d")
    shortlist_path = shortlist_dir / f"{name}.json"
    loaded = load_shortlist(shortlist_path)
    if loaded is None:
        log(f"[!] 旧清单无法解析，拒绝合并: {shortlist_path.relative_to(ROOT)}")
        return 1
    existing, file_consumed = loaded
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
        "最低分": min_score,
        "每页降分": decay_step,
        "录取线": final_line,
        "库存候选": inv_count,
        "联网新抓": fetched_new,
        "跳过已见": skipped,
        "候选总数": len(candidates),
        "达线条数": len(admitted),
        "清单新增": len(picked),
        "清单总条数": len(existing),
        "报错数": len(errors),
    }
    readme = build_readme(fetch_info, filt, img)
    save_shortlist(shortlist_path, existing, file_consumed, readme)
    log(f"[3/4] 选品清单 -> {shortlist_path.relative_to(ROOT)}（本次新增 {len(picked)} 条，共 {len(existing)} 条）")

    # 4) 只给本次新进清单的条目下图
    if img["下载图片"] and picked:
        log("[4/4] 下载图片…")
        download_images(
            picked, img, ROOT / out["图片目录"],
            delay=grab["请求间隔秒"], timeout=grab["请求超时秒"], errors=errors,
        )
        save_shortlist(shortlist_path, existing, file_consumed, build_readme(fetch_info, filt, img))
    else:
        log("[4/4] 跳过图片下载")

    if errors:
        err_path = pool_dir / f"mj_{stamp}_errors.json"
        write_json(err_path, errors)
        log(f"\n[!] {len(errors)} 个错误，详见 {err_path.relative_to(ROOT)}")

    log("\n完成。下一步：打开选品清单，删掉不要的条目、按需手动补条目。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
