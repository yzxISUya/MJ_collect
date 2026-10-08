#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MJ 采集器入口（命令行参数可临时覆盖 config/collector.json）
eg:
  python tools/collect_mj.py
  python tools/collect_mj.py --target 8 --no-images
  python tools/collect_mj.py --intent "少女肖像10，偏抽象2，个性化5" --name vol1
  python tools/collect_mj.py --scorer embedding --intent "荒诞超现实"
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
from collector.filter import passes_hard, heat_score, fuse_score  # noqa: E402
from collector.schema import parse_entry, build_readme  # noqa: E402
from collector.scorers import (  # noqa: E402
    build_scorers, ChainScorer, set_env_path, parse_intent, ScorerUnavailable,
)
from collector.store import (  # noqa: E402
    write_json, load_seen_ids, load_consumed, load_inventory,
    load_shortlist, save_shortlist,
)


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """CLI 解析器。所有［默认 …］都动态取自配置文件当前值——命令行只是临时覆盖。"""
    grab, filt, img = cfg["抓取"], cfg["筛选"], cfg["图片"]
    score_cfg = cfg["评分"]

    def d(v, none="不限"):
        return none if v is None else v

    p = argparse.ArgumentParser(
        description="MJ 采集器：抓取 Midjourney 热榜作品，评分择优产出可编辑的选品 JSON。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "默认值来自 config/collector.json（命令行只做临时覆盖），"
            "上方［默认 …］即该文件的当前值。\n"
            "评分: S = 相关度R × (1 + 热度权重×热度H)；R 由评分器给出（llm/embedding/lexical）。\n"
            "示例:\n"
            "  python tools/collect_mj.py --intent \"少女肖像，偏写实\" --name vol1\n"
            "  python tools/collect_mj.py --scorer embedding --intent \"荒诞超现实\"\n"
        ),
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"配置文件路径［默认 {DEFAULT_CONFIG.relative_to(ROOT)}］")
    p.add_argument("--target", type=int,
                   help=f"清单新增条数，择优取 top-N（0=不限）［默认 {d(filt.get('目标条数'))}］")
    p.add_argument("--intent", type=str,
                   help=f"选品意图分面：'少女肖像10，偏抽象2'（尾随数字=权重）/ '少女肖像=10' / 裸词权重1"
                        f"［默认 {filt.get('意图') or '空'}］")
    p.add_argument("--scorer", choices=["embedding", "hybrid", "llm", "lexical", "auto"],
                   help=f"评分器：embedding=本地向量(零token) / hybrid=向量粗筛+LLM精评top / llm=纯LLM / lexical=词法"
                        f"［默认 {score_cfg.get('评分器', 'embedding')}］")
    p.add_argument("--beta", type=float,
                   help=f"热度权重 β（很低，仅相近时起作用）［默认 {d(score_cfg.get('热度权重'), '0.1')}］")
    p.add_argument("--min-rel", type=float,
                   help=f"软门槛：相关度下限，不够优雅放宽保产出［默认 {d(score_cfg.get('最低相关分'), '0.2')}］")
    p.add_argument("--include-seen", action="store_true",
                   help=f"关闭去重，联网可含历史已抓作品（重看热榜）［默认 去重={'开' if filt.get('去重', True) else '关'}］")
    p.add_argument("--start-page", type=int,
                   help=f"起始页码［默认 {d(grab.get('起始页'), '1')}］")
    p.add_argument("--delay", type=float,
                   help=f"请求间隔秒，勿低于 1［默认 {d(grab.get('请求间隔秒'), '1.5')}］")
    p.add_argument("--timeout", type=float,
                   help=f"单请求超时秒［默认 {d(grab.get('请求超时秒'), '20')}］")
    p.add_argument("--keyword", nargs="+",
                   help=f"包含关键词：词法模式打分素材；语义模式=必须全命中硬过滤［默认 {filt.get('包含关键词') or '无'}］")
    p.add_argument("--weight", nargs="+", type=float,
                   help=f"词法模式关键词权重，同序［默认 {filt.get('关键词权重') or '100 50 25 10…'}］")
    p.add_argument("--exclude", nargs="+",
                   help=f"排除关键词，命中任一即弃（硬条件）［默认 {filt.get('排除关键词') or '无'}］")
    p.add_argument("--ar", nargs="+",
                   help=f"画幅筛选，如 9:16 3:4（硬条件）［默认 {filt.get('画幅') or '全收'}］")
    p.add_argument("--min-len", type=int,
                   help=f"prompt 字数下限［默认 {d(filt.get('prompt字数下限'), '0')}］")
    p.add_argument("--max-len", type=int,
                   help=f"prompt 字数上限［默认 {d(filt.get('prompt字数上限'))}］")
    p.add_argument("--no-images", action="store_true",
                   help=f"不下载图片［默认 下载={'开' if img.get('下载图片', True) else '关'}］")
    p.add_argument("--image-width", type=int, choices=[384, 640],
                   help=f"图片宽度档［默认 {d(img.get('宽度档'), '640')}］")
    p.add_argument("--grid", type=int,
                   help=f"图片格子序号（2×2 网格）［默认 {d(img.get('格子'), '0')}］")
    p.add_argument("--name", help="选品清单文件名（不含扩展名；同名自动追加合并）［默认 今天日期］")
    return p


def main() -> int:
    # Windows 控制台编码兜底
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    # 先取配置文件（--help 要动态显示它的当前值），再全量解析
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    args = build_parser(cfg).parse_args()
    cfg = merge_cli_config(cfg, args)
    grab, filt, img, out = cfg["抓取"], cfg["筛选"], cfg["图片"], cfg["输出"]
    score_cfg = cfg["评分"]

    set_env_path(ROOT / ".env")
    fetched_at = now_iso()
    stamp = now_stamp()
    target = filt.get("目标条数")
    dedup = bool(filt.get("去重", True))
    intent_raw = filt.get("意图")
    facets = parse_intent(intent_raw)
    intent = (intent_raw if isinstance(intent_raw, str) else "；".join(
        f"{f['text']}×{f['weight']:g}" for f in facets)) or ""
    mode = (score_cfg.get("评分器") or "auto").lower()
    semantic = bool(facets) and mode != "lexical"
    beta = float(score_cfg.get("热度权重") if score_cfg.get("热度权重") is not None else 0.1)
    k_plateau = float(score_cfg.get("热度高原") or 50)
    m_end = float(score_cfg.get("热度终点") or 150)
    r_min = float(score_cfg.get("最低相关分") or 0.0)

    log(f"MJ 采集器 | 清单新增 {target if target else '不限'} 条 | 间隔 {grab['请求间隔秒']}s")
    log(f"意图: {intent or '—（词法/纯热榜）'}  评分器: {mode}  β={beta:g}  软门槛={r_min:g}")
    if facets:
        log("分面: " + "  ".join(f"{f['text']}×{f['weight']:g}" for f in facets))
    log(f"硬条件: 包含(AND)={filt['包含关键词'] or '—'}  排除={filt['排除关键词'] or '—'}"
        f"  画幅={filt['画幅'] or '—'}  字数={filt['prompt字数下限']}~{filt['prompt字数上限'] or '∞'}")

    # 0) 库存与台账
    pool_dir = ROOT / out["采集池目录"]
    shortlist_dir = ROOT / out["选品目录"]
    consumed = load_consumed(shortlist_dir)
    seen = load_seen_ids(pool_dir, shortlist_dir)
    seen0 = set(seen)
    if dedup:
        log(f"去重：开，历史已见 {len(seen)} 条（联网只捞新作品）")
    else:
        log(f"去重：关（--include-seen，联网可含已见 {len(seen)} 条）")

    # 1) 候选收集：库存=第 0 页 + 联网全扫（v3 无翻页门槛机器，全量收集后评分择优）
    log("\n[1/4] 收集候选…")
    candidates: list = []
    cand_ids: set = set()
    for e in load_inventory(pool_dir, consumed):
        if not passes_hard(e, filt, must_keywords=semantic):
            continue
        cand_ids.add(e["id"])
        candidates.append(e)
    inv_count = len(candidates)
    log(f"  库存候选 {inv_count} 条（过硬条件）")

    errors: list = []
    skipped = 0
    fetched_new = 0
    page = grab["起始页"]
    while True:
        items, err = fetch_page(page, grab["请求超时秒"])
        if items is None:
            errors.append({"环节": "抓取列表", "页码": page, "错误": err})
            log(f"  [!] 第 {page} 页抓取失败: {err}")
            break
        if not items:
            log(f"  第 {page} 页为空，可见数据已抓完")
            break
        new_count = 0
        for idx, raw in enumerate(items):
            job_id = raw.get("id", "")
            if dedup and job_id in seen:
                skipped += 1
                continue
            if dedup:
                seen.add(job_id)
            if job_id in cand_ids:
                continue
            rank = (page - 1) * 50 + idx + 1  # 热榜全局位置
            entry = parse_entry(raw, fetched_at, rank=rank)
            new_count += 1
            fetched_new += 1
            if not passes_hard(entry, filt, must_keywords=semantic):
                continue
            cand_ids.add(job_id)
            candidates.append(entry)
        log(f"  第 {page} 页: 新 {new_count} 跳过 {len(items) - new_count} | 累计候选 {len(candidates)}")
        page += 1
        time.sleep(grab["请求间隔秒"])

    if not candidates:
        if skipped:
            log(f"没有新作品：{skipped} 条全部已见，本次无候选。")
        else:
            log("没有候选数据，退出。")
        if errors:
            err_path = pool_dir / f"mj_{stamp}_errors.json"
            write_json(err_path, errors)
            log(f"[!] {len(errors)} 个错误，详见 {err_path.relative_to(ROOT)}")
        return 0 if skipped else 1

    # 2) 评分：R（相关度）→ 融合 H（热度）→ S
    log(f"\n[2/4] 评分（{len(candidates)} 条）…")
    chain = build_scorers(mode, filt, score_cfg)
    scorer = ChainScorer(chain) if len(chain) > 1 else chain[0]
    try:
        results = scorer.score(candidates)
    except ScorerUnavailable as e:
        log(f"[!] 评分失败: {e}")
        return 1
    used = getattr(scorer, "used", scorer.name)
    log(f"  实际评分器: {used}")

    scored: list = []  # [(S, R, entry)]
    for e, res in zip(candidates, results):
        r = res["relevance"]
        h = heat_score(e.get("feed_rank"), k_plateau, m_end)
        s = fuse_score(r, h, beta)
        e["match"] = {
            "relevance": round(r, 3),
            "heat": round(h, 3),
            "score": round(s, 3),
            "rank": e.get("feed_rank"),
            "why": res["why"],
            "facets": res["facets"],
        }
        scored.append((s, r, e))
    # 达软门槛优先，其次按总分降序（稳定排序，同分保持扫描序）
    scored.sort(key=lambda x: (x[1] >= r_min, x[0]), reverse=True)
    gate_passed = [x for x in scored if x[1] >= r_min]
    picked_rows = scored[:target] if target and target > 0 else scored
    if target and target > 0 and len(gate_passed) < target:
        log(f"  软门槛 {r_min:g} 只有 {len(gate_passed)} 条，放宽补足（共取 {len(picked_rows)}）")
    picked = [e for _, _, e in picked_rows]
    if picked:
        tops = ", ".join(f"{s:.2f}" for s, _, _ in picked_rows[:8])
        log(f"  取 top {len(picked)}（S: {tops}）")

    # 3) 口味库存：达软门槛者（含溢出）+ 被取走的放宽条目，只入本次新见
    picked_ids = {e["id"] for e in picked}
    pool_entries = [
        e for s, r, e in scored
        if (r >= r_min or e["id"] in picked_ids) and e["id"] not in seen0
    ]
    if pool_entries:
        pool_path = pool_dir / f"mj_{stamp}.json"
        write_json(pool_path, {
            "type": "pool",
            "fetched_at": fetched_at,
            "count": len(pool_entries),
            "entries": pool_entries,
        })
        log(f"[3/4] 采集池 -> {pool_path.relative_to(ROOT)}（入库 {len(pool_entries)} 条）")
    else:
        log("[3/4] 无新作品入库")

    # 4) 选品清单（同名追加合并 + 消费台账；已有条目一字不动）
    name = args.name or datetime.now().strftime("%Y-%m-%d")
    shortlist_path = shortlist_dir / f"{name}.json"
    loaded = load_shortlist(shortlist_path)
    if loaded is None:
        log(f"[!] 旧清单无法解析，拒绝合并: {shortlist_path.relative_to(ROOT)}")
        return 1
    existing, file_consumed = loaded
    for e in existing:
        if isinstance(e, dict) and e.get("id") and e["id"] not in file_consumed:
            file_consumed.append(e["id"])
    existing.extend(picked)
    file_consumed.extend(p["id"] for p in picked)
    fetch_info = {
        "时间": fetched_at,
        "目标条数": target if target else "不限",
        "去重": "开" if dedup else "关",
        "评分器": used,
        "意图": intent or None,
        "热度权重": beta,
        "热度高原/终点": f"{k_plateau:g}/{m_end:g}",
        "最低相关分": r_min,
        "库存候选": inv_count,
        "联网新抓": fetched_new,
        "跳过已见": skipped,
        "候选总数": len(candidates),
        "达门槛数": len(gate_passed),
        "清单新增": len(picked),
        "清单总条数": len(existing),
        "报错数": len(errors),
    }
    readme = build_readme(fetch_info, filt, img)
    save_shortlist(shortlist_path, existing, file_consumed, readme)
    log(f"[4/4] 选品清单 -> {shortlist_path.relative_to(ROOT)}（本次新增 {len(picked)} 条，共 {len(existing)} 条）")

    # 图片：只给本次新进清单的条目下
    if img["下载图片"] and picked:
        log("下载图片…")
        download_images(
            picked, img, ROOT / out["图片目录"],
            delay=grab["请求间隔秒"], timeout=grab["请求超时秒"], errors=errors,
        )
        save_shortlist(shortlist_path, existing, file_consumed, build_readme(fetch_info, filt, img))

    if errors:
        err_path = pool_dir / f"mj_{stamp}_errors.json"
        write_json(err_path, errors)
        log(f"\n[!] {len(errors)} 个错误，详见 {err_path.relative_to(ROOT)}")

    log("\n完成。下一步：打开选品清单，删掉不要的条目、按需手动补条目。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
