# -*- coding: utf-8 -*-
"""配置加载与命令行覆盖。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "config" / "collector.json"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_cli_config(cfg: dict, args) -> dict:
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
    if args.weight is not None:
        filt["关键词权重"] = args.weight
    if args.intent is not None:
        filt["意图"] = args.intent
    score = cfg["评分"]
    if args.scorer is not None:
        score["评分器"] = args.scorer
    if args.beta is not None:
        score["热度权重"] = args.beta
    if args.min_rel is not None:
        score["最低相关分"] = args.min_rel
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
