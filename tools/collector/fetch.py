# -*- coding: utf-8 -*-
"""HTTP 抓取：MJ explore 接口与图床下载（免登录，见项目笔记 §12）。"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from collector import log
from collector.config import ROOT
from collector.schema import CDN_IMAGE

EXPLORE_API = "https://www.midjourney.com/api/explore"
PER_PAGE = 50  # MJ explore 接口每页固定 50 条

# MJ 站点对非浏览器请求较敏感，统一用常见浏览器 UA
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


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


def fetch_page(page: int, timeout: float):
    """抓一页 explore 热榜，返回 (条目列表|None, 错误信息|None)。空页 = ([], None)。"""
    url = f"{EXPLORE_API}?page={page}&feed=top&_ql=explore"
    body, err = http_get_retry(url, timeout=timeout)
    if body is None:
        return None, err
    try:
        return json.loads(body), None
    except json.JSONDecodeError as e:
        return None, str(e)


def download_images(entries: list, img_cfg: dict, images_root: Path,
                    delay: float, timeout: float, errors: list) -> None:
    """给进清单的条目下图；已有本地文件的跳过。"""
    width = img_cfg["宽度档"]
    grid = img_cfg["格子"]
    day_dir = images_root / datetime.now().strftime("%Y-%m-%d")
    total = len(entries)
    for i, entry in enumerate(entries, 1):
        local = entry.get("image", {}).get("local")
        if local and (ROOT / local).exists():
            log(f"  [{i}/{total}] {entry['id'][:8]} 已有本地图片，跳过")
            continue
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
