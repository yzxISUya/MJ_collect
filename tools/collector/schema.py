# -*- coding: utf-8 -*-
"""条目结构：MJ 原始数据解析、手动录入模板、选品清单文档头。"""

from __future__ import annotations

from datetime import datetime

JOB_PAGE = "https://www.midjourney.com/jobs/{job_id}"
CDN_IMAGE = "https://cdn.midjourney.com/{job_id}/0_{grid}_{width}_N.webp"


def parse_entry(raw: dict, fetched_at: str, rank: int | None = None) -> dict:
    """把 MJ 原始条目整理成可读的选品条目结构。rank=热榜位置（页内序号的全局序）。"""
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
        "feed_rank": rank,
        "source_url": JOB_PAGE.format(job_id=job_id),
        "image": {
            "url": CDN_IMAGE.format(job_id=job_id, grid="0", width="640"),  # 占位，下载时按配置改写
            "local": None,
        },
        "fetched_at": fetched_at,
        "notes": "",
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
    "feed_rank": None,
    "match": None,
    "source_url": "作品源链接（必填）",
    "image": {"url": None, "local": None},
    "fetched_at": "录入时间",
    "notes": "",
}


def build_readme(fetch_info: dict, filt: dict, img_cfg: dict) -> dict:
    """选品清单文件头：操作指引 + 字段说明 + 本次运行参数回显。"""
    return {
        "怎么删": "删掉 entries 数组里不要的整个 { } 块（注意保留上下条目的逗号）。删除后不会被重新送上（_consumed_ids 台账仍记着它）；反悔可在 2 天 feed 窗口内用 --include-seen 重捞。",
        "怎么加": (
            "把文件末尾的 _manual_entry_template 整个复制进 entries 数组，"
            "改好内容；手动条目 source 写 'manual'（或其他平台名），id 自拟唯一即可。"
        ),
        "注意": [
            "author（作者署名）与 source_url（源链接）是合规红线，必留。",
            "entries 里的条目顺序即排版候选顺序，可自由挪动。",
            "同名再运行=新条目追加到 entries 末尾（分数排序择优），已有条目（含手改/手补）一字不动。",
            "删掉的条目不会再被自动抓回（_consumed_ids 台账记着）；反悔可在 2 天 feed 窗口内用 --include-seen 重捞。",
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
            "feed_rank": "热榜位置（页码×50+序号；手动条目为 null）",
            "match": "评分结果（工具维护）：relevance 相关度 / heat 热度 / score 总分 / why 理由 / facets 分面明细（text,weight,score,why）",
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
