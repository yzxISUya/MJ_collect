# -*- coding: utf-8 -*-
"""筛选与打分的数学层：硬条件布尔闸 + 词边界匹配 + 热度/总分公式。

约定（项目笔记 §13.8–13.10，详见《筛选算法.md》）：
- 硬条件（排除词/画幅/字数，语义模式下含包含关键词 AND 闸）与分数无关，不过就直接拒。
- 词边界匹配：girl 不会误中 cowgirl，realistic 不会误中 photorealistic。
  通配符只认首/尾：realistic*（词尾放宽）/ *realistic（词首放宽）/ *realistic*（子串）。
- 总分 S = R × (1 + β·H)。热度是加分项不是及格线（乘性加成：零相关永远垫底）；
  热榜头部是一片高原——前 K 名 H 满分持平，K~M 线性衰减。
- 关键词权重（100/50/25/10…）只服务词法评分器的归一化，语义模式下权重在『意图』分面上。
"""

from __future__ import annotations

import re

# 词边界：前后不能是字母数字（对中文关键词同样成立，CJK 不在 [a-z0-9] 内）
_HEAD_BOUND = r"(?<![a-z0-9])"
_TAIL_BOUND = r"(?![a-z0-9])"

DEFAULT_WEIGHTS = (100.0, 50.0, 25.0)  # 第 4 词起各 10
TAIL_WEIGHT = 10.0


def keyword_weights(filt: dict) -> list[float]:
    """按位置取权重：『关键词权重』覆盖的用覆盖值，否则默认 100/50/25/10…"""
    kws = filt.get("包含关键词") or []
    override = filt.get("关键词权重") or []
    out: list[float] = []
    for i in range(len(kws)):
        if i < len(override) and override[i] is not None:
            out.append(float(override[i]))
        elif i < len(DEFAULT_WEIGHTS):
            out.append(DEFAULT_WEIGHTS[i])
        else:
            out.append(TAIL_WEIGHT)
    return out


def compile_word(kw: str) -> re.Pattern:
    """关键词 → 正则。支持首/尾 * 通配（见模块 docstring）。"""
    k = (kw or "").lower().strip()
    head_free = k.startswith("*")
    tail_free = k.endswith("*")
    core = k.strip("*")
    if not core:
        return re.compile(r"(?!x)x")  # 空关键词：永不命中
    left = "" if head_free else _HEAD_BOUND
    right = "" if tail_free else _TAIL_BOUND
    return re.compile(left + re.escape(core) + right)


def passes_hard(entry: dict, filt: dict, must_keywords: bool = False) -> bool:
    """硬条件：排除词（命中任一即拒）/ 画幅 / prompt 字数；与分数无关。
    must_keywords=True（语义模式）时，『包含关键词』降级为硬过滤：全部必须命中（AND）。
    词法模式（must_keywords=False）下关键词是打分素材，不作硬闸。"""
    text = (entry.get("prompt_text") or "").lower()

    for kw in filt.get("排除关键词") or []:
        if compile_word(kw).search(text):
            return False

    if must_keywords:
        for kw in filt.get("包含关键词") or []:
            if not compile_word(kw).search(text):
                return False

    ars = filt.get("画幅") or []
    if ars and (entry.get("params") or {}).get("ar") not in ars:
        return False

    n = len(entry.get("prompt_text") or "")
    min_len = filt.get("prompt字数下限") or 0
    max_len = filt.get("prompt字数上限")
    if n < min_len:
        return False
    if max_len is not None and n > max_len:
        return False

    return True


# ---------------------------------------------------------------- 热度与融合
# S = R × (1 + β·H)。热度是加分项不是及格线（乘性加成：零相关永远垫底）；
# 热榜头部是一片高原——前 K 名 H 满分持平，K~M 线性衰减（项目笔记 §13.9）。


def heat_score(rank, k: float = 50, m: float = 150) -> float:
    """位置→热度 [0,1]。rank=None（无热榜位置，如手动条目）取 0.5 中性。"""
    if rank is None:
        return 0.5
    if rank <= k:
        return 1.0
    if rank <= m:
        return (m - rank) / (m - k)
    return 0.0


def fuse_score(r: float, h: float, beta: float = 0.1) -> float:
    """总分 = 相关度 × (1 + β·热度)。"""
    return r * (1.0 + beta * h)
