# -*- coding: utf-8 -*-
"""筛选与打分：硬条件布尔闸 + 关键词权重打分 + 衰减门槛。

约定（项目笔记 §13.8）：
- 权重默认：第 1 词 100、第 2 词 50、第 3 词 25、之后各 10（可用『关键词权重』按位覆盖）。
  关键词顺序即优先级——分数=命中词的权重之和，全命中者永远排在单命中者前面。
- 词边界匹配：girl 不会误中 cowgirl，realistic 不会误中 photorealistic。
  通配符只认首/尾：realistic*（词尾放宽）/ *realistic（词首放宽）/ *realistic*（子串）。
- 硬条件（排除词/画幅/字数）与分数无关，不过就直接拒。
- 默认最低分 = w1 + w最小（n=1 即 w1）——语义『主词必中 + 至少再中一个』（3 词时=125）。
- 每翻一页门槛降低 翻页降分比例 × 最低分，下限 = 最小权重（至少命中一词）；
  无关键词时最低分/下限 = 0，退化为先到先得。
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


def build_scorer(filt: dict):
    """返回 (score_fn, weights)。score_fn(entry) → 命中权重之和。"""
    kws = filt.get("包含关键词") or []
    weights = keyword_weights(filt)
    patterns = [compile_word(k) for k in kws]

    def score_fn(entry: dict) -> float:
        text = (entry.get("prompt_text") or "").lower()
        return sum(w for pat, w in zip(patterns, weights) if pat.search(text))

    return score_fn, weights


def passes_hard(entry: dict, filt: dict) -> bool:
    """硬条件：排除词（命中任一即拒）/ 画幅 / prompt 字数。与分数无关。"""
    text = (entry.get("prompt_text") or "").lower()

    for kw in filt.get("排除关键词") or []:
        if compile_word(kw).search(text):
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


def resolve_gate(filt: dict, weights: list[float]) -> tuple[float, float, float]:
    """返回 (最低分 min0, 每页降分 decay_step, 下限 floor)。"""
    if not weights:
        min_score = float(filt.get("最低分") or 0)
        floor = 0.0
    elif filt.get("最低分") is not None:
        min_score = float(filt["最低分"])
        floor = min(weights)
    else:
        min_score = weights[0] if len(weights) == 1 else weights[0] + min(weights)
        floor = min(weights)
    alpha_raw = filt.get("翻页降分比例")
    alpha = 0.5 if alpha_raw is None else float(alpha_raw)
    return min_score, min_score * alpha, floor


def hit_count(candidates: list, line: float) -> int:
    """candidates: [(score, entry)]，统计达线（score ≥ line）条数。"""
    return sum(1 for s, _ in candidates if s >= line)


def above_line(candidates: list, line: float) -> list:
    """返回达线候选 [(score, entry)]（保持扫描序）。"""
    return [(s, e) for s, e in candidates if s >= line]


def pick_top(candidates: list, target) -> list:
    """分数降序取前 target 条（同分保持扫描序：库存 FIFO 先于联网热榜序）。
    target 为 null/0 表示不限，全取。"""
    ordered = sorted(candidates, key=lambda c: -c[0])
    if target and target > 0:
        ordered = ordered[:target]
    return [e for _, e in ordered]
