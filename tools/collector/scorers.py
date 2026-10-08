# -*- coding: utf-8 -*-
"""相关度评分器：词法 / 本地向量 / LLM 评审——统一的『意图分面加权』架构（§13.10）。

意图不是一锅炖的整段文本，而是带权重的分面列表：
    "少女肖像10，偏抽象2，个性化5"  →  [(少女肖像,10), (偏抽象,2), (个性化,5)]
每个评分器对每个分面打 R_i∈[0,1]（附理由），再统一聚合：
    R = Σ(w_i·R_i)/Σw_i     加权平均——主分面（大权重）主导，弱分面只做微调
绝对必须的条件请用『包含关键词』硬闸（AND 一票否决），不混进权重体系。

评分器（模式）：
  llm       C 方案：LLM 分面评审（意图理解最强，附分面理由）——OpenAI 兼容接口
  embedding B 方案：本地向量（离线、排序几何好）
  lexical   A 方案：词法命中（零依赖兜底，中文分面是整词匹配，较糙）
  auto      依次尝试 llm → embedding → lexical，失败自动降级

向量选型实测（2026-10-08）：paraphrase-multilingual-MiniLM-L12-v2（fastembed 量化
ONNX 0.22GB）：分离度 0.397 胜 potion（0.209），128-token 截断天然压制 prompt 尾部
关键词堆砌。校准 R_i = clip((cos - 0.20) / 0.45, 0, 1)。
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request

from collector import log
from collector.filter import compile_word, keyword_weights

# ---------------------------------------------------------------- 意图分面


def parse_intent(intent) -> list:
    """解析意图为分面列表 [{text, weight}]。

    三种写法（分隔符 ，,、;；）：
      '少女肖像10，偏抽象2'   尾随数字
      '少女肖像=10，Y2K=5'    显式等号（分面文本自带数字时用）
      '少女肖像,偏抽象'       裸分面，权重 1
    也可传结构化列表 [{"文本": "...", "权重": 10}, ...]。
    """
    if isinstance(intent, list):
        out = []
        for it in intent:
            if isinstance(it, dict):
                t = str(it.get("文本") or it.get("text") or "").strip()
                w = it.get("权重", it.get("weight", 1))
            else:
                t, w = str(it).strip(), 1
            if t:
                out.append({"text": t, "weight": float(w or 1)})
        return out
    if not (intent or "").strip():
        return []
    out = []
    for seg in re.split(r"[，,、;；\n]+", str(intent)):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^(.*?)[=：:]\s*([\d.]+)\s*$", seg)  # 显式：文本=权重
        if m and m.group(1).strip():
            out.append({"text": m.group(1).strip(), "weight": float(m.group(2))})
            continue
        m = re.match(r"^(.*?)([\d.]+)\s*$", seg)  # 尾随数字：文本10
        if m and m.group(1).strip():
            out.append({"text": m.group(1).strip(), "weight": float(m.group(2))})
            continue
        out.append({"text": seg, "weight": 1.0})  # 裸分面
    return out


def finalize(facet_rows: list) -> dict:
    """分面成绩 → {relevance, why, facets}。加权平均 + 按权重降序展示。"""
    rows = sorted(facet_rows, key=lambda f: -f["weight"])
    total_w = sum(f["weight"] for f in rows) or 1.0
    rel = sum(f["weight"] * f["score"] for f in rows) / total_w
    top = rows[0] if rows else {"text": "—", "why": ""}
    return {
        "relevance": float(rel),
        "why": f"主分面[{top['text']}] {top['why']}".strip(),
        "facets": [
            {
                "text": f["text"],
                "weight": f["weight"],
                "score": round(float(f["score"]), 3),
                "why": f["why"],
            }
            for f in rows
        ],
    }


class ScorerUnavailable(Exception):
    """评分器当前不可用（缺依赖/缺密钥/缺意图）。"""


def load_dotenv(path) -> dict:
    """极简 .env 读取（不覆盖已存在的环境变量）。"""
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass
    return out


class BaseScorer:
    name = "base"

    def score(self, entries: list) -> list:
        """返回与 entries 等长的 [{relevance, why, facets}]。"""
        raise NotImplementedError


# ---------------------------------------------------------------- A 词法


class LexicalScorer(BaseScorer):
    """分面文本词法命中（1/0）→ 加权平均。无分面时退化为关键词权重归一。"""

    name = "lexical"

    def __init__(self, filt: dict, facets: list):
        self.facets = list(facets or [])
        if not self.facets:
            kws = [k.lower() for k in (filt.get("包含关键词") or [])]
            self.facets = [
                {"text": k, "weight": w} for k, w in zip(kws, keyword_weights(filt))
            ]
        self.patterns = [compile_word(f["text"]) for f in self.facets]

    def score(self, entries: list) -> list:
        if not self.facets:  # 无意图无关键词：纯热榜
            return [finalize([{"text": "全部", "weight": 1.0, "score": 1.0,
                               "why": "纯热榜模式（无意图无关键词）"}]) for _ in entries]
        out = []
        for e in entries:
            text = (e.get("prompt_text") or "").lower()
            rows = []
            for f, p in zip(self.facets, self.patterns):
                hit = bool(p.search(text))
                rows.append({"text": f["text"], "weight": f["weight"],
                             "score": 1.0 if hit else 0.0,
                             "why": "命中" if hit else "未命中"})
            out.append(finalize(rows))
        return out


# ---------------------------------------------------------------- B 向量

# 校准：无关≈-0.03~0.18、堆砌≈0.29、相关≈0.32~0.75（2026-10-08 实测）
_EMB_LO = 0.20
_EMB_HI = 0.65


class EmbeddingScorer(BaseScorer):
    """各分面单独向量化，cos(prompt, 分面) 校准后加权平均。需要 fastembed。"""

    name = "embedding"

    def __init__(self, facets: list, model_name: str):
        if not facets:
            raise ScorerUnavailable("语义评分需要『意图』文本")
        self.facets = facets
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ScorerUnavailable(f"fastembed 未安装（pip install fastembed）: {e}")
        self._model = TextEmbedding(model_name=self.model_name)

    def score(self, entries: list) -> list:
        self._ensure()
        docs = [(e.get("prompt_text") or "")[:2000] for e in entries]
        q_vecs = list(self._model.embed([f["text"] for f in self.facets]))
        d_vecs = list(self._model.embed(docs))

        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return num / den if den else 0.0

        out = []
        for dv in d_vecs:
            rows = []
            for f, qv in zip(self.facets, q_vecs):
                c = float(cos(qv, dv))  # numpy 标量 → Python float（JSON 可序列化）
                r = max(0.0, min(1.0, (c - _EMB_LO) / (_EMB_HI - _EMB_LO)))
                rows.append({"text": f["text"], "weight": f["weight"],
                             "score": float(r), "why": f"向量相似度 {c:.3f}"})
            out.append(finalize(rows))
        return out


# ---------------------------------------------------------------- C LLM

_LLM_SYSTEM = (
    "你是 AI 生图作品选品评审员。给定『选品意图分面』（每个分面带权重）与若干 "
    "Midjourney 作品的 prompt，对每条作品按每个分面分别打贴合分（0-10 的整数）"
    "并给一句中文理由。\n"
    "评分锚点：0=无关；2=沾边；4=部分相关；6=比较贴合；8=很贴合；10=完美贴合。\n"
    "要求：拉开区分度（禁止全部给 6-7 分）；权重表示该分面的重要程度，大权重分面"
    "贴合与否对总分影响最大；只看 prompt 文本能推断的内容；prompt 尾部的关键词堆砌"
    "不算贴合。\n"
    '严格只输出 JSON：{"results":[{"i":1,"facets":[{"f":1,"score":8,"why":"…"}]}]}\n'
    "i 是作品编号，f 是分面编号（都从 1 开始，每个作品要覆盖全部分面）。"
)


class LLMScorer(BaseScorer):
    """OpenAI 兼容 chat 接口分面评审；两段式（全量打分 + top 重校准）。"""

    name = "llm"

    def __init__(self, facets: list, base_url: str, model: str, key_env: str,
                 timeout: float = 90.0, batch: int = 30, recalibrate_top: int = 30):
        if not facets:
            raise ScorerUnavailable("语义评分需要『意图』文本")
        self.facets = facets
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch = batch
        self.recalibrate_top = recalibrate_top
        self.key = os.environ.get(key_env) or (
            load_dotenv(_ROOT_ENV).get(key_env) if _ROOT_ENV else None
        )
        if not self.key:
            raise ScorerUnavailable(f"未找到 API 密钥（环境变量 {key_env} 或 .env）")

    # -- HTTP --
    def _chat(self, user_content: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": 6000,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"LLM 输出不含 JSON: {text[:120]}")
        return json.loads(m.group(0))

    def _facet_header(self) -> list:
        lines = ["选品意图分面（编号. 文本 × 权重）："]
        for i, f in enumerate(self.facets, 1):
            lines.append(f"{i}. {f['text']} × {f['weight']:g}")
        lines.append("")
        return lines

    def _batch_score(self, prompts: list) -> list:
        lines = self._facet_header()
        for i, p in enumerate(prompts, 1):
            lines.append(f"[作品{i}] {p[:1000]}")
        raw = self._chat("\n".join(lines))
        data = self._parse_json(raw)
        by_i = {int(r["i"]): r for r in data.get("results", [])}
        out = []
        for i in range(1, len(prompts) + 1):
            r = by_i.get(i) or {}
            f_by = {int(x.get("f", 0)): x for x in (r.get("facets") or [])}
            rows = []
            for fi, f in enumerate(self.facets, 1):
                x = f_by.get(fi) or {}
                s = float(x.get("score", 0) or 0)
                rows.append({"text": f["text"], "weight": f["weight"],
                             "score": max(0.0, min(1.0, s / 10.0)),
                             "why": str(x.get("why") or "LLM 未返回")[:60]})
            out.append(finalize(rows))
        return out

    def score(self, entries: list) -> list:
        prompts = [(e.get("prompt_text") or "") for e in entries]
        results = []
        for start in range(0, len(prompts), self.batch):
            results.extend(self._batch_score(prompts[start:start + self.batch]))
        # 两段式：对 top 重校准（放在一起比着打分，强制区分度）
        if len(results) >= 10:
            order = sorted(range(len(results)),
                           key=lambda i: -results[i]["relevance"])[: self.recalibrate_top]
            try:
                re_res = self._batch_score([prompts[i] for i in order])
                for i, res2 in zip(order, re_res):
                    if res2["relevance"] > 0:
                        results[i] = res2
            except Exception as e:  # noqa: BLE001 — 重校准失败不致命
                log(f"  LLM 二段重校准失败（沿用一段分）: {e}")
        return results


# ---------------------------------------------------------------- 组装

_ROOT_ENV = None  # 由 set_env_path 注入（避免循环依赖 config）


def set_env_path(path) -> None:
    global _ROOT_ENV
    _ROOT_ENV = path


def build_scorers(mode: str, filt: dict, score_cfg: dict) -> list:
    """按模式返回按优先级排列的评分器候选链（auto=llm→embedding→lexical）。"""
    facets = parse_intent(filt.get("意图"))
    lex = LexicalScorer(filt, facets)

    def llm():
        return LLMScorer(
            facets=facets,
            base_url=score_cfg.get("LLM接口") or "https://api.deepseek.com",
            model=score_cfg.get("LLM模型") or "deepseek-chat",
            key_env=score_cfg.get("LLM密钥") or "DEEPSEEK_API_KEY",
        )

    def emb():
        return EmbeddingScorer(
            facets=facets,
            model_name=score_cfg.get("嵌入模型")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    if mode == "lexical":
        return [lex]
    if not facets:  # 语义模式无意义，直接词法
        return [lex]
    if mode == "llm":
        return [llm()]
    if mode == "embedding":
        return [emb()]
    # auto：按 C → B → A 探测可用性
    chain = []
    for factory in (llm, emb):
        try:
            chain.append(factory())
        except ScorerUnavailable as e:
            log(f"  评分器探测：{e}")
    chain.append(lex)
    return chain


class ChainScorer(BaseScorer):
    """依次尝试候选链，失败自动降级；used 记录实际生效者。"""

    name = "auto"

    def __init__(self, chain: list):
        self.chain = chain
        self.used = "?"

    def score(self, entries: list) -> list:
        last = None
        for s in self.chain:
            try:
                res = s.score(entries)
                self.used = s.name
                return res
            except Exception as e:  # noqa: BLE001 — 降级链统一收集
                log(f"  评分器 {s.name} 失败：{e}")
                last = e
        raise ScorerUnavailable(f"所有评分器均失败: {last}")
