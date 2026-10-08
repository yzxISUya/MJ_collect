# -*- coding: utf-8 -*-
"""相关度评分器：词法 / 本地向量 / LLM 评审，统一输出 R∈[0,1] + 理由。

选型实测（2026-10-08，项目笔记 §13.9）：
- 向量模型定为 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  （fastembed 量化 ONNX，0.22GB，约 50 语种）：语义分离度 0.397 胜
  potion-multilingual（0.209），且 128-token 截断天然压制 prompt 尾部关键词堆砌
  （堆砌攻击样例 0.288 vs potion 0.520）。
- 向量校准：R = clip((cos - 0.20) / 0.45, 0, 1)。实测无关≈-0.03~0.18、
  堆砌≈0.29、相关≈0.32~0.75、跨语种命中≈0.61~0.65。
- LLM 评分走 OpenAI 兼容接口（DeepSeek 等），temp=0 + 锚点量规 + 二段式重校准。

模式（评分器）：
  llm       C 方案：LLM 评审（意图理解最强，附理由）
  embedding B 方案：本地向量（离线、排序几何好）
  lexical   A 方案：词法权重归一（零依赖兜底）
  auto      依次尝试 llm → embedding → lexical，失败自动降级
语义模式（llm/embedding）需要『意图』文本；无意图时只有词法模式有意义。
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request

from collector import log
from collector.filter import compile_word, keyword_weights

# ---------------------------------------------------------------- 通用


class ScorerUnavailable(Exception):
    """评分器当前不可用（缺依赖/缺密钥）。"""


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
        """返回与 entries 等长的 [(R∈[0,1], why)]。"""
        raise NotImplementedError


# ---------------------------------------------------------------- A 词法


class LexicalScorer(BaseScorer):
    """R = 命中关键词的权重和 / 总权重；无关键词 = 纯热榜（R 全 1）。"""

    name = "lexical"

    def __init__(self, filt: dict):
        self.kws = [k.lower() for k in (filt.get("包含关键词") or [])]
        self.weights = keyword_weights(filt)
        self.patterns = [compile_word(k) for k in self.kws]

    def score(self, entries: list) -> list:
        total = sum(self.weights) or 1.0
        out = []
        for e in entries:
            text = (e.get("prompt_text") or "").lower()
            hits = [(k, w) for k, w, p in zip(self.kws, self.weights, self.patterns)
                    if p.search(text)]
            if not self.kws:
                out.append((1.0, "纯热榜模式（无意图无关键词）"))
            else:
                got = sum(w for _, w in hits)
                tag = "+".join(k for k, _ in hits) or "无命中"
                out.append((got / total, f"词法 {got:g}/{total:g}（{tag}）"))
        return out


# ---------------------------------------------------------------- B 向量

# 校准：无关≈-0.03~0.18、堆砌≈0.29、相关≈0.32~0.75
_EMB_LO = 0.20
_EMB_HI = 0.65


class EmbeddingScorer(BaseScorer):
    """本地向量余弦 → 校准到 [0,1]。需要 fastembed 与意图文本。"""

    name = "embedding"

    def __init__(self, intent: str, model_name: str):
        if not (intent or "").strip():
            raise ScorerUnavailable("语义评分需要『意图』文本")
        self.intent = intent.strip()
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
        d_vec = list(self._model.embed(docs))
        q_vec = next(iter(self._model.embed([self.intent])))

        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return num / den if den else 0.0

        out = []
        for vec in d_vec:
            c = float(cos(q_vec, vec))  # numpy 标量 → Python float（JSON 可序列化）
            r = max(0.0, min(1.0, (c - _EMB_LO) / (_EMB_HI - _EMB_LO)))
            out.append((float(r), f"向量相似度 {c:.3f}"))
        return out


# ---------------------------------------------------------------- C LLM

_LLM_SYSTEM = (
    "你是 AI 生图作品选品评审员。给定『选品意图』与若干 Midjourney 作品的 prompt，"
    "输出每条作品与意图的贴合分（0-10 的整数）和一句中文理由。\n"
    "评分锚点：0=无关；2=沾边；4=部分相关；6=比较贴合；8=很贴合；10=完美贴合。\n"
    "要求：拉开区分度（禁止全部给 6-7 分）；只看 prompt 文本能推断的内容；"
    "prompt 尾部的关键词堆砌不算贴合。\n"
    '严格只输出 JSON：{"results":[{"i":1,"score":7,"why":"…"}]}'
)


class LLMScorer(BaseScorer):
    """OpenAI 兼容 chat 接口批量评审；两段式（全量打分 + top 重校准）。"""

    name = "llm"

    def __init__(self, intent: str, base_url: str, model: str, key_env: str,
                 timeout: float = 90.0, batch: int = 40, recalibrate_top: int = 30):
        if not (intent or "").strip():
            raise ScorerUnavailable("语义评分需要『意图』文本")
        self.intent = intent.strip()
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
            "max_tokens": 4000,
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

    def _batch_score(self, prompts: list, ids: list) -> list:
        lines = [f"选品意图：{self.intent}", ""]
        for i, p in enumerate(prompts, 1):
            lines.append(f"[{i}] {p[:1000]}")
        raw = self._chat("\n".join(lines))
        data = self._parse_json(raw)
        by_i = {int(r["i"]): r for r in data.get("results", [])}
        out = []
        for i in range(1, len(prompts) + 1):
            r = by_i.get(i)
            if not r:
                out.append((0.0, "LLM 未返回该条"))
                continue
            score = float(r.get("score", 0))
            out.append((max(0.0, min(1.0, score / 10.0)), str(r.get("why") or "")[:80]))
        return out

    def score(self, entries: list) -> list:
        prompts = [(e.get("prompt_text") or "") for e in entries]
        results = []
        for start in range(0, len(prompts), self.batch):
            chunk = prompts[start:start + self.batch]
            results.extend(self._batch_score(chunk, None))
        # 两段式：对 top 重校准（放在一起比着打分，强制区分度）
        if len(results) >= 10:
            order = sorted(range(len(results)), key=lambda i: -results[i][0])[: self.recalibrate_top]
            try:
                re_scores = self._batch_score([prompts[i] for i in order], order)
                for i, (r2, why2) in zip(order, re_scores):
                    if r2 > 0:
                        results[i] = (r2, why2 or results[i][1])
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
    intent = (filt.get("意图") or "").strip()
    lex = LexicalScorer(filt)

    def llm():
        return LLMScorer(
            intent=intent,
            base_url=score_cfg.get("LLM接口") or "https://api.deepseek.com",
            model=score_cfg.get("LLM模型") or "deepseek-chat",
            key_env=score_cfg.get("LLM密钥") or "DEEPSEEK_API_KEY",
        )

    def emb():
        return EmbeddingScorer(
            intent=intent,
            model_name=score_cfg.get("嵌入模型")
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

    if mode == "lexical":
        return [lex]
    if not intent:  # 语义模式无意义，直接词法
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
