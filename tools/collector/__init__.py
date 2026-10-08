# -*- coding: utf-8 -*-
"""MJ 采集器组件包。
模块：
config.py  配置加载与命令行覆盖
schema.py  条目结构（解析/模板/清单文档头）
fetch.py   HTTP 抓取
filter.py  筛选打分
store.py   本地 JSON 状态（已见/台账/库存/清单读写）
"""

from datetime import datetime


def log(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")
