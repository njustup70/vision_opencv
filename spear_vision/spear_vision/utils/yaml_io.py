"""
YAML 读写工具（spear_vision）

设计目标：
1) 统一使用 `yaml.safe_load/safe_dump`，避免执行任意对象反序列化带来的安全风险；
2) 默认不排序 key（sort_keys=False），让导出的 YAML 更贴近人类阅读顺序；
3) `allow_unicode=True`，保证中文注释/字段不会被转义成 \\uXXXX。
"""

from __future__ import annotations

import os
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    # 支持 "~" 家目录路径，方便在 launch/参数里写简短路径
    expanded = os.path.expanduser(path)
    with open(expanded, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        # 空文件视为 {}，便于上层用 dict.get(...)
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {expanded}")
    return data


def save_yaml(path: str, data: Any) -> None:
    # 自动创建输出目录（例如 ~/xxx/xxx.yaml），避免用户手动 mkdir -p
    expanded = os.path.expanduser(path)
    os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
    with open(expanded, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
