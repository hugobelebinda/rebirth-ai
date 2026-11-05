"""Language adapters (full v1.0 set via registry)."""

from rebirth_ai.adapters.base import LanguageAdapter, SandboxSpec
from rebirth_ai.adapters.cpp import CppAdapter
from rebirth_ai.adapters.csharp import CSharpAdapter
from rebirth_ai.adapters.go_lang import GoAdapter
from rebirth_ai.adapters.java_lang import JavaAdapter
from rebirth_ai.adapters.js_ts import JsTsAdapter
from rebirth_ai.adapters.python import PythonAdapter
from rebirth_ai.adapters.registry import (
    adapter_summaries,
    detect_adapter_id,
    get_adapter,
    get_adapter_for_project,
    language_label_for_file,
    list_adapter_ids,
)
from rebirth_ai.adapters.rust import RustAdapter
from rebirth_ai.adapters.swift import SwiftAdapter

__all__ = [
    "LanguageAdapter",
    "SandboxSpec",
    "JsTsAdapter",
    "PythonAdapter",
    "GoAdapter",
    "JavaAdapter",
    "CSharpAdapter",
    "RustAdapter",
    "CppAdapter",
    "SwiftAdapter",
    "adapter_summaries",
    "detect_adapter_id",
    "get_adapter",
    "get_adapter_for_project",
    "language_label_for_file",
    "list_adapter_ids",
]
