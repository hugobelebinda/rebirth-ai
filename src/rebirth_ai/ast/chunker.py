"""Tree-sitter based code chunking for RAG / context retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser


@dataclass(frozen=True)
class CodeChunk:
    """A structured slice of a source file."""

    kind: str  # import | function | class | method | export | other
    name: str
    start_line: int
    end_line: int
    text: str
    file_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "file_path": self.file_path,
        }


def _line_of(byte_offset: int, source_bytes: bytes) -> int:
    return source_bytes.count(b"\n", 0, byte_offset) + 1


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _identifier(node: Node | None, source_bytes: bytes) -> str:
    if node is None:
        return "<anonymous>"
    if node.type in {"identifier", "property_identifier", "type_identifier"}:
        return _node_text(node, source_bytes)
    for child in node.children:
        found = _identifier(child, source_bytes)
        if found != "<anonymous>":
            return found
    return "<anonymous>"


class CodeChunker:
    """Extract imports, functions, and classes from JS/TS (and extensible later)."""

    def __init__(self, language: str = "javascript") -> None:
        self.language_name = language
        self._language = self._load_language(language)
        self._parser = Parser(self._language)

    @staticmethod
    def _load_language(language: str) -> Language:
        key = language.lower()
        if key in {"javascript", "js", "jsx"}:
            import tree_sitter_javascript as tsjs

            return Language(tsjs.language())
        if key in {"typescript", "ts"}:
            import tree_sitter_typescript as tsts

            return Language(tsts.language_typescript())
        if key in {"tsx"}:
            import tree_sitter_typescript as tsts

            return Language(tsts.language_tsx())
        if key in {"python", "py"}:
            import tree_sitter_python as tspy

            return Language(tspy.language())
        if key in {"go"}:
            import tree_sitter_go as tsgo

            return Language(tsgo.language())
        if key in {"java"}:
            import tree_sitter_java as tsjava

            return Language(tsjava.language())
        if key in {"c_sharp", "csharp", "c#"}:
            import tree_sitter_c_sharp as tscs

            return Language(tscs.language())
        if key in {"rust"}:
            import tree_sitter_rust as tsrust

            return Language(tsrust.language())
        if key in {"cpp", "c++", "cxx"}:
            import tree_sitter_cpp as tscpp

            return Language(tscpp.language())
        if key in {"swift"}:
            import tree_sitter_swift as tsswift

            return Language(tsswift.language())
        raise ValueError(f"Unsupported language for chunking: {language}")

    def chunk_file(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        text = source if source is not None else path.read_text(encoding="utf-8")
        return self.chunk_source(text, file_path=str(path))

    def parses(self, source: str) -> bool:
        """True if Tree-sitter builds a tree with no ERROR / missing nodes."""
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)

        def bad(node: Node) -> bool:
            if node.type == "ERROR" or node.is_missing:
                return True
            return any(bad(c) for c in node.children)

        return not bad(tree.root_node)

    def chunk_source(self, source: str, file_path: str = "<memory>") -> list[CodeChunk]:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        chunks: list[CodeChunk] = []
        self._walk(tree.root_node, source_bytes, file_path, chunks)
        return chunks

    def _walk(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        chunks: list[CodeChunk],
    ) -> None:
        kind = self._classify(node)
        if kind is not None:
            decl = self._declaration_node(node)
            name = self._name_for(decl, kind, source_bytes)
            chunks.append(
                CodeChunk(
                    kind=kind,
                    name=name,
                    start_line=_line_of(node.start_byte, source_bytes),
                    end_line=_line_of(max(node.end_byte - 1, node.start_byte), source_bytes),
                    text=_node_text(node, source_bytes),
                    file_path=file_path,
                )
            )
            # Do not descend into function/class bodies for nested extraction in v1
            # except we still want methods on classes — handle separately.
            if kind == "class":
                self._emit_class_members(decl, source_bytes, file_path, chunks)
            return

        for child in node.children:
            self._walk(child, source_bytes, file_path, chunks)

    def _declaration_node(self, node: Node) -> Node:
        """Unwrap export_statement to the inner declaration when present."""
        if node.type != "export_statement":
            return node
        for child in node.children:
            if child.type in {
                "function_declaration",
                "generator_function_declaration",
                "class_declaration",
                "lexical_declaration",
            }:
                return child
        return node

    def _emit_class_members(
        self,
        class_node: Node,
        source_bytes: bytes,
        file_path: str,
        chunks: list[CodeChunk],
    ) -> None:
        body = class_node.child_by_field_name("body")
        if body is None:
            # Fallback: find a class_body child (JS/TS) or block (Python)
            for child in class_node.children:
                if child.type in {"class_body", "block"}:
                    body = child
                    break
        if body is None:
            return
        for member in body.children:
            member_kind = self._classify_member(member)
            if member_kind is None:
                continue
            chunks.append(
                CodeChunk(
                    kind=member_kind,
                    name=self._name_for(member, member_kind, source_bytes),
                    start_line=_line_of(member.start_byte, source_bytes),
                    end_line=_line_of(max(member.end_byte - 1, member.start_byte), source_bytes),
                    text=_node_text(member, source_bytes),
                    file_path=file_path,
                )
            )

    def _classify(self, node: Node) -> str | None:
        t = node.type
        if self.language_name in {"python", "py"}:
            return self._classify_python(node)
        if self.language_name == "go":
            return self._classify_go(node)
        if self.language_name == "java":
            return self._classify_java(node)
        if self.language_name in {"c_sharp", "csharp", "c#"}:
            return self._classify_csharp(node)
        if self.language_name == "rust":
            return self._classify_rust(node)
        if self.language_name in {"cpp", "c++", "cxx"}:
            return self._classify_cpp(node)
        if self.language_name == "swift":
            return self._classify_swift(node)
        if t in {
            "import_statement",
            "import_clause",
            "lexical_declaration",  # handled below if it's export-ish
        }:
            if t == "import_statement":
                return "import"
        if t == "export_statement":
            # Prefer classifying the declaration inside when possible
            for child in node.children:
                inner = self._classify(child)
                if inner in {"function", "class"}:
                    return inner
            return "export"
        if t in {
            "function_declaration",
            "generator_function_declaration",
            "function",
            "arrow_function",
        }:
            # Only top-level function_declaration / generator — arrow assigned via lexical
            if t in {"function_declaration", "generator_function_declaration"}:
                return "function"
        if t in {"class_declaration", "class"}:
            if t == "class_declaration":
                return "class"
        if t == "lexical_declaration":
            # const foo = () => {} or function expr
            for child in node.children:
                if child.type == "variable_declarator":
                    value = child.child_by_field_name("value")
                    if value is not None and value.type in {
                        "arrow_function",
                        "function",
                        "function_expression",
                    }:
                        return "function"
        return None

    def _classify_python(self, node: Node) -> str | None:
        t = node.type
        if t in {"import_statement", "import_from_statement"}:
            return "import"
        if t in {"function_definition", "async_function_definition"}:
            return "function"
        if t == "class_definition":
            return "class"
        return None

    def _classify_go(self, node: Node) -> str | None:
        t = node.type
        if t == "import_declaration":
            return "import"
        if t in {"function_declaration", "method_declaration"}:
            return "function" if t == "function_declaration" else "method"
        if t == "type_declaration":
            return "class"
        return None

    def _classify_java(self, node: Node) -> str | None:
        t = node.type
        if t == "import_declaration":
            return "import"
        if t in {"method_declaration", "constructor_declaration"}:
            return "method"
        if t in {"class_declaration", "interface_declaration", "enum_declaration"}:
            return "class"
        return None

    def _classify_csharp(self, node: Node) -> str | None:
        t = node.type
        if t in {"using_directive", "import_directive"}:
            return "import"
        if t in {"method_declaration", "constructor_declaration", "local_function_statement"}:
            return "method"
        if t in {"class_declaration", "struct_declaration", "interface_declaration", "record_declaration"}:
            return "class"
        return None

    def _classify_rust(self, node: Node) -> str | None:
        t = node.type
        if t in {"use_declaration", "extern_crate_declaration"}:
            return "import"
        if t in {"function_item", "function_signature_item"}:
            return "function"
        if t in {"struct_item", "enum_item", "impl_item", "trait_item"}:
            return "class"
        return None

    def _classify_cpp(self, node: Node) -> str | None:
        t = node.type
        if t in {"preproc_include", "using_declaration"}:
            return "import"
        if t in {"function_definition", "declaration"}:
            # Only treat clear function_definition as function
            if t == "function_definition":
                return "function"
        if t in {"class_specifier", "struct_specifier"}:
            return "class"
        return None

    def _classify_swift(self, node: Node) -> str | None:
        t = node.type
        if t in {"import_declaration"}:
            return "import"
        if t in {"function_declaration", "init_declaration"}:
            return "function"
        if t in {"class_declaration", "struct_declaration", "enum_declaration", "protocol_declaration"}:
            return "class"
        return None

    def _classify_member(self, node: Node) -> str | None:
        if self.language_name in {"python", "py"}:
            if node.type in {"function_definition", "async_function_definition"}:
                return "method"
            return None
        if self.language_name in {"java", "c_sharp", "csharp", "c#", "swift", "go"}:
            if node.type in {
                "method_declaration",
                "constructor_declaration",
                "function_declaration",
                "init_declaration",
                "local_function_statement",
            }:
                return "method"
            return None
        if self.language_name == "rust":
            if node.type in {"function_item"}:
                return "method"
            return None
        if node.type in {
            "method_definition",
            "public_field_definition",
            "field_definition",
            "property_signature",
            "method_signature",
        }:
            if "method" in node.type:
                return "method"
            return "field"
        return None

    def _name_for(self, node: Node, kind: str, source_bytes: bytes) -> str:
        if kind == "import":
            if self.language_name in {"python", "py"}:
                # Prefer dotted_name / relative_import module path
                for child in node.children:
                    if child.type in {"dotted_name", "relative_import", "string"}:
                        return _node_text(child, source_bytes).strip("'\"")
                return "import"
            # Prefer the module specifier string
            for child in node.children:
                if child.type == "string":
                    return _node_text(child, source_bytes).strip("'\"")
            return "import"
        if kind in {"function", "class", "method", "field"}:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return _node_text(name_node, source_bytes)
            # lexical const name = ...
            if node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        n = child.child_by_field_name("name")
                        if n is not None:
                            return _node_text(n, source_bytes)
            return _identifier(node, source_bytes)
        return kind
