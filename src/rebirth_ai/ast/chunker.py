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
        raise ValueError(f"Unsupported language for chunking: {language}")

    def chunk_file(self, path: Path, source: str | None = None) -> list[CodeChunk]:
        text = source if source is not None else path.read_text(encoding="utf-8")
        return self.chunk_source(text, file_path=str(path))

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
            name = self._name_for(node, kind, source_bytes)
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
                body = node.child_by_field_name("body")
                members = body.children if body is not None else node.children
                for member in members:
                    member_kind = self._classify_member(member)
                    if member_kind is None:
                        # class_body wrapper
                        if member.type == "class_body":
                            for nested in member.children:
                                nested_kind = self._classify_member(nested)
                                if nested_kind is None:
                                    continue
                                chunks.append(
                                    CodeChunk(
                                        kind=nested_kind,
                                        name=self._name_for(
                                            nested, nested_kind, source_bytes
                                        ),
                                        start_line=_line_of(nested.start_byte, source_bytes),
                                        end_line=_line_of(
                                            max(nested.end_byte - 1, nested.start_byte),
                                            source_bytes,
                                        ),
                                        text=_node_text(nested, source_bytes),
                                        file_path=file_path,
                                    )
                                )
                        continue
                    chunks.append(
                        CodeChunk(
                            kind=member_kind,
                            name=self._name_for(member, member_kind, source_bytes),
                            start_line=_line_of(member.start_byte, source_bytes),
                            end_line=_line_of(
                                max(member.end_byte - 1, member.start_byte),
                                source_bytes,
                            ),
                            text=_node_text(member, source_bytes),
                            file_path=file_path,
                        )
                    )
            return

        for child in node.children:
            self._walk(child, source_bytes, file_path, chunks)

    def _classify(self, node: Node) -> str | None:
        t = node.type
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

    def _classify_member(self, node: Node) -> str | None:
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
