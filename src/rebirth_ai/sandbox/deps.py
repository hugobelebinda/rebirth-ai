"""Project dependencies for the sandbox's setup step, and which packages a migration may install.

The setup step (``SandboxRunner``) runs with network *before* the model's draft is written in, so
only the project's own declared dependencies are downloaded there. A package the tests later report
missing is installed only if the project declares it or the user's original file already imports it:
a package that only the model's draft imports (possibly a hallucinated or look-alike name) is refused
unless ``REBIRTH_ALLOW_UNDECLARED_PACKAGES=1``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import tomllib
from pathlib import Path

# import name → PyPI distribution, where they differ (installing the import name would fetch an
# unrelated package, or nothing)
PY_DISTRIBUTIONS = {
    "yaml": "PyYAML", "cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "attr": "attrs", "jwt": "PyJWT",
    "Crypto": "pycryptodome", "dotenv": "python-dotenv", "magic": "python-magic",
    "serial": "pyserial", "usb": "pyusb", "docx": "python-docx", "pptx": "python-pptx",
    "OpenSSL": "pyOpenSSL", "google.protobuf": "protobuf", "MySQLdb": "mysqlclient",
}


def allow_undeclared() -> bool:
    return os.environ.get("REBIRTH_ALLOW_UNDECLARED_PACKAGES", "").strip().lower() in ("1", "true", "yes")


def normalize(name: str) -> str:
    """PEP 503 name (``Foo_Bar`` → ``foo-bar``); npm names are kept as written."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------------------------------------------------------------- JavaScript / TypeScript

_NPM_SECTIONS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")


def npm_package(module: str) -> str:
    """``lodash/fp`` → ``lodash``; ``@scope/pkg/sub`` → ``@scope/pkg``."""
    parts = module.split("/")
    return "/".join(parts[:2]) if module.startswith("@") else parts[0]


def npm_declared(project_dir: Path) -> set[str]:
    p = Path(project_dir) / "package.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {name for sec in _NPM_SECTIONS for name in (data.get(sec) or {})}


def npm_setup(project_dir: Path) -> str | None:
    """Install the project's declared packages (lockfile when there is one)."""
    if not npm_declared(project_dir):
        return None
    if (Path(project_dir) / "package-lock.json").is_file():
        return "npm ci --no-audit --no-fund"
    return "npm install --no-audit --no-fund --no-package-lock"


# ---------------------------------------------------------------- Python

def _req_files(project_dir: Path) -> list[Path]:
    d = Path(project_dir)
    names = sorted({*d.glob("requirements*.txt"), *d.glob("*-requirements.txt"), *d.glob("requirements/*.txt")})
    return [p for p in names if p.is_file()]


def _pyproject(project_dir: Path) -> dict:
    p = Path(project_dir) / "pyproject.toml"
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _pep508_name(req: str) -> str | None:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return m.group(1) if m else None


def pyproject_requirements(project_dir: Path) -> list[str]:
    """PEP 621 dependencies + optional groups, and Poetry dependencies (names only), in file order."""
    data = _pyproject(project_dir)
    proj = data.get("project") or {}
    reqs = list(proj.get("dependencies") or [])
    for group in (proj.get("optional-dependencies") or {}).values():
        reqs += list(group or [])
    for group in (data.get("dependency-groups") or {}).values():  # PEP 735
        reqs += [r for r in group or [] if isinstance(r, str)]
    poetry = (data.get("tool") or {}).get("poetry") or {}
    sections = [poetry.get("dependencies") or {}, poetry.get("dev-dependencies") or {}]
    sections += [(g or {}).get("dependencies") or {} for g in (poetry.get("group") or {}).values()]
    reqs += [name for sec in sections for name in sec if name.lower() != "python"]
    seen, out = set(), []
    for r in reqs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def python_declared(project_dir: Path) -> set[str]:
    names = {_pep508_name(r) for r in pyproject_requirements(project_dir)}
    for f in _req_files(project_dir):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                names.add(_pep508_name(line))
    return {normalize(n) for n in names if n}


def pip_setup(project_dir: Path) -> str | None:
    """``pip install --user`` the requirements files and the pyproject's dependencies (not the project)."""
    d = Path(project_dir)
    args = [f"-r {shlex.quote(str(f.relative_to(d)))}" for f in _req_files(d)]
    args += [shlex.quote(r) for r in pyproject_requirements(d)]
    if not args:
        return None
    return "pip install --user --no-cache-dir --disable-pip-version-check -q " + " ".join(args)


def py_distribution(module: str) -> str:
    """PyPI name to install for an import name (``yaml`` → ``PyYAML``; ``a.b`` → ``a``)."""
    if module in PY_DISTRIBUTIONS:
        return PY_DISTRIBUTIONS[module]
    top = module.split(".")[0]
    return PY_DISTRIBUTIONS.get(top, top)


# ---------------------------------------------------------------- who may install what

def imports_module(code: str, module: str, language: str) -> bool:
    """Whether ``code`` (the user's original file) imports ``module`` itself."""
    m = re.escape(module)
    if language == "python":
        top = re.escape(module.split(".")[0])
        return bool(re.search(rf"^\s*(?:import\s+{top}\b|from\s+{top}(?:\.[\w.]+)?\s+import\b)", code, re.M))
    pat = rf"""(?:\bfrom\s*|\brequire\s*\(\s*|\bimport\s*\(\s*|^\s*import\s+)['"]{m}(?:/[^'"]*)?['"]"""
    return bool(re.search(pat, code, re.M))


_NPM_NAME = re.compile(r"^(?:@[a-z0-9][a-z0-9._~-]*/)?[a-z0-9][a-z0-9._~-]*$")
_PY_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def valid_package_name(name: str, language: str) -> bool:
    return bool((_PY_NAME if language == "python" else _NPM_NAME).match(name))


def install_decision(project_dir: Path, language: str, module: str, original_code: str) -> tuple[str | None, str]:
    """(package to install or None, reason) for a module the tests report missing."""
    if not valid_package_name(npm_package(module) if language != "python" else py_distribution(module), language):
        return None, "not a valid package name"
    if language == "python":
        pkg, declared = py_distribution(module), python_declared(project_dir)
        is_declared = normalize(pkg) in declared or normalize(module.split(".")[0]) in declared
    else:
        pkg = npm_package(module)
        is_declared = pkg in npm_declared(project_dir)
    if is_declared:
        return pkg, "declared by the project"
    if imports_module(original_code or "", module, language):
        return pkg, "imported by the original file"
    if allow_undeclared():
        return pkg, "REBIRTH_ALLOW_UNDECLARED_PACKAGES=1"
    return None, ("only the draft imports it and the project does not declare it "
                  "(REBIRTH_ALLOW_UNDECLARED_PACKAGES=1 allows it)")
