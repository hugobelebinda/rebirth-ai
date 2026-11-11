"""Fail fast, with a sentence instead of a traceback, when Ollama or Docker is not there."""

from __future__ import annotations


def ollama_error(host: str | None = None) -> str | None:
    import httpx

    from rebirth_ai.generate.ollama_client import ollama_host

    url = ollama_host(host)
    try:
        httpx.get(f"{url}/api/tags", timeout=3.0).raise_for_status()
        return None
    except Exception as exc:  # connection refused, timeout, HTTP error
        return (f"Ollama is not reachable at {url} ({type(exc).__name__}). Start it (`ollama serve` or the "
                "Ollama app; OLLAMA_HOST points elsewhere), then run `rebirth doctor`.")


def docker_error() -> str | None:
    try:
        import docker

        docker.from_env().ping()
        return None
    except Exception as exc:  # daemon down, socket missing, no permission
        reason = str(exc).split("(")[0].strip() or type(exc).__name__
        return (f"Docker is not reachable ({reason}). Start Docker and check that your user can use it "
                "(`docker ps`), then run `rebirth doctor`.")
