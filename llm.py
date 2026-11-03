"""Minimal client for a locally running Ollama server (REST API, no extra SDK)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import requests

from . import config


class LLMError(Exception):
    """Base class. ``str(error)`` is always safe to show to the analyst."""


class OllamaUnavailable(LLMError):
    pass


class ModelNotInstalled(LLMError):
    pass


class GenerationFailed(LLMError):
    pass


class BadModelOutput(LLMError):
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


@dataclass
class ServerStatus:
    reachable: bool
    models: list[str] = field(default_factory=list)
    message: str = ""


def model_installed(model: str, installed: list[str]) -> bool:
    wanted = model.strip()
    names = {m for m in installed}
    if wanted in names:
        return True
    if ":" not in wanted and f"{wanted}:latest" in names:
        return True
    return False


def parse_json_output(content: str) -> dict:
    """Parse JSON from a model reply, tolerating think-tags and code fences."""
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("Top-level JSON value is not an object", text, 0)
    return value


class OllamaClient:
    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        host: str | None = None,
        num_ctx: int = config.DEFAULT_NUM_CTX,
        temperature: float = config.TEMPERATURE,
        timeout: int = config.REQUEST_TIMEOUT_S,
    ):
        self.model = model
        self.host = (host or config.OLLAMA_URL).rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------ status ----
    def server_status(self, timeout: float = 4.0) -> ServerStatus:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=timeout)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            return ServerStatus(True, sorted(m for m in models if m))
        except requests.exceptions.RequestException as exc:
            return ServerStatus(False, [], f"{type(exc).__name__}")
        except ValueError:
            return ServerStatus(False, [], "The server answered, but not like Ollama.")

    def ensure_ready(self) -> None:
        """Raise a clear error if Ollama or the selected model is not available."""
        status = self.server_status()
        if not status.reachable:
            raise OllamaUnavailable(
                f"Ollama is not reachable at {self.host}. Install it from https://ollama.com, "
                "start it (run `ollama serve`, or open the Ollama app), then try again."
            )
        if not model_installed(self.model, status.models):
            available = ", ".join(status.models) if status.models else "none"
            raise ModelNotInstalled(
                f"The model '{self.model}' is not installed in Ollama (installed: {available}). "
                f"Run `ollama pull {self.model}` in a terminal, or choose an installed model "
                "in the sidebar."
            )

    # -------------------------------------------------------- generation ----
    def _post(self, payload: dict) -> requests.Response:
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=(5, self.timeout),
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailable(
                f"Ollama is not reachable at {self.host}. Start Ollama and try again."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise GenerationFailed(
                "The local model took too long to answer. Try a smaller context size, "
                "fewer or shorter sources, or a smaller model."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise GenerationFailed(f"The request to Ollama failed ({type(exc).__name__}).") from exc
        return response

    def _chat_once(
        self,
        system: str,
        user: str,
        schema: dict,
        num_predict: int,
        seed: int | None,
        on_progress: Callable[[int], None] | None,
    ) -> tuple[str, str]:
        payload = {
            "model": self.model,
            "stream": True,
            "think": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
                **({"seed": seed} if seed is not None else {}),
            },
        }
        response = self._post(payload)
        if response.status_code >= 400 and "think" in response.text.lower():
            payload.pop("think")  # older Ollama versions: retry without the field
            response = self._post(payload)

        if response.status_code == 404:
            raise ModelNotInstalled(
                f"The model '{self.model}' was not found. Run `ollama pull {self.model}`."
            )
        if response.status_code >= 400:
            detail = response.text[:300].strip()
            raise GenerationFailed(f"Ollama returned an error (HTTP {response.status_code}): {detail}")

        content, done_reason = [], ""
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if "error" in obj:
                    message = str(obj["error"])
                    if "not found" in message.lower():
                        raise ModelNotInstalled(
                            f"The model '{self.model}' was not found. Run `ollama pull {self.model}`."
                        )
                    raise GenerationFailed(f"Ollama reported an error: {message}")
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    content.append(piece)
                    if on_progress:
                        on_progress(sum(len(c) for c in content))
                if obj.get("done"):
                    done_reason = obj.get("done_reason", "")
                    break
        except requests.exceptions.RequestException as exc:
            raise GenerationFailed(
                "The connection to Ollama was interrupted during generation."
            ) from exc
        except json.JSONDecodeError as exc:
            raise GenerationFailed("Ollama sent a malformed streaming response.") from exc
        finally:
            response.close()
        return "".join(content), done_reason

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        num_predict: int,
        on_progress: Callable[[int], None] | None = None,
        attempts: int = 2,
    ) -> tuple[dict, str]:
        """Ask the model for schema-constrained JSON. Returns ``(parsed, raw_text)``.

        Raises :class:`BadModelOutput` if no valid JSON object can be obtained.
        """
        last_raw, last_reason = "", ""
        for attempt in range(attempts):
            raw, reason = self._chat_once(
                system, user, schema, num_predict, seed=attempt + 1, on_progress=on_progress
            )
            last_raw, last_reason = raw, reason
            try:
                return parse_json_output(raw), raw
            except json.JSONDecodeError:
                continue
        hint = (
            " The answer was cut off because the output limit was reached."
            if last_reason == "length"
            else ""
        )
        raise BadModelOutput(
            f"The model did not return valid structured output after {attempts} attempts.{hint} "
            "No findings were generated. Try again, use a larger context size, or select a "
            "different model.",
            raw=last_raw,
        )
