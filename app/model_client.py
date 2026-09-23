import httpx
import base64

from app.config import Settings
from app.errors import AgentError


def analyze_model(body, files: list[bytes], key: str, settings: Settings):
    require_model_config(settings)
    payload = body.model_dump(mode="json")
    for document, content in zip(payload["documents"], files, strict=True):
        document.pop("storage_key")
        document["content_base64"] = base64.b64encode(content).decode()
    try:
        with httpx.Client(timeout=httpx.Timeout(settings.model_request_timeout_seconds, connect=settings.model_connect_timeout_seconds), follow_redirects=False, trust_env=False) as client:
            with client.stream("POST", settings.model_api_url, json=payload, headers={"Authorization": f"Bearer {settings.model_api_key.get_secret_value()}", "Idempotency-Key": key}) as response:
                if response.status_code != 200:
                    raise AgentError(503, "MODEL_UNAVAILABLE", "Remote model is unavailable")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 1024 * 1024:
                        raise AgentError(502, "MODEL_INVALID_RESPONSE", "Model response is too large")
                return httpx.Response(200, content=bytes(raw)).json()
    except httpx.TimeoutException:
        raise AgentError(504, "MODEL_TIMEOUT", "Remote model timed out") from None
    except httpx.HTTPError:
        raise AgentError(503, "MODEL_UNAVAILABLE", "Remote model is unavailable") from None
    except ValueError:
        raise AgentError(502, "MODEL_INVALID_RESPONSE", "Model response is not JSON") from None


def require_model_config(settings: Settings) -> None:
    if not settings.model_api_url or not settings.model_api_key.get_secret_value():
        raise AgentError(503, "MODEL_NOT_CONFIGURED", "Remote model is not configured")


def check_model_ready(settings: Settings) -> None:
    require_model_config(settings)
    if not settings.model_health_url:
        raise AgentError(503, "MODEL_HEALTH_NOT_CONFIGURED", "Remote model health endpoint is not configured")
    timeout = httpx.Timeout(settings.model_connect_timeout_seconds)
    # No retries or redirects. Never forward credentials to a redirect destination.
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
            with client.stream("GET", settings.model_health_url, headers={
                "Authorization": f"Bearer {settings.model_api_key.get_secret_value()}"
            }) as response:
                if response.status_code != 200:
                    raise AgentError(503, "MODEL_UNAVAILABLE", "Remote model is unavailable")
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 64 * 1024:
                        raise AgentError(502, "MODEL_INVALID_RESPONSE", "Invalid remote model response")
                payload = httpx.Response(200, content=bytes(content)).json()
                if not isinstance(payload, dict) or payload.get("status") != "ok":
                    raise AgentError(503, "MODEL_NOT_READY", "Remote model is not ready")
    except httpx.TimeoutException:
        raise AgentError(504, "MODEL_TIMEOUT", "Remote model timed out") from None
    except httpx.HTTPError:
        raise AgentError(503, "MODEL_UNAVAILABLE", "Remote model is unavailable") from None
    except (ValueError, UnicodeError):
        raise AgentError(502, "MODEL_INVALID_RESPONSE", "Invalid remote model response") from None
