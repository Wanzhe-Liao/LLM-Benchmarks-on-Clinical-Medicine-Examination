from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import httpx
import time
import asyncio


@dataclass
class LLMResponse:
    content: str
    model: str
    latency_ms: int
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.content != ''


class BaseLLMAdapter(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', '')
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', '')
        self.timeout = config.get('timeout', 120)
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 1024)
        retry_count = config.get('retry_count', 3)
        self.retry_count = int(3 if retry_count is None else retry_count)
        retry_delay = config.get('retry_delay', 1)
        self.retry_delay = float(1 if retry_delay is None else retry_delay)
        self.extra_body = config.get('extra_body') or {}

    @abstractmethod
    async def complete(self, prompt: str) -> LLMResponse:
        pass

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def _build_request_body(self, prompt: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _parse_response(self, response: Dict[str, Any], latency_ms: int) -> LLMResponse:
        pass

    async def _complete_with(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> LLMResponse:
        start_time = time.time()
        try:
            response = await self._make_request(url, headers, body)
            latency_ms = int((time.time() - start_time) * 1000)
            return self._parse_response(response, latency_ms)
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            msg = str(e) if e is not None else ""
            if not msg:
                msg = repr(e)
            return LLMResponse(
                content="",
                model=self.model_name,
                latency_ms=latency_ms,
                error=msg,
            )

    async def _make_request(self, url: str, headers: Dict[str, str],
                            body: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Exception | None = None
        attempts = max(1, int(self.retry_count))

        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=body)

                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    text = ""
                    try:
                        text = response.text
                    except Exception:
                        text = ""
                    snippet = (text[:800] + "...") if len(text) > 800 else text
                    err = httpx.HTTPStatusError(
                        f"{e} | status={response.status_code} | body={snippet}",
                        request=e.request,
                        response=e.response,
                    )

                    status = response.status_code
                    retriable = status == 429 or status >= 500
                    if not retriable:
                        raise err from None
                    raise err from None

                return response.json()

            except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                    status = e.response.status_code
                    if status < 500 and status != 429:
                        break

                if attempt < attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise last_exc if last_exc else RuntimeError("request failed")


class OpenAICompatibleAdapter(BaseLLMAdapter):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _build_request_body(self, prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if isinstance(self.extra_body, dict) and self.extra_body:
            body.update(self.extra_body)
        return body

    def _parse_response(self, response: Dict[str, Any], latency_ms: int) -> LLMResponse:
        try:
            msg = response["choices"][0]["message"]
            content = msg.get("content", "")
            if not content:
                # Some providers (e.g. Kimi / DeepSeek reasoning variants) may return content in a separate field.
                content = msg.get("reasoning_content", "") or msg.get("reasoning", "") or content
            usage = response.get("usage", {})
            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency_ms,
                usage=usage
            )
        except (KeyError, IndexError) as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                latency_ms=latency_ms,
                error=f"Parse error: {str(e)}"
            )

    async def complete(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        return await self._complete_with(url, self._get_headers(), self._build_request_body(prompt))


class AnthropicAdapter(BaseLLMAdapter):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

    def _build_request_body(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

    def _parse_response(self, response: Dict[str, Any], latency_ms: int) -> LLMResponse:
        try:
            content = response["content"][0]["text"]
            usage = {
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0)
            }
            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency_ms,
                usage=usage
            )
        except (KeyError, IndexError) as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                latency_ms=latency_ms,
                error=f"Parse error: {str(e)}"
            )

    async def complete(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url}/messages"
        return await self._complete_with(url, self._get_headers(), self._build_request_body(prompt))


class GoogleAdapter(BaseLLMAdapter):
    def _get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_request_body(self, prompt: str) -> Dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

    def _parse_response(self, response: Dict[str, Any], latency_ms: int) -> LLMResponse:
        try:
            content = response["candidates"][0]["content"]["parts"][0]["text"]
            usage = response.get("usageMetadata", {})
            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency_ms,
                usage={
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0)
                }
            )
        except (KeyError, IndexError) as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                latency_ms=latency_ms,
                error=f"Parse error: {str(e)}"
            )

    async def complete(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
        return await self._complete_with(url, self._get_headers(), self._build_request_body(prompt))


class DashScopeAdapter(BaseLLMAdapter):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _build_request_body(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "input": {
                "messages": [{"role": "user", "content": prompt}]
            },
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        }

    def _parse_response(self, response: Dict[str, Any], latency_ms: int) -> LLMResponse:
        try:
            content = response["output"]["text"]
            usage = response.get("usage", {})
            return LLMResponse(
                content=content,
                model=self.model_name,
                latency_ms=latency_ms,
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0)
                }
            )
        except (KeyError, IndexError) as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                latency_ms=latency_ms,
                error=f"Parse error: {str(e)}"
            )

    async def complete(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url}/services/aigc/text-generation/generation"
        return await self._complete_with(url, self._get_headers(), self._build_request_body(prompt))


class MiniMaxAdapter(OpenAICompatibleAdapter):
    pass
