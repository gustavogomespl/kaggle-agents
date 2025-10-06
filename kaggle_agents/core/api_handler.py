"""API Handler atualizado para Responses API + gpt-5-mini, com retry robusto."""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import httpx
import openai
from openai import OpenAI

# LangSmith tracing (opcional)
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Constantes
MAX_ATTEMPTS = 5
RETRY_DELAY = 30
DEFAULT_MAX_LENGTH = 100_000

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class APISettings:
    """Configurações padrão para geração."""
    max_completion_tokens: int
    temperature: Optional[float] = 0.7  # pode ser ignorado por alguns modelos
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None

    @property
    def timeout(self) -> int:
        return (self.max_completion_tokens // 1000 + 1) * 30


def load_api_config() -> Tuple[str, Optional[str]]:
    """Carrega OPENAI_API_KEY e OPENAI_BASE_URL do ambiente ou de api_key.txt."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_key:
        return api_key, base_url

    api_key_file = Path(__file__).parent.parent.parent / "api_key.txt"
    if api_key_file.exists():
        with open(api_key_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        api_key = lines[0] if lines else None
        base_url = lines[1] if len(lines) > 1 else None
        if api_key:
            return api_key, base_url

    raise ValueError(
        "API key não encontrada. Defina OPENAI_API_KEY ou crie api_key.txt."
    )


def _to_responses_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Converte mensagens no formato chat clássico para o formato da Responses API:
    [{"role": "user"|"system"|"assistant", "content": [{"type":"text","text":"..."}]}]
    """
    converted = []
    for m in messages:
        content = m.get("content", "")
        # já aceita lista? mantém; senão, empacota como text
        if isinstance(content, list):
            converted.append({"role": m["role"], "content": content})
        else:
            converted.append(
                {"role": m["role"], "content": [{"type": "text", "text": str(content)}]}
            )
    return converted


@traceable(name="OpenAI_API_Call", run_type="llm")
def generate_response(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    settings: APISettings,
    response_type: str = "text",
) -> Any:
    """
    Gera resposta usando Responses API para gpt-5 e cai para Chat Completions para legados.
    """
    logger.info(f"Gerando com modelo: {model}")
    start = time.time()

    is_gpt5 = model.startswith("gpt-5")
    logger.info(f"Usando {'Responses' if is_gpt5 else 'Chat Completions'} API")

    # Monta kwargs de geração (evitando params possivelmente não suportados)
    # Sempre use max_output_tokens na Responses API
    base_kwargs: Dict[str, Any] = {}

    # Alguns modelos gpt-5 podem rejeitar temperature/top_p/etc. Passamos só se não None.
    # Se vier 400, faremos retry removendo-os.
    gen_params: Dict[str, Any] = {}
    if settings.temperature is not None:
        gen_params["temperature"] = settings.temperature
    if settings.top_p is not None:
        gen_params["top_p"] = settings.top_p
    if settings.frequency_penalty is not None:
        gen_params["frequency_penalty"] = settings.frequency_penalty
    if settings.presence_penalty is not None:
        gen_params["presence_penalty"] = settings.presence_penalty
    if settings.stop:
        gen_params["stop"] = settings.stop

    try:
        if response_type != "text":
            raise ValueError(f"Unsupported response type: {response_type}")

        if is_gpt5:
            # Responses API
            resp = client.responses.create(
                model=model,
                input=_to_responses_messages(messages),
                max_output_tokens=settings.max_completion_tokens,
                **gen_params,
            )
        else:
            # Chat Completions (legado)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=settings.max_completion_tokens,
                temperature=settings.temperature if settings.temperature is not None else 0.7,
                top_p=settings.top_p if settings.top_p is not None else 1.0,
                frequency_penalty=settings.frequency_penalty or 0.0,
                presence_penalty=settings.presence_penalty or 0.0,
                stop=settings.stop,
                timeout=settings.timeout,
            )
    except openai.BadRequestError as e:
        # Re-tenta sem parâmetros opcionais que podem não ser suportados pelo gpt-5
        if is_gpt5 and any(k in str(e).lower() for k in ["temperature", "top_p", "penalty"]):
            logger.warning("Parâmetros de amostragem não suportados. Re-tentando sem eles.")
            resp = client.responses.create(
                model=model,
                input=_to_responses_messages(messages),
                max_output_tokens=settings.max_completion_tokens,
            )
        else:
            raise

    elapsed = time.time() - start

    # Métricas de uso (Responses e Chat)
    try:
        usage = getattr(resp, "usage", None)
        if usage:
            # Responses API costuma expor input_tokens/output_tokens/total_tokens
            pt = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
            ct = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
            tt = getattr(usage, "total_tokens", 0) or (pt or 0) + (ct or 0)
            logger.info(
                f"Metrics - Model: {model} | {elapsed:.2f}s | "
                f"Input: {pt} | Output: {ct} | Total: {tt}"
            )
        else:
            logger.info(f"Resposta em {elapsed:.2f}s (sem métricas de uso).")
    except Exception:
        logger.debug("Não foi possível ler métricas de uso.", exc_info=True)

    return resp


class APIHandler:
    """Handler com retry + truncamento para Responses API."""

    def __init__(self, model: str, verify_ssl: bool = True):
        self.model = model
        self.api_key, self.base_url = load_api_config()

        http_client = None
        if not verify_ssl:
            http_client = httpx.Client(verify=False)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
        )

    def _save_long_message(self, messages: List[Dict[str, str]], save_dir: Optional[Path] = None):
        save_dir = Path(save_dir) if save_dir else Path.cwd()
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = save_dir / f"long_message_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for m in messages:
                f.write(f"Role: {m.get('role')}\nContent: {m.get('content')}\n\n")
        logger.info(f"Mensagem longa salva em {filename}")

    def _truncate_messages(self, messages: List[Dict[str, str]], max_length: int = DEFAULT_MAX_LENGTH) -> List[Dict[str, str]]:
        total_len = sum(len(str(m.get("content", ""))) for m in messages)
        if total_len <= max_length:
            return messages

        truncated = messages[:-1]
        last = messages[-1]
        rem = max_length - sum(len(str(m.get("content", ""))) for m in truncated)

        if rem > 100:
            content = str(last.get("content", ""))
            newc = content[: rem - 3] + "..."
            truncated.append({"role": last.get("role", "user"), "content": newc})
            logger.warning(f"Truncado último conteúdo de {len(content)} para {len(newc)} chars.")
        else:
            logger.warning("Sem espaço útil para incluir a última mensagem truncada.")
        return truncated

    def _extract_text(self, response: Any) -> Optional[str]:
        """
        Extrai texto do objeto de resposta:
        - Responses API: response.output_text (preferido)
        - Fallback: navegar em response.output / content
        - Chat Completions: response.choices[0].message.content
        """
        # Responses API
        text = getattr(response, "output_text", None)
        if text:
            return text

        # Fallback Responses (output -> content[])
        try:
            output = getattr(response, "output", None)
            if output and isinstance(output, list):
                # procura primeiro bloco de texto
                for item in output:
                    parts = item.get("content") if isinstance(item, dict) else None
                    if isinstance(parts, list):
                        for p in parts:
                            if isinstance(p, dict) and p.get("type") == "output_text":
                                txt = p.get("text") or p.get("content") or ""
                                if txt:
                                    return txt
        except Exception:
            pass

        # Chat Completions
        try:
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
        except Exception:
            pass
        return None

    def get_output(
        self,
        messages: List[Dict[str, str]],
        settings: APISettings,
        response_type: str = "text",
        save_dir: Optional[Path] = None,
    ) -> str:
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = generate_response(
                    self.client, self.model, messages, settings, response_type
                )
                content = self._extract_text(resp)
                if not content:
                    logger.error(f"Conteúdo vazio. Response: {resp}")
                    # tenta identificar motivo (Responses tem status/reason às vezes)
                    reason = getattr(resp, "status", "") or getattr(resp, "reason", "")
                    return f"Error: resposta vazia. {('Motivo: ' + str(reason)) if reason else ''}".strip()
                return content

            except openai.BadRequestError as e:
                msg = str(e).lower()
                last_error = e
                if any(x in msg for x in ["string too long", "maximum context length", "context_length_exceeded"]):
                    logger.error("Mensagem muito longa. Tentando truncar…")
                    self._save_long_message(messages, save_dir)
                    messages = self._truncate_messages(messages)
                    continue
                logger.error(f"Tentativa {attempt}/{MAX_ATTEMPTS} falhou (400): {e}")

            except (TimeoutError, openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                last_error = e
                logger.error(f"Tentativa {attempt}/{MAX_ATTEMPTS} falhou: {e}")

            if attempt < MAX_ATTEMPTS:
                logger.info(f"Retry em {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)
            else:
                return f"Error: Máximo de tentativas atingido. Último erro: {last_error}"

        return "Error: todas as tentativas falharam."


if __name__ == "__main__":
    # Exemplo rápido
    handler = APIHandler("gpt-5-mini")
    msgs = [
        {"role": "system", "content": "Você é um assistente útil."},
        {"role": "user", "content": "Resuma em 1 frase por que a Responses API é útil."},
    ]
    settings = APISettings(max_completion_tokens=64, temperature=0.7)
    print(handler.get_output(messages=msgs, settings=settings))
