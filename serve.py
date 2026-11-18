"""HTTP inference API for a trained IVA Nano checkpoint.

Run with:
  IVA_CHECKPOINT=checkpoints/artifacts/iva_nano_40m_checkpoint.pt \
  IVA_TOKENIZER=checkpoints/iva_tokenizer.model \
  IVA_API_TOKEN=<long-random-secret> uvicorn serve:app --host 0.0.0.0 --port 8000

This service does inference only. It does not train the model.
"""
import hmac
import os
from pathlib import Path
from threading import Lock

import sentencepiece as spm
import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from iva_nano.model import IVANano
from chat import answer as generate_answer
from iva_nano.prompts import DEFAULT_SYSTEM_PROMPT

app = FastAPI(title="IVA Nano Inference API", version="0.1.0")
_model = None
_tokenizer = None
_device = None
_lock = Lock()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=12)
    max_new_tokens: int = Field(default=160, ge=1, le=256)
    temperature: float = Field(default=0.7, gt=0.0, le=1.5)
    top_k: int = Field(default=40, ge=1, le=100)


def load_runtime():
    global _model, _tokenizer, _device
    if _model is not None:
        return
    checkpoint_path = Path(os.environ.get(
        "IVA_CHECKPOINT", "checkpoints/artifacts/iva_nano_40m_checkpoint.pt"
    ))
    tokenizer_path = Path(os.environ.get(
        "IVA_TOKENIZER", "checkpoints/iva_tokenizer.model"
    ))
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise RuntimeError("IVA Nano checkpoint/tokenizer is not installed")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    _model = IVANano(checkpoint["model_config"]).to(_device)
    _model.load_state_dict(checkpoint["model"])
    _model.eval()
    _tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def require_token(authorization: str | None = Header(default=None)):
    expected = os.environ.get("IVA_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="IVA_API_TOKEN is not configured")
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid API token")


@app.get("/health")
def health():
    checkpoint = Path(os.environ.get(
        "IVA_CHECKPOINT", "checkpoints/artifacts/iva_nano_40m_checkpoint.pt"
    ))
    tokenizer = Path(os.environ.get(
        "IVA_TOKENIZER", "checkpoints/iva_tokenizer.model"
    ))
    return {
        "service": "iva-nano-inference",
        "model_id": "iva-nano-40m",
        "parameters": 40015008,
        "checkpoint_present": checkpoint.is_file(),
        "tokenizer_present": tokenizer.is_file(),
        "ready": checkpoint.is_file() and tokenizer.is_file(),
    }


@app.post("/v1/chat")
def chat(request: ChatRequest, _auth: None = Depends(require_token)):
    try:
        with _lock:
            load_runtime()
            messages = [(m.role, m.content) for m in request.messages]
            result = generate_answer(
                _model, _tokenizer, messages,
                DEFAULT_SYSTEM_PROMPT,
                _device, request.max_new_tokens, request.temperature, request.top_k,
            )
        return {"model": "iva-nano-40m", "text": result}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
