import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import mlx_lm

# --- State ---
# This dict is module-level. Lives for the lifetime of the process.
# Model loads once here, never again.
model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup before any request is served
    print("Loading model...")
    model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    model_state["model"] = model
    model_state["tokenizer"] = tokenizer
    print("Model loaded. Server ready.")
    yield
    # Runs once at shutdown
    model_state.clear()

app = FastAPI(lifespan=lifespan)

# --- Request / Response shapes ---
class GenerateRequest(BaseModel):
    tenant_id: str
    prompt: str
    max_tokens: int = 150

class GenerateResponse(BaseModel):
    tenant_id: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

# --- Endpoint ---
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    model = model_state["model"]
    tokenizer = model_state["tokenizer"]

    # Count input tokens
    input_token_ids = tokenizer.encode(request.prompt)
    input_token_count = len(input_token_ids)

    # Generate
    start = time.time()
    response_text = mlx_lm.generate(
        model,
        tokenizer,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        verbose=False
    )
    latency_ms = (time.time() - start) * 1000

    # Count output tokens
    output_token_ids = tokenizer.encode(response_text)
    output_token_count = len(output_token_ids)

    return GenerateResponse(
        tenant_id=request.tenant_id,
        response=response_text,
        input_tokens=input_token_count,
        output_tokens=output_token_count,
        latency_ms=round(latency_ms, 2)
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "model" in model_state}
