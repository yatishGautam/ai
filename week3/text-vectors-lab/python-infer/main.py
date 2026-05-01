from typing import List, Optional
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="embedding-inference-service", version="0.1")

MODEL_NAME_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
model: Optional[SentenceTransformer] = None

class EmbeddingRequests(BaseModel):
    model : str = Field(default = MODEL_NAME_DEFAULT, min_length = 1)
    texts : List[str] = Field(min_length = 1, max_length = 64)
    normalize : Optional[bool] = Field(default = True)

class EmbeddingResponse(BaseModel):
    model: str
    dim: int
    vectors: List[List[float]]
    usage : dict
    timing_ms : dict

@app.on_event("startup")
def _startup():
    global model
    model = SentenceTransformer(MODEL_NAME_DEFAULT)


@app.get("/health/live")
def live():
    return {"status": "alive"}

@app.get("/health/ready")
def ready():
    return {"status": "ready", "model": model is not None}

@app.post("/internal/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequests):
    global model

    if any((t is None or not t.strip()) for t in req.texts):
        raise HTTPException(status_code=400, detail="Texts must not be empty")
    
    if model is None or getattr(model, "model_card", None) is None:
        pass

    t0 = time.time()
    if req.model != MODEL_NAME_DEFAULT:
        model = SentenceTransformer(req.model)
    t1 = time.time()
    vecs = model.encode(
        req.texts,
        batch_size=min(len(req.texts), 32),
        normalize_embeddings = req.normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    t2 = time.time()

    vecs = np.asarray(vecs, dtype=np.float32)
    vectors = vecs.tolist()

    dim = int(vecs.shape[1]) if vecs.ndim == 2 else int(vecs.shape[0])
    usage = {"texts": len(req.texts)}

    return EmbeddingResponse(
        model=req.model,
        dim=dim,
        vectors=vectors,
        usage=usage,
        timing_ms={
            "model_load": int((t1 - t0) * 1000),
            "inference": int((t2 - t1) * 1000),
            "total": int((t2 - t0) * 1000),
        },
    )

    