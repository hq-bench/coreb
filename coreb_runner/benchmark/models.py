"""
Code Embedding Model Wrappers

Model wrappers for different embedding frameworks to work with CoIR evaluation.
All wrappers implement a consistent interface for encoding queries and corpus.
"""

import hashlib
import logging
import os
import re
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

try:
    from easyllm_kit.utils import get_logger

    logger = get_logger('code_emb_models')
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger("code_emb_models")
    logger.warning("easyllm_kit logger unavailable; using stdlib logging fallback.")


class BaseEmbeddingWrapper(ABC):
    """Base class for embedding model wrappers."""

    @abstractmethod
    def encode_queries(self, queries: List[str], batch_size: int = 12, **kwargs) -> np.ndarray:
        """Encode queries to embeddings."""
        pass

    @abstractmethod
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 12, **kwargs) -> np.ndarray:
        """Encode corpus documents to embeddings."""
        pass


class HuggingFaceWrapper(BaseEmbeddingWrapper):
    """Wrapper for HuggingFace Transformers models compatible with CoIR."""

    def __init__(self, model_name: str, **kwargs):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )

        # Extract wrapper-only kwargs before forwarding to from_pretrained
        device_cfg         = kwargs.pop("device", None)
        self._normalize    = kwargs.pop("normalize_embeddings", True)
        self._default_task = kwargs.pop("default_task", None)
        kwargs.pop("delete_after_eval", None)
        kwargs.pop("batch_size", None)

        def _resolve_complete_local_snapshot(name: str, cache_dir: str):
            """
            Resolve a complete local HF snapshot if cache refs/main is incomplete.

            This avoids transformers crashes like:
            AttributeError: 'NoneType' object has no attribute 'endswith'
            when refs/main points to a tokenizer-only snapshot.
            """
            if not cache_dir or "/" not in name:
                return None

            model_cache_dir = Path(cache_dir) / f"models--{name.replace('/', '--')}"
            snapshots_dir = model_cache_dir / "snapshots"
            refs_main = model_cache_dir / "refs" / "main"
            if not snapshots_dir.exists():
                return None

            need_jina_v4_adapters = "jina-embeddings-v4" in name.lower()

            def _is_complete(snapshot_dir: Path) -> bool:
                markers = (
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors.index.json",
                    "pytorch_model.bin.index.json",
                )
                has_weights = any((snapshot_dir / m).exists() for m in markers)
                if not has_weights:
                    return False
                if not need_jina_v4_adapters:
                    return True
                # Jina v4 requires local LoRA adapter files during model load.
                return (snapshot_dir / "adapters" / "adapter_config.json").exists()

            if refs_main.exists():
                try:
                    ref_id = refs_main.read_text(encoding="utf-8").strip()
                    ref_snapshot = snapshots_dir / ref_id
                    if ref_snapshot.exists() and _is_complete(ref_snapshot):
                        return str(ref_snapshot)
                except Exception:
                    pass

            candidates = [p for p in snapshots_dir.iterdir() if p.is_dir() and _is_complete(p)]
            if not candidates:
                return None
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(candidates[0])

        model_source = model_name
        cache_dir = kwargs.get("cache_dir")
        local_snapshot = _resolve_complete_local_snapshot(model_name, cache_dir)
        requires_jina_v4_adapters = "jina-embeddings-v4" in model_name.lower()
        if local_snapshot:
            logger.info(f"Using complete local snapshot for {model_name}: {local_snapshot}")
            model_source = local_snapshot
            kwargs.pop("cache_dir", None)
        elif requires_jina_v4_adapters and cache_dir:
            model_cache_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
            snapshots_dir = model_cache_dir / "snapshots"
            if snapshots_dir.exists():
                raise FileNotFoundError(
                    f"{model_name} cache exists but no complete snapshot with required "
                    f"`adapters/adapter_config.json` found under {snapshots_dir}. "
                    "Re-download this model to cache (including adapters), or use another model."
                )

        logger.info(f"Loading HuggingFace model: {model_source}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
        self.model = AutoModel.from_pretrained(model_source, **kwargs)
        self.model_name = model_name
        # jina-embeddings-v4: forward(task_label=...) required; pooling via single_vec_emb
        self._jina_v4 = "jina-embeddings-v4" in model_name.lower()
        self._jina_task = self._default_task
        if self._jina_v4 and not self._jina_task:
            # Keep backward compatibility with older configs that omitted default_task.
            self._jina_task = "retrieval"
            logger.warning(
                "default_task not set for jina-embeddings-v4; falling back to "
                "task_label='retrieval'."
            )

        import torch
        if device_cfg and device_cfg != "auto":
            self.device = device_cfg
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on device: {self.device}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        import torch
        # Some models (e.g. C2LLM) already return a pooled embedding directly
        # under the 'sentence_embedding' key — no pooling needed.
        if isinstance(model_output, dict) and "sentence_embedding" in model_output:
            return model_output["sentence_embedding"]
        if hasattr(model_output, "sentence_embedding"):
            return model_output.sentence_embedding
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_queries(self, queries: List[str], batch_size: int = 12, **kwargs) -> np.ndarray:
        """Encode queries."""
        logger.info(f"Encoding {len(queries)} queries with batch_size={batch_size}")
        return self._encode_texts(queries, batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 12, **kwargs) -> np.ndarray:
        """Encode corpus documents."""
        texts = [doc['text'] for doc in corpus]
        logger.info(f"Encoding {len(texts)} corpus documents with batch_size={batch_size}")
        return self._encode_texts(texts, batch_size, **kwargs)

    def _encode_texts(self, texts: List[str], batch_size: int = 12, max_length: int = 512, **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        import torch
        from tqdm import tqdm

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                forward_kw = dict(encoded)
                if self._jina_v4:
                    forward_kw['task_label'] = self._jina_task
                model_output = self.model(**forward_kw)
                if self._jina_v4 and getattr(model_output, 'single_vec_emb', None) is not None:
                    embeddings = model_output.single_vec_emb
                else:
                    embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # NumPy has no native bfloat16; models (e.g. Jina v4) often output bf16.
            all_embeddings.append(embeddings.detach().float().cpu().numpy())

        return np.vstack(all_embeddings)


class GeminiEmbeddingWrapper(BaseEmbeddingWrapper):
    """
    Wrapper for Google Gemini embedding models (google-genai SDK).

    Features
    --------
    - Incremental on-disk caching: embeddings are saved as .npy files after
      every batch so a run can be resumed after interruption.
    - Per-minute rate-limit handling: parses retryDelay from the API error
      and sleeps accordingly.
    - Daily quota handling: if the free-tier daily quota is exhausted the
      wrapper sleeps until 5 min past the next UTC midnight.
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-2-preview",
        api_key: str = "",
        batch_size: int = 100,
        cache_dir: str = "scripts/cache",
        cache_namespace: str = "",
        max_retries: int = 5,
        retry_delay: int = 60,
        **kwargs,
    ):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai not installed. "
                "Install with: pip install google-genai"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._genai_types = genai_types

        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=api_key)

        safe_name = model_name.replace("/", "_").replace("-", "_")
        # Keep cache separated per dataset/release to avoid accidental cross-dataset reuse.
        raw_namespace = cache_namespace or kwargs.pop("dataset_name", "") or "default_dataset"
        safe_namespace = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_namespace)).strip("._-")
        safe_namespace = safe_namespace or "default_dataset"

        self._emb_cache_dir = Path(cache_dir) / "gemini_embeddings" / safe_namespace / safe_name
        self._emb_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"GeminiEmbeddingWrapper: model={model_name}, "
            f"namespace={safe_namespace}, cache={self._emb_cache_dir}"
        )

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_path(self, cache_key: str, texts: List[str]) -> Path:
        """Unique .npy path derived from cache_key + texts fingerprint."""
        fingerprint = hashlib.md5(
            (str(texts[:3]) + str(len(texts))).encode()
        ).hexdigest()[:10]
        return self._emb_cache_dir / f"{cache_key}_{fingerprint}.npy"

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def _encode_texts(
        self,
        texts: List[str],
        task_type: str,
        cache_key: str,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Call Gemini with one ``UserContent`` per string."""
        bs = batch_size if batch_size is not None else self.batch_size
        emb_file = self._cache_path(cache_key, texts)

        # Resume from partial / load complete cache
        completed: List[np.ndarray] = []
        if emb_file.exists():
            cached = np.load(str(emb_file))
            if len(cached) == len(texts):
                logger.info(f"Loaded {len(cached)} embeddings from cache: {emb_file}")
                return cached
            completed = list(cached)
            logger.info(f"Resuming from {len(completed)}/{len(texts)} cached embeddings")

        start = len(completed)
        minute_attempts = 0

        for i in range(start, len(texts), bs):
            batch = texts[i: i + bs]
            attempt = 0
            while True:
                try:
                    contents_for_api = [
                        self._genai_types.UserContent(
                            parts=[
                                self._genai_types.Part.from_text(
                                    text=t if (t is not None and str(t).strip()) else " "
                                )
                            ]
                        )
                        for t in batch
                    ]
                    resp = self._client.models.embed_content(
                        model=self.model_name,
                        contents=contents_for_api,
                        config=self._genai_types.EmbedContentConfig(task_type=task_type),
                    )
                    batch_embs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
                    if batch_embs.shape[0] != len(batch):
                        raise RuntimeError(
                            f"Gemini returned {batch_embs.shape[0]} embeddings for {len(batch)} "
                            "inputs (expected one per string)."
                        )
                    completed.extend(batch_embs)
                    minute_attempts = 0

                    # Save progress after every batch
                    np.save(str(emb_file), np.array(completed, dtype=np.float32))
                    logger.info(f"  [{len(completed)}/{len(texts)}] batch encoded & saved")
                    break

                except Exception as e:
                    err = str(e).lower()

                    # Daily quota exhausted — sleep until 5 min past next UTC midnight
                    if "perday" in err or "per_day" in err or "daily" in err:
                        now_utc = datetime.now(timezone.utc)
                        next_midnight = now_utc.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                        from datetime import timedelta
                        next_midnight += timedelta(days=1)
                        wait = (next_midnight - now_utc).total_seconds() + 300
                        wait = max(wait, 300)
                        logger.warning(
                            f"Daily quota exhausted. Sleeping {wait/3600:.1f} h "
                            f"until {next_midnight.isoformat()}+5min."
                        )
                        time.sleep(wait)
                        minute_attempts = 0
                        continue

                    # Per-minute rate limit
                    if "429" in err or "rate" in err or "resource_exhausted" in err:
                        if minute_attempts >= self.max_retries:
                            raise
                        # Try to parse retryDelay from the error message
                        import re
                        m = re.search(r"retrydelay[^0-9]*([0-9]+)", err)
                        wait = int(m.group(1)) + 2 if m else self.retry_delay
                        logger.warning(
                            f"Rate limit hit (attempt {minute_attempts+1}). "
                            f"Sleeping {wait}s ..."
                        )
                        time.sleep(wait)
                        minute_attempts += 1
                        continue

                    # Other API errors
                    attempt += 1
                    if attempt >= self.max_retries:
                        raise
                    logger.warning(f"API error (attempt {attempt}): {e}. Retrying in {self.retry_delay}s ...")
                    time.sleep(self.retry_delay)

        return np.array(completed, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public interface (CoIR-compatible)
    # ------------------------------------------------------------------

    def encode_queries(self, queries: List[str], batch_size: int = 100, **kwargs) -> np.ndarray:
        logger.info(f"Encoding {len(queries)} queries with Gemini ({self.model_name})")
        return self._encode_texts(
            queries, task_type="RETRIEVAL_QUERY", cache_key="queries", batch_size=batch_size
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 100, **kwargs) -> np.ndarray:
        texts = [doc["text"] for doc in corpus]
        logger.info(f"Encoding {len(texts)} corpus docs with Gemini ({self.model_name})")
        return self._encode_texts(
            texts, task_type="RETRIEVAL_DOCUMENT", cache_key="corpus", batch_size=batch_size
        )


def create_model_wrapper(model_name: str, model_type: str = "huggingface", **kwargs) -> BaseEmbeddingWrapper:
    """
    Factory function to create model wrappers.

    Args:
        model_name:  Name / path of the model.
        model_type:  ``"gemini"`` for Google Gemini API, ``"huggingface"`` (default)
                     for any HuggingFace / Sentence-Transformers model.
        **kwargs:    Passed through to the wrapper constructor.

    Returns:
        BaseEmbeddingWrapper instance.
    """
    if model_type == "gemini" or "gemini" in model_name.lower():
        return GeminiEmbeddingWrapper(model_name=model_name, **kwargs)
    return HuggingFaceWrapper(model_name, **kwargs)
