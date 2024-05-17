import os
import logging
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

RANKING_MODEL = os.environ.get("RANKING_MODEL", "BAAI/bge-reranker-v2-m3")

@lru_cache
def get_ranking_model():
    """Create the ranking model."""
    logger.info(f"Using {RANKING_MODEL} as model engine for ranking")
    tokenizer_rerank = AutoTokenizer.from_pretrained(RANKING_MODEL)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RANKING_MODEL)
    if torch.cuda.is_available():
        rerank_model = rerank_model.to('cuda')
    rerank_model.eval()
    return rerank_model, tokenizer_rerank
