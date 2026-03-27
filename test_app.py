"""Unit tests for app.py logic functions."""

import sys
import types
from unittest.mock import MagicMock, patch

# Stub out streamlit before importing app so st.set_page_config doesn't run
_st_stub = types.ModuleType("streamlit")
_st_stub.set_page_config = lambda **kw: None
_st_stub.cache_resource = lambda **kw: (lambda f: f)
_st_stub.cache_data = lambda **kw: (lambda f: f)
sys.modules.setdefault("streamlit", _st_stub)

import numpy as np
import pytest
import torch

from app import preprocess_text_for_model, classify_utterance, CLASSIFICATION_THRESHOLDS


class TestPreprocessTextForModel:
    def test_bge_query_adds_prefix(self):
        result = preprocess_text_for_model("hello world", "BAAI/bge-large-en-v1.5", "query")
        assert result == "Represent this sentence for retrieval: hello world"

    def test_bge_passage_no_prefix(self):
        result = preprocess_text_for_model("hello world", "BAAI/bge-large-en-v1.5", "passage")
        assert result == "hello world"

    def test_e5_query_adds_query_prefix(self):
        result = preprocess_text_for_model("check balance", "intfloat/e5-base-v2", "query")
        assert result == "query: check balance"

    def test_e5_passage_adds_passage_prefix(self):
        result = preprocess_text_for_model("check balance", "intfloat/e5-base-v2", "passage")
        assert result == "passage: check balance"

    def test_other_model_returns_text_unchanged(self):
        text = "some utterance"
        result = preprocess_text_for_model(text, "sentence-transformers/all-MiniLM-L6-v2", "query")
        assert result == text


class TestClassifyUtterance:
    def _make_mock_model(self, similarity_value: float):
        """Return a mock SentenceTransformer whose encode returns a unit vector."""
        mock_model = MagicMock()
        # Return a tensor that when cosine-sim'd with the intent embeddings gives similarity_value
        # We'll use identity vectors: query = e0, intents = [e0, e1, e2]
        # cos_sim(e0, e0) = 1.0, cos_sim(e0, e1) = 0.0, etc.
        # But we want a specific similarity value, so we control via a dot-product trick.
        vec = torch.tensor([similarity_value, (1 - similarity_value**2)**0.5, 0.0])
        mock_model.encode.return_value = vec
        return mock_model

    def _make_intent_embeddings(self):
        """Three orthogonal unit vectors as intent embeddings."""
        return np.eye(3, dtype=np.float32)

    def test_high_confidence_classification(self):
        intent_names = ["Intent_A", "Intent_B", "Intent_C"]
        intent_embeddings = self._make_intent_embeddings()
        # query vector aligns strongly with Intent_A (first row)
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.tensor([0.9, 0.1, 0.0])

        result = classify_utterance(
            "some text", mock_model, "other-model",
            intent_names, intent_embeddings
        )

        assert result["classification_level"] == "HIGH_CONFIDENCE"
        assert result["final_intent"] == "Intent_A"
        assert result["best_confidence"] >= CLASSIFICATION_THRESHOLDS["high_confidence"]

    def test_no_match_classification(self):
        intent_names = ["Intent_A", "Intent_B", "Intent_C"]
        # All embeddings point in a direction orthogonal to the query
        intent_embeddings = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5],
        ], dtype=np.float32)
        mock_model = MagicMock()
        # query is [1, 0, 0] — orthogonal to all intents → similarity ≈ 0
        mock_model.encode.return_value = torch.tensor([1.0, 0.0, 0.0])

        result = classify_utterance(
            "gibberish", mock_model, "other-model",
            intent_names, intent_embeddings
        )

        assert result["classification_level"] == "NO_MATCH"
        assert result["final_intent"] == "No_Intent_Match"

    def test_top_results_are_sorted_descending(self):
        intent_names = ["A", "B", "C", "D", "E"]
        # Each intent is a unit vector in a different dimension
        intent_embeddings = np.eye(5, dtype=np.float32)
        mock_model = MagicMock()
        # query has varying similarity to each intent
        mock_model.encode.return_value = torch.tensor([0.8, 0.5, 0.3, 0.1, 0.0])

        result = classify_utterance(
            "test", mock_model, "other-model",
            intent_names, intent_embeddings
        )

        confidences = [r["confidence"] for r in result["top_results"]]
        assert confidences == sorted(confidences, reverse=True)
        assert len(result["top_results"]) == 5