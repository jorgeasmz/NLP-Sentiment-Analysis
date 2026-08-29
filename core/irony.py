"""Runs the fine-tuned irony head through ONNX Runtime."""

import numpy as np
import onnxruntime
from transformers import AutoTokenizer

IRONIC = 1


class IronyClassifier:
    """
    Scores text for irony from an exported graph.

    The session is built once and reused: ONNX Runtime allocates its arenas at
    construction, so creating one per call would pay that cost on every request.
    """

    def __init__(self, onnx_path: str, tokenizer_source: str, threshold: float, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.threshold = threshold
        self.max_length = max_length

        options = onnxruntime.SessionOptions()
        # One thread per physical core is the default; free-tier containers get a
        # fraction of a core, where extra threads only add contention.
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            onnx_path, options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {tensor.name for tensor in self.session.get_inputs()}

    def predict(self, texts: list[str]) -> list[dict]:
        """Returns one dict per input with the ironic probability and the decision."""
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="np",
        )
        feeds = {
            name: value.astype(np.int64)
            for name, value in encoded.items()
            if name in self.input_names
        }
        logits = self.session.run(None, feeds)[0]
        probabilities = _softmax(logits)[:, IRONIC]

        return [
            {"ironic": bool(probability >= self.threshold), "score": float(probability)}
            for probability in probabilities
        ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)
