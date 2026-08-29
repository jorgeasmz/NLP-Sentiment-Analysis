"""
Exports the fine-tuned irony head to ONNX and produces an int8 copy.

Usage: python -m training.export
"""

import shutil

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.export import Dim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training import config


class LogitsOnly(torch.nn.Module):
    """
    Narrows the model to the two tensors serving actually sends and the one it reads.

    Exporting the classifier directly captures its optional inputs and dataclass
    output, which the runtime would then have to satisfy on every call.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> None:
    if not config.TORCH_MODEL_DIR.exists():
        raise SystemExit(
            f"No fine-tuned model at {config.TORCH_MODEL_DIR}. Run training.train first."
        )

    config.ONNX_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(config.TORCH_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(config.TORCH_MODEL_DIR)).eval()

    sample = tokenizer(
        ["oh good, another meeting", "the delivery arrived on time"],
        padding=True,
        return_tensors="pt",
    )
    # Both axes stay symbolic and shared: batch size varies per request, the
    # collator pads to the batch rather than to a fixed width, and sharing the
    # objects records that the two inputs agree in shape.
    axes = {0: Dim("batch"), 1: Dim("sequence")}

    torch.onnx.export(
        LogitsOnly(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(config.ONNX_FP32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=(axes, axes),
        external_data=False,
    )

    # The exporter leaves shape metadata describing tensors as they were before
    # its own optimisation pass, which the quantiser's shape inference then
    # contradicts. Dropping the entries lets it recompute them from the graph.
    graph = onnx.load(str(config.ONNX_FP32))
    del graph.graph.value_info[:]
    onnx.save(graph, str(config.ONNX_FP32))

    # Dynamic quantisation only touches the weights; activation ranges are
    # computed per call, so no calibration set is needed and the accuracy cost
    # falls entirely on the matrix multiplies the benchmark then measures.
    quantize_dynamic(
        str(config.ONNX_FP32),
        str(config.ONNX_INT8),
        weight_type=QuantType.QInt8,
    )

    tokenizer.save_pretrained(str(config.ONNX_DIR))
    shutil.copy(config.TORCH_MODEL_DIR / "config.json", config.ONNX_DIR / "config.json")

    for path in (config.ONNX_FP32, config.ONNX_INT8):
        print(f"{path.name:<20}{path.stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
