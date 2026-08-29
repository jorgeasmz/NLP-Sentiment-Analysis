"""
Fine-tunes DistilBERT to detect irony and records the run in MLflow.

The service already documents that a sentence-level sentiment classifier inverts
its answer on ironic text while keeping maximum confidence. Training a second
head is the response to that finding: the system stops guessing harder and
starts recognising the input it cannot be trusted on.

Usage: python -m training.train [--freeze-layers 4] [--run-name frozen-4]
"""

import argparse
import json
import math

import mlflow
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training import config
from training.data import build_collator, build_tokenizer, encode, load_splits
from training.metrics import (
    classification_metrics,
    confusion,
    ironic_probability,
    trainer_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=0,
        help="Freeze the embeddings and the first N transformer blocks.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Names the MLflow run and the output directory. Defaults to the freezing depth.",
    )
    return parser.parse_args()


def freeze(model, layers: int) -> int:
    """
    Freezes the embeddings and the lowest transformer blocks, returning the
    number of parameters left trainable.

    The lower blocks encode lexical and syntactic regularities that transfer
    unchanged; irony is decided higher up, so holding them fixed cuts the
    backward pass without touching what the task depends on.
    """
    if layers <= 0:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    for parameter in model.distilbert.embeddings.parameters():
        parameter.requires_grad = False
    for block in model.distilbert.transformer.layer[:layers]:
        for parameter in block.parameters():
            parameter.requires_grad = False

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    args = parse_args()
    default_name = "full" if args.freeze_layers == 0 else f"frozen-{args.freeze_layers}"
    run_name = args.run_name or default_name
    output_dir = config.ARTIFACTS / f"irony-{run_name}"
    set_seed(args.seed)

    tokenizer = build_tokenizer()
    splits = encode(load_splits(), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_CHECKPOINT,
        num_labels=len(config.LABELS),
        id2label=dict(enumerate(config.LABELS)),
        label2id={label: index for index, label in enumerate(config.LABELS)},
    )
    trainable = freeze(model, args.freeze_layers)
    total = sum(p.numel() for p in model.parameters())

    # TrainingArguments takes an absolute warmup length, so the ratio is turned
    # into steps here rather than being restated as a magic number per run.
    steps_per_epoch = math.ceil(len(splits["train"]) / args.batch_size)
    warmup_steps = int(steps_per_epoch * args.epochs * config.WARMUP_RATIO)

    arguments = TrainingArguments(
        output_dir=str(config.ARTIFACTS / "checkpoints" / run_name),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=config.METRIC_FOR_BEST,
        greater_is_better=True,
        logging_steps=25,
        seed=args.seed,
        report_to=["mlflow"],
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=build_collator(tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "base_checkpoint": config.BASE_CHECKPOINT,
                "dataset": f"{config.DATASET_ID}:{config.DATASET_CONFIG}",
                "max_length": config.MAX_LENGTH,
                "frozen_layers": args.freeze_layers,
                "warmup_steps": warmup_steps,
                "trainable_parameters": trainable,
                "total_parameters": total,
                "test_size": len(splits["test"]),
            }
        )

        trainer.train()

        # The validation split chose the checkpoint, so what gets reported comes
        # from the split that had no part in that choice. Reported at argmax,
        # which is the rule the published TweetEval numbers use and therefore
        # the only one comparable to them and to the baselines; the operating
        # point is set later, on the artifact that actually gets served.
        predictions = trainer.predict(splits["test"])
        labels = predictions.label_ids
        predicted = (ironic_probability(predictions.predictions) >= 0.5).astype(int)
        test_metrics = classification_metrics(labels, predicted)

        mlflow.log_metrics({f"test_{name}": value for name, value in test_metrics.items()})
        mlflow.log_text(
            json.dumps(confusion(labels, predicted).tolist()), "test_confusion_matrix.json"
        )

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    print(f"\nTest set ({len(labels)} tweets, argmax)")
    for name, value in test_metrics.items():
        print(f"  {name:<18}{value:.4f}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
