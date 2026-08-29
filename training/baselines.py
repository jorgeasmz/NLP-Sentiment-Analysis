"""
Establishes what the fine-tuned head has to beat, and records each baseline in MLflow.

Three references, in increasing order of what they concede to the task: the
majority class, a linear model over character and word n-grams, and the
sentiment checkpoint the service already runs. The third answers a question the
first two cannot, namely whether the deployed model carries any signal about
irony at all.

Usage: python -m training.baselines
"""

import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

from core.config import MAX_LENGTH as SENTIMENT_MAX_LENGTH
from core.model_loader import get_model
from training import config
from training.data import load_splits
from training.metrics import classification_metrics

SENTIMENT_BATCH = 32


def majority_class(train_labels, test_labels) -> dict:
    """Predicts the training majority for everything. The floor any model must clear."""
    majority = int(np.bincount(train_labels).argmax())
    return classification_metrics(test_labels, np.full(len(test_labels), majority))


def tfidf_logistic(train_texts, train_labels, test_texts, test_labels) -> dict:
    """
    Word and character n-grams into logistic regression.

    Character n-grams are included because irony markers on this corpus are often
    sub-lexical: elongations, punctuation runs and casing survive them while a
    word-level vocabulary drops them.
    """
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                        (
                            "char",
                            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    model.fit(train_texts, train_labels)
    return classification_metrics(test_labels, model.predict(test_texts))


def sentiment_signal(texts: list[str]) -> np.ndarray:
    """
    Turns the served sentiment model into one feature: confidence signed by polarity.

    A value near +1 means the checkpoint is certain the text is positive, near -1
    certain it is negative. If irony were legible to it, ironic tweets would sit
    somewhere distinctive on that axis.
    """
    pipeline = get_model()
    outputs = pipeline(
        texts, truncation=True, max_length=SENTIMENT_MAX_LENGTH, batch_size=SENTIMENT_BATCH
    )
    signed = [
        output["score"] if output["label"] == "POSITIVE" else -output["score"]
        for output in outputs
    ]
    return np.asarray(signed).reshape(-1, 1)


def sentiment_probe(train_texts, train_labels, test_texts, test_labels) -> dict:
    """
    Fits a classifier on the sentiment model's own output and scores it on irony.

    This is the most favourable reading available to the deployed checkpoint: it
    is not asked to label irony, only to expose a feature a supervised model can
    exploit. Whatever it scores here is an upper bound on the irony information
    its output carries.
    """
    probe = LogisticRegression(max_iter=1000)
    probe.fit(sentiment_signal(train_texts), train_labels)

    test_features = sentiment_signal(test_texts)
    metrics = classification_metrics(test_labels, probe.predict(test_features))
    metrics["roc_auc"] = float(roc_auc_score(test_labels, probe.predict_proba(test_features)[:, 1]))
    return metrics


def main() -> None:
    splits = load_splits()
    train_texts = list(splits["train"]["text"])
    train_labels = np.asarray(splits["train"]["label"])
    test_texts = list(splits["test"]["text"])
    test_labels = np.asarray(splits["test"]["label"])

    print(f"train {len(train_texts)} · test {len(test_texts)}")

    results = {
        "majority-class": majority_class(train_labels, test_labels),
        "tfidf-logistic": tfidf_logistic(train_texts, train_labels, test_texts, test_labels),
        "sentiment-probe": sentiment_probe(train_texts, train_labels, test_texts, test_labels),
    }

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    for name, metrics in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("family", name)
            mlflow.log_param("dataset", f"{config.DATASET_ID}:{config.DATASET_CONFIG}")
            mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})

    header = f"{'Baseline':<18}{'Accuracy':>10}{'F1 macro':>10}{'F1 ironic':>11}"
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<18}{metrics['accuracy']:>10.3f}{metrics['f1_macro']:>10.3f}"
            f"{metrics['f1_ironic']:>11.3f}"
        )
    print(f"\nSentiment probe ROC-AUC: {results['sentiment-probe']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
