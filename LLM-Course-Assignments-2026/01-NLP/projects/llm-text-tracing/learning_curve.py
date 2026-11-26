"""Refit the character detection and attribution heads on nested, balanced training samples."""
import argparse
import hashlib
from pathlib import Path

from experiment import id_hash, write_json
from tracing.data import balanced_sample, read_corpus
from tracing.metrics import evaluate
from tracing.models import Classifiers, text_vectors, text_windows


SEEDS = (20261119, 20261120, 20261121)
PER_FAMILY = (8, 16, 32, 64, 128)


def run(args):
    training, dev = read_corpus(args.data)
    dev_ids = sorted(dev)
    dev_text = text_windows([dev[i]["text"] for i in dev_ids])
    trials = []
    for seed in SEEDS:
        previous = set()
        for per_family in PER_FAMILY:
            subset = balanced_sample(training, per_family, seed=seed)
            ids = {r["id"] for r in subset}
            if not previous.issubset(ids):
                raise ValueError("同一种子的大样本必须包含小样本")
            previous = ids
            vectorizer, matrix = text_vectors([r["text"] for r in subset])
            heads = Classifiers.fit(matrix, subset)
            predictions = heads.predict(vectorizer.transform(dev_text), dev_ids)
            metrics = evaluate(dev, predictions)
            digest = hashlib.sha256("".join(f"{i},{predictions[i]['label']},{predictions[i]['family']}\n"
                                            for i in dev_ids).encode("ascii")).hexdigest()
            trials.append({"seed": seed, "size": len(subset), "train_ids_sha256": id_hash(subset),
                           "predictions_sha256": digest, "vocabulary_size": len(vectorizer.vocabulary_),
                           "metrics": metrics})
            print(seed, len(subset), metrics["joint_score"], flush=True)
    write_json(args.out, {"method": "char-tfidf", "train_pool": len(training), "validation_n": len(dev),
                          "validation_ids_sha256": id_hash([dev[i] for i in dev_ids]), "seeds": SEEDS,
                          "trials": trials})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.zip"))
    parser.add_argument("--out", type=Path, default=Path("results/learning_curve.json"))
    run(parser.parse_args())
