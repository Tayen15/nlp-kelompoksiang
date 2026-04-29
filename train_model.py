from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def parse_train_data(file_path: Path) -> pd.DataFrame:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ::: ", maxsplit=3)
            if len(parts) != 4:
                continue
            sample_id, title, genre, description = parts
            rows.append(
                {
                    "id": sample_id,
                    "title": title,
                    "genre": genre,
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=60000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1200)),
        ]
    )


def main() -> None:
    data_file = Path("genre-classification-dataset-imdb/Genre Classification Dataset/train_data.txt")
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_file}")

    df = parse_train_data(data_file)
    if df.empty:
        raise ValueError("Dataset kosong atau gagal diparse.")

    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna(""))

    x_train, x_val, y_train, y_val = train_test_split(
        df["text"],
        df["genre"],
        test_size=0.2,
        random_state=42,
        stratify=df["genre"],
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "genre_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
