from pathlib import Path

import gradio as gr
import joblib

MODEL_PATH = Path("models/genre_model.joblib")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model belum ada. Jalankan train_model.py dulu untuk membuat models/genre_model.joblib"
    )

model = joblib.load(MODEL_PATH)


def predict_genre(title: str, description: str) -> str:
    title = (title or "").strip()
    description = (description or "").strip()

    if not title and not description:
        return "Mohon isi judul atau deskripsi film."

    text = f"{title} {description}".strip()
    pred = model.predict([text])[0]
    return str(pred)


app = gr.Interface(
    fn=predict_genre,
    inputs=[
        gr.Textbox(label="Judul Film", placeholder="Contoh: The Dark Knight"),
        gr.Textbox(
            label="Deskripsi Film",
            lines=5,
            placeholder="Masukkan sinopsis singkat film...",
        ),
    ],
    outputs=gr.Textbox(label="Prediksi Genre"),
    title="IMDb Genre Classification",
    description="Prediksi genre film berdasarkan judul dan deskripsi.",
)


if __name__ == "__main__":
    app.launch()
