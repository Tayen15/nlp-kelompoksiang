# NLP Kelompok Siang

Proyek ini adalah tugas klasifikasi genre film berbasis NLP menggunakan dataset IMDb dari Kaggle (hijest/genre-classification-dataset-imdb).
Link dataset: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## Ringkasan Pengerjaan
- Data collection: menggunakan dataset Kaggle dengan jumlah data >10.000.
- Text preprocessing: case folding, tokenization, stopword removal, dan lemmatization.
- Feature extraction: TF-IDF dan Word2Vec embedding.
- Modeling: Naive Bayes, Logistic Regression, dan SVM.
- Evaluasi: Accuracy, Precision, Recall, dan F1-score.
- Deployment sederhana: antarmuka prediksi menggunakan Gradio.

## Isi Repository
- Notebook utama: `main.ipynb`
- Script training model: `train_model.py`
- File model hasil training: `models/genre_model.joblib`
- Script deployment app Gradio: `app.py`
- Daftar dependency: `requirements.txt`

## Cara Menjalankan
1. Install dependency:
	`pip install -r requirements.txt`
2. Train dan simpan model:
	`python train_model.py`
3. Jalankan app deployment lokal (Gradio):
	`python app.py`
