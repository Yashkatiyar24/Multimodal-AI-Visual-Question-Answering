# Visual Question Answering (VQA) — Multimodal AI Demo

A small local demo that uses the Hugging Face `dandelin/vilt-b32-finetuned-vqa` model to answer natural-language questions about images.

This repository contains a Gradio web app (`app.py`) which loads the ViLT VQA model and exposes a simple UI for uploading images and asking questions.

## Contents
- `app.py` — Gradio Blocks app that runs the VQA model locally
- `requirements.txt` — Python dependencies to install

> Note: example images referenced by the UI (`examples/*.jpg`) are not included in this repository by default. You can add your own example images to an `examples/` folder if you'd like the example buttons to load images.

## Quick start (macOS, zsh)
1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the URL printed in the console (usually http://127.0.0.1:7860) in your browser.

## How it works
- The app uses `ViltProcessor` and `ViltForQuestionAnswering` from the `transformers` library.
- When you upload an image and ask a question, the app tokenizes/encodes inputs with the processor, runs the model (in evaluation mode), and returns the most likely answers along with a small top-k visualization.

## Notes & troubleshooting
- First run will download the model weights from Hugging Face — this can take a few minutes and requires an internet connection.
- If you want GPU acceleration, install a CUDA-compatible `torch` wheel (see https://pytorch.org/get-started/locally/) and ensure `torch.cuda.is_available()` returns `True`.
- Common issues:
  - Import errors for `transformers` or `torch`: try upgrading packages with `pip install -U transformers torch`.
  - Slow startup: model download on first run or CPU-based inference.

## Customization ideas
- Swap the model for another pretrained multimodal model from Hugging Face.
- Persist recent Q&A pairs in the UI or add a small database to log queries.
- Add Grad-CAM / attention visualization overlays on the image.

## License
This repository contains demo code. Replace or add a license file as needed for your project.

---
If you want, I can also:
- Add an example images folder with 3 sample images used by the UI.
- Create a `.gitignore` that excludes the virtualenv and other common files.
- Commit these files into your (currently-empty) GitHub repo and push them for you.

Which of those would you like next?
