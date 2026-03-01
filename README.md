# Petals to the Metal - Flower Classification on TPU

> Classify 104 types of flowers from images using deep learning on TPUs.

## About

| | |
|---|---|
| **Type** | Computer Vision |
| **Metric** | Accuracy |
| **Kaggle** | [Petals to the Metal - Flower Classification on TPU](https://www.kaggle.com/competitions/tpu-getting-started) |
| **Live Demo** | [HuggingFace Space](https://huggingface.co/spaces/yusufbodur/KG_Petals_To_The_Metal_CV) |

## Project Files

| File | Purpose |
|------|---------|
| `notebook.ipynb` | End-to-end ML pipeline |
| `streamlit_app.py` | Interactive demo app |
| `data/` | Dataset files *(not tracked by Git)* |
| `models/` | Saved model artifacts |
| `outputs/` | Submission CSV |

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add data** — Download from the [Kaggle competition page](https://www.kaggle.com/competitions/tpu-getting-started) and place files in `data/`.

**3. Run the notebook**
```bash
jupyter notebook notebook.ipynb
```

**4. Launch the demo app**
```bash
streamlit run streamlit_app.py
```
