# Manchu OCR with Qwen3VL



Experimental code for **Manchu word recognition** using **Qwen3-VL** (2B / 4B, QLoRA) and **TrOCR**. Given a cropped Manchu word image, the model predicts the corresponding **romanization**.

This repository is notebook-centric and covers synthetic-data training, unseen-word generalization evaluation, and real historical-document testing with error analysis.

This work is inspired by https://arxiv.org/pdf/2507.06761, the dataset used in this paper [mic7ch/manchu_sub1](https://huggingface.co/datasets/mic7ch/manchu_sub1) which is a part from https://github.com/tyotakuki/ManchuOCR helps me a lot. 

---

## Resources

| Resource | Link |
|----------|------|
| Dataset | [Um1neko/Manchu_OCR](https://huggingface.co/datasets/Um1neko/Manchu_OCR) (It's a adjusted version of [mic7ch/manchu_sub1](https://huggingface.co/datasets/mic7ch/manchu_sub1))|
| Qwen3-VL-4B LoRA | [Um1neko/Qwen3VL-Instruct-4B-ManchuOCR](https://huggingface.co/Um1neko/Qwen3VL-Instruct-4B-ManchuOCR) |
| Qwen3-VL-2B LoRA | [Um1neko/Qwen3VL-Instruct-2B-ManchuOCR](https://huggingface.co/Um1neko/Qwen3VL-Instruct-2B-ManchuOCR) |
| TrOCR | [Um1neko/TrOCR_Manchu](https://huggingface.co/Um1neko/TrOCR_Manchu) |

> **Dataset note:** After downloading, **unzip manually** and do **not** change the folder structure.

---

## Task & Data

### Task

- **Input:** Manchu word images (synthetic or real document crops)
- **Output:** Romanization string (`roman` field in the parquet files)
- **Prompt** (Qwen3-VL): `识别满文单词：`
- **Metrics:** CER (Character Error Rate), WER (Word Error Rate); some scripts also report exact-match counts / Word Accuracy

### Recommended layout

After unzipping, keep a structure similar to the following (update absolute paths in the notebooks for your machine):

```text
Manchu_OCR/
├── train.parquet          # synthetic training labels (filename, roman, ...)
├── train/                 # synthetic training images
├── test.parquet           # real-document test labels (~218 samples in the examples)
└── test/                  # real-document test images
```

### Split conventions

| Model | Split | Notes |
|-------|-------|-------|
| Qwen3-VL 2B / 4B | `train_test_split(..., test_size=0.05, random_state=42)` | ~57,000 train / 3,000 val |
| TrOCR | `test_size=0.1, random_state=42` | ~54,000 train / 6,000 val |

For synthetic validation, the scripts audit **roman-label overlap** between train and val. If the overlap rate exceeds a threshold (default 5%), samples with labels seen in training are filtered out, and evaluation runs on the **Unseen Words** subset for a closer estimate of true generalization.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `training and test on sythetic data_qwen3vl4b.ipynb` | **Qwen3-VL-4B** QLoRA training + synthetic unseen-word validation |
| `test_on_real_documents_qwen3vl4b.ipynb` | **Qwen3-VL-4B** evaluation & error analysis on real `test.parquet` |
| `train_and_test_synthetic_and_real_qwen3vl2b.ipynb` | **Qwen3-VL-2B** synthetic unseen eval, real-doc eval, and QLoRA training |
| `trocr_train.ipynb` | **TrOCR** two-phase training (custom Manchu tokenizer + differential LRs) |
| `trocr_test_on_synthetic_and_real.ipynb` | **TrOCR** synthetic unseen zero-shot sample eval + real-document test |

---

## Methods

### 1. Qwen3-VL + QLoRA

- Bases: `Qwen/Qwen3-VL-2B-Instruct` or `Qwen/Qwen3-VL-4B-Instruct`
- Quantization: NF4 4-bit (BitsAndBytes) with bf16 compute
- LoRA: `r=64`, `lora_alpha=128`, `lora_dropout=0.05`, `target_modules="all-linear"`, with `embed_tokens` / `lm_head` saved
- Pixel budget: `min_pixels=32*32`, `max_pixels=384*384` (training)
- Example configs targeting a single **RTX 3090 24GB**:
  - 2B: `batch_size=192` (notes also mention batch=96), 9 epochs, lr=`2e-4`
  - 4B: `batch_size=64`, 9 epochs, lr=`2e-4`
- Training masks the prompt tokens (`ignore_index=-100`) so loss is computed only on the assistant romanization
- Inference: `do_sample=False`; synthetic eval often uses `max_new_tokens=30`, real-doc eval often uses `64`

### 2. TrOCR two-phase fine-tuning

Built on `microsoft/trocr-base-printed`:

1. Train a **custom Manchu BPE tokenizer** (vocab ≈ 8,000) and smart-initialize embeddings (exact match / substring average / random)
2. **Phase 1:** Adapt to the new vocabulary (high LR on embeddings; transformer nearly frozen)
3. **Phase 2:** Joint full-model fine-tuning (differential LRs for embedding / decoder / encoder)
4. `trocr_train.ipynb` also includes a **historical-document degradation visualization** (ink bleed/fade, aged paper tint, noise, blur) to illustrate the synthetic-vs-real domain gap

At inference, assemble the processor from Microsoft’s image processor + the Manchu tokenizer; generation uses `num_beams=4`.

---

## Results (from executed notebook outputs)

> Numbers are taken from notebook run outputs for easy reproduction checks. Results may vary slightly with environment, checkpoint paths, or sampling.

### Synthetic · Unseen-word validation

| Model | Setting | CER ↓ | WER / Acc |
|-------|---------|------:|----------|
| Qwen3-VL-4B | Unseen val (1,914) | **0.37%** | WER **2.61%** |
| Qwen3-VL-2B | Unseen val (1,914) | 1.02% | WER 6.58% |
| TrOCR Phase 2 | Full validation set | 7.51% | Word Acc 75.43% |
| TrOCR Phase 2 | Unseen sample of 500 | 7.68% | Word Acc 71.80% |

### Real historical documents (218 images)

| Model | CER ↓ | WER ↓ | Exact match |
|-------|------:|------:|------------:|
| **Qwen3-VL-4B** | **3.37%** | **13.30%** | **189 / 218** |
| Qwen3-VL-2B | 22.51% | 59.63% | 88 / 218 |
| TrOCR | 29.51% | 67.43% | Acc 32.57% |

**Takeaway:** On both synthetic unseen words and real documents, **Qwen3-VL-4B QLoRA** clearly outperforms the 2B variant and TrOCR. Domain shift to real documents remains the main challenge (larger drops for 2B / TrOCR).

---

## Environment

Use a recent PyTorch + CUDA build, then install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # adjust for your CUDA
pip install transformers peft accelerate bitsandbytes
pip install qwen-vl-utils datasets jiwer evaluate
pip install pandas pyarrow scikit-learn pillow tqdm
pip install tokenizers opencv-python matplotlib  # TrOCR / degradation viz
# Optional: FlashAttention 2 (notebooks use attn_implementation="flash_attention_2")
```

If Hugging Face is hard to reach (e.g., in mainland China), notebooks already demonstrate:

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

Hardware reference: a single **24GB** GPU (e.g., RTX 3090) can run the QLoRA setups above. TrOCR needs less peak VRAM but Phase-2 full fine-tuning takes longer.

---

## Quick start

### 1. Prepare the dataset

1. Download [Manchu_OCR](https://huggingface.co/datasets/Um1neko/Manchu_OCR) from Hugging Face
2. Unzip and keep the original structure
3. Point notebook paths to your local copies, e.g.:

```python
'data_parquet': "/path/to/Manchu_OCR/train.parquet",
'image_dir': "/path/to/Manchu_OCR/train",
'test_parquet': "/path/to/Manchu_OCR/test.parquet",
```

### 2. Evaluate released checkpoints

- **Qwen3-VL-4B on real docs:** open `test_on_real_documents_qwen3vl4b.ipynb`  
  - Set `lora_path` to `Um1neko/Qwen3VL-Instruct-4B-ManchuOCR` or a local `final_lora`
- **Qwen3-VL-2B synthetic + real:** first two cells in `train_and_test_synthetic_and_real_qwen3vl2b.ipynb`  
  - Default LoRA: `Um1neko/Qwen3VL-Instruct-2B-ManchuOCR`
- **TrOCR:** `trocr_test_on_synthetic_and_real.ipynb`  
  - Real-test default: `Um1neko/TrOCR_Manchu`

Scripts print CER / WER. Qwen real-document notebooks also write error-analysis CSVs (e.g. `error_analysis.csv` / `error_analysis_2b.csv`).

### 3. Train yourself

1. **Qwen3-VL-4B:** training cell in `training and test on sythetic data_qwen3vl4b.ipynb`
2. **Qwen3-VL-2B:** last cell in `train_and_test_synthetic_and_real_qwen3vl2b.ipynb`
3. **TrOCR:** `trocr_train.ipynb` (tokenizer / Phase 1, then Phase 2)

After training, point `lora_path` / `MODEL_PATH` at your local output directories and reuse the eval notebooks.

---

## Notes

1. Notebooks hard-code AutoDL-style absolute paths (`/root/autodl-fs/...`); update them for your machine.
2. When loading Qwen LoRA for eval, the processor is usually loaded from the **LoRA directory** (includes the pixel constraints used in training).
3. Loading custom TrOCR weights requires assembling `TrOCRProcessor` from `microsoft/trocr-base-printed`’s image processor + the Manchu `PreTrainedTokenizerFast` (already done in the eval notebook).
4. FlashAttention 2, large `batch_size`, and high `dataloader_num_workers` should be tuned to your VRAM/CPU; if you OOM, reduce `batch_size` or `max_pixels` first.
5. Unseen-word filtering for synthetic validation depends on the **same** `random_state` and `test_size` as training—do not change them casually, or the split will no longer match.
