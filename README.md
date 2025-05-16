# ExpGuard: LLM Content Moderation in Specialized Domains

This repository is the official implementation of *ExpGuard: LLM Content Moderation in Specialized Domains*.

**Important Notice for Reviewers:** During the reviewing period, both the code and dataset are kept private due to the sensitive nature of the content. Upon acceptance, we plan to release them with gated access to ensure responsible usage, given the potentially harmful nature of the training and evaluation data. This approach aligns with ethical AI development practices and helps prevent misuse of the content. (The code repository itself is not harmful and will be made public without gated access, but the model weights below need a careful consideration.)

## Requirements

To install requirements:
```bash
conda create -n expguard python=3.12
conda activate expguard
pip install -r requirements.txt
```

It is recommended to set up the environment where CUDA version is 12+ for enabling flash attention. Otherwise, exclude installing flash attention.

## Hardware Requirements

### Training
- **Recommended Setup (Used in Paper)**: 4 × NVIDIA H200 144GB GPUs
- **Minimum Requirements (tested)**: 2 × NVIDIA H200 144GB GPUs
- The training process requires significant GPU memory due to the large model size and long sequence length (4096).

### Evaluation
- A single NVIDIA A6000 or A100 GPU is sufficient for evaluation
- Evaluation can be run on smaller GPUs with appropriate batch size adjustments

## Data

Download our ExpGuardMix (`train.jsonl`, `val.jsonl`, `test.jsonl`) and put them in the `data/expguardmix` folder.
- [Kaggle Private Data Download Link](https://kaggle.com/datasets/99644af2eb4febcdb6f20f8b2bec5a16279eb205ce6565fe3f5c831bc9c47d49)

Then, the project structure will be as follows:
```
ExpGuard/
├── data/
│   └── expguardmix/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
├── scripts/
│   ├── train.sh
│   └── safety_eval.sh
├── safety-eval/
│   └── ... (other evaluation scripts)
├── train.py
├── ... (other training scripts)
└── README.md
```

## Training

To train the model in the paper:
1. Configure your training parameters in `scripts/train.sh`. Key parameters include:
    - `dataset_name`: Name of the dataset (default: `expguardmix`).
    - `model_type`: Type of the model (default: `qwen2.5-7b`).
    - `model_name_or_path`: Pretrained model name or path (default: `Qwen/Qwen2.5-7B`).
    - `per_device_train_batch_size`: Batch size per device (default: `4`).
    - `gradient_accumulation_steps`: Gradient accumulation steps (default: `2`).
    - `max_seq_len`: Maximum sequence length (default: `4096`).
    - `learning_rate`: Learning rate (default: `5e-6`).
    - `epochs`: Number of training epochs (default: `3`).
    - `attn_implementation`: Attention implementation (default: `flash_attention_2`). Set to `sdpa` or `eager` if flash attention is not available.
2. Run the training script:

```bash
bash scripts/train.sh
```

## Evaluation

To evaluate model safety:

1. Configure evaluation parameters in `scripts/safety_eval.sh`.
2. Set the model path (`model_path`) to the local path where you have downloaded the ExpGuard's model weights. If you are evaluating a model available on HuggingFace Hub, you can leave this empty and the script will download it automatically.
3. Set `task_type` to `prompt` or `response` for prompt and response evaluation, respectively.
4. Set `model_name` to the model you wish to evaluate. The default is our model `ExpGuard`. The baseline model names can be found in `safety-eval/src/classifier_models/loader.py` (e.g., `OpenAIModeration`, `PerspectiveAPI`, `LlamaGuard3`, `WildGuard`, etc.). You need API keys to run API-based guardrails. These should be set as environment variables (`OPENAI_API_KEY`, `PERSPECTIVE_API_KEY`, `AZURE_CONTENT_SAFETY_KEY`, `AZURE_CONTENT_SAFETY_ENDPOINT`) in `scripts/safety_eval.sh`. All API tools are free, except Azure which is commercial but offers some free credit for new accounts.
5. Run the evaluation script:
```
bash scripts/safety_eval.sh
```
6. The results will be saved in the `./classfication_results` directory.

## Model Weights

You can download ExpGuard's model weights here:
- [Google Drive Model Weights Download Link](https://drive.google.com/file/d/1oKI67clgR96s8YOPzonmTjNSjZsf1yXO/view?usp=sharing)

## Results

The following is the reported scores of ExpGuard:

| Model | Public Prompt Harm Avg. F1 | Public Response Harm Avg. F1 | ExpGuardTest Prompt Harm Total F1 | ExpGuardTest Response Harm Total F1 |
|---|---|---|---|---|
| ExpGuard | 85.7 | 78.5 | 93.3 | 92.7 |

To reproduce the scores in the paper, we use **early stopping** at the end of epoch 2. Please refer to Appendix for more hyperparameter details.

**Note:** The reported values may not be perfectly reproducible and might vary slightly due to factors such as the vLLM inference library, CUDA versions, specific GPU devices used, and other stochastic elements in the evaluation process.
