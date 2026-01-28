# ExpGuard: LLM Content Moderation in Specialized Domains

This repository is the official implementation for our ICLR 2026 paper: [ExpGuard: LLM Content Moderation in Specialized Domains](https://openreview.net/forum?id=t5cYJlV6aJ).

<img width="3153" height="1588" alt="expguard_overview" src="https://github.com/user-attachments/assets/ab94544a-4314-4d3d-97d2-c59b8a1f8dff" />

## 🔧 Requirements

To install requirements:
```bash
conda create -n expguard python=3.12.9
conda activate expguard
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

## ⌛ Hardware Requirements

### Training
- **Recommended Setup (Used in Paper)**: 4 × NVIDIA H200 144GB GPUs
- **Minimum Requirements (tested)**: 2 × NVIDIA H200 144GB GPUs
- The training process requires significant GPU memory due to the large model size and long sequence length (4096).

### Evaluation
- A single NVIDIA A6000 or A100 GPU is sufficient for evaluation
- Evaluation can be run on smaller GPUs with appropriate batch size adjustments

## 🤗 Data

Get our ExpGuardMix dataset from [Hugging Face Hub](https://huggingface.co/datasets/6rightjade/expguardmix).

WildChat, LMSYS-Chat-1M, and Suicide Detection subsets in Aegis 2.0 have been removed from the distributed version of the dataset. Users can access the subsets used in the paper through the [Google Drive link](https://drive.google.com/drive/folders/1VPJoQpwe3RZuQoFWg5a_5_9LhqGesRVE?usp=drive_link). Put the subsets in the `data/expguardmix` folder.

## 📁 Project Structure

```
ExpGuard/
├── data/
│   └── expguardmix/
│       ├── lmsys.jsonl
│       ├── suicide.jsonl
│       └── wildchat.jsonl
├── scripts/
│   ├── train.sh
│   └── safety_eval.sh
├── safety-eval/
│   └── ... (other evaluation scripts)
├── train.py
├── ... (other training scripts)
└── README.md
```

## 🔥 Training

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

## 📏 Evaluation

To evaluate model safety:

1. Configure evaluation parameters in `scripts/safety_eval.sh`.
2. Set the model path (`model_path`) to the local path where you have downloaded ExpGuard's model weights. If you are evaluating a model available on HuggingFace Hub, you can leave this empty, and the script will download it automatically.
3. Set `task_type` to `prompt` or `response` for prompt and response evaluation, respectively.
4. Set `model_name` to the model you wish to evaluate. The default is our model `ExpGuard`. The baseline model names can be found in `safety-eval/src/classifier_models/loader.py` (e.g., `OpenAIModeration`, `PerspectiveAPI`, `LlamaGuard3`, `WildGuard`, etc.). You need API keys to run API-based guardrails. These should be set as environment variables (`OPENAI_API_KEY`, `PERSPECTIVE_API_KEY`, `AZURE_CONTENT_SAFETY_KEY`, `AZURE_CONTENT_SAFETY_ENDPOINT`) in `scripts/safety_eval.sh`. All API tools are free, except Azure, which is commercial but offers some free credit for new accounts.
5. Run the evaluation script:
```
bash scripts/safety_eval.sh
```
6. The results will be saved in the `./classfication_results` directory.

## 🔢 Results

The following are the reported scores of ExpGuard:

| Model | Public Prompt Harm Avg. F1 | Public Response Harm Avg. F1 | ExpGuardTest Prompt Harm Total F1 | ExpGuardTest Response Harm Total F1 |
|---|---|---|---|---|
| ExpGuard | 85.7 | 78.5 | 93.3 | 92.7 |

To reproduce the scores in the paper, we use **early stopping** at the end of epoch 2. Please refer to the Appendix for more hyperparameter details.

**Note:** The reported values may not be perfectly reproducible and might vary slightly due to factors such as the vLLM inference library, CUDA versions, specific GPU devices used, and other stochastic elements in the evaluation process.

## Acknowledgements

The evaluation code is adapted from [safety-eval](https://github.com/allenai/safety-eval). Props to the Ai2 team 👍

## Citation

```bibtex
@inproceedings{
  choi2026expguard,
  title={ExpGuard: {LLM} Content Moderation in Specialized Domains},
  author={Choi, Minseok and Kim, Dongjin and Yang, Seungbin and Kim, Subin and Kwak, Youngjun and Oh, Juyoung and Choo, Jaegul and Son, Jungmin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=t5cYJlV6aJ}
}
```
