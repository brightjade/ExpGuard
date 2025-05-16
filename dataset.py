import os.path as osp
import json
import torch
from datasets import Dataset


def load_data(args, tokenizer):
    with open(osp.join(args.data_dir, "train.jsonl"), "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(osp.join(args.data_dir, "val.jsonl"), "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    # concat train and eval after hyperparameter tuning to maximize performance
    data = train_data + eval_data
    # convert all none to empty string to convert to huggingface dataset
    data = [{k: v if v is not None else "" for k, v in d.items()} for d in data]
    dataset = Dataset.from_list(data)
    
    formatting_func = CustomFormatter(tokenizer, model_type=args.model_type)
    dataset = dataset.map(formatting_func, batched=True)

    train_dataset = CustomDataset(args, tokenizer, dataset)
    return train_dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, data):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        inputs = self.tokenizer(data["text"], max_length=self.max_seq_len, truncation=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        # add padding if needed
        if inputs["input_ids"].shape[1] < self.max_seq_len:
            padding_length = self.max_seq_len - inputs["input_ids"].shape[1]
            padding = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=torch.long)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], padding], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
            labels = torch.cat([labels, torch.full((1, padding_length), -100, dtype=torch.long)], dim=1)

        # change label to -100 for question tokens
        num_question_tokens = len(self.tokenizer.tokenize(data["eval_text"]))
        labels[0, :num_question_tokens] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class CustomFormatter:
    def __init__(self, tokenizer, model_type):
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __call__(self, batch):
        prompts, responses = batch["prompt"], batch["response"]
        prompt_labels, response_labels = batch["prompt_label"], batch["response_label"]
        formatted_inputs = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}] if response else \
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt, response in zip(prompts, responses)
        ]
        formatted_inputs_train = [
            formatted_input + f"User Safety Assessment: {prompt_label}\nAgent Safety Assessment: {response_label}" + self.tokenizer.eos_token if response_label else \
            formatted_input + f"User Safety Assessment: {prompt_label}" + self.tokenizer.eos_token
            for formatted_input, prompt_label, response_label in zip(formatted_inputs, prompt_labels, response_labels)
        ]
        return {"eval_text": formatted_inputs, "text": formatted_inputs_train}


def custom_train_collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_masks = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]
    return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_masks), "labels": torch.stack(labels)}
