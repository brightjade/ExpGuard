import abc
from collections import defaultdict

from datasets import load_dataset

from evaluation.schemas import ClassificationTaskBase
from evaluation.tasks.metrics import get_average, get_f1
from src.classifier_models.base import SafetyClassifierBase, PromptHarmfulness


class PromptHarmfulnessEvalBase(ClassificationTaskBase, abc.ABC):
    @abc.abstractmethod
    def load(self) -> list[dict]:
        raise NotImplementedError

    def required_input_fields(self) -> list[str]:
        return ["prompt"]

    @property
    def label_field(self) -> str:
        return "prompt_harmfulness"

    def required_output_fields(self) -> list[list[str]]:
        return [["prompt_harmfulness"]]

    def _evaluate(self, model: SafetyClassifierBase) -> tuple[dict, list[dict]]:
        classification_results = model.classify(self.data)
        pred_is_correct = defaultdict(list)
        individual_results = []
        for gt, pred in zip(self.data, classification_results):
            pred_label = pred.prompt_harmfulness
            gt_label = gt[self.label_field]
            individual_results.append({
                "id": gt["id"],
                "prompt": gt["prompt"],
                "gt_prompt_harmfulness": gt_label.value,
                **pred.asdict(),
            })
            if pred.is_parsing_error:
                pred_is_correct["parsing_error"].append(1)
                pred_is_correct["micro_acc"].append(0)
                gt_label_is_harmful = gt_label == PromptHarmfulness.HARMFUL
                pred_is_correct["f1"].append({"pred": not gt_label_is_harmful,
                                              "gt": gt_label_is_harmful})
                pred_is_correct[gt_label.value].append(0)
                if "prompt_type" in gt:
                    pred_is_correct[f"{gt['prompt_type']}_f1"].append({"pred": not gt_label_is_harmful,
                                                                       "gt": gt_label_is_harmful})
                    pred_is_correct[gt["prompt_type"]].append(0)
            else:
                pred_is_correct["parsing_error"].append(0)
                pred_is_correct["micro_acc"].append(gt_label == pred_label)
                pred_is_correct["f1"].append({"pred": pred_label == PromptHarmfulness.HARMFUL,
                                              "gt": gt_label == PromptHarmfulness.HARMFUL})
                pred_is_correct[gt_label.value].append(gt_label == pred_label)
                if "prompt_type" in gt:
                    pred_is_correct[f"{gt['prompt_type']}_f1"].append(
                        {"pred": pred_label == PromptHarmfulness.HARMFUL,
                         "gt": gt_label == PromptHarmfulness.HARMFUL})
                    pred_is_correct[gt["prompt_type"]].append(gt_label == pred_label)
           

        return {k: get_average(v) if "f1" not in k else get_f1(v) for k, v in
                pred_is_correct.items()}, individual_results


class XSTestPromptHarmfulnessEval(PromptHarmfulnessEvalBase):
    def load(self) -> list[dict]:
        dataset = load_dataset("Paul/XSTest", split="train")


        return [{
            "prompt": row["prompt"],
            "response": "",
            "prompt_harmfulness": PromptHarmfulness.HARMFUL if row["label"] == "unsafe" else PromptHarmfulness.UNHARMFUL,
            "prompt_type": row["type"],
        } for row in dataset]
