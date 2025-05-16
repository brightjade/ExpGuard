import time

import fire

from evaluation.schemas import ClassificationTaskBase
from src.classifier_models.loader import load_classifier_model
from evaluation.tasks import load_evaluation_tasks, EvalMode
from evaluation.utils import maybe_load_evaluation_report, save_evaluation_report


def classifiers(model_name: str,
                tasks: str,
                report_output_path: str,
                save_individual_results_path: str | None = None,
                override_model_path: str | None = None,
                override_existing_report: bool = False,
                ):
    """
    Evaluate a model using command line arguments.
    Evaluation report will be saved into the json format, with the keys
    as the task names and the values as the dictionary of evaluation results per model.

    Example usage:
        python eval.py classifiers \
            --model_name ./finetuned_olmo_7b \
            --tasks ai2_prompt_safety,ai2_prompt_safety:vanilla \
            --report_output_path ./evaluation_report_olmo_classification.json

    Args:
        model_name: Name of the model to evaluate.
            It will be loading a safety classifier from oe-safety (if it is a classification task).

        tasks: List of tasks to evaluate the model on, split by comma.
            each task can be xxx or xxx:yyy where xxx is the task name and yyy is the task config file name.
            if yyy is not provided, it will be assumed as xxx/default.yaml.
            Then the yaml file for the task config will be loaded from tasks/classification/xxx/yyy.yaml.

        report_output_path: Path to save the evaluation report.

        save_individual_results_path: Path to save the individual results.

        override_model_path: Path to the model to override the model path.
            this is to replace the model path easily without implementing the new model in oe-safety.

        override_existing_report: Whether to override the existing report or not. If true, it will override the existing report.
    """
    start_time = time.time()
    model = load_classifier_model(model_name, local_model_path=override_model_path)
    if isinstance(tasks, tuple):
        tasks = list(tasks)
    elif isinstance(tasks, str):
        tasks = tasks.split(",")
    evaluation_tasks: list[ClassificationTaskBase] = load_evaluation_tasks(EvalMode.CLASSIFICATION, tasks)
    evaluation_report = maybe_load_evaluation_report(report_output_path)
    all_individual_results = maybe_load_evaluation_report(save_individual_results_path)

    for task_name, eval_task in zip(tasks, evaluation_tasks):
        if task_name in evaluation_report and not override_existing_report:
            print(f"Skipping {task_name} as it is already evaluated.")
            continue

        try:
            eval_task.validate_if_model_is_compatible(model)
            report, individual_results = eval_task.evaluate(model)
            evaluation_report[task_name] = report
            all_individual_results[task_name] = individual_results
        except AssertionError as e:
            print(f"Skipping {task_name} due to the error: {e}")

        save_evaluation_report(report_output_path, evaluation_report)
        if save_individual_results_path:
            save_evaluation_report(save_individual_results_path, all_individual_results)
        print(f" *** {task_name} is evaluated. Evaluation report is saved to {report_output_path}. "
              f"Spent {time.time() - start_time:.2f} seconds.")
    print(f" *** All tasks are DONE! Evaluation report is saved to {report_output_path}. "
          f"Spent {time.time() - start_time:.2f} seconds.")


ALL_FUNCTIONS = {
    "classifiers": classifiers,
}

if __name__ == "__main__":
    fire.Fire(ALL_FUNCTIONS)
