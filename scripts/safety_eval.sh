# !/bin/bash

export PYTHONPATH=$(pwd)/safety-eval
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=""                  # needed to run OpenAIModeration
export PERSPECTIVE_API_KEY=""             # needed to run PerspectiveAPI
export AZURE_CONTENT_SAFETY_KEY=""        # needed to run AzureContentSafety
export AZURE_CONTENT_SAFETY_ENDPOINT=""   # needed to run AzureContentSafety

current_time=$(date +"%Y-%m-%d_%H-%M-%S")

model_name="ExpGuard"
task_type="prompt" # 'prompt' or 'response'
model_path=""      # replace with the path to the model to evaluate

# Define task lists based on type
prompt_tasks=("expguardtest" "toxicchat" "openai_mod" "xstest_prompt_harm" "harmbench" "aegis_safety_dataset2" "wildguardtest_prompt")
response_tasks=("expguardtest" "beavertails" "saferlhf" "harmbench" "aegis_safety_dataset2" "wildguardtest_response")

# Select tasks based on task_type
if [ "$task_type" == "prompt" ]; then
  task_list=("${prompt_tasks[@]}")
elif [ "$task_type" == "response" ]; then
  task_list=("${response_tasks[@]}")
else
  echo "Invalid task_type: $task_type"
  exit 1
fi

# Format tasks as task_name:task_type
tasks=$(IFS=','; echo "${task_list[*]/%/:$task_type}")

# Run eval.py with selected configuration
echo "======================================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Running evaluation for model: $model_name"
echo "Data type: $task_type"
echo "Task: $tasks"
echo "Time: $current_time"
echo "======================================================"

python safety-eval/evaluation/eval.py classifiers \
  --model_name $model_name \
  --tasks $tasks \
  --report_output_path "./classification_results/${model_name}/metrics_${task_type}_${current_time}.json" \
  --save_individual_results_path "./classification_results/${model_name}/all_${task_type}_${current_time}.json" \
  --override_model_path $model_path