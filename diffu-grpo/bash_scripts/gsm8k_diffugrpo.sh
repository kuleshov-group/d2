DATASET="gsm8k"
RUN_NAME=${DATASET}_diffugrpo
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
NUM_ITER=12 # number of policy gradient inner updates iterations
OUTPUT_DIR=./outputs/gsm8k_diffugrpo


accelerate launch \
    --config_file ./bash_scripts/accelerate_a100.yaml \
    --main_process_port 12346 ./diffu_grpo_train.py \
    --config ./bash_scripts/train.yaml \
    --save_steps 120 \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --trainer_name diffu-GRPO \
    --max_completion_length 256 \
    --diffusion_steps 128
