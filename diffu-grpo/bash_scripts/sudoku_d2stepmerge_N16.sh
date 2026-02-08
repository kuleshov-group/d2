DATASET="sudoku"
RUN_NAME=${DATASET}_d2stepmerge_N16
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
NUM_ITER=8 # number of policy gradient inner updates iterations
OUTPUT_DIR=./outputs/sudoku_d2stepmerge_N16


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
    --trainer_name d2-StepMerge \
    --max_completion_length 128 \
    --diffusion_steps 64 \
    --temperature 0.3 \
    --N 16 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 160
