DATASET="gsm8k"
RUN_NAME=anyorder_${DATASET}_diffugrpo
MODEL_PATH=GuanghanWang/d2_anyorder_causal_llada_intellectsft_gsm8k
NUM_ITER=2 # number of policy gradient inner updates iterations
output_dir=./outputs/anyorder_gsm8k_diffugrpo


accelerate launch \
    --config_file ./bash_scripts/accelerate.yaml \
    --main_process_port 12346 ./diffu_grpo_train_ao.py \
    --config ./bash_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir $output_dir \
    --trainer_name diffu-GRPO \
    --max_completion_length 256 \
    --save_steps 120 \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --per_device_train_batch_size 8 