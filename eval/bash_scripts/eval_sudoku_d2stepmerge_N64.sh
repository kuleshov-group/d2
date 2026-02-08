for i in {120..1560..120}
do
    MASTER_PORT=29411 # START FROM 29411
    task="sudoku"
    batch_size=8
    gen_length=128
    OUTPUT_DIR="./results/sudoku_d2stepmerge_N64"
    MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
    CHECKPOINT_PATH="../diffu-grpo/outputs/sudoku_d2stepmerge_N64/checkpoint-$i"


    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 \
        --master_port $MASTER_PORT \
        eval.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --output_dir $OUTPUT_DIR \
        --model_path $MODEL_PATH \
        --checkpoint_path $CHECKPOINT_PATH
done