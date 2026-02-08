for i in {240..26160..240}
do
    MASTER_PORT=29411 # START FROM 29411
    task="gsm8k"
    batch_size=8
    gen_length=256
    OUTPUT_DIR="./results/gsm8k_diffugrpo"
    MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
    CHECKPOINT_PATH="../diffu-grpo/outputs/gsm8k_diffugrpo/checkpoint-$i"


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