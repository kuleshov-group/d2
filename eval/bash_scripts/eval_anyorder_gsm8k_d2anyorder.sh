for i in {120..2280..120}
do
    MASTER_PORT=29411 # START FROM 29411
    task="gsm8k"
    batch_size=1
    gen_length=256
    OUTPUT_DIR="./results/anyorder_gsm8k_d2anyorder"
    MODEL_PATH="GuanghanWang/d2_anyorder_causal_llada_intellectsft_gsm8k"
    CHECKPOINT_PATH="../diffu-grpo-ao/outputs/anyorder_gsm8k_d2anyorder/checkpoint-$i"


    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 \
        --master_port $MASTER_PORT \
        eval_anyorder.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --output_dir $OUTPUT_DIR \
        --model_path $MODEL_PATH \
        --block_length 32 
done
