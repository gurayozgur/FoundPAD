#!/bin/bash
export OMP_NUM_THREADS=8
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_HOME="./cache/"

declare -a dataset_names=("idiap" "casia" "msu" "oulu" "mi" "ocm" "oci" "omi" "icm" "synthaspoof")
declare -a backbone_sizes=("ViT-B/16" "ViT-L/14")
declare -a training_types=("PAD_training" "PAD_training_only_header" "PAD_training_scratch")

# Define batch sizes
default_batch_size=64
vit_l_scratch_batch_size=32

# Define the specific experiments
experiments=(
  "1e-6 1e-3 0.40 8 8"
)

for experiment in "${experiments[@]}"; do
  IFS=' ' read -r lr_model lr_header lora_dropout lora_r lora_a <<< "$experiment"
  for backbone_size in "${backbone_sizes[@]}"; do
    for training_type in "${training_types[@]}"; do
      for dataset_name in "${dataset_names[@]}"; do
        # Set batch size based on model and training type
        if [[ "$backbone_size" == "ViT-L/14" ]] && [[ "$training_type" == "PAD_training_scratch" ]]; then
          batch_size=$vit_l_scratch_batch_size
        else
          batch_size=$default_batch_size
        fi
        # Convert scientific notation to decimal and compare using awk
        if awk -v header="$lr_header" -v model="$lr_model" 'BEGIN{exit !(header >= model)}'; then
          CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 \
          --node_rank=0 --master_addr="localhost" --master_port=12355 pad/src/config.py --debug=False \
          --backbone_size="$backbone_size" --dataset_name="$dataset_name" --training_type="$training_type" \
          --lr_model="$lr_model" --lr_header="$lr_header" --lora_dropout="$lora_dropout" --lora_r="$lora_r" --lora_a="$lora_a"  --batch_size="$batch_size"
        fi
      done
    done
  done
done
