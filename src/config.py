import argparse
import os
import random
import sys

import numpy as np
import torch
from easydict import EasyDict as edict

# Get paths and validate
try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(file_dir)
    workspace_root = os.path.join("/workspace")  # Docker mount point

    paths_to_add = [
        project_root,
        workspace_root,
        os.path.join(workspace_root, "pad")
    ]

    # Add paths if they exist and aren't already in sys.path
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to Python path: {path}")
        else:
            print(f"Path does not exist or already in sys.path: {path}")

    print(f"Project root: {project_root}")
    print(f"Python path: {os.environ.get('PYTHONPATH', '')}")

except Exception as e:
    print(f"Failed to setup paths: {str(e)}")
    raise

def get_config(args):

    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    config = edict(vars(args))

    if config.training_type == "PAD_training":
        config.use_lora = True
        config.train_scratch = False
    elif config.training_type == "PAD_training_only_header":
        config.use_lora = False
        config.train_scratch = False
    elif config.training_type == "PAD_training_scratch":
        config.use_lora = False
        config.train_scratch = True

    train_dataset_paths = {
        "idiap": "/workspace/FacePAD/Protocols/replayattack.csv",
        "casia": "/workspace/FacePAD/Protocols/casia.csv",
        "msu": "/workspace/FacePAD/Protocols/msu.csv",
        "oulu": "/workspace/FacePAD/Protocols/oulu_npu.csv",
        "mi": "/workspace/FacePAD/Protocols/cross_domain_cropface/mi_c/train.csv",
        "ocm": "/workspace/FacePAD/Protocols/cross_domain_cropface/ocm_i/train.csv",
        "oci": "/workspace/FacePAD/Protocols/cross_domain_cropface/oci_m/train.csv",
        "omi": "/workspace/FacePAD/Protocols/cross_domain_cropface/omi_c/train.csv",
        "icm": "/workspace/FacePAD/Protocols/cross_domain_cropface/icm_o/train.csv",
        "synthaspoof": "/workspace/FacePAD/Protocols/synthaspoof.csv"
    }

    test_dataset_paths = {
        "casia": "/workspace/FacePAD/Protocols/casia.csv",
        "msu": "/workspace/FacePAD/Protocols/msu.csv",
        "idiap": "/workspace/FacePAD/Protocols/replayattack.csv",
        "oulu": "/workspace/FacePAD/Protocols/oulu_npu.csv",
        "celeba": "/workspace/FacePAD/Protocols/cross_domain_cropface/mco_ca/test.csv"
    }

    if config.dataset_name == "idiap":
        config.dataset_path = train_dataset_paths["idiap"]
        config.test_dataset_path = [
            test_dataset_paths["casia"],
            test_dataset_paths["msu"],
            test_dataset_paths["oulu"]
        ]
        config.test_data = ["casia", "msu", "oulu"]
    elif config.dataset_name == "casia":
        config.dataset_path = train_dataset_paths["casia"]
        config.test_dataset_path = [
            test_dataset_paths["idiap"],
            test_dataset_paths["msu"],
            test_dataset_paths["oulu"]
        ]
        config.test_data = ["idiap", "msu", "oulu"]
    elif config.dataset_name == "msu":
        config.dataset_path = train_dataset_paths["msu"]
        config.test_dataset_path = [
            test_dataset_paths["casia"],
            test_dataset_paths["idiap"],
            test_dataset_paths["oulu"]
        ]
        config.test_data = ["casia", "idiap", "oulu"]
    elif config.dataset_name == "oulu":
        config.dataset_path = train_dataset_paths["oulu"]
        config.test_dataset_path = [
            test_dataset_paths["casia"],
            test_dataset_paths["msu"],
            test_dataset_paths["idiap"]
        ]
        config.test_data = ["casia", "msu", "idiap"]
    elif config.dataset_name == "mi":
        config.dataset_path = train_dataset_paths["mi"]
        config.test_dataset_path = [
            test_dataset_paths["casia"],
            test_dataset_paths["oulu"]
        ]
        config.test_data = ["casia", "oulu"]
    elif config.dataset_name == "ocm":
        config.dataset_path = train_dataset_paths["ocm"]
        config.test_dataset_path = [
            test_dataset_paths["celeba"],
            test_dataset_paths["idiap"]
        ]
        config.test_data = ["celeba", "idiap"]
    elif config.dataset_name == "oci":
        config.dataset_path = train_dataset_paths["oci"]
        config.test_dataset_path = [
            test_dataset_paths["msu"]
        ]
        config.test_data = ["msu"]
    elif config.dataset_name == "omi":
        config.dataset_path = train_dataset_paths["omi"]
        config.test_dataset_path = [
            test_dataset_paths["casia"]
        ]
        config.test_data = ["casia"]
    elif config.dataset_name == "icm":
        config.dataset_path = train_dataset_paths["icm"]
        config.test_dataset_path = [
            test_dataset_paths["oulu"]
        ]
        config.test_data = ["oulu"]
    elif config.dataset_name == "synthaspoof":
        config.dataset_path = train_dataset_paths["synthaspoof"]
        config.test_dataset_path = [
            test_dataset_paths["msu"],
            test_dataset_paths["casia"],
            test_dataset_paths["idiap"],
            test_dataset_paths["oulu"]
        ]
        config.test_data = ["msu", "casia", "idiap", "oulu"]
    config.num_classes = 2
    if config.backbone_size == "ViT-B/16" or config.backbone_size == "ViT-B/32":
        # "ViT-B/32", "ViT-B/16", "ViT-L/14"
        config.training_desc = f'ViT-B16/{config.training_type}/{config.dataset_name}'
    elif config.backbone_size == "ViT-L/14":
        config.training_desc = f'ViT-L14/{config.training_type}/{config.dataset_name}'
    config.output_path = "/output/pad_last/" + config.training_desc

    if config.training_type == "PAD_training":
        config.output_path = (
            f"{config.output_path}/lrm{config.lr_model:.0e}_lrh{config.lr_header:.0e}_d{config.lora_dropout:.0e}_a{config.lora_a}_r{config.lora_r}"
        )
    elif config.training_type == "PAD_training_only_header":
        config.output_path = f"{config.output_path}/lrh{config.lr_header:.0e}"
    elif config.training_type == "PAD_training_scratch":
        config.output_path = (
            f"{config.output_path}/lrm{config.lr_model:.0e}_lrh{config.lr_header:.0e}"
        )

    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # cudnn.benchmark = True
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description="Distributed training job")
    parser.add_argument("--local-rank", type=int, help="local_rank")
    parser.add_argument(
        "--mode",
        default="training",
        choices=["training", "evaluation"],
        help="train or eval mode",
    )
    parser.add_argument(
        "--debug", default=False, type=bool, help="Log additional debug informations"
    )

    parser.add_argument("--backbone_size", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--training_type", type=str, required=True)

    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr_model", type=float, default=1e-6)
    parser.add_argument("--lr_header", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eta_min", type=float, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0.4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_a", type=int, default=8)
    parser.add_argument("--max_norm", type=float, default=5)
    parser.add_argument("--loss", type=str, default="BinaryCrossEntropy")
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup", type=bool, default=True)
    parser.add_argument("--num_warmup_epochs", type=int, default=5)
    parser.add_argument("--T_0", type=int, default=5)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--lr_func_drop", type=int, nargs="+", default=[22, 30, 40])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q", "v"]
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--normalize_type", type=str, default="clip")
    parser.add_argument("--interpolation_type", type=str, default="bicubic")
    parser.add_argument(
        "--eval_path", type=str, default="/home/chettaou/data/validation"
    )
    parser.add_argument("--val_targets", type=str, nargs="+", default=[])
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--batch_size_eval", type=int, default=16)
    parser.add_argument("--horizontal_flip", type=bool, default=True)
    parser.add_argument("--rand_augment", type=bool, default=True)
    args = parser.parse_args()
    config = get_config(args)
    from src.train import main
    main(config)
