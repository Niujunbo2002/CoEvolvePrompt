import argparse
from huggingface_hub import hf_hub_download
import shutil


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=["MATH500", "MATH_train"],
    required=True,
    help="Which dataset to download"
)
args = parser.parse_args()
dataset = args.dataset


if dataset == "MATH_train":
    split = "train"
else:
    split = "test"


cached_path = hf_hub_download(
    repo_id=f"yinjiewang/{dataset}",
    repo_type="dataset",
    filename=f"{split}/{dataset}.json"
)
shutil.copy(cached_path, f"./{dataset}.json")