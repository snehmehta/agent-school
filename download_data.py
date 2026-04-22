"""Download sakhiya-callcenter-data from HuggingFace Hub."""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

token = os.environ.get("HUGGINGFACE_API_KEY")
output_dir = Path("raw-data/sakhiya-callcenter-data")

print(f"Downloading to {output_dir} ...")
snapshot_download(
    repo_id="snehmehta/sakhiya-callcenter-data",
    repo_type="dataset",
    local_dir=str(output_dir),
    token=token,
)
print(f"Done. Files in {output_dir}:")
for f in sorted(output_dir.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(output_dir)}")
