from datasets import load_dataset
import json, pathlib

root = pathlib.Path("data/raw")
root.mkdir(parents=True, exist_ok=True)

ds = load_dataset("HuggingFaceM4/ChartQA", "default", split="train+validation+test")
# 按 sample_id 建索引
index = {}
for row in ds:
    sid = int(row["id"])          # ChartQA meta 里自带
    img_path = root / f"{sid}.png"
    # 保存图像文件
    row["image"].save(img_path)
    index[sid] = {"img_path": str(img_path), "meta": row}

with open(root / "index_by_id.json", "w") as f:
    json.dump(index, f)