import json, pathlib
pot_path = pathlib.Path("data/pot/chartqa_pot_dataset.jsonl")
index = json.loads(open("data/raw/index_by_id.json").read())

merged = []
with pot_path.open() as f:
    for line in f:
        item = json.loads(line)
        sid = item["sample_id"]
        if sid not in index:
            continue      # 若缺图像，直接跳过或报警
        merged.append({
            "image": index[sid]["img_path"],
            "question": item["query"],
            "answer": str(item["label"]),
            "code": item["python_code"],     # PoT 训练时用
        })

json.dump(merged, open("data/merged_all.json", "w"))
print(f"Merged {len(merged)} samples.")