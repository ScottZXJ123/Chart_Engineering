import json, random
random.seed(42)

data = json.load(open("data/merged_all.json"))
random.shuffle(data)
n = len(data)
train, val, test = data[:int(0.8*n)], data[int(0.8*n):int(0.9*n)], data[int(0.9*n):]

for split, arr in zip(("train","val","test"), (train, val, test)):
    json.dump(arr, open(f"data/{split}.json", "w"))