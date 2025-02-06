from datasets import load_dataset

output = load_dataset("ThatsGroes/dialog-topics")
df = output["train"].to_pandas()
df.to_csv("test.csv", index=False)