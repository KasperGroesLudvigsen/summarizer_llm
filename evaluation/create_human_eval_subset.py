"""
S
"""
import json
import random
from datasets import load_dataset 


output = load_dataset("ThatsGroes/LLM-summary-evaluation")
df = output["test"].to_pandas()


sample = df.sample(n=100)

# TODO: Refactor this now that there is only one model column
summary_columns = [col for col in sample.columns if "summary_by_" in col]
random.shuffle(summary_columns)

#uuids = [str(uuid.uuid1()) for i in range(len(summary_columns))]
uuids = [i+1 for i in range(len(df["model"].unique()))]

#mapping = [{"model" : model, "uuid" : uid} for model, uid in zip(summary_columns, uuids)]
mapping = {}
for model, uid in zip(summary_columns, uuids):
    mapping[model] = uid

with open('model_to_uuid_map.json', 'w') as fout:
    json.dump(mapping, fout)

# Rename columns
sample = sample.rename(columns=mapping)

# Reorder DataFrame

keep_col = ["dialog", "summary"]
keep_col.extend(uuids)
print(keep_col)
# drop cols and reorder
sample = sample[keep_col]
print(sample.head())
sample.to_csv("samples_human_eval.csv", index=False)
