import pandas as pd


df = pd.read_csv("habitual.txt", sep="\t", header=None, names=["AAE", "SAE"])
print(df.head())
df.to_csv("aae_sae.tsv", sep="\t", index=False)
