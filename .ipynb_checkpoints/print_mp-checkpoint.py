import pandas as pd
import numpy as np

# --- Load both files ---
gpt_df = pd.read_csv("GPT_data/GPT2_hoffmanagreement_byword.tsv", sep="\t")
aae_df = pd.read_csv("AAE_SAE_MP.tsv", sep="\t")


print(aae_df["sentid"].max())


final_words = gpt_df.groupby("sentid")["wordpos"].max().reset_index(drop=True).to_numpy()

aae_df["ROI"] = final_words

merged_df.to_csv("AAE_SAE_MP_fixed.tsv", sep="\t", index=False)

print("done!")

