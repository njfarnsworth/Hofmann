import pandas as pd


# 1. Load files


print("code running")
bypair = pd.read_csv('GPT_data/GPT2_hoffmanagreement_bypair.tsv', delimiter='\t').iloc[:100]
mp_df = pd.read_csv('AAE_SAE_MP.tsv', delimiter='\t').iloc[:200]

# 2. Build a list of occupations by pairid

print("on step 2")
max_pairid = mp_df['pairid'].max()
occupations = []

for i in range(max_pairid + 1):
    first_row = mp_df[mp_df['pairid'] == i].iloc[0]
    occupations.append(first_row['occupation'])


# 3. Add occupations to bypairs DataFrame

print("on step 3")
bypair['occupation'] = [occupations[i] for i in bypair['pairid']]


# 4. Convert surprisal -> probability

print("on step 4")
bypair['P_SAE'] = 2 ** (-bypair['expected'])
bypair['P_AAVE'] = 2 ** (-bypair['unexpected'])

pd.set_option('display.max_rows', None)    # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', None)       # Disable line wrapping
pd.set_option('display.max_colwidth', None) # Show full column content

print(bypair)


# 5. Optional: average probabilities by occupation

print("on step 5")

occupation_probs = {}

for occ in occupations:
    # Filter rows for this occupation
    rows = bypair[bypair['occupation'] == occ]
    # Extract the probabilities
    sae_probs = rows['P_SAE'].tolist()
    aave_probs = rows['P_AAVE'].tolist()
    avg_sae = sum(sae_probs) / len(sae_probs)
    avg_aave = sum(aave_probs) / len(aave_probs) 
    occupation_probs[occ] = [avg_sae, avg_aave]


# 6. Display results

for occ, prob in occupation_probs.items():
    print(f"occupation: {occ}")
    print(f"probs: {prob}")

# -----------------------------
# 7. Save to file if needed
# -----------------------------
# bypair.to_csv('GPT_data/bypair_with_probs.tsv', sep='\t', index=False)
