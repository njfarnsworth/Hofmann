import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load data ---
df_raw = pd.read_csv("results/gpt_pred_small_bypair.tsv", sep="\t")
avg_probs = df_raw.groupby('occupation')[["diff"]].mean().reset_index()
mean_diff_dict = avg_probs.set_index('occupation')['diff'].to_dict()

# --- 2. Prestige scores ---
prestige_scores = {
    'accountant': 5.7, 'actor': 5.7, 'actress': 5.5, 'administrator': 4.8, 'analyst': 5.0,
    'architect': 6.7, 'artist': 5.4, 'assistant': 4.8, 'athlete': 6.2, 'author': 4.7,
    'chef': 4.4, 'chief': 5.9, 'clergy': 4.1, 'clerk': 3.9, 'dentist': 6.9,
    'designer': 4.6, 'developer': 5.5, 'doctor': 7.4, 'driver': 3.6, 'engineer': 6.3,
    'farmer': 4.4, 'journalist': 4.4, 'judge': 5.1, 'lawyer': 6.6, 'manager': 5.4,
    'mechanic': 4.0, 'model': 3.8, 'musician': 5.0, 'nurse': 5.0, 'photographer': 4.5,
    'physician': 7.4, 'pilot': 6.6, 'professor': 6.0, 'psychologist': 5.2, 'scientist': 6.5,
    'secretary': 4.2, 'singer': 4.1, 'student': 3.2, 'supervisor': 4.8, 'surgeon': 7.7,
    'teacher': 5.0, 'writer': 5.1
}

df = pd.DataFrame({
    'occupation': list(prestige_scores.keys()),
    'prestige': [prestige_scores[occ] for occ in prestige_scores.keys()],
    'diff': [mean_diff_dict[occ] for occ in prestige_scores.keys()]
})

# --- 3. Fit linear regression ---
X = sm.add_constant(df['diff'])  # shape (n, 2), adds intercept
y = df['prestige']

model = sm.OLS(y, X)
results = model.fit()

# --- 4. Regression summary ---
print(results.summary())  # note parentheses

# --- 5. Predicted values ---
df['predicted'] = results.predict(X)

# --- 6. Plot ---
plt.figure(figsize=(8,6))
plt.scatter(df['diff'], df['prestige'], color='blue', label='Observed')
plt.plot(df['diff'], df['predicted'], color='red', label='Regression line')
plt.xlabel('AAVE-association score (diff)')
plt.ylabel('Occupational prestige')
plt.title('Linear Regression of Prestige on AAVE-association')
plt.legend()
plt.grid(True)
plt.savefig('figures/prestige_vs_diff.png', dpi=300, bbox_inches='tight')
#
