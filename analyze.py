import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

matplotlib.rc("font", **{"size": 15})

nr_files = 0
# soam
tconv_ok = 0
tconv_single_ok = 0
jaccard1, jaccard2 = [], []
soam_tables = []
# qap
coefficients = []
qap_pvalues = []

base_path = "../data/foursquare"
for folder in os.listdir(base_path):
    if folder[0] == ".":
        continue
    path_soam = os.path.join(base_path, folder, "soam_fitted.csv")
    path_qap = os.path.join(base_path, folder, "qap_fitted")

    # check qap
    coeff_3, pvalues_3 = [], []
    for j in range(3):
        qap = pd.read_csv(path_qap + f"_{j}.csv")
        qap["x"] = ["Intercept", "Distances", "Dist from home", "Same purpose"]
        qap = qap.set_index("x").drop(columns=["Unnamed: 0"])
        coeff_3.append(qap["ESt."].values)
        pvalues_3.append(qap["p-value"].values)

        if folder == "310" and j == 0:
            print(qap)
            print(qap.to_latex())
    coefficients.append(coeff_3)
    qap_pvalues.append(pvalues_3)

    # check soam
    soam = pd.read_csv(path_soam)
    # check overall convergence
    if soam.loc[0, "tconv_max"] < 0.2:
        tconv_ok += 1
    single_conv = soam["t.conv"].dropna().values
    # check single convergences
    if np.all(single_conv < 0.1):
        tconv_single_ok += 1
    # save jaccard
    jaccard1.append(soam.loc[0, "jac1"])
    jaccard2.append(soam.loc[0, "jac2"])
    part_soam = soam[["effect", "theta", "p.value"]]
    part_soam.rename(columns={"theta": "theta" + str(nr_files), "p.value": "p" + str(nr_files)}, inplace=True)
    soam_tables.append(part_soam)

    nr_files += 1

print("Number of files:", nr_files)
print("overall convergence rate is okay in", tconv_ok / float(nr_files))
print("single convergence rate are okay in", tconv_single_ok / float(nr_files))
print("__________________")
print("Jaccard index analysis")
print(np.mean(jaccard1), np.std(jaccard1))
print(np.mean(jaccard2), np.std(jaccard2))
plt.figure(figsize=(7, 5))
plt.hist(jaccard1, bins=20)
plt.savefig("../results/hist_jac1.pdf")
plt.figure(figsize=(7, 5))
plt.hist(jaccard2, bins=20)
plt.savefig("../results/hist_jac2.pdf")
print("Pearsonr", pearsonr(jaccard1, jaccard2))
plt.figure(figsize=(7, 5))
plt.scatter(jaccard1, jaccard2)
plt.savefig("../results/correlation_jaccards.pdf")

### Analyze QAP coefficients and p values of all users
coefficients = np.array(coefficients)
qap_pvalues = np.array(qap_pvalues)
var_intra_user = np.std(coefficients, axis=1)
var_inter_user = np.std(coefficients, axis=0)
print("----------")
print("Comparison of intra user and inter user variance")
print(var_intra_user.shape, var_inter_user.shape)
print(np.around(np.mean(var_intra_user), 5))
print(np.around(np.mean(var_inter_user), 5))
print("By variable:")
print(np.around(np.mean(var_intra_user, axis=0), 5))
print(np.around(np.mean(var_inter_user, axis=0), 5))

# print fraction of positive and negative correlations
for i in range(4):
    coeff_purpose = coefficients[:, :, i]
    pvals_purpose = qap_pvalues[:, :, i]
    pos_coeff_purpose = pvals_purpose[coeff_purpose > 0]
    print("Positive correlation significant in", sum(pos_coeff_purpose < 0.05) / 75)
    neg_coeff_purpose = pvals_purpose[coeff_purpose < 0]
    print("Negative correlation significant in", sum(neg_coeff_purpose < 0.05) / 75)

# Average over time bins
avg_coeff = np.mean(coefficients, axis=1)
avg_qap_pvalues = np.mean(qap_pvalues, axis=1)

matplotlib.rc("font", **{"size": 22})
plt.figure(figsize=(15, 10))
for j, col in enumerate(["Intercept", "Distances", "Distance from home", "Same purpose"]):
    plt.subplot(2, 2, j + 1)
    sns.histplot(avg_coeff[:, j], bins=10)
    plt.title(col, fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.xticks(fontsize=20)
    plt.tight_layout()
plt.savefig("../results/over_users.pdf")
# plt.show()


#### SOAM
print()
print("_______ SOAM _______________")

from functools import reduce

df_merged = reduce(lambda left, right: pd.merge(left, right, on=["effect"], how="outer"), soam_tables)

p_table = df_merged[["p" + str(i) for i in range(25)]]
p_table.fillna(1)
p_table = np.array(p_table) < 0.05

theta_table = (df_merged[["effect"] + ["theta" + str(i) for i in range(25)]]).set_index("effect")

# Compute mean and standard deviations of effects
new_df_dict = []
for i, effect in enumerate(theta_table.index):
    row = theta_table.loc[effect]
    # print(effect, np.mean(row), np.std(row))
    reduced_row = row[p_table[i]]
    # print("Number significant", len(reduced_row))
    if len(reduced_row) > 0:
        print("sigificant", effect, np.mean(reduced_row), np.std(reduced_row))
    new_df_dict.append(
        {
            "Effect": effect,
            "Ratio significant": len(reduced_row) / len(row),
            "Mean overall": np.mean(row),
            "Std overall": np.std(row),
            "Mean significant": np.mean(reduced_row),
            "Std significant": np.std(reduced_row),
        }
    )
    # print()
results = pd.DataFrame(new_df_dict)
results.to_csv("../results/results_soam.csv")
