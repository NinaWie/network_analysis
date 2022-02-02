import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
import seaborn as sns

matplotlib.rc("font", **{"size": 15})

# Main parameter
ds_name = "foursquare"
filter_out_unconverged = True

# collect results
nr_files = 0
# soam
tconv_ok = 0
tconv_single_ok = 0
jaccard1, jaccard2 = [], []
soam_tables = []
# qap
coefficients = []
qap_pvalues = []
# Remember which dataset
dataset = []
dataset_dict = {"gc1": "Green Class", "foursquare": "Foursquare"}
# remember whether soam was included or not
soam_included = []

os.makedirs(os.path.join("..", f"results_{ds_name}"), exist_ok=True)

f = open(os.path.join("..", f"results_{ds_name}", "terminal.txt"), "w")
sys.stdout = f

if ds_name == "both":
    base_path_list = [os.path.join("..", "data", "gc1_120"), os.path.join("..", "data", "foursquare_120")]
else:
    # get the path of the correct dataset
    base_path_list = [os.path.join("..", "data", f"{ds_name}_120")]

print("\n READING DATA ...")
for base_path in base_path_list:
    print("---------> Load from path", base_path)
    for folder in os.listdir(base_path):
        # read files
        if folder[0] == ".":
            continue
        path_soam = os.path.join(base_path, folder, "soam_fitted.csv")
        path_qap = os.path.join(base_path, folder, "qap_fitted")

        nr_files += 1

        # check qap
        coeff_3, pvalues_3 = [], []
        for j in range(3):
            qap = pd.read_csv(path_qap + f"_{j}.csv")
            if pd.isna(qap.iloc[3]["ESt."]):
                print(folder)
            qap["x"] = ["Intercept", "Distances", "Dist from home", "Same purpose"]
            qap = qap.set_index("x").drop(columns=["Unnamed: 0"])
            coeff_3.append(qap["ESt."].values)
            pvalues_3.append(qap["p-value"].values)

            if folder == "310" and j == 0:
                print("USER 310 - QAP")
                print(qap)
                print(qap.drop(columns=["exp(Est.)"]).to_latex(float_format="%.2f"))
        coefficients.append(coeff_3)
        qap_pvalues.append(pvalues_3)

        # check soam
        soam = pd.read_csv(path_soam)

        # save jaccard in any case
        jaccard1.append(soam.loc[0, "jac1"])
        jaccard2.append(soam.loc[0, "jac2"])

        dataset.append(dataset_dict[(base_path.split(os.sep)[-1]).split("_")[0]])

        # check overall convergence and filter out if required
        if soam.loc[0, "tconv_max"] < 0.2:
            tconv_ok += 1
        elif filter_out_unconverged:
            continue
        single_conv = soam["t.conv"].dropna().values
        # check single convergences and filter out if required
        if np.all(single_conv < 0.1):
            tconv_single_ok += 1
        elif filter_out_unconverged:
            continue

        if folder == "310":
            print("USER 310 - SOAM")
            print(
                soam.drop(columns=["dependent", "jac1", "jac2", "sig.", "tconv_max", "Unnamed: 0"])
                .set_index("effect")
                .to_latex(float_format="%.2f")
            )

        part_soam = soam[["effect", "theta", "p.value"]]
        part_soam.rename(
            columns={"theta": "theta" + str(nr_files - 1), "p.value": "p" + str(nr_files - 1)}, inplace=True
        )
        soam_tables.append(part_soam)
        soam_included.append(nr_files - 1)


print("Number of files:", nr_files)
print("\n 1) Jaccard index results\n")
print("SOAM: overall convergence rate is okay in", tconv_ok, "files")  #  / float(nr_files))
print("SOAM: single convergence rate are okay in", tconv_single_ok, "files")
print("__________________")
print("Jaccard index analysis")
print(np.mean(jaccard1), np.std(jaccard1))
print(np.mean(jaccard2), np.std(jaccard2))
plt.figure(figsize=(7, 5))
plt.hist(jaccard1, bins=20)
plt.savefig(os.path.join("..", f"results_{ds_name}", "hist_jac1.pdf"))
plt.figure(figsize=(7, 5))
plt.hist(jaccard2, bins=20)
plt.savefig(os.path.join("..", f"results_{ds_name}", "hist_jac2.pdf"))

# Jaccard correlation
print("Pearsonr", pearsonr(jaccard1, jaccard2))
plt.figure(figsize=(7, 5))
# plt.scatter(jaccard1, jaccard2, label=dataset)
jac_data = pd.DataFrame()
jac_data["jac1"] = jaccard1
jac_data["jac2"] = jaccard2
jac_data["Dataset"] = dataset
sns.scatterplot(data=jac_data, x="jac1", y="jac2", hue="Dataset")
plt.xlabel("Jaccard index - 1st and 2nd time period")
plt.ylabel("Jaccard index - 2nd and 3rd time period")
plt.tight_layout()
plt.savefig(os.path.join("..", f"results_{ds_name}", "correlation_jaccards.pdf"))

### Analyze QAP coefficients and p values of all users
print("\n 2) QAP coefficient analysis \n")
coefficients = np.array(coefficients)
fine_rows = [row for row in range(len(coefficients)) if ~np.any(np.isnan(coefficients[row]))]
print(coefficients.shape, coefficients[fine_rows].shape)
qap_pvalues = np.array(qap_pvalues)
var_intra_user = np.std(coefficients[fine_rows], axis=1)
var_inter_user = np.std(coefficients[fine_rows], axis=0)
print("----------")
print("QAP: Comparison of intra user and inter user variance")
print(var_intra_user.shape, var_inter_user.shape)
print(np.around(np.mean(var_intra_user), 5))
print(np.around(np.mean(var_inter_user), 5))
print("(By variable:)")
print(np.around(np.mean(var_intra_user, axis=0), 5))
print(np.around(np.mean(var_inter_user, axis=0), 5))

# Average over time bins
avg_coeff = np.mean(coefficients, axis=1)
avg_qap_pvalues = np.mean(qap_pvalues, axis=1)

# print fraction of positive and negative correlations
for i in range(4):
    coeff_purpose = avg_coeff[:, i]
    pvals_purpose = avg_qap_pvalues[:, i]
    overall_coeff = len(coeff_purpose)
    pos_coeff_purpose = pvals_purpose[coeff_purpose > 0]
    print(f"Positive correlation significant (from {overall_coeff}) in", sum(pos_coeff_purpose < 0.05) / overall_coeff)
    neg_coeff_purpose = pvals_purpose[coeff_purpose < 0]
    print("Negative correlation significant in", sum(neg_coeff_purpose < 0.05) / overall_coeff)


matplotlib.rc("font", **{"size": 20})
plt.figure(figsize=(15, 10))
for j, col in enumerate(
    ["Intercept", "Distance (between A and B)", "Distance from home (of B)", "Same purpose (of A and B)"]
):
    plt.subplot(2, 2, j + 1)
    if "Distance" in col:
        bins = 140
    else:
        bins = 40
    temp_df = pd.DataFrame()
    temp_df["temp"] = avg_coeff[:, j]
    temp_df["Dataset"] = dataset
    histplt = sns.histplot(data=temp_df, x="temp", hue="Dataset", bins=bins, multiple="stack", stat="percent")
    plt.xlabel(col)
    plt.ylabel("Percent of users", fontsize=20)
    plt.xticks(fontsize=20)
    if "Distance" in col:
        plt.xlim(-0.25, 0.05)
    plt.tight_layout()
plt.savefig(os.path.join("..", f"results_{ds_name}", "results_qap.pdf"))
# plt.show()


#### SOAM
print()
print("_______ SOAM _______________")
print("Files in soam", len(soam_included))

from functools import reduce

df_merged = reduce(lambda left, right: pd.merge(left, right, on=["effect"], how="outer"), soam_tables)

p_table = df_merged[["p" + str(i) for i in soam_included]]
print("P table shape", p_table.shape)
p_table.fillna(1)
p_table = np.array(p_table) < 0.05

theta_table = (df_merged[["effect"] + ["theta" + str(i) for i in soam_included]]).set_index("effect")

# Compute mean and standard deviations of effects
new_df_dict = []
for i, effect in enumerate(theta_table.index):
    row = theta_table.loc[effect]
    # print(effect, np.mean(row), np.std(row))
    reduced_row = row[p_table[i]]
    # divided into gc and foursquare
    gc1_vals = [val for ind, val in row.items() if dataset[int(ind[5:])] == "Green Class"]
    foursquare_vals = [val for ind, val in row.items() if dataset[int(ind[5:])] == "Foursquare"]
    assert len(gc1_vals) + len(foursquare_vals) == len(row)
    # check if there is any that are significant in the wrong direction:
    print(effect, len(reduced_row[reduced_row >= 0]), len(reduced_row[reduced_row <= 0]))

    pval_diff = ttest_ind(gc1_vals, foursquare_vals)[1]
    if pval_diff < 0.05:
        print(
            "t-test (ind) of GC1 and Foursquare",
            round(np.mean(gc1_vals), 2),
            round(np.mean(foursquare_vals), 2),
            pval_diff,
        )

    new_df_dict.append(
        {
            "Effect": effect,
            "Ratio significant > 0": len(reduced_row[reduced_row >= 0]) / len(row),
            "Ratio significant < 0": len(reduced_row[reduced_row <= 0]) / len(row),
            "Mean overall": np.mean(row),
            "Std overall": np.std(row),
            "Mean significant": np.mean(reduced_row),
            "Std significant": np.std(reduced_row),
            "Mean GC1": np.mean(gc1_vals),
            "Std GC1": np.std(gc1_vals),
            "Mean Foursquare": np.mean(foursquare_vals),
            "Std Foursquare": np.std(foursquare_vals),
        }
    )
    # print()
results = pd.DataFrame(new_df_dict)
results.to_csv(os.path.join("..", f"results_{ds_name}", "results_soam.csv"))
f.close()
