# Made by Robin Mesman
# Date: 01-06-2026 untill 16-06-2026
#
# Results Visualization and Affinity Comparison
# Loads paired prediction outputs and matches entries by shared identifier.
# Combines corresponding values into a single dataset and visualizes them
# as a scatter plot to compare two measured properties for each entry.
#
# Available dataset_config parameters:
# cAMP: file path for first condition results
# cGMP: file path for second condition results
# prefix: adds text to avoid name collisions
# color: plotting color for dataset points
# filter_keep: keeps selected structure names only
# label: legend name for dataset group

import matplotlib.pyplot as plt
import pandas as pd

# Define input CSV files, colors, labels, and optional filtering settings
# for each dataset included in the affinity comparison.
datasets_config = [
    {
        "cAMP": "../results/EPAC1_wt_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/EPAC1_wt_cGMP_Boltz2_scores.csv",
        "prefix": "",
        "color": "green",
        "label": "EPAC1",  
    },
    {
        "cAMP": "../results/generated_structures_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/generated_structures_cGMP_Boltz2_scores.csv",
        "prefix": "",
        "color": "blue",
        "label": "Random_Designs",
    },
    {
        "cAMP": "../results/Known_proteins_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/Known_proteins_cGMP_Boltz2_scores.csv",
        "prefix": "",
        "color": "black",
        "filter_keep": ["EPAC1_mutant", "PKG_Ibeta", "PKA_RIalfa_A"],
        "label": "Known_proteins_designs",
    },
    {
        "cAMP": "../results/GS_cAMPf_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/GS_cAMPf_cGMP_Boltz2_scores.csv",
        "prefix": "NieuwSet1_",
        "color": "pink",
        "label": "cAMPf_designs",
    },
    {
        "cAMP": "../results/GS_cGMPf_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/GS_cGMPf_cGMP_Boltz2_scores.csv",
        "prefix": "NieuwSet2_",
        "color": "purple",
        "label": "cGMPf_designs",
    },
    {
        "cAMP": "../results/GS_cGMPw_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/GS_cGMPw_cGMP_Boltz2_scores.csv",
        "prefix": "NieuwSet3_",
        "color": "red",
        "label": "cGMPw_designs",
    },
    {
        "cAMP": "../results/GS_cAMPw_cAMP_Boltz2_scores.csv",
        "cGMP": "../results/GS_cAMPw_cGMP_Boltz2_scores.csv",
        "prefix": "NieuwSet4_",
        "color": "yellow",
        "label": "cAMPw_designs",
    },
]

# specific color overrides for individual proteins
# used to highlight important structures such as mutants
specific_colors = {"EPAC1_mutant": "green"}

# proteins that should receive text labels in the final plot
proteins_to_label = ["EPAC1_wt", "EPAC1_mutant", "PKG_Ibeta", "PKA_RIalfa_A"]


# create figure and axis for the scatter plot
fig, ax = plt.subplots(figsize=(10, 7))  # Iets breder gemaakt voor de legenda

# iterate through all configured datasets
for config in datasets_config:

     # load cAMP and cGMP affinity prediction results
    df_camp_single = pd.read_csv(config["cAMP"])
    df_cgmp_single = pd.read_csv(config["cGMP"])

    # optionally retain only selected structures
    if "filter_keep" in config:
        df_camp_single = df_camp_single[
            df_camp_single["Structure"].isin(config["filter_keep"])
        ]
        df_cgmp_single = df_cgmp_single[
            df_cgmp_single["Structure"].isin(config["filter_keep"])
        ]

    # add dataset-specific prefixes when required
    if config["prefix"]:
        df_camp_single["Structure"] = (
            config["prefix"] + df_camp_single["Structure"].astype(str)
        )
        df_cgmp_single["Structure"] = (
            config["prefix"] + df_cgmp_single["Structure"].astype(str)
        )

    # merge cAMP and cGMP predictions based on structure name
    df_merged = pd.merge(
        df_camp_single,
        df_cgmp_single,
        on="Structure",
        suffixes=("_cAMP", "_cGMP"),
    )

    # separate standard points from structures with custom colors
    df_normal = df_merged[~df_merged["Structure"].isin(specific_colors.keys())]
    df_special = df_merged[df_merged["Structure"].isin(specific_colors.keys())]

    # plot standard dataset points
    
    ax.scatter(
        df_normal["Affinity (log10 IC50 µM)_cAMP"],
        df_normal["Affinity (log10 IC50 µM)_cGMP"],
        color=config["color"],
        label=config["label"],
        s=70,
        alpha=0.85,
    )

    # plot highlighted structures using custom colors
    for _, row in df_special.iterrows():
        name = row["Structure"]
        specific_color = specific_colors[name]
        ax.scatter(
            row["Affinity (log10 IC50 µM)_cAMP"],
            row["Affinity (log10 IC50 µM)_cGMP"],
            color=specific_color,
            s=70,
            alpha=0.85,
        )

    # add text labels to selected proteins
    for _, row in df_merged.iterrows():
        name = row["Structure"]
        if name in proteins_to_label:
            x = row["Affinity (log10 IC50 µM)_cAMP"]
            y = row["Affinity (log10 IC50 µM)_cGMP"]
            ax.annotate(
                name,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
            )

# add diagonal reference line representing equal affinity
ax.plot([-1, 1], [-1, 1], linestyle="--", color="red", linewidth=1)

# configure axis limits and labels
ax.set_xlim(-1.5, 1.2)
ax.set_ylim(-1.5, 1.2)
ax.set_xlabel("cAMP (log10 IC50 µM)")
ax.set_ylabel("cGMP (log10 IC50 µM)")
ax.set_title("cAMP vs cGMP IC50")

# place legend outside the plot area for readability
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Datasets")

# optimize layout and display figure
plt.tight_layout()
plt.show()