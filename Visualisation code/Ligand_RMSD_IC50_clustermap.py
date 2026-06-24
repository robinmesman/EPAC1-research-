# Made by Robin Mesman
# Date: 18-05-2026 until 16-06-2026
#
# Structural Cluster Analysis and Affinity Integration Pipeline
# Loads PDB/CIF structure files and maps each structure to its corresponding IC50 affinity score.
# Selects the top-scoring models per ligand condition, computes a pairwise backbone RMSD matrix
# to quantify structural differences, and visualizes the results using a hierarchical clustermap
# annotated with ligand type and affinity color strips.
#
# Required Dataset Inputs:
# cAMP Affinity Data: CSV with 'Structure' and 'Affinity (log10 IC50 µM)'
# cGMP Affinity Data: CSV with 'Structure' and 'Affinity (log10 IC50 µM)'
# cAMP Structure Files: directory with PDB/CIF models
# cGMP Structure Files: directory with PDB/CIF models


from Bio.PDB import PDBParser, MMCIFParser, Superimposer
import os
from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


def load_structures(pattern, file_format):
    """
    Load all structure files matching a glob pattern and return them as a list
    of BioPython structure objects.
    """
    # normalize format string for consistent parsing
    fmt = file_format.lower()

    # select correct parser depending on structure format
    if fmt == "pdb":
        parser = PDBParser(QUIET=True)
    elif fmt == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("choose 'pdb' or 'cif'")

    # list for parsed structures
    structures = []

    # loop through all matching files in sorted order for reproducibility
    for filepath in sorted(glob(pattern)):
        
        # strip directory path and file extension to get a clean structure ID
        structure_id = os.path.basename(filepath).replace(f".{fmt}", "")

        # parse file into BioPython Structure object
        structure = parser.get_structure(structure_id, filepath)

        # store structure in list
        structures.append(structure)

    return structures


def extract_backbone_atoms(structure):
    """
    Extract backbone atoms (N, CA, C, O) from a structure.
    Only backbone atoms are used for RMSD so that side chains don't influence the comparison.
    """
    backbone_atom_names = ["N", "CA", "C", "O"]

    # list where the matching backbone atoms will be stored
    backbone_atoms = []

    # loop through all atoms in the structure and keep only backbone atoms
    for atom in structure.get_atoms():
        if atom.name in backbone_atom_names:
            backbone_atoms.append(atom)

    return backbone_atoms


def compute_pairwise_rmsd_matrix(structures):
    """
    Build an n x n matrix of RMSD values for all pairwise structure comparisons.
    Each structure is superimposed onto the other before computing RMSD,
    so only structural shape differences are measured.
    """
    # get the number of structures to set the matrix dimensions
    n = len(structures)

    # initialize an empty square matrix filled with zeros
    rmsd_matrix = np.zeros((n, n))

    # loop through all pairs of structures to fill the matrix
    for i in range(n):
        for j in range(n):

            # extract backbone atoms for both structures in this pair
            atoms_i = extract_backbone_atoms(structures[i])
            atoms_j = extract_backbone_atoms(structures[j])

            # superimpose structure j onto i and compute the RMSD
            superimposer = Superimposer()
            superimposer.set_atoms(atoms_i, atoms_j)
            rmsd_matrix[i, j] = round(superimposer.rms, 3)

    return rmsd_matrix


def make_short_label(ligand_prefix, structure):
    """
    Build a clean display label like 'cAMP_5' from a structure object.
    The raw ID contains '_model' suffixes and 'structure_' which are stripped for readability.
    """
    # remove the model suffix first, then remove the word 'structure_'
    base = structure.id.split("_model")[0].replace("structure_", "")

    return ligand_prefix + "_" + base


def build_top5_text(title, top5_rows):
    """
    Format the top 5 models and their IC50 scores into a text block for the plot annotation.
    """
    # start with the title and a separator line
    text = title + "\n" + "-" * 25 + "\n"

    for _, row in top5_rows.iterrows():

        # remove 'structure_' from the name to match the short label style used in the plot
        short_name = row["Name"].replace("structure_", "")

        score = row["Affinity (log10 IC50 µM)"]
        text += f"{short_name:<12}: {score:>6.3f}\n"

    return text

def combined_dataset(df1, df2, column_name):
    """
    This function reads two CSV files, normalizes the structure identifiers,
    prepends 'cAMP_' and 'cGMP_' prefixes, and 
    merges the results into a single DataFrame.
    """

    # load datasets
    camp_scores_df = pd.read_csv(df1)
    cgmp_scores_df = pd.read_csv(df2)

    # normalize structure naming for consistency
    camp_scores_df[column_name] = camp_scores_df[column_name].str.split("_model").str[0]
    cgmp_scores_df[column_name] = cgmp_scores_df[column_name].str.split("_model").str[0]

    # add ligand prefix to ensure unique identifiers
    camp_scores_df["Name"] = "cAMP_" + camp_scores_df[column_name]
    cgmp_scores_df["Name"] = "cGMP_" + cgmp_scores_df[column_name]

    # combine both datasets into one DataFrame for easier sorting and filtering
    combined_df = pd.concat([
        camp_scores_df[["Name", "Affinity (log10 IC50 µM)"]],
        cgmp_scores_df[["Name", "Affinity (log10 IC50 µM)"]],
    ])

    return combined_df

# creating a combined data set of two CSV files
ic50_df = combined_dataset("../results/GS_cAMPw_cGMP_Boltz2_scores.csv", "../results/GS_cGMPw_cGMP_Boltz2_scores.csv", "Structure")

# create a dictionary for IC50 score by structure name
ic50_by_name = dict(zip(ic50_df["Name"], ic50_df["Affinity (log10 IC50 µM)"]))

# sort structures by best binding (lowest IC50 first)
ic50_df_sorted = ic50_df.sort_values("Affinity (log10 IC50 µM)")

# select the 10 best structures based on lowest IC50 per ligand separately, then merge into one set
top10_camp_names = set(ic50_df_sorted[ic50_df_sorted["Name"].str.startswith("cAMP")].head(10)["Name"])
top10_cgmp_names = set(ic50_df_sorted[ic50_df_sorted["Name"].str.startswith("cGMP")].head(10)["Name"])
top10_all_names = top10_camp_names | top10_cgmp_names

# load all available structure files for each ligand type
all_camp_structures = load_structures("../results/GS_cAMPw_cGMP_boltz_models/*.cif", "cif")
all_cgmp_structures = load_structures("../results/GS_cGMPw_cGMP_boltz_models/*.cif", "cif")

def keep_top10_structures(name, top10, structures):
    """
    This function iterates through the provided structures, checks if their
    identifier matches any name in the top 10 set, and returns the filtered list.
    """

    top10_structures = []
    
    # keep only the structures whose name appears in the top 10 selection
    for s in structures:
        if name + s.id.split("_model")[0] in top10:
            top10_structures.append(s)

    return top10_structures

# get top 10 structures of each dataset
camp_structures = keep_top10_structures('cAMP_', top10_all_names, all_camp_structures)
cgmp_structures = keep_top10_structures('cGMP_', top10_all_names, all_cgmp_structures)

# combine into one list with cAMP structures first, then cGMP
selected_structures = camp_structures + cgmp_structures

# build short display labels in the same order as selected_structures
model_labels = []
for s in camp_structures:
    model_labels.append(make_short_label("cAMP", s))
for s in cgmp_structures:
    model_labels.append(make_short_label("cGMP", s))

# build an IC50 lookup using the short labels so scores can be retrieved by plot label
ic50_by_short_label = {}
for s in camp_structures:
    short_label = make_short_label("cAMP", s)
    original_name = "cAMP_" + s.id.split("_model")[0]
    ic50_by_short_label[short_label] = ic50_by_name[original_name]

for s in cgmp_structures:
    short_label = make_short_label("cGMP", s)
    original_name = "cGMP_" + s.id.split("_model")[0]
    ic50_by_short_label[short_label] = ic50_by_name[original_name]


# compute structural similarity matrix
rmsd_matrix = compute_pairwise_rmsd_matrix(selected_structures)

# convert to labeled dataframe for seaborn
rmsd_df = pd.DataFrame(rmsd_matrix, index=model_labels, columns=model_labels)

# assign ligand-based colors
ligand_colors = []
for name in model_labels:
    if name.startswith("cAMP"):
        ligand_colors.append("red")
    else:
        ligand_colors.append("blue")

# collect the IC50 score for each structure in the same order as model_labels
ic50_values = []
for name in model_labels:
    ic50_values.append(ic50_by_short_label[name])

ic50_colormap = plt.get_cmap("YlGn_r")

# the colormap expects values between 0 and 1, but IC50 scores are negative
# shifting by the minimum moves the range to start at 0, dividing by the new
# maximum scales it to [0, 1] while preserving the relative differences between scores
ic50_min = min(ic50_values)

ic50_shifted = []
for v in ic50_values:
    ic50_shifted.append(v - ic50_min)

ic50_max_shifted = max(ic50_shifted)

ic50_scaled = []
for v in ic50_shifted:
    ic50_scaled.append(v / ic50_max_shifted)

# map the scaled values to actual RGBA colors using the colormap
ic50_colors = ic50_colormap(ic50_scaled)

# the color strip DataFrame must share the same index as rmsd_df so seaborn aligns them correctly
color_strip_df = pd.DataFrame(
    {"Ligand": ligand_colors, "IC50": list(ic50_colors)},
    index=model_labels,
)

# get the 5 best scoring structures per ligand from the sorted DataFrame
top5_camp_rows = ic50_df_sorted[ic50_df_sorted["Name"].str.startswith("cAMP")].head(5)
top5_cgmp_rows = ic50_df_sorted[ic50_df_sorted["Name"].str.startswith("cGMP")].head(5)

# build the annotation text block for both ligands
top5_text = build_top5_text("IC50 of top 5 cAMPw_designs:", top5_camp_rows)
top5_text += "\n" + build_top5_text("IC50 of top 5 cGMPw_designs:", top5_cgmp_rows)

# clustermap inputs
clustermap = sns.clustermap(
    rmsd_df,
    xticklabels=model_labels,
    yticklabels=model_labels,
    cmap="rocket_r",
    linewidths=0.5,
    figsize=(13, 10),
    cbar_pos=None,              
    row_colors=color_strip_df,  
    col_colors=color_strip_df,  
)

# place the RMSD colorbar in the upper left; [left, bottom, width, height] in figure coordinates
rmsd_cbar_ax = clustermap.figure.add_axes([0.05, 0.75, 0.02, 0.12])
rmsd_cbar = clustermap.figure.colorbar(
    clustermap.ax_heatmap.collections[0],   
    cax=rmsd_cbar_ax,
    orientation="vertical",
)
rmsd_cbar.set_label("RMSD (Å)", fontsize=9, weight="bold")

# place the IC50 colorbar just below the RMSD colorbar
ic50_cbar_ax = clustermap.figure.add_axes([0.05, 0.58, 0.02, 0.12])
ic50_scalar_mappable = plt.cm.ScalarMappable(cmap=ic50_colormap)
ic50_scalar_mappable.set_array(ic50_values)

# set_clim makes the colorbar ticks show the real IC50 values, not the shifted/scaled ones
ic50_scalar_mappable.set_clim(vmin=min(ic50_values), vmax=max(ic50_values))
ic50_cbar = clustermap.figure.colorbar(ic50_scalar_mappable, cax=ic50_cbar_ax, orientation="vertical")
ic50_cbar.set_label("IC50 (log10 µM)", fontsize=9, weight="bold")


# place the annotation box in the bottom left corner of the figure
clustermap.figure.text(
    0.02, 0.15, top5_text,
    fontsize=8.5,
    fontfamily="monospace",
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", alpha=0.9),
)

# rotate x labels to 90 degrees so they read straight down like the Ligand/IC50 strip labels
plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90, ha="center", fontsize=9)
plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

# get the single best model name per ligand to know which label to make bold
best_camp_label = top5_camp_rows.iloc[0]["Name"].replace("structure_", "")
best_cgmp_label = top5_cgmp_rows.iloc[0]["Name"].replace("structure_", "")

# color tick labels by ligand type bold the best model per ligand
for tick in clustermap.ax_heatmap.get_xticklabels():
    label = tick.get_text()
    if label == best_camp_label or label == best_cgmp_label:
        if label.startswith("cAMP"):
            tick.set_color("red")
        else:
            tick.set_color("blue")
        tick.set_fontweight("bold")

for tick in clustermap.ax_heatmap.get_yticklabels():
    label = tick.get_text()
    if label == best_camp_label or label == best_cgmp_label:
        if label.startswith("cAMP"):
            tick.set_color("red")
        else:
            tick.set_color("blue")
        tick.set_fontweight("bold")

# place the ligand color legend on the left side below the IC50 colorbar
camp_patch = mpatches.Patch(color="red", label="cAMP")
cgmp_patch = mpatches.Patch(color="blue", label="cGMP")
clustermap.figure.legend(
    handles=[camp_patch, cgmp_patch],
    loc="upper left",
    bbox_to_anchor=(0.16, 0.85),  
    frameon=True,
    fontsize=11,                  
)

# layout
plt.suptitle(
    "Hierarchical Clustering of Top 10 cAMP & cGMP Structures (RMSD & IC50)",
    fontsize=14,
    weight="bold",
    y=0.96,
)

# adjust margins to leave room for the colorbars on the left and the dendrogram on top
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.25, right=0.85)
plt.show()