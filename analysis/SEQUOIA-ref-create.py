import os
import pandas as pd
import numpy as np

data_dir = "/blue/cis6930/reardons/data/TCGA2025/TCGA-BLCA/Transcriptome_Profiling/Gene_Expression_Quantification/"
matched_path = "/blue/cis6930/reardons/data/TCGA2025/matched_gene_metadata.csv"
matched_csv = pd.read_csv(matched_path)

matched_wsi_path = "/blue/cis6930/reardons/data/TCGA2025/matched_wsi_metadata.csv"
matched_wsi_csv = pd.read_csv(matched_wsi_path)

gene_list_csv_path = "/blue/tkahveci/reardons/CANCER-TRANS-2025/sequoia-pub/evaluation/gene_list.csv"
gene_list_csv = pd.read_csv(gene_list_csv_path, header=0)

seq_df = pd.DataFrame(columns=["wsi_file_name", "patient_id", gene_list_csv["gene"]])

def preprocess_tsv(tsv):
    tsv = tsv.drop([0,1,2,3]) # drop N_* columns
    tsv_drop_index = tsv[(tsv["gene_type"] != "protein_coding") & (tsv["gene_type"] != "miRNA") & (tsv["gene_type"] != "lncRNA")].index
    tsv = tsv.drop(tsv_drop_index)


for file in os.listdir(data_dir):
    data_tsv = pd.read_csv(data_dir + file, sep='\t', header=1, index_col=False)
    
    print("Dropping rows from " + file)
    data_tsv = preprocess_tsv(data_tsv)

    temp_df = data_tsv[["gene_name", "fpkm_uq_unstranded"]].set_index("gene_name")
    temp_df = temp_df.transpose()
    temp_df.reset_index(inplace=True, drop=True)

    # Grabbing the file id
    file_index = matched_csv["file_name"][matched_csv["file_name"] == file].index
    file_id_name = matched_csv["file_id"][file_index].values[0]

    # Trying to get the WSI file name
    case_submitter_id = matched_csv["cases.submitter_id"][file_index].values[0]
    cases_id_index = matched_wsi_csv["cases"][matched_wsi_csv["cases"] == case_submitter_id].index
    wsi_file_name = matched_wsi_csv["file_name"][cases_id_index].values[0]

    temp_df["patient_id"] = [file_id_name]
    temp_df["wsi_file_name"] = [wsi_file_name]
    temp_df = temp_df.set_index("wsi_file_name")
    



    