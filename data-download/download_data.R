library(BiocManager)
library(TCGAbiolinks)
library(dplyr)
library(DT)

csv_path = "/blue/cis6930/reardons/data/TCGA2025"

# gene_csv <- read.csv(paste(csv_path, "/matched_gene_metadata.csv", sep=""))
# wsi_csv <- read.csv(paste(csv_path, "/matched_wsi_metadata.csv", sep=""))
clinical_csv <- read.csv(paste(csv_path, "/matched_clinical_metadata.csv", sep=""))

# query_gene <- GDCquery(
#   project = "TCGA-BLCA", 
#   data.category = "Transcriptome Profiling",
#   data.type = "Gene Expression Quantification",
#   barcode = c(gene_csv$cases.submitter_id)
# )
  
# GDCdownload(query_gene, directory = csv_path)

# query_wsi <- GDCquery(
#   project = "TCGA-BLCA", 
#   data.category = "Biospecimen", 
#   data.type = "Slide Image",
#   experimental.strategy = "Diagnostic Slide",
#   barcode = c(wsi_csv$cases)
# )

query_clinical <- GDCquery(
  project = "TCGA-BLCA",
  data.category = "Clinical",
  data.type = "Clinical Supplement",
  data.format = "bcr xml",
  barcode = c(clinical_csv$cases)
)

GDCdownload(query_clinical, directory = csv_path)
