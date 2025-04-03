library(BiocManager)
library(TCGAbiolinks)
library(dplyr)

# Query TCGA-BLCA gene expression 
query_all_gene <- GDCquery(
  project = "TCGA-BLCA", 
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
)
gene_metadata <- query_all_gene %>% getResults
gene_barcodes <- gene_metadata$cases.submitter_id

# Query TCGA-BLCA WSI
query_all_wsi <- GDCquery(
  project = "TCGA-BLCA", 
  data.category = "Biospecimen", 
  data.type = "Slide Image",
  experimental.strategy = "Diagnostic Slide",
)
wsi_metadata <- query_all_wsi %>% getResults
wsi_barcodes <- wsi_metadata$cases

# Query TCGA-BLCA clinical data
query_all_clinical <- GDCquery(
  project = "TCGA-BLCA",
  data.category = "Clinical",
  data.type = "Clinical Supplement",
  data.format = "bcr xml"
)
clinical_metadata <- query_all_clinical %>% getResults
clinical_barcodes <- clinical_metadata$cases
print(length(clinical_barcodes))

# Find the matching barcodes between the two
matched_barcodes <- intersect(gene_barcodes, wsi_barcodes)
matched_gene_metadata <- gene_metadata[gene_barcodes %in% matched_barcodes, ]
matched_wsi_metadata <- wsi_metadata[wsi_barcodes %in% matched_barcodes, ]
matched_clinical_metadata <- clinical_metadata[clinical_barcodes %in% matched_barcodes, ]

print(dim(matched_gene_metadata))
print(dim(matched_wsi_metadata))
print(dim(matched_clinical_metadata))

# Download the .csv with the data
csv_path = "/blue/cis6930/reardons/data/TCGA2025"

write.csv(matched_gene_metadata, paste(csv_path, "/matched_gene_metadata.csv", sep=""), row.names = FALSE)
write.csv(matched_wsi_metadata, paste(csv_path, "/matched_wsi_metadata.csv", sep=""), row.names = FALSE)
write.csv(matched_clinical_metadata, paste(csv_path, "/matched_clinical_metadata.csv", sep=""), row.names = FALSE)
write.csv(matched_barcodes, paste(csv_path, "/matched_barcodes.csv", sep=""), row.names = FALSE)

