args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
if (length(script_path) > 0 && nzchar(script_path[1])) {
  setwd(dirname(normalizePath(script_path[1])))
} else if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

input_m6a_path <- "step01_scDART_hg38_WT_MAE_validated_m6a.rds"
input_transcript_path <- "step01_scDART_hg38_WT_MAE_validated_transcript.rds"
output_dir <- "step02_input"
run_id <- "no_site_filter_all135k"

selected_site_cap <- 135300L
heldout_fraction <- 0.20
expression_gene_cap <- 2000L
seed <- 42L
cell_min_coverage_sites <- 1500L
site_min_coverage_cells <- 0L
site_min_positive_cells <- 0L
min_total_for_coverage <- 20
min_total_for_positive <- 20
site_min_mean_total <- 0

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
invisible(mem.maxVSize(120000))

suppressPackageStartupMessages(library(SummarizedExperiment))
suppressPackageStartupMessages(library(GenomicRanges))

write_csv_plain <- function(df, path) {
  write.csv(df, path, row.names = FALSE, quote = FALSE)
}

matrix_to_cell_df <- function(mat, row_labels, col_labels) {
  df <- as.data.frame(t(mat), check.names = FALSE)
  colnames(df) <- row_labels
  data.frame(cell_id = col_labels, df, check.names = FALSE)
}

pick_gene_labels <- function(transcript_obj) {
  gene_rd <- as.data.frame(rowData(transcript_obj))
  fallback <- rownames(transcript_obj)
  if (is.null(fallback)) fallback <- paste0("gene_", seq_len(nrow(transcript_obj)))
  for (candidate in c("symbol", "gene_name", "gene_id", "gene")) {
    if (candidate %in% colnames(gene_rd)) {
      labels <- as.character(gene_rd[[candidate]])
      labels[is.na(labels) | labels == ""] <- fallback[is.na(labels) | labels == ""]
      return(labels)
    }
  }
  fallback
}

observed_row_sd <- function(mat, mask) {
  observed_n <- rowSums(mask)
  observed_sum <- rowSums(mat * mask)
  observed_mean <- observed_sum / pmax(observed_n, 1)
  observed_ss <- rowSums(((mat - observed_mean) ^ 2) * mask)
  sqrt(observed_ss / pmax(observed_n - 1, 1))
}

safe_col <- function(df, name, n, default = NA_character_) {
  if (name %in% colnames(df)) as.character(df[[name]]) else rep(default, n)
}

set.seed(seed)

validated_m6a <- readRDS(input_m6a_path)
required_assays <- c("m6A", "Total", "m6ASiteProb", "AdjPvalue")

site_gr <- rowRanges(validated_m6a)
m6a_mat <- assay(validated_m6a, "m6A")
total_mat <- assay(validated_m6a, "Total")
prob_mat <- assay(validated_m6a, "m6ASiteProb")
padj_mat <- assay(validated_m6a, "AdjPvalue")

site_ids <- rownames(validated_m6a)
if (is.null(site_ids) || any(site_ids == "")) {
  site_ids <- paste0(
    as.character(seqnames(site_gr)), ":",
    start(site_gr), "-",
    end(site_gr), ":",
    as.character(strand(site_gr))
  )
}
cell_ids_all <- colnames(validated_m6a)

coverage_all <- total_mat >= min_total_for_coverage
cell_coverage_sites_all <- colSums(coverage_all)
cell_keep <- cell_coverage_sites_all >= cell_min_coverage_sites

cell_ids <- cell_ids_all[cell_keep]
m6a_mat <- m6a_mat[, cell_keep, drop = FALSE]
total_mat <- total_mat[, cell_keep, drop = FALSE]
prob_mat <- prob_mat[, cell_keep, drop = FALSE]
padj_mat <- padj_mat[, cell_keep, drop = FALSE]

coverage_mask <- total_mat >= min_total_for_coverage
positive_prob_mask <- (prob_mat > 0.5) & (total_mat >= min_total_for_positive)
padj_sig_mask <- (padj_mat < 0.05) & (total_mat >= min_total_for_positive)
significance_mask <- positive_prob_mask & padj_sig_mask

significance_mask[is.na(significance_mask)] <- FALSE
positive_prob_mask[is.na(positive_prob_mask)] <- FALSE
padj_sig_mask[is.na(padj_sig_mask)] <- FALSE

ratio_mat <- matrix(0, nrow = nrow(m6a_mat), ncol = ncol(m6a_mat), dimnames = list(site_ids, cell_ids))
ratio_mat[coverage_mask] <- m6a_mat[coverage_mask] / total_mat[coverage_mask]

coverage_cells <- rowSums(coverage_mask)
positive_prob_cells <- rowSums(positive_prob_mask)
padj_sig_cells <- rowSums(padj_sig_mask)
significant_cells <- rowSums(significance_mask)
mean_total_reads <- rowMeans(total_mat)
mean_ratio_detected <- rowSums(ratio_mat * coverage_mask) / pmax(coverage_cells, 1)
ratio_sd_detected <- observed_row_sd(ratio_mat, coverage_mask)

candidate_idx <- which(
  coverage_cells >= site_min_coverage_cells &
    significant_cells >= site_min_positive_cells &
    mean_total_reads >= site_min_mean_total
)

ranked_idx <- candidate_idx[order(
  significant_cells[candidate_idx],
  positive_prob_cells[candidate_idx],
  ratio_sd_detected[candidate_idx],
  coverage_cells[candidate_idx],
  mean_total_reads[candidate_idx],
  decreasing = TRUE
)]

selected_site_count <- min(selected_site_cap, length(ranked_idx))
selected_idx <- ranked_idx[seq_len(selected_site_count)]
selected_site_ids <- site_ids[selected_idx]

selected_m6a_mat <- m6a_mat[selected_idx, , drop = FALSE]
selected_total_mat <- total_mat[selected_idx, , drop = FALSE]
selected_prob_mat <- prob_mat[selected_idx, , drop = FALSE]
selected_padj_mat <- padj_mat[selected_idx, , drop = FALSE]
selected_ratio_mat <- ratio_mat[selected_idx, , drop = FALSE]
selected_coverage_mask <- coverage_mask[selected_idx, , drop = FALSE]
selected_positive_prob_mask <- positive_prob_mask[selected_idx, , drop = FALSE]
selected_padj_sig_mask <- padj_sig_mask[selected_idx, , drop = FALSE]
selected_significance_mask <- significance_mask[selected_idx, , drop = FALSE]

heldout_mask <- matrix(FALSE, nrow = selected_site_count, ncol = length(cell_ids), dimnames = list(selected_site_ids, cell_ids))
for (cell_idx in seq_along(cell_ids)) {
  observed_idx <- which(selected_coverage_mask[, cell_idx])
  if (length(observed_idx) >= 2) {
    hold_n <- max(1L, floor(length(observed_idx) * heldout_fraction))
    hold_n <- min(hold_n, length(observed_idx) - 1L)
    heldout_mask[sample(observed_idx, size = hold_n), cell_idx] <- TRUE
  }
}

train_mask <- selected_coverage_mask & !heldout_mask
empty_train_sites <- which(rowSums(train_mask) == 0 & rowSums(selected_coverage_mask) > 0)
for (site_idx in empty_train_sites) {
  held_cells <- which(heldout_mask[site_idx, ])
  if (length(held_cells) > 0) heldout_mask[site_idx, held_cells[1]] <- FALSE
}
train_mask <- selected_coverage_mask & !heldout_mask

cell_group <- ifelse(
  grepl("^YTHmut_", cell_ids),
  "YTHmut_control",
  ifelse(grepl("^YTH_", cell_ids), "YTH_test", "unknown")
)

site_rd <- as.data.frame(rowData(validated_m6a))
site_metadata <- data.frame(
  site_id = selected_site_ids,
  selected_rank = seq_len(selected_site_count),
  coverage_cells = coverage_cells[selected_idx],
  positive_prob_cells = positive_prob_cells[selected_idx],
  padj_significant_cells = padj_sig_cells[selected_idx],
  significant_cells = significant_cells[selected_idx],
  mean_total_reads = mean_total_reads[selected_idx],
  mean_ratio_detected = mean_ratio_detected[selected_idx],
  ratio_sd_detected = ratio_sd_detected[selected_idx],
  train_observed_cells = rowSums(train_mask),
  heldout_observed_cells = rowSums(heldout_mask),
  heldout_significant_cells = rowSums(heldout_mask & selected_significance_mask),
  orthogonal_validation_status = safe_col(site_rd, "orthogonal_validation_status", nrow(site_rd))[selected_idx],
  reference_site_source = safe_col(site_rd, "reference_site_source", nrow(site_rd))[selected_idx],
  seqnames = as.character(seqnames(site_gr)[selected_idx]),
  start = start(site_gr)[selected_idx],
  end = end(site_gr)[selected_idx],
  strand = as.character(strand(site_gr)[selected_idx]),
  stringsAsFactors = FALSE
)

cell_metadata <- data.frame(
  cell_id = cell_ids,
  cell_group = cell_group,
  cell_index = seq_along(cell_ids),
  original_cell_coverage_sites = cell_coverage_sites_all[cell_keep],
  detected_sites = colSums(selected_coverage_mask),
  significant_sites = colSums(selected_significance_mask),
  train_observed_sites = colSums(train_mask),
  heldout_observed_sites = colSums(heldout_mask),
  heldout_significant_sites = colSums(heldout_mask & selected_significance_mask),
  total_reads = colSums(selected_total_mat),
  methylated_reads = colSums(selected_m6a_mat),
  mean_ratio_detected = colSums(selected_ratio_mat * selected_coverage_mask) / pmax(colSums(selected_coverage_mask), 1),
  stringsAsFactors = FALSE
)

transcript_obj <- readRDS(input_transcript_path)
transcript_assay_names <- assayNames(transcript_obj)
expression_assay_name <- if ("RPKM" %in% transcript_assay_names) "RPKM" else if ("ReadCounts" %in% transcript_assay_names) "ReadCounts" else transcript_assay_names[[1]]
expression_mat <- assay(transcript_obj, expression_assay_name)
expression_mat <- expression_mat[, cell_ids, drop = FALSE]

gene_labels_raw <- pick_gene_labels(transcript_obj)
log_expression <- log1p(expression_mat)
gene_variance <- apply(log_expression, 1, var)
anchor_gene_count <- min(expression_gene_cap, nrow(log_expression))
anchor_gene_idx <- order(gene_variance, decreasing = TRUE)[seq_len(anchor_gene_count)]
anchor_gene_labels <- make.unique(gene_labels_raw[anchor_gene_idx])

expression_anchor_df <- as.data.frame(t(log_expression[anchor_gene_idx, , drop = FALSE]), check.names = FALSE)
colnames(expression_anchor_df) <- anchor_gene_labels
expression_anchor_df <- data.frame(cell_id = cell_ids, expression_anchor_df, check.names = FALSE)

canonical_regulators <- c(
  "METTL3", "METTL14", "WTAP", "VIRMA", "RBM15", "RBM15B", "ZC3H13",
  "ALKBH5", "FTO", "YTHDF1", "YTHDF2", "YTHDF3", "YTHDC1", "YTHDC2",
  "HNRNPC", "HNRNPA2B1", "IGF2BP1", "IGF2BP2", "IGF2BP3"
)

upper_gene_labels <- toupper(gene_labels_raw)
available_regulators <- Filter(function(gene) any(upper_gene_labels == gene), canonical_regulators)

regulator_df <- data.frame(cell_id = cell_ids, stringsAsFactors = FALSE)
for (gene in available_regulators) {
  gene_idx <- which(upper_gene_labels == gene)
  if (length(gene_idx) > 1) gene_idx <- gene_idx[which.max(gene_variance[gene_idx])]
  regulator_df[[gene]] <- as.numeric(log_expression[gene_idx, ])
}

write_csv_plain(matrix_to_cell_df(selected_ratio_mat, selected_site_ids, cell_ids), file.path(output_dir, "step02_ratio_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_m6a_mat, selected_site_ids, cell_ids), file.path(output_dir, "step02_m6a_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_total_mat, selected_site_ids, cell_ids), file.path(output_dir, "step02_total_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_prob_mat, selected_site_ids, cell_ids), file.path(output_dir, "step02_m6a_probability_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_padj_mat, selected_site_ids, cell_ids), file.path(output_dir, "step02_adj_pvalue_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_coverage_mask * 1L, selected_site_ids, cell_ids), file.path(output_dir, "step02_coverage_mask_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_positive_prob_mask * 1L, selected_site_ids, cell_ids), file.path(output_dir, "step02_positive_probability_mask_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_padj_sig_mask * 1L, selected_site_ids, cell_ids), file.path(output_dir, "step02_padj_significance_mask_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(selected_significance_mask * 1L, selected_site_ids, cell_ids), file.path(output_dir, "step02_significance_mask_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(train_mask * 1L, selected_site_ids, cell_ids), file.path(output_dir, "step02_train_mask_cells_by_sites.csv"))
write_csv_plain(matrix_to_cell_df(heldout_mask * 1L, selected_site_ids, cell_ids), file.path(output_dir, "step02_heldout_mask_cells_by_sites.csv"))
write_csv_plain(site_metadata, file.path(output_dir, "step02_site_metadata.csv"))
write_csv_plain(cell_metadata, file.path(output_dir, "step02_cell_metadata.csv"))
write_csv_plain(expression_anchor_df, file.path(output_dir, "step02_expression_anchor_cells_by_genes.csv"))
write_csv_plain(regulator_df, file.path(output_dir, "step02_regulator_expression_by_cell.csv"))

summary_lines <- c(
  "step=02_improved",
  paste0("run_id=", run_id),
  paste0("input_m6a=", input_m6a_path),
  paste0("input_transcript=", input_transcript_path),
  paste0("output_dir=", output_dir),
  "site_rule=coverage_plus_sample_level_significance_plus_variability",
  paste0("selected_site_cap=", selected_site_cap),
  paste0("selected_site_count=", selected_site_count),
  paste0("candidate_site_count=", length(candidate_idx)),
  paste0("cells_before_qc=", length(cell_ids_all)),
  paste0("cells_after_qc=", length(cell_ids)),
  paste0("cell_min_coverage_sites=", cell_min_coverage_sites),
  paste0("site_min_coverage_cells=", site_min_coverage_cells),
  paste0("site_min_positive_cells=", site_min_positive_cells),
  paste0("min_total_for_coverage=", min_total_for_coverage),
  paste0("min_total_for_positive=", min_total_for_positive),
  paste0("site_min_mean_total=", site_min_mean_total),
  paste0("coverage_fraction_selected=", round(mean(selected_coverage_mask), 6)),
  paste0("significance_fraction_selected=", round(mean(selected_significance_mask), 6)),
  paste0("heldout_fraction_target=", heldout_fraction),
  paste0("heldout_observed_entries=", sum(heldout_mask)),
  paste0("heldout_significant_entries=", sum(heldout_mask & selected_significance_mask)),
  paste0("train_observed_entries=", sum(train_mask)),
  paste0("expression_assay=", expression_assay_name),
  paste0("expression_anchor_gene_count=", anchor_gene_count),
  paste0("available_regulators=", paste(available_regulators, collapse = ",")),
  paste0("seed=", seed)
)
writeLines(summary_lines, con = file.path(output_dir, "step02_summary.txt"))
