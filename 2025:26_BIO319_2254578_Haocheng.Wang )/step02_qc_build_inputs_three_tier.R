#!/usr/bin/env Rscript

# step02_qc_build_inputs_observe5_three_tier.R
# Full-rebuild Step 2 for FYPmain-4.
# Purpose:
#   Build coverage-aware, relaxed observed-entry-centered model inputs from the fixed Step 1 RDS files.
#   This script does NOT modify Step 1, does NOT destructively filter validated sites,
#   and does NOT treat unobserved/low-coverage entries as zero methylation.
#
# Expected working directory:
#   The same folder that contains:
#     step01_scDART_hg38_WT_MAE_validated_m6a.rds
#     step01_scDART_hg38_WT_MAE_validated_transcript.rds
#
# Run:
#   cd /path/to/FYPmain-4
#   Rscript step02_qc_build_inputs_observe5_three_tier.R

options(stringsAsFactors = FALSE)
options(warn = 1)

get_script_dir <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- cmd[grepl("^--file=", cmd)]
  if (length(file_arg) > 0L) {
    return(normalizePath(dirname(sub("^--file=", "", file_arg[[1L]])), mustWork = TRUE))
  }
  normalizePath(".", mustWork = TRUE)
}

is_absolute_path <- function(path) {
  grepl("^(/|~)", path) || grepl("^[A-Za-z]:[/\\\\]", path)
}

resolve_from_script <- function(path, script_dir, mustWork = FALSE) {
  if (!is_absolute_path(path)) path <- file.path(script_dir, path)
  normalizePath(path, mustWork = mustWork)
}

SCRIPT_DIR <- get_script_dir()

# -----------------------------
# 0. Parameters
# -----------------------------
PARAMS <- list(
  expected_m6a_file = "step01_scDART_hg38_WT_MAE_validated_m6a.rds",
  expected_tx_file  = "step01_scDART_hg38_WT_MAE_validated_transcript.rds",
  output_dir = "step2-output",
  report_dir = "step2-output",
  # Three-tier Step2 v3 thresholds:
  #   observed entries: Total >= 5
  #   candidate m6A entries: Total >= 5 AND (m6ASiteProb >= 0.5 OR AdjPvalue < 0.05)
  #   high-confidence significant entries: Total >= 5 AND m6ASiteProb >= 0.5 AND AdjPvalue < 0.05
  #
  # Intended use:
  #   observed_entries.tsv.gz      -> reconstruction / denoising training
  #   candidate_m6a_entries.tsv.gz -> optional soft-gate / soft-evidence supervision
  #   significant_entries.tsv.gz   -> high-confidence pseudo-labels for Step3 significance learning and final evaluation
  min_total_observed = 5L,
  min_total_candidate = 5L,
  min_total_significant = 5L,
  prob_positive = 0.5,
  padj_significant = 0.05,
  total_cap_for_loss = 100,
  epsilon = 1e-6,
  eb_alpha = 1.0,
  n_anchor_genes = 2000L,
  dense_panel_cap = NA_integer_,
  cell_filter_tukey_k = 1.5,
  min_model_panel_sites = 1000L,
  chunk_cells = 20L,
  canonical_regulators = c(
    "METTL3", "METTL14", "WTAP", "VIRMA", "KIAA1429", "RBM15", "RBM15B", "ZC3H13",
    "FTO", "ALKBH5", "YTHDF1", "YTHDF2", "YTHDF3", "YTHDC1", "YTHDC2",
    "HNRNPC", "HNRNPA2B1", "IGF2BP1", "IGF2BP2", "IGF2BP3"
  )
)

PARAMS$output_dir <- resolve_from_script(PARAMS$output_dir, SCRIPT_DIR, mustWork = FALSE)
PARAMS$report_dir <- resolve_from_script(PARAMS$report_dir, SCRIPT_DIR, mustWork = FALSE)

# -----------------------------
# 1. Package loading
# -----------------------------
required_pkgs <- c(
  "SummarizedExperiment", "GenomicRanges", "S4Vectors",
  "data.table", "matrixStats", "jsonlite"
)

missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0L) {
  dir.create(PARAMS$report_dir, recursive = TRUE, showWarnings = FALSE)
  dep_file <- file.path(PARAMS$report_dir, "dependency_check.md")
  writeLines(c(
    "# Dependency check failed",
    "",
    "The following R packages are missing:",
    paste0("- ", missing_pkgs),
    "",
    "Install example:",
    "",
    "```r",
    "install.packages(c('data.table', 'matrixStats', 'jsonlite'))",
    "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')",
    "BiocManager::install(c('SummarizedExperiment', 'GenomicRanges', 'S4Vectors'))",
    "```"
  ), dep_file)
  stop("Missing required packages: ", paste(missing_pkgs, collapse = ", "),
       ". See ", dep_file, call. = FALSE)
}

suppressPackageStartupMessages({
  library(SummarizedExperiment)
  library(GenomicRanges)
  library(S4Vectors)
  library(data.table)
  library(matrixStats)
  library(jsonlite)
})

# -----------------------------
# 2. Helper functions
# -----------------------------
log_msg <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse = ""))
}

as_scalar_character <- function(x) {
  if (length(x) == 0L) return(NA_character_)
  if (length(x) == 1L) return(as.character(x))
  paste(as.character(x), collapse = ";")
}

sanitize_df <- function(df) {
  # Convert S4 DataFrame/data.frame to data.table and collapse list columns.
  if (is.null(df)) return(data.table())
  out <- as.data.frame(df, stringsAsFactors = FALSE, optional = TRUE)
  if (ncol(out) == 0L) return(data.table())
  for (nm in names(out)) {
    if (is.list(out[[nm]]) && !is.data.frame(out[[nm]])) {
      out[[nm]] <- vapply(out[[nm]], as_scalar_character, character(1))
    } else if (inherits(out[[nm]], "Rle")) {
      out[[nm]] <- as.vector(out[[nm]])
    } else {
      out[[nm]] <- as.character(out[[nm]])
      out[[nm]][out[[nm]] == "NA"] <- NA_character_
    }
  }
  as.data.table(out)
}

locate_file <- function(workdir, expected_name, fallback_regex) {
  exact <- file.path(workdir, expected_name)
  if (file.exists(exact)) return(normalizePath(exact))

  candidates <- list.files(workdir, pattern = fallback_regex, full.names = TRUE, ignore.case = TRUE)
  candidates <- candidates[grepl("\\.rds$", candidates, ignore.case = TRUE)]
  if (length(candidates) != 1L) {
    stop(
      "Cannot uniquely locate Step 1 RDS for pattern '", fallback_regex, "'.\n",
      "Expected exact file: ", expected_name, "\n",
      "Candidates found: ", ifelse(length(candidates) == 0L, "<none>", paste(basename(candidates), collapse = ", ")),
      call. = FALSE
    )
  }
  normalizePath(candidates)
}

require_assays <- function(se, required, label) {
  available <- SummarizedExperiment::assayNames(se)
  missing <- setdiff(required, available)
  if (length(missing) > 0L) {
    stop(label, " missing assay(s): ", paste(missing, collapse = ", "),
         ". Available assays: ", paste(available, collapse = ", "), call. = FALSE)
  }
}

safe_qlogis <- function(p, eps = 1e-6) {
  qlogis(pmin(pmax(p, eps), 1 - eps))
}

safe_row_var <- function(mat) {
  # matrixStats::rowVars handles NA with na.rm=TRUE, but rows with <2 valid values should be NA.
  v <- matrixStats::rowVars(mat, na.rm = TRUE)
  valid_n <- rowSums(!is.na(mat))
  v[valid_n < 2L] <- NA_real_
  v
}

make_site_id <- function(m6a_se) {
  rn <- rownames(m6a_se)
  if (!is.null(rn) && length(rn) == nrow(m6a_se) && all(!is.na(rn)) && length(unique(rn)) == length(rn)) {
    return(list(site_id = rn, rebuilt = FALSE, warning = NULL))
  }
  rr <- rowRanges(m6a_se)
  if (length(rr) != nrow(m6a_se)) stop("rowRanges length does not match number of m6A rows.", call. = FALSE)
  seqs <- as.character(seqnames(rr))
  starts <- start(rr)
  ends <- end(rr)
  strands <- as.character(strand(rr))
  if (all(starts == ends, na.rm = TRUE)) {
    site_id <- paste0(seqs, ":", starts, ":", strands)
  } else {
    site_id <- paste0(seqs, ":", starts, "-", ends, ":", strands)
  }
  if (anyDuplicated(site_id)) {
    stop("Rebuilt site_id is not unique. Check rowRanges.", call. = FALSE)
  }
  list(site_id = site_id, rebuilt = TRUE, warning = "m6A rownames were missing/invalid; site_id rebuilt from rowRanges.")
}

get_gene_labels <- function(tx_se) {
  rd <- sanitize_df(rowData(tx_se))
  n <- nrow(tx_se)
  fallback <- rownames(tx_se)
  if (is.null(fallback) || length(fallback) != n) fallback <- paste0("gene_", seq_len(n))

  label <- rep(NA_character_, n)
  for (field in c("symbol", "gene_name", "gene_id", "gene")) {
    if (field %in% names(rd)) {
      x <- as.character(rd[[field]])
      x[x == "" | is.na(x)] <- NA_character_
      label[is.na(label)] <- x[is.na(label)]
    }
  }
  label[is.na(label)] <- fallback[is.na(label)]
  make.unique(label)
}

get_match_labels_upper <- function(tx_se) {
  rd <- sanitize_df(rowData(tx_se))
  n <- nrow(tx_se)
  vals <- vector("list", n)
  for (i in seq_len(n)) vals[[i]] <- character(0)
  for (field in c("symbol", "gene_name", "gene_id", "gene")) {
    if (field %in% names(rd)) {
      x <- as.character(rd[[field]])
      x[is.na(x)] <- ""
      for (i in seq_len(n)) {
        if (nzchar(x[i])) vals[[i]] <- unique(c(vals[[i]], toupper(x[i])))
      }
    }
  }
  rn <- rownames(tx_se)
  if (!is.null(rn) && length(rn) == n) {
    for (i in seq_len(n)) vals[[i]] <- unique(c(vals[[i]], toupper(rn[i])))
  }
  vals
}

compress_tsv_if_possible <- function(path_tsv, warnings_env) {
  gz_path <- paste0(path_tsv, ".gz")
  if (file.exists(gz_path)) file.remove(gz_path)
  gz <- Sys.which("gzip")
  if (nzchar(gz)) {
    status <- system2(gz, args = c("-f", path_tsv))
    if (status == 0L && file.exists(gz_path)) return(gz_path)
  }
  warnings_env$warnings <- c(warnings_env$warnings, paste0("gzip command unavailable/failed; left uncompressed file: ", path_tsv))
  path_tsv
}

append_dt <- function(dt, path) {
  data.table::fwrite(dt, file = path, sep = "\t", append = file.exists(path), col.names = !file.exists(path))
}

# -----------------------------
# 3. Main script
# -----------------------------
main <- function(workdir = ".") {
  workdir <- normalizePath(workdir, mustWork = TRUE)
  dir.create(PARAMS$output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(PARAMS$report_dir, recursive = TRUE, showWarnings = FALSE)

  warn_env <- new.env(parent = emptyenv())
  warn_env$warnings <- character(0)

  log_msg("Locating Step 1 RDS files in: ", workdir)
  m6a_path <- locate_file(workdir, PARAMS$expected_m6a_file, "validated.*m6a|m6a.*validated")
  tx_path  <- locate_file(workdir, PARAMS$expected_tx_file,  "validated.*transcript|transcript.*validated")

  log_msg("Reading m6A RDS: ", basename(m6a_path))
  m6a_se <- readRDS(m6a_path)
  log_msg("Reading transcript RDS: ", basename(tx_path))
  tx_se <- readRDS(tx_path)

  if (!inherits(m6a_se, "SummarizedExperiment")) stop("m6A RDS is not a SummarizedExperiment/RangedSummarizedExperiment.", call. = FALSE)
  if (!inherits(tx_se, "SummarizedExperiment")) stop("Transcript RDS is not a SummarizedExperiment/RangedSummarizedExperiment.", call. = FALSE)

  require_assays(m6a_se, c("m6A", "Total", "m6ASiteProb", "AdjPvalue"), "m6A RDS")
  if (!any(c("RPKM", "ReadCounts") %in% assayNames(tx_se))) {
    stop("Transcript RDS must contain at least one of assays: RPKM, ReadCounts. Available: ",
         paste(assayNames(tx_se), collapse = ", "), call. = FALSE)
  }

  if (!setequal(colnames(m6a_se), colnames(tx_se))) {
    stop("m6A and transcript RDS do not contain the same cell IDs.", call. = FALSE)
  }
  if (!identical(colnames(m6a_se), colnames(tx_se))) {
    log_msg("Transcript columns have same cell IDs but different order; reordering transcript to match m6A.")
    tx_se <- tx_se[, colnames(m6a_se)]
    warn_env$warnings <- c(warn_env$warnings, "Transcript columns reordered to match m6A colnames.")
  }

  n_sites <- nrow(m6a_se)
  n_cells <- ncol(m6a_se)
  site_id_info <- make_site_id(m6a_se)
  site_id <- site_id_info$site_id
  if (isTRUE(site_id_info$rebuilt)) warn_env$warnings <- c(warn_env$warnings, site_id_info$warning)
  cell_id <- colnames(m6a_se)

  log_msg("Input dimensions: ", n_sites, " sites x ", n_cells, " cells")

  # Extract assays. Keep original direction: sites x cells.
  Y <- assay(m6a_se, "m6A")
  N <- assay(m6a_se, "Total")
  Prob <- assay(m6a_se, "m6ASiteProb")
  Padj <- assay(m6a_se, "AdjPvalue")

  # Basic numeric checks.
  if (!all(dim(Y) == c(n_sites, n_cells))) stop("m6A assay dimension mismatch.", call. = FALSE)
  if (!all(dim(N) == c(n_sites, n_cells))) stop("Total assay dimension mismatch.", call. = FALSE)
  if (any(N < 0, na.rm = TRUE) || any(Y < 0, na.rm = TRUE)) stop("Negative count values detected in m6A/Total assays.", call. = FALSE)
  if (any(Y > N, na.rm = TRUE)) {
    warn_env$warnings <- c(warn_env$warnings, "Some m6A counts are greater than Total coverage; downstream ratios are clamped through pmax(N, 1). Please inspect source data.")
  }

  log_msg("Building three-tier observed, candidate, and high-confidence significant masks")
  obs <- N >= PARAMS$min_total_observed
  obs[is.na(obs)] <- FALSE

  candidate_cov <- N >= PARAMS$min_total_candidate
  candidate_cov[is.na(candidate_cov)] <- FALSE
  prob_pos_candidate <- candidate_cov & !is.na(Prob) & Prob >= PARAMS$prob_positive
  prob_pos_candidate[is.na(prob_pos_candidate)] <- FALSE
  padj_sig_candidate <- candidate_cov & !is.na(Padj) & Padj < PARAMS$padj_significant
  padj_sig_candidate[is.na(padj_sig_candidate)] <- FALSE

  cand <- candidate_cov & (prob_pos_candidate | padj_sig_candidate)
  cand[is.na(cand)] <- FALSE

  significant_cov <- N >= PARAMS$min_total_significant
  significant_cov[is.na(significant_cov)] <- FALSE
  prob_pos_sig <- significant_cov & !is.na(Prob) & Prob >= PARAMS$prob_positive
  prob_pos_sig[is.na(prob_pos_sig)] <- FALSE
  padj_sig_strict <- significant_cov & !is.na(Padj) & Padj < PARAMS$padj_significant
  padj_sig_strict[is.na(padj_sig_strict)] <- FALSE

  sig <- significant_cov & prob_pos_sig & padj_sig_strict
  sig[is.na(sig)] <- FALSE

  n_obs <- sum(obs)
  n_cand <- sum(cand)
  n_sig <- sum(sig)
  if (n_obs <= 0L) stop("No observed entries found under Total >= ", PARAMS$min_total_observed, call. = FALSE)
  if (n_cand <= 0L) warn_env$warnings <- c(warn_env$warnings, "No candidate m6A entries found under Total >= 5 OR-evidence rule.")
  if (n_sig <= 0L) stop("No high-confidence significant entries found under Total >= ", PARAMS$min_total_significant,
                         ", prob >= ", PARAMS$prob_positive,
                         " and padj < ", PARAMS$padj_significant, call. = FALSE)

  log_msg("Computing QC summaries")
  observed_sites <- colSums(obs)
  candidate_sites <- colSums(cand)
  significant_sites <- colSums(sig)
  total_coverage_sum_cell <- colSums(N, na.rm = TRUE)
  mean_total_observed <- colSums(N * obs, na.rm = TRUE) / pmax(observed_sites, 1)
  median_total_observed <- vapply(seq_len(n_cells), function(j) {
    v <- N[, j]
    m <- obs[, j]
    if (!any(m)) return(NA_real_)
    as.numeric(stats::median(v[m], na.rm = TRUE))
  }, numeric(1))
  cell_qc <- data.table(
    cell_idx = seq_len(n_cells) - 1L,
    cell_id = cell_id,
    observed_sites = as.integer(observed_sites),
    candidate_sites = as.integer(candidate_sites),
    significant_sites = as.integer(significant_sites),
    total_coverage_sum = as.numeric(total_coverage_sum_cell),
    mean_total_observed = as.numeric(mean_total_observed),
    median_total_observed = as.numeric(median_total_observed),
    pct_candidate_among_observed = as.numeric(candidate_sites / pmax(observed_sites, 1)),
    pct_significant_among_observed = as.numeric(significant_sites / pmax(observed_sites, 1))
  )

  # Conservative two-dimensional cell-level QC trimming.
  # The filter is label-independent and operates on log10-transformed coverage
  # metrics. A cell is removed only if both its observed m6A site breadth and
  # total m6A coverage are lower-tail Tukey outliers. This avoids removing cells
  # with low observed-site breadth but adequate sequencing depth.
  safe_log10_metric <- function(x) log10(pmax(as.numeric(x), 1))
  log_observed_sites <- safe_log10_metric(cell_qc$observed_sites)
  log_total_coverage <- safe_log10_metric(cell_qc$total_coverage_sum)
  q1_log_observed_sites <- as.numeric(stats::quantile(log_observed_sites, probs = 0.25, na.rm = TRUE, names = FALSE))
  q3_log_observed_sites <- as.numeric(stats::quantile(log_observed_sites, probs = 0.75, na.rm = TRUE, names = FALSE))
  iqr_log_observed_sites <- q3_log_observed_sites - q1_log_observed_sites
  q1_log_total_coverage <- as.numeric(stats::quantile(log_total_coverage, probs = 0.25, na.rm = TRUE, names = FALSE))
  q3_log_total_coverage <- as.numeric(stats::quantile(log_total_coverage, probs = 0.75, na.rm = TRUE, names = FALSE))
  iqr_log_total_coverage <- q3_log_total_coverage - q1_log_total_coverage
  cell_filter_threshold_log_observed_sites <- q1_log_observed_sites - PARAMS$cell_filter_tukey_k * iqr_log_observed_sites
  cell_filter_threshold_log_total_coverage <- q1_log_total_coverage - PARAMS$cell_filter_tukey_k * iqr_log_total_coverage
  cell_filter_threshold_observed_sites <- 10 ^ cell_filter_threshold_log_observed_sites
  cell_filter_threshold_total_coverage <- 10 ^ cell_filter_threshold_log_total_coverage
  low_quality_cells <- cell_qc[
    observed_sites < cell_filter_threshold_observed_sites &
      total_coverage_sum < cell_filter_threshold_total_coverage
  ]
  keep_cells <- !(cell_id %in% low_quality_cells$cell_id)
  if (sum(keep_cells) < 3L) {
    stop("Cell QC filtering retained fewer than three cells; check cell_filter_tukey_k.", call. = FALSE)
  }
  cell_qc[, cell_qc_status := fifelse(cell_id %in% low_quality_cells$cell_id, "removed_low_observed_and_total_coverage", "retained")]
  cell_qc[, cell_filter_threshold_observed_sites := cell_filter_threshold_observed_sites]
  cell_qc[, cell_filter_threshold_total_coverage_sum := cell_filter_threshold_total_coverage]
  cell_qc[, cell_filter_threshold_log10_observed_sites := cell_filter_threshold_log_observed_sites]
  cell_qc[, cell_filter_threshold_log10_total_coverage_sum := cell_filter_threshold_log_total_coverage]
  cell_qc[, cell_filter_tukey_k := PARAMS$cell_filter_tukey_k]
  cell_qc_all <- copy(cell_qc)
  cell_qc_removed <- copy(low_quality_cells)
  if (nrow(cell_qc_removed) > 0L) {
    cell_qc_removed[, cell_qc_status := "removed_low_observed_and_total_coverage"]
    cell_qc_removed[, cell_filter_threshold_observed_sites := cell_filter_threshold_observed_sites]
    cell_qc_removed[, cell_filter_threshold_total_coverage_sum := cell_filter_threshold_total_coverage]
    cell_qc_removed[, cell_filter_threshold_log10_observed_sites := cell_filter_threshold_log_observed_sites]
    cell_qc_removed[, cell_filter_threshold_log10_total_coverage_sum := cell_filter_threshold_log_total_coverage]
    cell_qc_removed[, cell_filter_tukey_k := PARAMS$cell_filter_tukey_k]
  }
  log_msg(
    "Cell QC log10 two-metric Tukey AND filter: observed_sites threshold=", round(cell_filter_threshold_observed_sites, 2),
    ", total_coverage_sum threshold=", round(cell_filter_threshold_total_coverage, 2),
    ", k=", PARAMS$cell_filter_tukey_k,
    ", removed=", nrow(cell_qc_removed), "/", n_cells, " cells"
  )

  if (nrow(cell_qc_removed) > 0L) {
    m6a_se <- m6a_se[, keep_cells]
    tx_se <- tx_se[, keep_cells]
    Y <- Y[, keep_cells, drop = FALSE]
    N <- N[, keep_cells, drop = FALSE]
    Prob <- Prob[, keep_cells, drop = FALSE]
    Padj <- Padj[, keep_cells, drop = FALSE]
    obs <- obs[, keep_cells, drop = FALSE]
    cand <- cand[, keep_cells, drop = FALSE]
    sig <- sig[, keep_cells, drop = FALSE]
    cell_id <- cell_id[keep_cells]
    n_cells_before_filter <- n_cells
    n_cells <- length(cell_id)

    observed_sites <- colSums(obs)
    candidate_sites <- colSums(cand)
    significant_sites <- colSums(sig)
    total_coverage_sum_cell <- colSums(N, na.rm = TRUE)
    mean_total_observed <- colSums(N * obs, na.rm = TRUE) / pmax(observed_sites, 1)
    median_total_observed <- vapply(seq_len(n_cells), function(j) {
      v <- N[, j]
      m <- obs[, j]
      if (!any(m)) return(NA_real_)
      as.numeric(stats::median(v[m], na.rm = TRUE))
    }, numeric(1))
    cell_qc <- data.table(
      cell_idx = seq_len(n_cells) - 1L,
      cell_id = cell_id,
      observed_sites = as.integer(observed_sites),
      candidate_sites = as.integer(candidate_sites),
      significant_sites = as.integer(significant_sites),
      total_coverage_sum = as.numeric(total_coverage_sum_cell),
      mean_total_observed = as.numeric(mean_total_observed),
      median_total_observed = as.numeric(median_total_observed),
      pct_candidate_among_observed = as.numeric(candidate_sites / pmax(observed_sites, 1)),
      pct_significant_among_observed = as.numeric(significant_sites / pmax(observed_sites, 1)),
      cell_qc_status = "retained",
      cell_filter_threshold_observed_sites = cell_filter_threshold_observed_sites,
      cell_filter_threshold_total_coverage_sum = cell_filter_threshold_total_coverage,
      cell_filter_threshold_log10_observed_sites = cell_filter_threshold_log_observed_sites,
      cell_filter_threshold_log10_total_coverage_sum = cell_filter_threshold_log_total_coverage,
      cell_filter_tukey_k = PARAMS$cell_filter_tukey_k
    )
  } else {
    n_cells_before_filter <- n_cells
  }

  n_obs <- sum(obs)
  n_cand <- sum(cand)
  n_sig <- sum(sig)
  if (n_obs <= 0L) stop("No observed entries remained after cell QC filtering.", call. = FALSE)
  if (n_cand <= 0L) warn_env$warnings <- c(warn_env$warnings, "No candidate m6A entries remained after cell QC filtering.")
  if (n_sig <= 0L) stop("No high-confidence significant entries remained after cell QC filtering.", call. = FALSE)

  observed_cells <- rowSums(obs)
  candidate_cells <- rowSums(cand)
  significant_cells <- rowSums(sig)
  total_coverage_sum_site <- rowSums(N, na.rm = TRUE)

  # Site mean for empirical Bayes smoothing.
  Y_obs_sum <- rowSums(Y * obs, na.rm = TRUE)
  N_obs_sum <- rowSums(N * obs, na.rm = TRUE)
  global_mean <- sum(Y_obs_sum, na.rm = TRUE) / sum(N_obs_sum, na.rm = TRUE)
  site_mean <- Y_obs_sum / pmax(N_obs_sum, 1)
  site_mean[observed_cells == 0L | is.na(site_mean)] <- global_mean
  site_mean <- pmin(pmax(site_mean, PARAMS$epsilon), 1 - PARAMS$epsilon)

  # Mean/variance of observed raw ratios by row, computed in row chunks to reduce peak memory.
  log_msg("Computing site-level ratio mean/variance in chunks")
  mean_ratio_observed <- rep(NA_real_, n_sites)
  var_ratio_observed <- rep(NA_real_, n_sites)
  row_chunk <- 5000L
  starts <- seq.int(1L, n_sites, by = row_chunk)
  for (s in starts) {
    e <- min(s + row_chunk - 1L, n_sites)
    idx <- s:e
    ratio_chunk <- Y[idx, , drop = FALSE] / pmax(N[idx, , drop = FALSE], 1)
    ratio_chunk[!obs[idx, , drop = FALSE]] <- NA_real_
    mean_ratio_observed[idx] <- rowMeans(ratio_chunk, na.rm = TRUE)
    var_ratio_observed[idx] <- safe_row_var(ratio_chunk)
  }
  mean_ratio_observed[is.nan(mean_ratio_observed)] <- NA_real_

  rr <- rowRanges(m6a_se)
  rd_m6a <- sanitize_df(rowData(m6a_se))
  site_metadata <- data.table(
    site_idx = seq_len(n_sites) - 1L,
    site_id = site_id,
    seqnames = as.character(seqnames(rr)),
    start = start(rr),
    end = end(rr),
    strand = as.character(strand(rr))
  )
  if (ncol(rd_m6a) > 0L) site_metadata <- cbind(site_metadata, rd_m6a)

  if (!"reference_site_source" %in% names(site_metadata)) {
    site_metadata[, reference_site_source := NA_character_]
    warn_env$warnings <- c(warn_env$warnings, "rowData(m6A) missing reference_site_source; output filled with NA.")
  }
  if (!"orthogonal_validation_status" %in% names(site_metadata)) {
    site_metadata[, orthogonal_validation_status := NA_character_]
    warn_env$warnings <- c(warn_env$warnings, "rowData(m6A) missing orthogonal_validation_status; output filled with NA.")
  }

  site_qc <- data.table(
    site_idx = seq_len(n_sites) - 1L,
    site_id = site_id,
    observed_cells = as.integer(observed_cells),
    candidate_cells = as.integer(candidate_cells),
    significant_cells = as.integer(significant_cells),
    total_coverage_sum = as.numeric(total_coverage_sum_site),
    mean_ratio_observed = as.numeric(mean_ratio_observed),
    var_ratio_observed = as.numeric(var_ratio_observed),
    site_mean_all_observed = as.numeric(site_mean),
    reference_site_source = site_metadata$reference_site_source,
    orthogonal_validation_status = site_metadata$orthogonal_validation_status,
    seqnames = site_metadata$seqnames,
    start = site_metadata$start,
    end = site_metadata$end,
    strand = site_metadata$strand
  )

  cd_m6a <- sanitize_df(colData(m6a_se))
  cell_metadata <- data.table(cell_idx = seq_len(n_cells) - 1L, cell_id = cell_id)
  if (ncol(cd_m6a) > 0L) cell_metadata <- cbind(cell_metadata, cd_m6a)
  constant_coldata_fields <- names(cd_m6a)[vapply(cd_m6a, function(x) length(unique(x[!is.na(x)])) <= 1L, logical(1))]
  if (length(constant_coldata_fields) > 0L) {
    warn_env$warnings <- c(warn_env$warnings, paste0("Constant colData fields not suitable as biological labels: ", paste(constant_coldata_fields, collapse = ", ")))
  }
  if (!any(grepl("^YTH|^YTHmut", cell_id))) {
    warn_env$warnings <- c(warn_env$warnings, "Cell IDs are SRR-like and do not encode YTH/YTHmut groups; do not treat cell_group as biological label.")
  }

  # Model panel sites: non-destructive auxiliary view only.
  log_msg("Selecting model panel sites as auxiliary view")
  min_observed_cells <- max(5L, ceiling(0.005 * n_cells))
  finite_var <- site_qc$var_ratio_observed[is.finite(site_qc$var_ratio_observed)]
  var_cut <- if (length(finite_var) > 0L) as.numeric(stats::quantile(finite_var, probs = 0.90, na.rm = TRUE)) else Inf
  model_panel <- copy(site_qc)
  model_panel[, selected_by_observed_cells := observed_cells >= min_observed_cells]
  model_panel[, selected_by_candidate_cells := candidate_cells >= 2L]
  model_panel[, selected_by_significant_cells := significant_cells >= 2L]
  model_panel[, selected_by_high_variance := is.finite(var_ratio_observed) & var_ratio_observed >= var_cut]
  model_panel <- model_panel[selected_by_observed_cells | selected_by_candidate_cells | selected_by_significant_cells | selected_by_high_variance]
  data.table::setorder(model_panel, -significant_cells, -candidate_cells, -observed_cells, -var_ratio_observed, site_id)
  if (!is.na(PARAMS$dense_panel_cap) && PARAMS$dense_panel_cap > 0L && nrow(model_panel) > PARAMS$dense_panel_cap) {
    model_panel <- model_panel[seq_len(PARAMS$dense_panel_cap)]
  }
  model_panel[, panel_rank := seq_len(.N)]
  if (nrow(model_panel) < 5000L) {
    warn_env$warnings <- c(warn_env$warnings, paste0("Model panel contains fewer than 5000 sites: ", nrow(model_panel)))
  }
  if (nrow(model_panel) < PARAMS$min_model_panel_sites) {
    stop("Model panel too small: ", nrow(model_panel), " sites. Minimum required: ", PARAMS$min_model_panel_sites, call. = FALSE)
  }

  # Expression anchor and regulator expression.
  log_msg("Building expression anchor and regulator expression")
  tx_assay <- if ("RPKM" %in% assayNames(tx_se)) "RPKM" else "ReadCounts"
  expr <- assay(tx_se, tx_assay)
  if (tx_assay == "ReadCounts") {
    lib <- colSums(expr, na.rm = TRUE)
    expr <- t(t(expr) / pmax(lib, 1) * 1e6)
    warn_env$warnings <- c(warn_env$warnings, "Transcript RPKM assay missing; used CPM-like normalized ReadCounts for expression reference.")
  }
  expr_log <- log1p(expr)
  gene_labels <- get_gene_labels(tx_se)
  rv <- matrixStats::rowVars(expr_log, na.rm = TRUE)
  rv[is.na(rv)] <- -Inf
  n_anchor <- min(PARAMS$n_anchor_genes, length(rv))
  anchor_idx <- order(rv, decreasing = TRUE)[seq_len(n_anchor)]
  anchor_labels <- make.unique(gene_labels[anchor_idx])
  expression_anchor <- as.data.table(t(expr_log[anchor_idx, , drop = FALSE]))
  data.table::setnames(expression_anchor, anchor_labels)
  expression_anchor <- cbind(data.table(cell_id = cell_id), expression_anchor)

  regulator_aliases <- list(
    VIRMA = c("VIRMA", "KIAA1429"),
    KIAA1429 = c("KIAA1429", "VIRMA")
  )
  match_labels <- get_match_labels_upper(tx_se)
  regulator_dt <- data.table(cell_id = cell_id)
  regulator_map <- data.table(regulator = character(), matched_row = integer(), matched_label = character())
  missing_regulators <- character(0)
  for (reg in PARAMS$canonical_regulators) {
    aliases <- toupper(unique(c(reg, regulator_aliases[[reg]])))
    hits <- which(vapply(match_labels, function(z) any(z %in% aliases), logical(1)))
    if (length(hits) == 0L) {
      missing_regulators <- c(missing_regulators, reg)
      next
    }
    # Choose the most variable matched row for this dataset.
    best <- hits[which.max(rv[hits])]
    regulator_dt[[reg]] <- as.numeric(expr_log[best, ])
    regulator_map <- rbind(regulator_map, data.table(
      regulator = reg,
      matched_row = best,
      matched_label = gene_labels[best]
    ))
  }
  if (length(missing_regulators) > 0L) {
    warn_env$warnings <- c(warn_env$warnings, paste0("Missing canonical regulators in transcript annotation: ", paste(missing_regulators, collapse = ", ")))
  }

  # Write static outputs.
  log_msg("Writing metadata and QC outputs")
  data.table::fwrite(cell_metadata, file.path(PARAMS$output_dir, "cell_metadata.csv"))
  data.table::fwrite(site_metadata, file.path(PARAMS$output_dir, "site_metadata.csv"))
  data.table::fwrite(cell_qc, file.path(PARAMS$output_dir, "cell_qc.csv"))
  data.table::fwrite(cell_qc_all, file.path(PARAMS$output_dir, "cell_qc_all_before_filter.csv"))
  data.table::fwrite(cell_qc_removed, file.path(PARAMS$output_dir, "cell_qc_removed_low_observed_sites.csv"))
  data.table::fwrite(site_qc, file.path(PARAMS$output_dir, "site_qc.csv"))
  data.table::fwrite(model_panel, file.path(PARAMS$output_dir, "model_panel_sites.csv"))
  data.table::fwrite(expression_anchor, file.path(PARAMS$output_dir, "expression_anchor.csv"))
  data.table::fwrite(regulator_dt, file.path(PARAMS$output_dir, "regulator_expression.csv"))
  data.table::fwrite(regulator_map, file.path(PARAMS$output_dir, "regulator_match_map.csv"))

  # Write observed/candidate/significant long tables in chunks.
  log_msg("Writing observed_entries.tsv, candidate_m6a_entries.tsv, and significant_entries.tsv in cell chunks")
  obs_path_tmp <- file.path(PARAMS$output_dir, "observed_entries.tsv")
  cand_path_tmp <- file.path(PARAMS$output_dir, "candidate_m6a_entries.tsv")
  sig_path_tmp <- file.path(PARAMS$output_dir, "significant_entries.tsv")
  high_conf_path_tmp <- file.path(PARAMS$output_dir, "high_conf_significant_entries.tsv")
  stale_imp_path_tmp <- file.path(PARAMS$output_dir, "important_entries.tsv")
  for (path in c(obs_path_tmp, cand_path_tmp, sig_path_tmp, high_conf_path_tmp, stale_imp_path_tmp)) {
    if (file.exists(path)) file.remove(path)
    if (file.exists(paste0(path, ".gz"))) file.remove(paste0(path, ".gz"))
  }

  source_vec <- site_metadata$reference_site_source
  val_vec <- site_metadata$orthogonal_validation_status
  cell_starts <- seq.int(1L, n_cells, by = PARAMS$chunk_cells)
  for (cs in cell_starts) {
    ce <- min(cs + PARAMS$chunk_cells - 1L, n_cells)
    cols <- cs:ce
    idx <- which(obs[, cols, drop = FALSE], arr.ind = TRUE)
    if (nrow(idx) == 0L) next
    site_pos <- idx[, 1]
    cell_pos <- cols[idx[, 2]]
    yv <- as.numeric(Y[cbind(site_pos, cell_pos)])
    nv <- as.numeric(N[cbind(site_pos, cell_pos)])
    pv <- as.numeric(Prob[cbind(site_pos, cell_pos)])
    av <- as.numeric(Padj[cbind(site_pos, cell_pos)])
    is_candidate <- as.integer(cand[cbind(site_pos, cell_pos)])
    is_sig <- as.integer(sig[cbind(site_pos, cell_pos)])
    ratio_raw <- yv / pmax(nv, 1)
    ratio_eb <- (yv + PARAMS$eb_alpha * site_mean[site_pos]) / (nv + PARAMS$eb_alpha)
    ratio_eb <- pmin(pmax(ratio_eb, PARAMS$epsilon), 1 - PARAMS$epsilon)
    x_logit <- safe_qlogis(ratio_eb, PARAMS$epsilon)
    coverage_channel <- log1p(pmin(nv, PARAMS$total_cap_for_loss)) / log1p(PARAMS$total_cap_for_loss)

    dt <- data.table(
      cell_idx = cell_pos - 1L,
      site_idx = site_pos - 1L,
      cell_id = cell_id[cell_pos],
      site_id = site_id[site_pos],
      Y = yv,
      N = nv,
      prob = pv,
      padj = av,
      is_observed = 1L,
      is_candidate = is_candidate,
      is_significant = is_sig,
      ratio_raw = ratio_raw,
      ratio_eb = ratio_eb,
      x_logit = x_logit,
      coverage_channel = coverage_channel,
      reference_site_source = source_vec[site_pos],
      orthogonal_validation_status = val_vec[site_pos]
    )
    append_dt(dt, obs_path_tmp)
    dt_cand <- dt[is_candidate == 1L]
    if (nrow(dt_cand) > 0L) append_dt(dt_cand, cand_path_tmp)
    dt_sig <- dt[is_significant == 1L]
    if (nrow(dt_sig) > 0L) {
      append_dt(dt_sig, sig_path_tmp)
      append_dt(dt_sig, high_conf_path_tmp)
    }
    log_msg("  wrote cells ", cs, "-", ce, " / ", n_cells)
  }

  obs_final <- compress_tsv_if_possible(obs_path_tmp, warn_env)
  cand_final <- compress_tsv_if_possible(cand_path_tmp, warn_env)
  sig_final <- compress_tsv_if_possible(sig_path_tmp, warn_env)
  high_conf_final <- compress_tsv_if_possible(high_conf_path_tmp, warn_env)

  # Summary JSON.
  summary_obj <- list(
    created_at = as.character(Sys.time()),
    input = list(
      workdir = workdir,
      m6a_rds = m6a_path,
      transcript_rds = tx_path,
      m6a_dim = c(n_sites, n_cells),
      transcript_dim = c(nrow(tx_se), ncol(tx_se)),
      m6a_assays = assayNames(m6a_se),
      transcript_assays = assayNames(tx_se),
      transcript_assay_used = tx_assay
    ),
    thresholds = list(
      min_total_observed = PARAMS$min_total_observed,
      min_total_candidate = PARAMS$min_total_candidate,
      min_total_significant = PARAMS$min_total_significant,
      observed_rule = "Total >= min_total_observed",
      candidate_rule = "Total >= min_total_candidate AND (m6ASiteProb >= prob_positive OR AdjPvalue < padj_significant)",
      significant_rule = "Total >= min_total_significant AND m6ASiteProb >= prob_positive AND AdjPvalue < padj_significant",
      prob_positive = PARAMS$prob_positive,
      padj_significant = PARAMS$padj_significant,
      total_cap_for_loss = PARAMS$total_cap_for_loss,
      epsilon = PARAMS$epsilon,
      eb_alpha = PARAMS$eb_alpha,
      cell_filter_rule = "remove cells with log10(observed_sites) and log10(total_coverage_sum) both below Q1 - cell_filter_tukey_k * IQR",
      cell_filter_metric = "observed_sites AND total_coverage_sum",
      cell_filter_transform = "log10",
      cell_filter_tukey_k = PARAMS$cell_filter_tukey_k,
      cell_filter_log10_observed_sites_q1 = q1_log_observed_sites,
      cell_filter_log10_observed_sites_q3 = q3_log_observed_sites,
      cell_filter_log10_observed_sites_iqr = iqr_log_observed_sites,
      cell_filter_log10_observed_sites_threshold = cell_filter_threshold_log_observed_sites,
      cell_filter_observed_sites_threshold = cell_filter_threshold_observed_sites,
      cell_filter_log10_total_coverage_sum_q1 = q1_log_total_coverage,
      cell_filter_log10_total_coverage_sum_q3 = q3_log_total_coverage,
      cell_filter_log10_total_coverage_sum_iqr = iqr_log_total_coverage,
      cell_filter_log10_total_coverage_sum_threshold = cell_filter_threshold_log_total_coverage,
      cell_filter_total_coverage_sum_threshold = cell_filter_threshold_total_coverage
    ),
    counts = list(
      n_sites_input = n_sites,
      n_cells_before_filter = n_cells_before_filter,
      n_cells_after_filter = n_cells,
      n_cells_removed_low_observed_and_total_coverage = nrow(cell_qc_removed),
      n_sites_archive = n_sites,
      n_observed_entries = as.numeric(n_obs),
      n_candidate_entries = as.numeric(n_cand),
      n_significant_entries = as.numeric(n_sig),
      coverage_fraction = as.numeric(n_obs / (n_sites * n_cells)),
      candidate_fraction_all_entries = as.numeric(n_cand / (n_sites * n_cells)),
      candidate_fraction_observed_entries = as.numeric(n_cand / n_obs),
      significant_fraction_all_entries = as.numeric(n_sig / (n_sites * n_cells)),
      significant_fraction_observed_entries = as.numeric(n_sig / n_obs),
      n_model_panel_sites = nrow(model_panel),
      model_panel_cap = ifelse(is.na(PARAMS$dense_panel_cap), NA, PARAMS$dense_panel_cap),
      n_anchor_genes = n_anchor,
      n_regulators_found = ncol(regulator_dt) - 1L,
      n_regulators_missing = length(missing_regulators)
    ),
    warnings = warn_env$warnings,
    missing_regulators = missing_regulators,
    constant_coldata_fields = constant_coldata_fields,
    output_files = list(
      observed_entries = obs_final,
      candidate_m6a_entries = cand_final,
      significant_entries = sig_final,
      high_conf_significant_entries = high_conf_final,
      cell_metadata = file.path(PARAMS$output_dir, "cell_metadata.csv"),
      site_metadata = file.path(PARAMS$output_dir, "site_metadata.csv"),
      cell_qc = file.path(PARAMS$output_dir, "cell_qc.csv"),
      cell_qc_all_before_filter = file.path(PARAMS$output_dir, "cell_qc_all_before_filter.csv"),
      cell_qc_removed_low_observed_and_total_coverage = file.path(PARAMS$output_dir, "cell_qc_removed_low_observed_sites.csv"),
      site_qc = file.path(PARAMS$output_dir, "site_qc.csv"),
      expression_anchor = file.path(PARAMS$output_dir, "expression_anchor.csv"),
      regulator_expression = file.path(PARAMS$output_dir, "regulator_expression.csv"),
      model_panel_sites = file.path(PARAMS$output_dir, "model_panel_sites.csv")
    )
  )
  jsonlite::write_json(summary_obj, file.path(PARAMS$output_dir, "step02_summary.json"),
                       pretty = TRUE, auto_unbox = TRUE, na = "null")

  # Human-readable report.
  report_lines <- c(
    "# Step 2 QC and coverage-aware input construction report",
    "",
    paste0("Generated at: ", Sys.time()),
    "",
    "## Input files",
    paste0("- m6A RDS: `", m6a_path, "`"),
    paste0("- Transcript RDS: `", tx_path, "`"),
    "",
    "## Dimensions",
    paste0("- m6A: ", n_sites, " sites × ", n_cells, " cells"),
    paste0("- Transcript: ", nrow(tx_se), " features × ", ncol(tx_se), " cells"),
    "",
    "## Assays",
    paste0("- m6A assays: ", paste(assayNames(m6a_se), collapse = ", ")),
    paste0("- Transcript assays: ", paste(assayNames(tx_se), collapse = ", ")),
    paste0("- Transcript assay used for expression reference: ", tx_assay),
    "",
    "## Thresholds",
    paste0("- Observed entry: Total >= ", PARAMS$min_total_observed),
    paste0("- Candidate m6A entry: Total >= ", PARAMS$min_total_candidate, " AND (m6ASiteProb >= ", PARAMS$prob_positive, " OR AdjPvalue < ", PARAMS$padj_significant, ")"),
    paste0("- High-confidence significant entry: Total >= ", PARAMS$min_total_significant, " AND m6ASiteProb >= ", PARAMS$prob_positive, " AND AdjPvalue < ", PARAMS$padj_significant),
    paste0("- EB alpha: ", PARAMS$eb_alpha),
    paste0("- Coverage channel cap: ", PARAMS$total_cap_for_loss),
    paste0("- Cell QC: remove cells where both log10(observed_sites) and log10(total_coverage_sum) are below Q1 - ", PARAMS$cell_filter_tukey_k, " × IQR"),
    paste0("- Cell QC observed_sites threshold: ", round(cell_filter_threshold_observed_sites, 2)),
    paste0("- Cell QC total_coverage_sum threshold: ", round(cell_filter_threshold_total_coverage, 2)),
    "",
    "## Counts",
    paste0("- Cells before QC: ", n_cells_before_filter),
    paste0("- Cells removed by low observed-site and low total-coverage QC: ", nrow(cell_qc_removed)),
    paste0("- Cells after QC: ", n_cells),
    paste0("- Observed entries: ", format(n_obs, big.mark = ",")),
    paste0("- Candidate m6A entries: ", format(n_cand, big.mark = ",")),
    paste0("- High-confidence significant entries: ", format(n_sig, big.mark = ",")),
    paste0("- Coverage fraction: ", signif(n_obs / (n_sites * n_cells), 5)),
    paste0("- Candidate / all entries: ", signif(n_cand / (n_sites * n_cells), 5)),
    paste0("- Candidate / observed entries: ", signif(n_cand / n_obs, 5)),
    paste0("- High-confidence significant / all entries: ", signif(n_sig / (n_sites * n_cells), 5)),
    paste0("- High-confidence significant / observed entries: ", signif(n_sig / n_obs, 5)),
    paste0("- Model panel sites: ", nrow(model_panel)),
    paste0("- Model panel cap: ", ifelse(is.na(PARAMS$dense_panel_cap), "none", PARAMS$dense_panel_cap)),
    paste0("- Expression anchor genes: ", n_anchor),
    paste0("- Regulators found: ", ncol(regulator_dt) - 1L),
    paste0("- Regulators missing: ", length(missing_regulators)),
    "",
    "## Important interpretation notes",
    "- This Step 2 does not destructively filter the validated site universe.",
    "- Cell-level QC uses label-independent coverage metrics before model training and removes only cells that are low in both observed-site breadth and total m6A coverage.",
    "- `model_panel_sites.csv` is an auxiliary training/view panel, not a replacement for the full archive matrix.",
    "- Unobserved or entries below the relaxed coverage threshold are not exported as zero methylation.",
    "- `candidate_m6a_entries.tsv.gz` stores relaxed OR-evidence entries and is intended for optional soft-gate supervision.",
    paste0("- `significant_entries.tsv.gz` stores Step2-defined AND-evidence entries under the relaxed coverage threshold Total>=", PARAMS$min_total_significant, " and is intended for significance pseudo-labeling and final evaluation."),
    "- `high_conf_significant_entries.tsv.gz` is a copy of `significant_entries.tsv.gz` with an explicit high-confidence name.",
    "- Transcript data are exported only as external expression anchors and regulator expression, not as AE training input.",
    "- No train/validation/test split is created here; splitting belongs to Step 3.",
    "",
    "## Warnings",
    if (length(warn_env$warnings) == 0L) "- None" else paste0("- ", warn_env$warnings),
    "",
    "## Output files",
    paste0("- `", obs_final, "`"),
    paste0("- `", cand_final, "`"),
    paste0("- `", sig_final, "`"),
    paste0("- `", high_conf_final, "`"),
    paste0("- `", file.path(PARAMS$output_dir, "cell_metadata.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "site_metadata.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "cell_qc.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "cell_qc_all_before_filter.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "cell_qc_removed_low_observed_sites.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "site_qc.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "model_panel_sites.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "expression_anchor.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "regulator_expression.csv"), "`"),
    paste0("- `", file.path(PARAMS$output_dir, "step02_summary.json"), "`")
  )
  writeLines(report_lines, file.path(PARAMS$report_dir, "step02_qc_report.md"))

  # Minimal integrity checks on outputs.
  required_outputs <- c(
    file.path(PARAMS$output_dir, "cell_metadata.csv"),
    file.path(PARAMS$output_dir, "site_metadata.csv"),
    file.path(PARAMS$output_dir, "cell_qc.csv"),
    file.path(PARAMS$output_dir, "cell_qc_all_before_filter.csv"),
    file.path(PARAMS$output_dir, "cell_qc_removed_low_observed_sites.csv"),
    file.path(PARAMS$output_dir, "site_qc.csv"),
    file.path(PARAMS$output_dir, "model_panel_sites.csv"),
    file.path(PARAMS$output_dir, "expression_anchor.csv"),
    file.path(PARAMS$output_dir, "regulator_expression.csv"),
    file.path(PARAMS$output_dir, "step02_summary.json"),
    file.path(PARAMS$report_dir, "step02_qc_report.md")
  )
  missing_outputs <- required_outputs[!file.exists(required_outputs)]
  if (length(missing_outputs) > 0L) {
    stop("Missing output files: ", paste(missing_outputs, collapse = ", "), call. = FALSE)
  }

  log_msg("Step 2 rebuild completed.")
  invisible(summary_obj)
}

# -----------------------------
# 4. Execute
# -----------------------------
args <- commandArgs(trailingOnly = TRUE)
workdir <- if (length(args) >= 1L) resolve_from_script(args[[1]], SCRIPT_DIR, mustWork = TRUE) else SCRIPT_DIR
main(workdir = workdir)
