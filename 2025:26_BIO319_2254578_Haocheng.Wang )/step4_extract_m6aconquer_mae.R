#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(mae = NULL, sites = NULL, output = NULL, dataset = "unknown")
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key, call. = FALSE)
    if (i == length(args)) stop("Missing value for ", key, call. = FALSE)
    val <- args[[i + 1L]]
    out[[sub("^--", "", key)]] <- val
    i <- i + 2L
  }
  if (is.null(out$mae) || is.null(out$sites) || is.null(out$output)) {
    stop("Required arguments: --mae, --sites, --output", call. = FALSE)
  }
  out
}

get_script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0L) {
    script_path <- sub("^--file=", "", file_arg[[1L]])
    return(dirname(normalizePath(script_path, mustWork = FALSE)))
  }
  normalizePath(".", mustWork = FALSE)
}

is_absolute_path <- function(path) {
  grepl("^(/|[A-Za-z]:[/\\\\])", path)
}

resolve_script_path <- function(path, script_dir) {
  path <- path.expand(path)
  if (!is_absolute_path(path)) {
    path <- file.path(script_dir, path)
  }
  normalizePath(path, mustWork = FALSE)
}

args <- parse_args()
script_dir <- get_script_dir()
args$mae <- resolve_script_path(args$mae, script_dir)
args$sites <- resolve_script_path(args$sites, script_dir)
args$output <- resolve_script_path(args$output, script_dir)
output_dir <- dirname(args$output)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (exists("mem.maxVSize", mode = "function")) mem.maxVSize(96000)

required <- c("MultiAssayExperiment", "SummarizedExperiment", "GenomicRanges")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0L) {
  stop("Missing R packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

suppressPackageStartupMessages({
  library(MultiAssayExperiment)
  library(SummarizedExperiment)
  library(GenomicRanges)
})

sites <- read.csv(args$sites, stringsAsFactors = FALSE)
required_cols <- c("site_key", "seqnames", "start", "strand")
missing_cols <- setdiff(required_cols, names(sites))
if (length(missing_cols) > 0L) {
  stop("Site table missing columns: ", paste(missing_cols, collapse = ", "), call. = FALSE)
}
sites$site_key <- as.character(sites$site_key)
target_keys <- unique(sites$site_key)

obj <- readRDS(args$mae)
if (!"m6AOmics" %in% names(experiments(obj))) {
  stop("m6AOmics experiment not found in ", args$mae, call. = FALSE)
}
m6a <- experiments(obj)[["m6AOmics"]]
rr <- rowRanges(m6a)
row_keys <- paste0(as.character(seqnames(rr)), ":", start(rr), ":", as.character(strand(rr)))
idx <- which(row_keys %in% target_keys)

if (length(idx) == 0L) {
  empty <- data.frame(
    site_key = character(), site_id = character(), dataset = character(),
    external_n_samples = integer(), external_observed_samples = integer(),
    external_total_sum = numeric(), external_m6a_sum = numeric(),
    external_ratio = numeric(), external_mean_prob = numeric(),
    external_min_padj = numeric(), stringsAsFactors = FALSE
  )
  write.csv(empty, args$output, row.names = FALSE)
  quit(save = "no", status = 0)
}

extract_assay <- function(name) {
  if (!name %in% assayNames(m6a)) return(NULL)
  assay(m6a, name)[idx, , drop = FALSE]
}

y <- extract_assay("m6A")
n <- extract_assay("Total")
prob <- extract_assay("m6ASiteProb")
padj <- extract_assay("AdjPvalue")

if (is.null(y) || is.null(n)) stop("m6A/Total assays are required.", call. = FALSE)

observed <- !is.na(n) & n > 0
total_sum <- rowSums(n, na.rm = TRUE)
m6a_sum <- rowSums(y, na.rm = TRUE)
ratio <- m6a_sum / pmax(total_sum, 1)
ratio[total_sum <= 0] <- NA_real_
mean_prob <- if (!is.null(prob)) rowMeans(prob, na.rm = TRUE) else rep(NA_real_, length(idx))
mean_prob[is.nan(mean_prob)] <- NA_real_
min_padj <- if (!is.null(padj)) apply(padj, 1, function(x) suppressWarnings(min(x, na.rm = TRUE))) else rep(NA_real_, length(idx))
min_padj[is.infinite(min_padj)] <- NA_real_

site_lookup <- sites[!duplicated(sites$site_key), c("site_key", "site_id")]
out <- data.frame(
  site_key = row_keys[idx],
  dataset = args$dataset,
  external_n_samples = ncol(m6a),
  external_observed_samples = rowSums(observed, na.rm = TRUE),
  external_total_sum = as.numeric(total_sum),
  external_m6a_sum = as.numeric(m6a_sum),
  external_ratio = as.numeric(ratio),
  external_mean_prob = as.numeric(mean_prob),
  external_min_padj = as.numeric(min_padj),
  stringsAsFactors = FALSE
)
out <- merge(out, site_lookup, by = "site_key", all.x = TRUE, sort = FALSE)
out <- out[, c("site_key", "site_id", "dataset", "external_n_samples", "external_observed_samples",
               "external_total_sum", "external_m6a_sum", "external_ratio", "external_mean_prob", "external_min_padj")]
write.csv(out, args$output, row.names = FALSE)
