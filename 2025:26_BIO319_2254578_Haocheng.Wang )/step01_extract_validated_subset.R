#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

if (exists("mem.maxVSize", mode = "function")) {
  mem.maxVSize(96000)
}

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

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(input = "scDART_hg38_WT_MAE.rds", outdir = ".", keep_status = "auto")
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Unexpected argument: ", key, call. = FALSE)
    if (i == length(args)) stop("Missing value for ", key, call. = FALSE)
    val <- args[[i + 1L]]
    out[[sub("^--", "", key)]] <- val
    i <- i + 2L
  }
  out
}

args <- parse_args()
script_dir <- get_script_dir()
args$input <- resolve_from_script(args$input, script_dir, mustWork = TRUE)
args$outdir <- resolve_from_script(args$outdir, script_dir, mustWork = FALSE)
dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

required <- c("MultiAssayExperiment", "SummarizedExperiment")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0L) {
  stop("Missing packages: ", paste(missing, collapse = ", "), ". Install with BiocManager.", call. = FALSE)
}

suppressPackageStartupMessages({
  library(MultiAssayExperiment)
  library(SummarizedExperiment)
})

input_path <- normalizePath(args$input, mustWork = TRUE)
output_m6a_path <- file.path(args$outdir, "step01_scDART_hg38_WT_MAE_validated_m6a.rds")
output_transcript_path <- file.path(args$outdir, "step01_scDART_hg38_WT_MAE_validated_transcript.rds")
summary_path <- file.path(args$outdir, "step01_summary.txt")

obj <- readRDS(input_path)
if (!"m6AOmics" %in% names(experiments(obj))) stop("m6AOmics experiment not found.", call. = FALSE)
if (!"TranscriptOmics" %in% names(experiments(obj))) stop("TranscriptOmics experiment not found.", call. = FALSE)

m6a_obj <- experiments(obj)[["m6AOmics"]]
transcript_obj <- experiments(obj)[["TranscriptOmics"]]

status <- as.character(rowData(m6a_obj)[["orthogonal_validation_status"]])
if (args$keep_status == "auto") {
  keep_status <- if ("Validated_Current" %in% status) {
    "Validated_Current"
  } else if ("Validated_Other" %in% status) {
    "Validated_Other"
  } else {
    stop("No Validated_Current or Validated_Other sites were found.", call. = FALSE)
  }
} else {
  keep_status <- args$keep_status
}

keep_idx <- !is.na(status) & status == keep_status
if (!any(keep_idx)) stop("No m6A sites retained for keep_status=", keep_status, call. = FALSE)

validated_m6a <- m6a_obj[keep_idx, ]
validated_transcript <- transcript_obj[, colnames(validated_m6a)]

saveRDS(validated_m6a, output_m6a_path)
saveRDS(validated_transcript, output_transcript_path)

writeLines(c(
  "step=01",
  paste0("input=", input_path),
  paste0("keep_status=", keep_status),
  paste0("m6a_output=", normalizePath(output_m6a_path, mustWork = FALSE)),
  paste0("transcript_output=", normalizePath(output_transcript_path, mustWork = FALSE)),
  paste0("m6a_dim=", paste(dim(validated_m6a), collapse = "x")),
  paste0("transcript_dim=", paste(dim(validated_transcript), collapse = "x"))
), summary_path)

cat("step01 completed\n")
cat("keep_status: ", keep_status, "\n", sep = "")
cat("m6a dim: ", paste(dim(validated_m6a), collapse = "x"), "\n", sep = "")
cat("transcript dim: ", paste(dim(validated_transcript), collapse = "x"), "\n", sep = "")
