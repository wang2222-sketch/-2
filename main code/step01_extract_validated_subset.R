args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", args, value = TRUE)[1])))

input_path <- file.path(script_dir, "scDART_hg38_WT_MAE.rds")
output_m6a_path <- file.path(script_dir, "step01_scDART_hg38_WT_MAE_validated_m6a.rds")
output_transcript_path <- file.path(script_dir, "step01_scDART_hg38_WT_MAE_validated_transcript.rds")
summary_path <- file.path(script_dir, "step01_summary.txt")

invisible(mem.maxVSize(120000))
suppressPackageStartupMessages(library(MultiAssayExperiment))
suppressPackageStartupMessages(library(SummarizedExperiment))

obj <- readRDS(input_path)
m6a_obj <- experiments(obj)[["m6AOmics"]]
transcript_obj <- experiments(obj)[["TranscriptOmics"]]

status <- as.character(rowData(m6a_obj)[["orthogonal_validation_status"]])
keep_status <- if ("Validated_Current" %in% status) {
  "Validated_Current"
} else if ("Validated_Other" %in% status) {
  "Validated_Other"
} else {
  stop("没有 Validated_Current 或 Validated_Other")
}

keep_idx <- !is.na(status) & status == keep_status
validated_m6a <- m6a_obj[keep_idx, ]
validated_transcript <- transcript_obj[, colnames(validated_m6a)]

saveRDS(validated_m6a, output_m6a_path)
saveRDS(validated_transcript, output_transcript_path)

writeLines(c(
  "step=01",
  paste0("keep_status=", keep_status),
  paste0("m6a_output=", output_m6a_path),
  paste0("transcript_output=", output_transcript_path),
  paste0("m6a_dim=", paste(dim(validated_m6a), collapse = "x")),
  paste0("transcript_dim=", paste(dim(validated_transcript), collapse = "x"))
), summary_path)

cat("step01 完成\n")
cat("keep_status: ", keep_status, "\n", sep = "")
cat("m6a dim: ", paste(dim(validated_m6a), collapse = "x"), "\n", sep = "")
cat("transcript dim: ", paste(dim(validated_transcript), collapse = "x"), "\n", sep = "")
