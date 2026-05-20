# m6AConquer External Data Content Overview

## Purpose In This Project

The files under `external-m6aconquer/raw/` support Step4 biological downstream analysis for the final3 scDART-seq autoencoder workflow. They are not Step1/Step2/Step3 inputs and should not be used to retrain the current model unless a new experiment is explicitly designed.

The intended Step4 biological story is:

```text
Autoencoder event_prob / residual / latent structure
-> external orthogonal m6A validation
-> genomic feature interpretation
-> FTO perturbation dynamic validation
```

This keeps the interpretation at site level, matching what the autoencoder predicts.

## Project Step1 Source Input

File:

- `../step01_scDART_hg38_WT_MAE_validated_transcript.rds`

Role in this project:

- This is the Step1 input for the current final3 scDART-seq autoencoder workflow.
- It is also the source dataset for this Autoencoder significance-analysis plan.
- This serves as the input for Step 1 and is also the source of the current Autoencoder methylation significance analysis.
- It should be treated as project-local upstream input, not as an external m6AConquer Step4 validation file.

Approximate content:

- R object class: `RangedSummarizedExperiment`.
- Dimensions: 58,650 transcript-level features by 1,000 samples.
- Assays: `ReadCounts` and `RPKM`.
- Row metadata includes `gene_id`, `gene_name`, `gene_biotype`, coordinate-system metadata, `symbol`, and `entrezid`.
- Sample metadata includes `SampleID`, `Title`, `SourceDatabase`, `TissueOrCellLine`, `Organism`, `Treatment`, `DetectionTechnique`, `DataProcessing`, `CurationDate`, and `BioSample`.

## Database Reference

Data source:

- m6AConquer website: <https://rnamd.org/m6aconquer/>
- m6AConquer download page: <https://rnamd.org/m6aconquer/download.html>

Recommended citation:

> Zhao X, Ye H, He D, Liu H, Li T, Rigden DJ, Wei Z. m6AConquer: a consistently quantified and orthogonally validated database for the N6-methyladenosine epitranscriptome. Nucleic Acids Research. 2026;54(D1):D204-D218. doi:10.1093/nar/gkaf1204.

The m6AConquer paper describes systematic reprocessing of raw data from 10 m6A profiling methods, including GLORI and eTAM-seq, and reports 135,300 orthogonally validated human m6A sites using reproducibility across technically orthogonal methods. It also provides matched multi-omics resources, downloadable analysis-ready matrices, and m6A QTL/disease links.

## Downloaded Content

### Orthogonally Validated m6A Sites

Files:

- `raw/m6A_OrthogonallyValidatedSites_Combined_hg38.csv`
- `raw/m6A_OrthogonallyValidatedSites_Combined_hg38.bed`
- `raw/m6A_OrthogonallyValidatedSites_HEK293T_Combined_hg38.csv`
- `raw/m6A_OrthogonallyValidatedSites_HEK293T_Combined_hg38.bed`

Approximate content:

- Human hg38 m6A sites reproducible across orthogonal technologies.
- CSV columns include `seqnames`, `start`, `end`, `width`, `strand`, `support_number`, and `support_technique`.
- BED files provide coordinate-level overlap targets.

Planned Step4 use:

- Test whether high AE `event_prob_residual` sites are enriched for combined or HEK293T orthogonally validated sites.
- Produce overlap counts, odds ratios, Fisher exact p-values, and enrichment figures.

### Genomic Feature Interpretation

Files:

- `raw/m6Aconquer_omicsFeatures_hg38.csv.gz`
- `raw/m6Aconquer_omicsFeatures_hg38.bed`
- `raw/m6AConquer_supplementary_row_data_hg38.csv`
- `raw/m6AConquer_supplementary_row_data_hg38.bed`

Approximate content:

- Human hg38 site-level genomic and transcriptomic features.
- `m6Aconquer_omicsFeatures_hg38.csv.gz` includes coordinate columns plus GC content, phastCons, transcript-region overlaps, exon/intron/UTR/CDS topology, motif-like columns, gene biotype indicators, histone marks, and m6A regulator/RBP-related feature columns.
- `m6AConquer_supplementary_row_data_hg38.csv` includes coordinate columns plus orthogonal validation status, reference site source, technique/sample support, GLORI support, eTAM-seq support, HEK293T support, and HeLa support.

Planned Step4 use:

- Test whether AE top residual sites are enriched for known m6A-compatible genomic contexts.
- Prefer transcript topology, DRACH-like motif, GLORI/eTAM support, and regulator/RBP feature summaries over generic GO enrichment.

### Orthogonal Quantification Datasets

Files:

- `raw/GLORI_hg38_WT_MAE.rds`
- `raw/eTAM_hg38_WT_MAE.rds`

Approximate content:

- Human hg38 wild-type MultiAssayExperiment RDS objects for GLORI and eTAM-seq.
- Intended to provide independent methylation-ratio evidence from technologies different from scDART-seq.

Planned Step4 use:

- Conservatively inspect/load only the needed site-level summaries.
- Correlate AE `event_prob`, `event_prob_residual`, and available foreground probability with GLORI/eTAM methylation evidence where coordinates overlap.

### FTO Perturbation Differential Methylation

Files:

- `raw/DMR_eTAM_GLORI_Control_FTO_hg38.csv`
- `raw/DMR_eTAM_GLORI_Control_FTO_hg38.bed`

Approximate content:

- Human hg38 differential methylation sites from eTAM-seq and GLORI under Control vs FTO perturbation.
- CSV columns include `seqnames`, `start`, `end`, `width`, `strand`, eTAM/GLORI methylation differences, ranks, indices, and `idr`.

Planned Step4 use:

- Test whether AE top residual sites preferentially overlap FTO-responsive DMRs.
- Interpret overlap as dynamic m6A-regulation support, not as training labels.

## Practical Interpretation Rules

- Background universe should be the Step4 analyzed site universe, not the whole genome.
- Primary AE top set should be top decile by `event_prob_residual`; fallback is top decile by `event_prob`.
- Orthogonal validation and FTO DMR are external biological evidence, not complete ground truth.
- GLORI/eTAM concordance should be reported only for overlapping sites and with coverage/missingness caveats.
- GO enrichment is intentionally not the main route because the model output is site-level, not gene-list-first.

## Acknowledgement

I deeply thank Prof. Wei Zhen and the m6AConquer team for developing and making this database publicly available. This resource provided critical external evidence for the orthogonal validation, genomic functional interpretation, and FTO perturbation dynamic validation in Step 4 of this project.
