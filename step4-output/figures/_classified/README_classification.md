# Step4 Figure Classification

This classification follows the paper-style Result narrative: show the strongest AE validation first, then biological support, then caveats. Weak or potentially distracting figures are kept out of the main folder.

## 00_method_design

- `fig03_step3_train_validation_test_split_summary.png` (present): Method figure: nested train/validation/test cells with observed, held-out, and significant entry counts.

## 01_best_results

- `fig0_step1_to_step4_evidence_chain.png` (present): Best opening result figure: connects Step1/2 labels, Step3 AE validation, and Step4 external support.
- `fig11_model_baseline_comparison.png` (present): Best model-validation figure: heldout AE lift plus permutation control and site-prior boundary.
- `fig12_heldout_pr_roc_curves.png` (present): Best curve-based evidence that AE scores rank heldout methylation-significant entries above random.
- `fig14_decile_lift.png` (present): Best ranking summary: top decile enrichment is easy to explain in Results.
- `fig_external_orthogonal_enrichment.png` (present): Best external-support figure: GLORI/eTAM/HEK293T top-vs-background enrichment.
- `fig_external_topology_support.png` (present): Best biology bridge: top AE residual sites align with support/topology/motif features.

## 02_supporting_biology

- `fig_external_feature_enrichment.png` (present): Grouped feature enrichment for biological interpretation and follow-up discussion.
- `fig3_cell_burden_heterogeneity.png` (present): Cell-level heterogeneity support; useful after the main AE result is established.
- `fig9_cell_clustering.png` (present): Cell clustering support; useful as backup for single-cell heterogeneity.

## 03_diagnostics_caveats

- `fig7_calibration_reliability.png` (present): Calibration caveat: scores are useful for ranking but not absolute probabilities.
- `fig8_threshold_analysis.png` (present): Threshold sensitivity and precision/recall operating-point audit.
- `fig2_ref_source_comparison.png` (present): Reference-source/coverage context; important for caveats.
- `fig6_beta_binomial_params.png` (present): Model-score distribution diagnostic, useful for methods/Q&A.
- `fig10_gene_clustering.png` (missing): Optional gene clustering diagnostic when trusted gene annotations exist.

## 04_not_recommended

- `fig1_chromosome_detection_rate.png` (present): Not recommended for main Results: chromosome-level differences may reflect source/coverage bias and distract from the core AE validation.
- `fig4_top_bottom_genes.png` (missing): Not recommended unless trusted gene symbols are present; current Step2 metadata does not support a gene-list-first claim.

## Missing Optional Figures

- `03_diagnostics_caveats/fig10_gene_clustering.png`: Optional gene clustering diagnostic when trusted gene annotations exist.
- `04_not_recommended/fig4_top_bottom_genes.png`: Not recommended unless trusted gene symbols are present; current Step2 metadata does not support a gene-list-first claim.
