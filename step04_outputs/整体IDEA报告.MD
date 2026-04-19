# FYPmain-3 Overall IDEA Report

## Task Decision

- The project now follows the cell-clustering / m6A regulatory-state autoencoder route.
- This is not a SigRM-style differential site calling workflow. SigRM informs the count-aware statistical mindset, but its test formula is not copied as the AE loss.
- This route was chosen because the existing code is already AE-based, earlier results showed usable expression-vs-AE ARI and YTHDF2-latent correlation, and DCA/scVI provide clear future baselines.

## Data Basis

Step1 was not rerun. Step2 uses the existing validated scDART/m6AConquer RDS inputs and writes plain `.csv` matrices for m6A count, Total coverage, ratio, masks, metadata, expression anchors, and regulator expression.

```text
step=02_improved
run_id=no_site_filter_all135k
input_m6a=step01_scDART_hg38_WT_MAE_validated_m6a.rds
input_transcript=step01_scDART_hg38_WT_MAE_validated_transcript.rds
output_dir=step02_input
site_rule=coverage_plus_sample_level_significance_plus_variability
selected_site_cap=135300
selected_site_count=135300
candidate_site_count=135300
cells_before_qc=1000
cells_after_qc=997
cell_min_coverage_sites=1500
site_min_coverage_cells=0
site_min_positive_cells=0
min_total_for_coverage=20
min_total_for_positive=20
site_min_mean_total=0
coverage_fraction_selected=0.195597
significance_fraction_selected=0.002124
heldout_fraction_target=0.2
heldout_observed_entries=5276405
heldout_significant_entries=57052
train_observed_entries=21108537
expression_assay=RPKM
expression_anchor_gene_count=2000
available_regulators=METTL3,METTL14,WTAP,RBM15,RBM15B,ZC3H13,ALKBH5,FTO,YTHDF1,YTHDF2,YTHDF3,YTHDC1,YTHDC2,HNRNPC,HNRNPA2B1,IGF2BP1,IGF2BP2,IGF2BP3
seed=42
```

## Step3 Model

- Model: `BinomialAutoencoder` trained on cell-by-site m6A matrices.
- Input channels: `ratio x mask`, `mask`, and `log1p(capped Total) x mask`.
- Loss: coverage-capped weighted binomial NLL.

```text
L_main = sum_ij m_ij w_ij [-y_ij log(p_ij) - (n_ij-y_ij) log(1-p_ij)] / sum_ij m_ij w_ij n_eff_ij
```

## Current Result

- Status: `display-ready preliminary result`.
- Cells/sites: `997` cells x `135300` sites.
- Expression vs AE ARI: `0.6590`.
- Expression-neighborhood overlap: `0.0575`.
- Top regulator-latent association: `YTHDF2 vs latent_2`, corr `-0.4895`.
- Significant-site NLL: AE `1.3678` vs site-mean `1.6604`.
- All-site NLL guardrail: AE `0.0978` vs site-mean `0.0950`.

## Figure Outputs

- `training_nll_curves`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_training_nll_curves.png`
- `method_nll_comparison`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_method_nll_comparison.png`
- `embedding_panel`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_embedding_panels_by_expression_cluster.png`
- `neighbor_overlap`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_neighbor_overlap.png`
- `failure_reflection`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_failure_reflection_all_nll.png`
- `regulator_correlations`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_regulator_correlations.png`
- `regulator_heatmap`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_regulator_correlation_heatmap.png`
- `dca_style_heatmap`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_dca_style_heatmap_panel.png`
- `dca_style_tsne`: `/Users/Zhuanz/Desktop/FYP/FYPmain-3/step04_outputs/plot_dca_style_tsne_panel.png`

## Limitation and Next Step

The current result passes the planned display thresholds, but NB/Beta-binomial objectives remain the next method-level improvement for over-dispersion.
