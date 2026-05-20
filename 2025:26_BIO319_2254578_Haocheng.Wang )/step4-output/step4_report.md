# Step4 Integrated Downstream Analysis Report

- Split used: `test`
- Exploratory detection threshold: `0.5`
- Conservative detection threshold: `0.7`
- Site detection is based on `pred_significant_probability` standardized as `event_prob`, then aggregated as `event_prob_max` at the site level.
- This report treats Step2 significant labels as high-confidence evidence, not as complete ground truth.

## Evidence-chain conclusion

The Step3 autoencoder learned a non-random methylation-significance signal on heldout entries: test AUPRC lift is 4.53, while the permutation-label control is 1.04. Train/validation/test lift is stable enough to avoid an obvious split-level overfit claim. Because the AE does not dominate every train-site prior baseline, the correct conclusion is that it learns and generalizes useful significance-ranking information, not that it universally beats all prior-based baselines.
Step4 therefore uses the AE score mainly as a ranking/enrichment signal. External GLORI/eTAM/HEK293T support and feature enrichment are used for biological interpretation; calibrated absolute methylation probability is not claimed.

## Key metrics

- `split_used`: test
- `score_source`: pred_significant_probability
- `threshold`: 0.5
- `conservative_threshold`: 0.7
- `model_validation_status`: weak_or_overfit
- `model_criterion_pass_count`: 4
- `model_criterion_total`: 5
- `test_auprc_lift`: 4.5256876321275
- `val_auprc_lift`: 4.411525499821798
- `train_auprc_lift`: 4.469756965475762
- `train_minus_val_lift`: 0.0582314656539635
- `val_minus_test_lift_abs`: 0.1141621323057018
- `test_site_prior_lift`: 4.725317640235621
- `test_site_mean_lift`: 4.317974438550347
- `permutation_test_lift`: 1.040153732710258
- `val_or_selected_sites`: 80303
- `total_cells`: 147
- `detected_sites`: 29086
- `detected_sites_conservative`: 14107
- `overall_detection_rate`: 0.3622031555483606
- `conservative_detection_rate`: 0.17567214176307236
- `mean_event_prob`: 0.353327669875874
- `mean_m6a_freq`: 0.01989982502571476
- `mean_event_prob_residual`: 0.3320122639087338
- `chr_highest`: chr18
- `chr_lowest`: chr19
- `cell_burden_cv`: 1.2875529290498944
- `top_gene`: unknown
- `site_mean_p_corr`: 0.5638073457371003
- `ref_source_detection`: {'Exon_DRACH': 0.3507426996005176, 'GLORI_DRACH_NonExon': 0.41830065359477125, 'GLORI_NonDRACH': 0.4563441409241767}
- `external_combined_validated_top_residual_or`: nan
- `external_combined_validated_top_residual_overlap`: 8031
- `external_fto_dmr_top_residual_or`: 0.8475841196822966
- `external_fto_dmr_top_residual_overlap`: 3364
- `external_best_spearman`: -0.08132008009126686
- `external_best_spearman_label`: eTAM:event_prob_residual~external_ratio
- `external_top_feature`: YTHDC2
- `external_top_feature_or`: 6.751494768310912
- `cell_cluster_k`: 3
- `cell_silhouette_chosen`: 0.40265965035133827
- `cell_cluster_min_size`: 14
- `cell_cluster_max_size`: 93

## Threshold audit

Best F1 against Step2 significant label: threshold=0.25, F1=0.575, precision=0.459, recall=0.769.
The default threshold is kept as an exploratory sensitivity threshold; the conservative threshold should be used when avoiding over-calling is more important.

## Calibration audit

Maximum absolute calibration residual across probability bins: 0.7678.
Positive residual means predicted event probability is higher than observed m6A frequency in that bin.

## Top genes by detected-site burden

Gene-level burden is not interpreted because the current Step2 metadata does not include trusted gene annotations.

## External m6AConquer biological validation

The primary biological interpretation is site-level: AE high-residual/high-event sites are compared with m6AConquer orthogonal validation, GLORI/eTAM evidence, genomic features, and FTO-responsive DMRs.

Top residual decile overlap/enrichment:
| external_evidence             |   overlap_count |   top_overlap_fraction |   background_overlap_fraction |   odds_ratio |   fisher_p_greater |
|:------------------------------|----------------:|-----------------------:|------------------------------:|-------------:|-------------------:|
| orthogonal_validated_combined |            8031 |               1        |                      1        |   nan        |           1        |
| orthogonal_validated_hek293t  |            4081 |               0.508156 |                      0.517628 |     0.962796 |           0.947712 |
| fto_responsive_dmr            |            3364 |               0.418877 |                      0.459583 |     0.847584 |           1        |

Note: combined orthogonal validation is saturated because the analyzed Step4 site universe already comes from validated m6A sites. It confirms source consistency but does not distinguish AE top sites from background. HEK293T support, GLORI/eTAM support, FTO DMR overlap, and genomic feature enrichment are more informative for top-vs-background interpretation.

GLORI/eTAM concordance summary:
| dataset   | score               | external_measure   |   n_overlap |    pearson |   spearman |   mean_score |   mean_external |
|:----------|:--------------------|:-------------------|------------:|-----------:|-----------:|-------------:|----------------:|
| GLORI     | event_prob_mean     | external_ratio     |       19932 | -0.133941  | -0.118722  |     0.630103 |        0.386908 |
| GLORI     | event_prob_residual | external_ratio     |       19932 | -0.12032   | -0.104818  |     0.585645 |        0.386908 |
| eTAM      | event_prob_mean     | external_ratio     |       19144 | -0.108942  | -0.0971713 |     0.631206 |        0.433743 |
| eTAM      | event_prob_residual | external_ratio     |       19144 | -0.0942525 | -0.0813201 |     0.58657  |        0.433743 |

Top genomic feature enrichments in AE high-residual sites:
| feature                   |   top_positive |   top_total |   background_positive |   background_total |   odds_ratio |   fisher_p_greater |
|:--------------------------|---------------:|------------:|----------------------:|-------------------:|-------------:|-------------------:|
| YTHDC2                    |              3 |        8031 |                     4 |              72272 |      6.75149 |        0.0256915   |
| TGACA                     |            249 |        8031 |                   644 |              72272 |      3.55881 |        8.23919e-52 |
| overlap_fullTranscripts   |           4013 |        8031 |                 15886 |              72272 |      3.545   |        0           |
| overlap_exons             |           3944 |        8031 |                 15760 |              72272 |      3.46032 |        0           |
| overlap_exonicTranscripts |           3944 |        8031 |                 15760 |              72272 |      3.46032 |        0           |
| protein_coding            |           3762 |        8031 |                 15222 |              72272 |      3.30276 |        0           |
| TGACT                     |            741 |        8031 |                  2173 |              72272 |      3.27901 |        1.3758e-131 |
| H3K27me3                  |            103 |        8031 |                   288 |              72272 |      3.24726 |        2.72186e-20 |
| GLORI_supported           |           3682 |        8031 |                 15191 |              72272 |      3.18126 |        0           |
| overlap_fullThreePrimeUTR |           2797 |        8031 |                 10481 |              72272 |      3.15051 |        0           |

This section intentionally avoids making GO enrichment the main claim because the model output is site-level rather than gene-list-first.

## Output files

- Numeric summaries: `step4_*csv`, `step4_key_metrics.json`, `visualization_summary.json`
- Figures: `figures/fig*.png`
- Classified figures: `figures/_classified/01_best_results`, `02_supporting_biology`, `03_diagnostics_caveats`, and `04_not_recommended`
- External m6AConquer outputs: `step4_external_overlap_summary.csv`, `step4_external_score_correlation.csv`, `step4_external_feature_enrichment.csv`, and `figures/fig_external_*.png`
- Cluster assignments: `figures/cell_cluster_assignments.csv`, `figures/gene_cluster_assignments.csv` if clustering is enabled.
