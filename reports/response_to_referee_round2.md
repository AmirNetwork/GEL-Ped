# Response to the referee

**Manuscript:** *GEL-Ped: Instability-aware guarded ensemble learning for short-horizon crowd prediction under flow shift*

We thank the referee for a careful and constructive report. The report exposed a genuine attribution problem in the previous version: the evidence did not distinguish a benefit of prospective geometry from the stabilizing effect of a bounded learner. We therefore changed both the method and the claim. GEL-Ped now uses a five-member graph ensemble as the default predictor and its local, past-only member spread as the guard signal. The graph receives unit weight inside the calibration envelope. Only an exceedance moves the forecast toward a bounded coarse behavioural residual model. Prospective closest-approach geometry is retained only as a control. The revised claim is consequently about limiting a detectable neural-instability tail, not about universal accuracy or geometry as the cause of the crossing result.

## Major comments

### 1. Crossing was development-informed

We now state this explicitly in the data chronology, the caption of Figure 1, the Discussion, and the Conclusion. Crossing is treated as a development-informed stress test, not as pristine confirmation. We added four bottleneck and eight unidirectional complete runs from the independent Tordeux archive and report them separately. GEL-Ped remains within 0.0003 m/s of the graph ensemble in these configurations while the coarse boosted model is 0.0159--0.0206 m/s worse. We also expanded the clean external evaluation from one ETH scene to all five official ETH/UCY leave-one-scene-out tasks. Those five tests were not used to select the mechanism and show essentially unchanged scene-macro accuracy, not a broad improvement. The manuscript makes this negative external result central to the scope of the claim.

### 2. Geometry-free coarse learner and mechanism attribution

The referee's alternative explanation was correct and changed the paper materially. The anchored coarse HGB is now a main comparator, and the primary fallback itself is the bounded coarse behavioural model rather than the prospective-geometry expert. In crossing, coarse HGB is marginally lower than GEL-Ped by 0.0007 m/s, a non-significant difference; however, it is 5--8% worse in the other four controlled regimes. GEL-Ped therefore supplies an operating point close to the better expert without choosing the regime from future error. We also added constant-velocity, prospective, graph--coarse-disagreement, learned mixture-of-experts, local graph-CV-disagreement, and Mahalanobis controls. The title and framing are now “instability-aware,” and the manuscript no longer attributes the gain to prospective geometry.

### 3. Crossing outlier and typical effect

Appendix Table B.1 now reports every crossing run, including graph mean, graph median, coarse HGB, GEL-Ped, GEL-minus-graph difference, graph-member seed SD, and realized mean graph weight. All 13 differences favour GEL-Ped. The mean difference is -0.0080 m/s, whereas the median is -0.0022 m/s. Leave-one-run-out means range from -0.0087 to -0.0045 m/s, and removing run d_1 leaves a 1.3% mean reduction. Run d_1 is diagnosed explicitly: one graph member reaches 0.8396 m/s RMSE while the other four lie between 0.3519 and 0.4176 m/s. The graph mean is 0.3899, coordinate median 0.3593, coarse HGB 0.3366, and GEL-Ped 0.3396 m/s. The text and abstract now distinguish the small typical gain from this larger instability event.

### 4. Deep ensemble baseline and within-family guard

The default neural predictor is now the mean of five independently initialized graph residual learners. A coordinate-wise median of the same five members is also reported. The guard is driven by within-family graph-member dispersion, measured locally per pedestrian and smoothed over at most five past observations. Thus the revised design directly tests whether ensemble disagreement can identify an unstable member average. Per-run graph seed SD and realized mean graph weight are provided in Appendix Table B.1. The highest-instability crossing stratum contains 2,875 forecasts; graph RMSE is 0.5654 m/s and GEL-Ped RMSE is 0.4665 m/s.

### 5. ETH/UCY protocol

We corrected the interpretation of ETH as a genuine leave-one-scene-out shift and now report ETH, HOTEL, UNIV, ZARA1, and ZARA2 under the same protocol. For each held scene, the other scenes train the experts and the supplied validation data calibrate the guard; the held scene is used once for testing. At 2.5 Hz, the 0.8-s history contains two intervals and the 0.4-s target one interval. Realized mean graph weights are reported for all scenes: 0.699, 0.994, 0.898, 0.999, and 0.986. GEL-Ped is slightly worse on ETH, ZARA1, and ZARA2, nearly equal on HOTEL, and better on UNIV; the equal-scene macro change is only -0.2% and is not significant. An independently cross-fitted Juelich ablation replacing the entrance axis with recent displacement preserves the crossing comparison, although absolute error increases.

### 6. Horizon and practical significance

The manuscript now reports collision-event counts rather than only percentages: GEL-Ped gives 72,265 true positives, 547 false positives, 786 misses, and 4,402 true negatives; the graph ensemble gives 72,243, 556, 808, and 4,393. The false-alarm similarity was coincidental rather than threshold-matched. We also added recursive audits at 0.8, 1.6, and 3.2 s. Graph/GEL-Ped displacement errors are 0.259/0.249, 0.646/0.589, and 1.900/1.559 m. We explicitly state that this is a compounding-error audit, not a closed-loop simulation: at 3.2 s, coarse HGB and constant velocity remain better than GEL-Ped, so the one-step model is not presented as a long-horizon replacement. The Introduction and Discussion now motivate mobile robots, AV--pedestrian interaction, and station/terminal monitoring without making a safety claim.

### 7. Guard design and comparators

The scene-level gate was replaced by a per-agent local guard. The graph weight is now exactly one inside the complete-run calibration envelope; therefore there is no permanent fallback contribution in supported conditions. Frozen threshold and scale sensitivity is reported on crossing, not only on calibration folds, and spans 0.3452--0.3478 m/s over the tested grid. A per-buffer oracle is added. Learned mixture-of-experts, Mahalanobis, graph--coarse disagreement, graph--CV disagreement, and prospective-fallback variants are reported together in Appendix Table C.1. The chosen rule is not always the lowest test score; it is retained because its signal is internal to the neural family, its in-support intervention is zero, and its operation is directly attributable to ensemble instability.

### 8. Physical and learned baselines

The mis-specified AVM and the weak physical-model table were removed from the main comparison. The graph is no longer called the strongest neural model in the literature. Instead, we report a calibration-only capacity/epoch sweep over six configurations from 6,019 to 51,939 parameters; width 48 and 25 epochs gives the best leave-one-complete-run-out calibration RMSE. Published long-horizon ETH/UCY ADE values are not mixed with the different 0.8-s-history/0.4-s-response task. The five-member graph mean, coordinate median, and coarse HGB provide protocol-matched comparators for the proposed guard.

### 9. Statistical reporting

All fifteen predeclared paired comparisons now appear in Appendix Table B.2, with wins, mean and median differences, exact paired p values, and Holm-adjusted p values. Bootstrap intervals from three or five clusters were removed; the paper reports complete-run ranges and per-run values instead. The former buffer-correlation claim was removed from the central argument. All main neural results use five seeds, and calibration is performed from the corresponding out-of-fold ensemble predictions. The future-speed filter is explicitly identified as an offline tracking-quality screen, and an all-speed audit removes it; GEL-Ped still wins 11 of 13 crossing runs. Conformal language has been corrected to “cross-conformal-like,” with no finite-sample guarantee claimed under dependent frames or topology shift. Guard-conditional empirical radii and their observed coverage are reported, including the failure to restore nominal coverage under crossing shift.

### 10. Run partition transparency

Appendix Table A.1 gives every calibration, familiar, and altered run with participant count, duration, entrance width, corridor length, exit width, nominal density, and number of forecasts. The text defines altered geometry a priori as a changed corridor length and/or exit arrangement relative to the 22-m, 5-m-exit calibration layout. It also explains that the seven calibration runs were selected to span entrance width and density before test-error inspection. Regime labels and complete-run partitions are immutable in the released configuration.

## Minor comments

1. Revision-history language has been removed from the manuscript.
2. “Reliable” and “geometry-guarded” have been removed from the title.
3. Table 1 defines every causal feature block and its operational meaning.
4. The former “conflict probability” is now described as a noisy-OR conflict index.
5. Relative-velocity expressions state their numerical regularization.
6. The guard equation uses a velocity norm, reports units of m/s, and defines its sampled population and calibration quantile.
7. The abstract reports the crossing effect, its outlier sensitivity, and the near-zero naturalistic macro effect.
8. The unpublished predecessor is no longer used to motivate or validate the method.
9. Unsupported MLP false-alarm prose was removed; the revised collision audit gives complete counts for the two central models.
10. Figure 3 defines CV and states that plotted ranges are complete-run ranges rather than confidence intervals; instability-bin counts are printed on the axis.
11. The contribution list now separates the method, operating point, and mechanism-focused evaluation.
12. The ETH/UCY source commit is stated in the Data Availability section.
13. The related-work section now discusses Tordeux et al. (2020), Bahari et al. (2021), Xu et al. (2022), Korbmacher and Tordeux (2022), Jacobs et al. (1991), and recent deployment-shift work. Relevant work by Milad Haghani is also integrated where transportability and empirical crowd-data limitations are discussed.

## Resulting scope

The revised manuscript supports a narrower but better-identified conclusion. GEL-Ped does not materially improve ordinary in-support averages, does not beat the coarse learner in the crossing mean, and does not show a significant aggregate advantage on the five untouched naturalistic scenes. Its value is that it keeps the stronger graph predictor fully active in supported flow while detecting and limiting a small, practically relevant tail of neural ensemble instability. This is the claim now made consistently in the title, abstract, results, discussion, and conclusion.
