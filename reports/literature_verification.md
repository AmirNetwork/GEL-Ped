# Literature verification for the referee-3 revision

Verified on 2 October 2026 (Australia/Sydney). This note records only metadata supported by a publisher DOI landing page, an official proceedings page, or an official author/institutional record. The accompanying `referee3_references.bib` is intentionally separate from the manuscript bibliography so that entries can be selected after the revision strategy is fixed.

## Verification rules used

- A DOI is included only when it resolves to, or is independently confirmed by, the publisher/official record for the named work.
- NeurIPS and PMLR papers without publisher-assigned DOIs use their official proceedings URLs; arXiv DOIs are not substituted for the conference record.
- Article numbers in *Transportation Research Part C* are recorded in the `pages` field because that is the convention used by the journal's metadata.
- A citation below supports only the claim described in the “safe use” column. Several works are classification or autonomous-driving papers, so they should motivate a design principle rather than be presented as pedestrian-forecasting evidence.

## Verified sources and safe manuscript use

### 1. Runtime assurance and disagreement-triggered recovery

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `sha2001simplicity` | Lui Sha, “Using simplicity to control complexity,” *IEEE Software*, 18(4), 20–28, 2001 | DOI: [10.1109/MS.2001.936213](https://doi.org/10.1109/MS.2001.936213); metadata also confirmed by the [University of Illinois record](https://experts.illinois.edu/en/publications/using-simplicity-to-control-complexity/) | Canonical motivation for a high-performance component supervised by a simpler, independently understandable fallback. Cite when describing GEL-Ped as a runtime-assurance or Simplex-like pattern. Do not claim that this paper studies trajectory prediction. |
| `filos2020distribution` | Angelos Filos, Panagiotis Tigkas, Rowan McAllister, Nicholas Rhinehart, Sergey Levine, and Yarin Gal, “Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?”, ICML 2020, PMLR 119, 3145–3153 | [Official PMLR record](https://proceedings.mlr.press/v119/filos20a.html); no publisher DOI | Direct precedent for using ensemble disagreement/epistemic uncertainty to identify novel driving scenes and trigger recovery or expert intervention. It supports the operational pattern, not a claim that the same score is calibrated for pedestrians. |

Recommended manuscript placement: one compact paragraph near the end of the introduction. Frame GEL-Ped as a *runtime safeguard for an existing predictor*, not as another general-purpose forecasting backbone. The central contrast is “replace or temper the backbone only when evidence suggests that its local residual is unreliable.”

### 2. Selective prediction and reject-option learning

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `geifman2017selective` | Yonatan Geifman and Ran El-Yaniv, “Selective Classification for Deep Neural Networks,” NeurIPS 2017, volume 30 | [Official NeurIPS record](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html); no publisher DOI | Establishes the risk–coverage view: abstention can trade coverage for lower retained risk. Because this is classification, cite it as conceptual selective-prediction provenance, not as a directly comparable regression method. |
| `cortes2016rejection` | Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri, “Learning with Rejection,” in *Algorithmic Learning Theory*, LNCS 9925, 67–82, 2016 | DOI: [10.1007/978-3-319-46379-7_5](https://doi.org/10.1007/978-3-319-46379-7_5); [official Springer volume](https://link.springer.com/book/10.1007/978-3-319-46379-7) confirms authors and pages | Formal reject-option background. Use to explain why a gate must be evaluated jointly with its coverage/usage rate and fallback cost. Do not imply that GEL-Ped implements their loss verbatim. |

Recommended manuscript placement: cite both in the definition of the retention curve and state that GEL-Ped performs *selection between experts* rather than pure abstention. Report coverage, retained/direct risk, routed risk, and full-system risk together.

### 3. Uncertainty under shift and risk–retention evaluation

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `ovadia2019uncertainty` | Yaniv Ovadia et al., “Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,” NeurIPS 2019, volume 32 | [Official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html); no publisher DOI | Supports the claim that in-distribution calibration does not by itself establish reliable uncertainty after distribution shift. It motivates evaluation on unseen scenes/shifts and not merely random splits. |
| `malinin2021shifts` | Andrey Malinin et al., “Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks,” NeurIPS 2021 Datasets and Benchmarks Track | [Official NeurIPS record](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ad61ab143223efbc24c7d2583be69251-Abstract-round2.html) and [official project citation](https://shifts.ai/dataset); no publisher DOI | Strong source for retention-based assessment under real shift, including motion prediction. Use for error/risk–retention curves and summary area measures. Make clear whether GEL-Ped orders samples by uncertainty, disagreement, or routing priority. |

Recommended manuscript placement: evaluation section. Define the ordering score, the retained fraction, the loss averaged on the retained subset, and the direction of improvement. If reporting an area under a risk–retention curve, specify numerical integration and whether lower is better. Do not label a routing curve “calibration” unless an actual probabilistic calibration quantity is tested.

### 4. Mahalanobis out-of-distribution scores

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `lee2018mahalanobis` | Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin, “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks,” NeurIPS 2018, volume 31 | [Official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html); no publisher DOI | Provenance for a Gaussian-feature Mahalanobis confidence/OOD score. State explicitly that Lee et al. use class-conditional deep features, whereas GEL-Ped applies a related distance in its own feature/residual space. The citation does not validate GEL-Ped's threshold or Gaussian assumption. |

Recommended manuscript placement: method section immediately after the gate score. Define the reference mean/covariance, regularisation, fitting split, and threshold selection. Call it a “Mahalanobis shift score” unless the paper demonstrates OOD-detection performance on labelled in/out distributions.

### 5. Online expert weighting

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `herbster1998tracking` | Mark Herbster and Manfred K. Warmuth, “Tracking the Best Expert,” *Machine Learning*, 32(2), 151–178, 1998 | DOI: [10.1023/A:1007424614876](https://doi.org/10.1023/A:1007424614876) | Canonical source for tracking a changing best expert by online loss updates. Cite if GEL-Ped updates neural/geometry expert weights after delayed outcomes arrive. |
| `cesabianchi2006prediction` | Nicolò Cesa-Bianchi and Gábor Lugosi, *Prediction, Learning, and Games*, Cambridge University Press, 2006 | DOI: [10.1017/CBO9780511546921](https://doi.org/10.1017/CBO9780511546921); [official Cambridge record](https://www.cambridge.org/core/books/prediction-learning-and-games/A05C9F6ABC752FAB8954C885D0065C8F) | General source for prediction with expert advice and regret-based sequential aggregation. Use for the update rule and regret terminology, not to claim a regret guarantee unless GEL-Ped's assumptions and proof match the cited setting. |

Recommended manuscript placement: method section. Give the exact loss, learning rate, update timing, delayed-label protocol, normalisation, initial weights, and reset policy. A lagged online-expert baseline with the same feedback timing is essential; otherwise the gain cannot be attributed to the proposed geometry signal.

### 6. Adaptive conformal uncertainty

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `gibbs2021adaptive` | Isaac Gibbs and Emmanuel Candès, “Adaptive Conformal Inference Under Distribution Shift,” NeurIPS 2021, volume 34 | [Official NeurIPS record](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html); no publisher DOI | Establishes adaptive online conformal inference under time-varying distributions. Cite as a principled alternative/extension for uncertainty sets, not as evidence that a heuristic GEL-Ped gate has conformal coverage. |
| `dixit2023adaptive` | Anushri Dixit, Lars Lindemann, Skylar X. Wei, Matthew Cleaveland, George J. Pappas, and Joel W. Burdick, “Adaptive Conformal Prediction for Motion Planning among Dynamic Agents,” L4DC 2023, PMLR 211, 300–314 | [Official PMLR record](https://proceedings.mlr.press/v211/dixit23a.html); no publisher DOI | Particularly relevant precedent for delayed observations, multi-step trajectory uncertainty sets, and downstream motion planning. It motivates a coverage experiment and a planner-facing interpretation of the safeguard. |

Recommended manuscript placement: related work and limitations. Unless conformal sets are implemented and empirically covered, say “complementary future direction,” not “calibrated guarantee.”

### 7. Online/test-time trajectory adaptation

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `ivanovic2023adaptive` | Boris Ivanovic, James Harrison, and Marco Pavone, “Expanding the Deployment Envelope of Behavior Prediction via Adaptive Meta-Learning,” ICRA 2023, 7786–7793 | DOI: [10.1109/ICRA48891.2023.10161155](https://doi.org/10.1109/ICRA48891.2023.10161155); publication venue confirmed by the [official NVIDIA research record](https://research.nvidia.com/publication/2023-05_expanding-deployment-envelope-behavior-prediction-adaptive-meta-learning) | Direct precedent for adapting a trajectory/behavior predictor to unseen domains through an adaptive Bayesian layer. Use to distinguish parameter adaptation from GEL-Ped's routing/aggregation. |
| `park2024t4p` | Daehee Park, Jaeseok Jeong, Sung-Hoon Yoon, Jaewoo Jeong, and Kuk-Jin Yoon, “T4P: Test-Time Training of Trajectory Prediction via Masked Autoencoder and Actor-specific Token Memory,” CVPR 2024, 15065–15076 | DOI: [10.1109/CVPR52733.2024.01427](https://doi.org/10.1109/CVPR52733.2024.01427); [official CVF record](https://openaccess.thecvf.com/content/CVPR2024/html/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.html) | Recent trajectory-specific test-time training under cross-dataset shift. Cite when positioning GEL-Ped as lightweight expert selection rather than deep representation adaptation. It is an important comparator in discussion; implementation is only necessary if inputs/datasets can be matched fairly. |

Recommended manuscript placement: related work. Explicitly distinguish three deployment responses: update the predictor (adaptive meta-learning/test-time training), quantify a prediction set (adaptive conformal), or retain a fixed backbone and route/aggregate with a transparent fallback (GEL-Ped).

### 8. Trajectory-forecasting surveys and benchmark context

| Key | Verified record | Primary locator | Safe use in GEL-Ped |
|---|---|---|---|
| `rudenko2020survey` | Andrey Rudenko, Luigi Palmieri, Michael Herman, Kris M. Kitani, Dariu M. Gavrila, and Kai O. Arras, “Human Motion Trajectory Prediction: A Survey,” *The International Journal of Robotics Research*, 39(8), 895–935, 2020 | DOI: [10.1177/0278364920917446](https://doi.org/10.1177/0278364920917446); [official SAGE record](https://journals.sagepub.com/doi/10.1177/0278364920917446) | Broad taxonomy of human motion prediction, datasets, metrics, and context levels. Use to establish the field and application need. |
| `kothari2022perspective` | Parth Kothari, Sven Kreiss, and Alexandre Alahi, “Human Trajectory Forecasting in Crowds: A Deep Learning Perspective,” *IEEE Transactions on Intelligent Transportation Systems*, 23(7), 7386–7400, 2022 | DOI: [10.1109/TITS.2021.3069362](https://doi.org/10.1109/TITS.2021.3069362); metadata/page range confirmed by the [TRID record](https://trid.trb.org/View/1993950) and linked IEEE record | Use for social-interaction modelling, TrajNet++, and socially aware evaluation. It supports the need to go beyond mean ADE/FDE, especially in crowded interactions. |

Recommended manuscript placement: opening related-work paragraph and evaluation design. These surveys should not be used as substitutes for citing the original baseline methods actually implemented.

### 9. Recent *Transportation Research Part C* positioning

| Key | Verified record | Primary locator | What it establishes for the story |
|---|---|---|---|
| `liu2024interpretable` | Qiujia Liu, Xiaodan Shi, Renhe Jiang, Haoran Zhang, Linjun Lu, and Ryosuke Shibasaki, “Modeling interpretable social interactions for pedestrian trajectory,” *TR-C*, 162, 104617, 2024 | DOI: [10.1016/j.trc.2024.104617](https://doi.org/10.1016/j.trc.2024.104617); [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S0968090X24001384) | Shows that interpretability and heterogeneous social-interaction modes are current transport-research concerns. Contrast its learned interaction modes with GEL-Ped's deployment-time safeguard; do not imply identical prediction horizons/tasks. |
| `mai2025diffusion` | Weiming Mai, Dorine Duives, and Serge Hoogendoorn, “A learning based pedestrian flow prediction approach with diffusion behavior,” *TR-C*, 179, 105243, 2025 | DOI: [10.1016/j.trc.2025.105243](https://doi.org/10.1016/j.trc.2025.105243); [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S0968090X25002475) | Recent hybrid, interpretable and online-calibrated pedestrian-flow work. It is highly relevant to the journal narrative but predicts aggregate flow, not individual short-horizon trajectories; make that distinction explicit. |
| `munir2025context` | Farzeen Munir and Tomasz Piotr Kucner, “Context-aware multi-task learning for pedestrian intent and trajectory prediction,” *TR-C*, 178, 105203, 2025 | DOI: [10.1016/j.trc.2025.105203](https://doi.org/10.1016/j.trc.2025.105203); [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S0968090X25002074) | Establishes the value of contextual and intention cues for road-crossing prediction. It uses rich visual/context features and different datasets; position it as complementary rather than a matched baseline unless those modalities are available. |
| `li2023cyclist` | Jianqiang Li, Ying Ni, and Jian Sun, “A two-layer integrated model for cyclist trajectory prediction considering multiple interactions with the environment,” *TR-C*, 155, 104304, 2023 | DOI: [10.1016/j.trc.2023.104304](https://doi.org/10.1016/j.trc.2023.104304); [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S0968090X23002930) | Strong transport-domain precedent for combining learned trajectory generation with an interpretable behavioural/utility layer to improve rare sudden-change cases. It is cyclist-specific, so use it to motivate the hybrid design principle rather than as a directly matched pedestrian baseline. |

Recommended manuscript placement: a short final paragraph in related work that identifies a precise gap: recent transport studies improve the *predictor itself* through interaction, context, intention, flow physics, or utility structure; GEL-Ped instead addresses *when a deployed short-horizon predictor should be trusted and how it should be safely complemented after only causally available feedback*. That claim is defensible only if the revised experiment enforces causal timing and reports system-level risk at matched coverage.

## Recommended citation bundle by manuscript section

1. **Introduction / operational need:** `rudenko2020survey`, `kothari2022perspective`, `sha2001simplicity`, `filos2020distribution`.
2. **Related work / transport positioning:** `liu2024interpretable`, `mai2025diffusion`, `munir2025context`, `li2023cyclist`, `ivanovic2023adaptive`, `park2024t4p`.
3. **Method / gate provenance:** `lee2018mahalanobis`, `herbster1998tracking`, `cesabianchi2006prediction`.
4. **Evaluation / selection under shift:** `geifman2017selective`, `cortes2016rejection`, `ovadia2019uncertainty`, `malinin2021shifts`.
5. **Uncertainty limitations and future safety integration:** `gibbs2021adaptive`, `dixit2023adaptive`.

### Integration note for the current bibliography

Four verified records are already present in `manuscript/overleaf_trc/references.bib` under different keys: `kothari2022`, `liu2024`, `mai2025`, and `munir2025`. When integrating the staging file, retain those existing keys (and enrich their fields if desired) rather than adding the corresponding `kothari2022perspective`, `liu2024interpretable`, `mai2025diffusion`, and `munir2025context` entries a second time. No other DOI in the staging file matched the current bibliography at verification time.

## Claims to avoid

- Do not call a Mahalanobis distance a calibrated probability or a conformal guarantee.
- Do not claim “runtime assurance” solely from improved mean ADE; assurance requires an explicit supervisor/fallback contract and failure-oriented evaluation.
- Do not present classification reject-option results as empirical evidence for pedestrian regression.
- Do not present aggregate pedestrian-flow prediction, cyclist prediction, or camera-based crossing-intent prediction as matched baselines.
- Do not claim that online expert weighting is novel. The novelty must reside in the causal geometry-derived signal, the deployment contract, and evidence that it improves routing beyond generic lagged-loss weighting.
- Do not claim robustness to distribution shift from a random holdout. Use a scene/domain/temporal shift that is untouched during threshold selection.

## Bibliographic cautions resolved

- The correct ICRA DOI for Ivanovic, Harrison, and Pavone is **10.1109/ICRA48891.2023.10161155**. A nearby DOI ending in `10160622` appears in an ambiguous profile listing and was not used.
- The official CVF page gives T4P's proceedings pages as **15065–15076**; the IEEE DOI is **10.1109/CVPR52733.2024.01427**.
- Kothari et al. is the 2022 paginated journal issue record, volume **23**, issue **7**, pages **7386–7400**, even though its DOI contains 2021 (online-first year).
- No publisher DOI was found for the cited NeurIPS or PMLR conference records; none has been invented.
