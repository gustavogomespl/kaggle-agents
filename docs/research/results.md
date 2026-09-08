# Results by competition

[← README](../../README.md#results) · [Evidence and provenance](evidence.md) · [Download CSV](results_by_competition.csv)

The table transcribes the reported scores and medals from the [preserved per-competition result table](sources/tab1_resultados_por_competicao.tex). Each cohort contains 22 tasks. These are historical outcomes, not freshly executed results or controlled replications. R1 is a search-enabled cohort with contamination evidence; see the evidence guide.

Scores retain the four decimal places of the source table. **G** = gold, **S** = silver, **B** = bronze, **—** = valid with no medal, **invalid** = no valid submission. ↑ means maximize and ↓ minimize. Medal labels are transcribed, not recomputed from rounded scores.

| Competition | Metric | R1 · search enabled | R2 · full | R3 · full | Without search |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [aerial-cactus-identification](https://www.kaggle.com/competitions/aerial-cactus-identification) | auc ↑ | 1.0000 · G | 1.0000 · G | 0.9999 · — | 1.0000 · G |
| [aptos2019-blindness-detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | quadratic weighted kappa ↑ | 0.8596 · — | 0.8678 · — | 0.8846 · — | 0.8884 · — |
| [denoising-dirty-documents](https://www.kaggle.com/competitions/denoising-dirty-documents) | rmse ↓ | 0.0208 · S | 0.0127 · G | 0.0151 · G | 0.0144 · G |
| [detecting-insults-in-social-commentary](https://www.kaggle.com/competitions/detecting-insults-in-social-commentary) | auc ↑ | 0.9381 · G | 0.9113 · G | 0.9104 · G | 0.9144 · G |
| [dog-breed-identification](https://www.kaggle.com/competitions/dog-breed-identification) | log loss ↓ | 0.4292 · — | 0.6236 · — | 0.3320 · — | 0.7124 · — |
| [dogs-vs-cats-redux-kernels-edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition) | log loss ↓ | 0.0170 · G | 0.0139 · G | 0.0153 · G | 0.0136 · G |
| [histopathologic-cancer-detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection) | auc ↑ | 0.9938 · G | 0.9972 · G | 0.9886 · G | 0.9953 · G |
| [jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) | auc ↑ | 0.9864 · B | 0.9777 · — | 0.9727 · — | 0.9724 · — |
| [leaf-classification](https://www.kaggle.com/competitions/leaf-classification) | log loss ↓ | 0.0378 · — | 0.0462 · — | 0.0044 · S | 0.0311 · — |
| [mlsp-2013-birds](https://www.kaggle.com/competitions/mlsp-2013-birds) | auc ↑ | 0.4943 · — | 0.5000 · — | 0.5538 · — | 0.5508 · — |
| [new-york-city-taxi-fare-prediction](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction) | rmse ↓ | 5.2454 · — | 5.0278 · — | 4.3534 · — | 4.2277 · — |
| [nomad2018-predict-transparent-conductors](https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors) | rmsle ↓ | 0.0594 · S | 0.0589 · S | 0.0597 · S | 0.0605 · S |
| [plant-pathology-2020-fgvc7](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7) | auc ↑ | 0.9936 · G | 0.9991 · G | 0.9979 · G | 0.9923 · G |
| [random-acts-of-pizza](https://www.kaggle.com/competitions/random-acts-of-pizza) | auc ↑ | 0.7870 · S | 0.7907 · S | 0.7972 · S | 0.7780 · S |
| [ranzcr-clip-catheter-line-classification](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification) | auc ↑ | 0.9383 · — | 0.8530 · — | 0.8612 · — | 0.8658 · — |
| [siim-isic-melanoma-classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) | auc ↑ | 0.8002 · — | 0.8697 · — | 0.8865 · — | invalid |
| [spooky-author-identification](https://www.kaggle.com/competitions/spooky-author-identification) | log loss ↓ | 0.2847 · B | 0.2890 · B | 0.2849 · B | 0.3807 · — |
| [tabular-playground-series-dec-2021](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021) | accuracy ↑ | 0.9592 · G | 0.9632 · G | 0.9633 · G | 0.9591 · G |
| [tabular-playground-series-may-2022](https://www.kaggle.com/competitions/tabular-playground-series-may-2022) | auc ↑ | 0.9871 · — | 0.9945 · — | 0.9944 · — | 0.9963 · — |
| [text-normalization-challenge-english-language](https://www.kaggle.com/competitions/text-normalization-challenge-english-language) | accuracy ↑ | 0.9906 · B | 0.9906 · B | 0.9904 · B | 0.9906 · B |
| [text-normalization-challenge-russian-language](https://www.kaggle.com/competitions/text-normalization-challenge-russian-language) | accuracy ↑ | 0.9700 · — | 0.9735 · — | 0.9735 · — | 0.9735 · — |
| [the-icml-2013-whale-challenge-right-whale-redux](https://www.kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux) | auc ↑ | 0.9781 · S | 0.9797 · S | 0.9519 · S | invalid |

## Reading the data

- Compare raw scores only within one competition; tasks use different metrics and scales.
- Count medal and valid-submission rates over all 22 tasks, including invalid outcomes.
- A displayed `1.0000` or a near-identical score does not establish the same unrounded result. Aerial Cactus illustrates why recorded medal labels matter.
- Right Whale Redux and SIIM-ISIC have empty numeric scores and `valid_submission=false` in the CSV for the retrieval-disabled cohort; they are not zero-score observations.
- The CSV is a transcription of the preserved result table. It is not a replacement for immutable raw grading reports, and the supplied notebook archive does not give an unambiguous one-to-one mapping for all 88 outcomes.

## Regenerate the charts

```bash
python docs/assets/render_results.py  # run from the repository root; requires matplotlib
```

The plotting script reads the CSV, retains invalid outcomes, and writes the [cohort chart](../assets/cohort-results.svg) and [competition matrix](../assets/competition-matrix.svg) as SVG and PNG. It does not run the agents or call a model.
