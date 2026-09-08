# Evidence and reproducibility

[← README](../../README.md) · [Full result table](results.md) · [CSV](results_by_competition.csv)

This guide explains what supports the README's claims, how the historical results were transcribed, and where preserved result tables and execution records disagree. The documentation review read saved notebook cells and outputs; **it did not execute notebooks, call models, train candidates, or rerun grading**.

## Sources and their roles

| Source | Used for | Availability here |
| :--- | :--- | :--- |
| `tex/tab1_resultados_por_competicao.tex` | Source for transcription of the reported 22 × 4 score/medal table | [Preserved table source](sources/tab1_resultados_por_competicao.tex) |
| `tex/tab3_agregados.tex` | Cross-check of cohort counts, medal breakdowns, and recorded mean times | [Preserved aggregate table](sources/tab3_agregados.tex) |
| `tex/figuras/pipeline.pdf` | Original architecture diagram | [Vector source](../assets/pipeline-paper.pdf), [PNG rendering](../assets/pipeline-paper.png) |
| Supplied `Kaggle_Agentes/notebooks/` archive | Stored execution evidence, seed/configuration observations, and historical protocol differences | 115 notebooks inspected; raw archive is not bundled |
| `siim_isic (1).ipynb` in the workspace and supplied repository copy | Additional evidence for the missing SIIM-ISIC submission in the retrieval-disabled configuration | Hash and cell references below; raw notebook is not bundled here |
| Supplied `outros/` CSV and XLSX | Check whether companion summaries independently cover the cohorts | Partial/stale snapshots; not used as the aggregate source |
| Current repository source | Architecture, CLI commands, output artifacts, and implemented controls | Linked from the README |

The preserved result tables are used as a **reported historical record**, not as an immutable raw experiment ledger. Stored execution evidence supports only the observations and matches described below; it does not independently verify every reported outcome.

## Reported counts

The [CSV](results_by_competition.csv) has 88 rows: one reported outcome for each of 22 tasks in each of four cohorts. Its scores and medals agree with the [preserved per-competition table](sources/tab1_resultados_por_competicao.tex). Aggregating its rows gives:

| Cohort | Valid | Gold | Silver | Bronze | Any medal |
| :--- | ---: | ---: | ---: | ---: | ---: |
| R1 | 22 | 6 | 4 | 3 | 13/22 = 59.1% |
| R2 | 22 | 7 | 3 | 2 | 12/22 = 54.5% |
| R3 | 22 | 6 | 4 | 2 | 12/22 = 54.5% |
| Without search | 20 | 7 | 2 | 1 | 10/22 = 45.5% |

Invalid outcomes stay in the denominator. They have an empty `score`, `medal=none`, and `valid_submission=false`, so a failed execution is distinguishable from a valid submission below all medal thresholds. Scores retain the source table's four decimal places. Medal labels are not regenerated from those rounded scores.

The [aggregate table](sources/tab3_agregados.tex) reports mean times of 6.92, 5.13, 5.19, and 4.68 hours per task. The available archive does not provide a complete immutable mapping of model, hardware, code, and duration for every reported outcome. Model identifiers also vary across stored logs.
