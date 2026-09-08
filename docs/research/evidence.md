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

## Reconciling results and execution records

### R1 protocol and test feedback

Saved execution evidence matching R1 includes target-competition retrieval and test feedback:

- `denoising-dirty-documents_3.ipynb`, `cells[12].outputs[0]`, searches for the target competition's winning solutions and analyzes retrieved notebooks. Its output later records `MLE-bench grade: score=0.02083 above_median=True`, followed immediately by `Objective reached (MLE-bench) - stopping remaining components`. The final 0.02083/Silver result matches the preserved table's R1 0.0208/Silver row.
- `siim-isic-melanoma-classification_3.ipynb`, `cells[19].outputs[0]`, grades intermediate candidates and updates `submission_best.csv` using the improved MLE-bench score. Its final score of 0.80018 matches the reported R1 0.8002.

Cell/output references are **zero-based Jupyter JSON indices**. These observations document contamination evidence in the search-enabled R1 cohort. They do not establish the exact history of every R1 task. Current code protections cannot retroactively establish a clean historical protocol.

### Seeds and configuration

The supplied notebook folder contains **63 seed-bearing summary objects**, all with seed **42**: 42 labeled `full` and 21 labeled `without-search`. These objects include duplicate or alternative saved runs; they must not be counted as independently established experimental units. They do not demonstrate multiple independent seeds.

Effective configurations also vary. Stored full-arm output headers include iteration ceilings of 4, 5, and 10; retrieval-disabled headers include 3, 4, and 10. For example, `dog-breed-identification_sem_busca.ipynb`, `cells[10].outputs[0]`, records seed 42, `without-search`, `Max iterations: 3`, and a component timeout of 3800 seconds before finishing with score 0.71244, matching the table's 0.7124.

### Uncertainty and causal interpretation

Pooling R1–R3 gives 37/66 = 56.1%. A binomial standard error of approximately 6.1 percentage points would treat those 66 observations as independent. It would not establish uncertainty across three independently seeded, fixed-protocol evaluations. The same tasks recur, and the configurations and historical protocols differ.

Accordingly, the README reports each cohort separately and omits a pooled error bar. Overlapping error bars across different studies do not establish architectural equivalence, and GPU-hour ratios across different accelerators, models, and execution setups do not establish relative efficiency.

Disabling search jointly removes three mechanisms: initial retrieval, reactive retrieval, and external data-loading examples. One heterogeneous historical cohort cannot isolate the causal effect of Search First or establish that retrieval caused improved robustness. Lower medal counts and invalid submissions remain useful descriptive observations.

## Coverage of the notebook evidence

Notebook filenames are not reliable identifiers of the recorded task. For example, `spooky-author-identification_2.ipynb` ends with a **Plant Pathology** result, score 0.99789/Gold. The review used recorded competition IDs, scores, medal labels, and arm indicators instead of assuming that filename suffixes identify rounds.

Recognizable final tables provide numerical/medal/arm matches for 8 R1 rows, 17 R2 rows, 18 R3 rows, and 21 retrieval-disabled rows. **Matching scores do not prove a unique mapping from notebook to cohort.** This is conservative extraction coverage, not a claim that unmatched outcomes never occurred. The extra SIIM notebook provides invalid-submission evidence but does not contain a comparable final cohort summary.

Two companion files do not resolve that gap:

- `mlebench_lite_mapa_performance.xlsx` contains a partial snapshot with 13 completed tasks and 9 pending; it is not a four-cohort ledger.
- `mlebench_results_summary.csv` contains a single failed Aerial Cactus run; it does not support aggregate results.

There is also an unresolved Aerial Cactus discrepancy: the `_12` and `_14` notebooks report 0.99997/no medal with execution time 27634.9 seconds; the spreadsheet reports 1.0/Gold with that same duration. The README preserves the source table's reported R1 Gold label, but does not call it independently verified by that notebook or spreadsheet pair.

### The two invalid retrieval-disabled outcomes

| Task | Directly available evidence | Limit of the diagnosis |
| :--- | :--- | :--- |
| Right Whale Redux | `the-icml-2013-whale-challenge-right-whale-redux_sem_busca.ipynb`, `cells[10].outputs[0]`: invalid submission and robustness validation failure; `cells[11].outputs[0]`: seed 42, `without-search`, zero valid submissions | Supports a gate-blocked candidate in that stored execution |
| SIIM-ISIC Melanoma | `siim_isic (1).ipynb`: `cells[6].source` enables `KAGGLE_AGENTS_ABLATE_SEARCH`; `cells[20].outputs[0]` records missing/invalid submission grading | The preserved notebook does not establish why finalization failed or prove its identity as the specific SIIM outcome in the result table |

The supplied SIIM notebook does not independently establish a root cause for the missing/invalid submission. Both notebook copies inspected have SHA-256 `7ad0449a8f2231851941c4f784a2350ad750483228435d733c67a7dce7268847`.

## From current implementation to a reproducible experiment

The README's architecture and controls were checked against the current checkout. A reproducible future study needs to record more than the current branch name:

1. Freeze code, dependency lockfile, prompts, retrieval corpus, dataset/grader versions, and model/provider identifiers.
2. Define task × seed × arm units in advance, with matched hardware, iteration ceilings, component timeouts, and externally enforced wall-clock limits.
3. Preserve final submissions and hashes, grading reports, allowed local validation artifacts, search audit, and configuration fingerprints.
4. Retain unsuccessful agent outcomes in the analysis. Separately identify provider, environment, and harness failures without selecting reruns by score.
5. Evaluate multiple seeds and report aggregate uncertainty at the appropriate evaluation-unit level, following the [official MLE-bench benchmarking guidance](https://github.com/openai/mle-bench#benchmarking).

The versioned notebooks are runnable entry points for new studies. They do not reconstruct all historical environment states or certify an already completed matched MLE-STAR baseline.

## Artifact checksums

SHA-256 identifies the exact supplied snapshots retained in this documentation; it does not certify the scientific validity of their claims.

| Preserved file | SHA-256 |
| :--- | :--- |
| [Per-competition TeX table](sources/tab1_resultados_por_competicao.tex) | `c65ae7aadb2c827e36acb8c5ae66a811bd7f132db1e7253f341eec36cf771797` |
| [Aggregate TeX table](sources/tab3_agregados.tex) | `aef6e273f4633f4f0e593096d5000a4c8fde6bb738861541fe891fdd9fec344b` |
| [Original pipeline PDF](../assets/pipeline-paper.pdf) | `e9b120dc9e7c7320cce65a6f717d333bbc2f7bec0b1cc5e2e06c5f63975ae2e9` |

The new plots are derived from the [CSV](results_by_competition.csv) by [the plotting script](../assets/render_results.py). Their SVG and PNG versions are static artifacts that can be regenerated without model calls. The overview is an editable SVG summarizing the workflow; the original architecture diagram is preserved separately.
