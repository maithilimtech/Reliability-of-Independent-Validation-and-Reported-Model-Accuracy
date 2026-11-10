# Predicting pathological complete response before surgery

Fourth ICABTTEC 2027 project. Fits Theme 05 (Cancer Biology, Targeted Therapy, Diagnostics) and Theme 07 (Precision Medicine). Abstracts due **01 Dec 2026**, full papers **05 Dec 2026**.

Results only — write the abstract and discussion yourself with your supervisor, and declare any AI assistance.

## Files

| File | Contents |
|---|---|
| `Neoadjuvant_pCR_Prediction.ipynb` | Full analysis, executed with figures inline. Colab or Kaggle. |
| `neoadjuvant_pcr_ml.py` | Same code as a plain script. |
| `data/` | Both GEO series matrices (`SOURCE.md` documents and cites them). |
| `figures/` | Figures 1–5, 300 dpi. |
| `results/` | Tables 1–5 as CSV; `key_numbers.json` holds every number quoted below. |

The notebook re-downloads anything missing from GEO, so `data/` is a convenience.

## Data

Two neoadjuvant breast-cancer cohorts on the same array (Affymetrix HG-U133A, GPL96, 22,283 shared probes), both recording pCR vs residual disease with matched clinical variables:

| | Patients | pCR | Rate |
|---|---|---|---|
| GSE25066 (train) | 488 | 99 | 20.3% |
| GSE20194 (test) | 278 | 56 | 20.1% |

## The finding that reframes everything else

**187 of the 278 "test" patients are the same arrays as training patients** — Pearson r > 0.99 across all 22,283 probes, median 0.999. GSE25066 pooled several neoadjuvant trials including the MD Anderson series that constitutes GSE20194.

Only **91** GSE20194 patients (25 pCR) are genuinely independent.

This pair is a standard train/test split in the pCR-prediction literature. Using it without a duplicate check reports a number that is mostly resubstitution.

## Results

**1. The contaminated "external validation" is wildly optimistic; internal CV was honest.**

| Model | Internal CV (GSE25066) | External, all 278 | External, 91 independent |
|---|---|---|---|
| ER status alone | 0.679 | 0.758 | **0.708** |
| Clinical only | 0.727 ± 0.010 | 0.798 | **0.713** |
| Expression only | 0.780 ± 0.010 | 0.959 | **0.789** |
| Clinical + expression | 0.780 ± 0.010 | 0.963 | **0.796** |

Cross-validation inside the training cohort predicted 0.780; true independent performance was 0.796. The *external* number was the misleading one, at 0.963.

This inverts the usual lesson. Internal validation is normally the optimistic estimate and an external cohort is the corrective. Here the external cohort was the trap, because it was not external.

**2. Clinical variables add nothing over ER status.** 0.713 against 0.708. Age, grade, T and N stage and HER2 contribute essentially nothing beyond receptor status — unsurprising given pCR rates of 35% in ER-negative against 11% in ER-positive disease.

**3. Expression adds modestly and genuinely.** 0.796 against 0.708 for ER alone, roughly +0.09 AUC. Within ER-negative patients (n = 49, 21 pCR) expression reaches 0.714 against 0.594 for clinical variables, so it is not purely recovering receptor status.

**4. The normalisation choice barely matters here.** Train-fitted scaling and per-cohort z-scoring give 0.796 and 0.793 on the independent subset. Worth reporting because the choice is usually unstated, and because per-cohort z-scoring is not available for a single incoming patient.

## Reading of the results

The headline is a data-provenance failure, not a modelling one. Two of the most widely used public neoadjuvant cohorts share two-thirds of their patients, and nothing in either GEO record says so. A duplicate check costs one correlation matrix and should precede any claim of external validation.

The secondary result is that a well-conducted internal cross-validation can be close to unbiased. The reflex that "internal CV is optimistic, external validation is truth" is wrong in both directions when provenance is unchecked.

For the clinical claim: an expression signature does add about 0.09 AUC over receptor status for predicting pCR. That is real but small, and at 91 patients the interval around it is wide.

## Caveats

- **91 patients and 25 events.** Every independent-cohort estimate has a wide confidence interval. The point estimate 0.796 should be read as "roughly 0.8, imprecisely".
- **The ER-positive stratum has 4 pCR events in 42 patients.** Its apparent AUC of 0.842 is noise and should not be quoted.
- Duplicate detection at r > 0.99 is conservative; the correlation histogram is clearly bimodal, so the threshold is not delicate, but near-duplicates below it may remain.
- Both cohorts are taxane/anthracycline era. Results do not transfer to current regimens including carboplatin, immunotherapy or anti-HER2 agents.
- pCR is a surrogate endpoint. Its relationship to survival differs by subtype.

## References

1. Hatzis C, Pusztai L, Valero V, et al. A genomic predictor of response and survival following taxane-anthracycline chemotherapy for invasive breast cancer. *JAMA.* 2011;305(18):1873-1881.
2. Shi L, Campbell G, Jones WD, et al. The MicroArray Quality Control (MAQC)-II study of common practices for the development and validation of microarray-based predictive models. *Nat Biotechnol.* 2010;28(8):827-838.
3. Cortazar P, Zhang L, Untch M, et al. Pathological complete response and long-term clinical benefit in breast cancer: the CTNeoBC pooled analysis. *Lancet.* 2014;384(9938):164-172.
4. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns.* 2023;4(9):100804.
5. Simon R, Radmacher MD, Dobbin K, McShane LM. Pitfalls in the use of DNA microarray data for diagnostic and prognostic classification. *J Natl Cancer Inst.* 2003;95(1):14-18.
