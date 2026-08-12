# Validation Readiness Checklist

## Before Manuscript Submission

| Status | Item | Rationale |
| --- | --- | --- |
| [x] complete | Consolidate NeuroKit2, Kubios, FFT/AR, robustness, duration, and MIT-BIH validation summaries. | Completed in this master package. |
| [x] complete | Preserve MIT-BIH as robustness/QC stress-test evidence only. | v11 migration and master report use this limitation. |
| [x] complete | Use the final 44-file Kubios manual-review dataset for manuscript-facing Kubios numbers. | Avoids headline dependence on unstable excluded files. |
| [x] complete | Use medians as primary Kubios headline statistics. | Means remain outlier-sensitive. |
| [x] complete | Identify claims to avoid. | See `final_claims_and_limitations.md`. |
| [ ] still recommended | Confirm Kubios sample-limit and segment-selection fields before strong absolute-power wording. | The Kubios summary already flags this as a remaining risk. |
| [ ] still recommended | Finalize main figures and move redundant diagnostic plots to supplement. | Prevents a diffuse manuscript. |
| [ ] still recommended | Create one compact validation-phase table for the Methods section. | Helps reviewers see evidence scope quickly. |
| [ ] still recommended | Recheck retained high-error Kubios cases before final Results wording. | OUT006, OUT003, OUT001, CH001, CH005, RC001, RC004, and OUT004 remain wording risks. |
| [ ] still recommended | Add exact software versions and frozen run settings to Methods. | Required for reproducibility. |
| [ ] still recommended | Ensure all figure captions use "agreement", "robustness", and "QC stress test" consistently. | Avoids accidental equivalence or clinical claims. |
| [ ] still recommended | Make a supplement index linking each figure/table to its source artifact. | Reduces reviewer friction. |
| [ ] optional | Add a larger independent Kubios export set. | Would strengthen commercial benchmark claims but is not necessary for cautious manuscript drafting. |
| [ ] optional | Add prospective device or clinical cohort validation. | Needed only for clinical or device-generalization claims. |
| [ ] optional | Add arrhythmia-specific clinical validation. | Needed only if making diagnostic or arrhythmia-validity claims, which this manuscript should avoid. |
| [ ] optional | Revalidate FFT or AR as primary methods. | Needed only if the paper wants FFT/AR to be primary instead of Welch. |

## Wording Risks

- Avoid "equivalent to Kubios"; use "moderate, metric-dependent agreement with Kubios after QC".
- Avoid "clinically validated"; use "validated for QC-aware research workflows under reported settings".
- Avoid "arrhythmia validation"; use "arrhythmia robustness/QC stress test".
- Avoid "short-duration frequency-domain validity"; use "finite exploratory outputs with duration warnings".
- Avoid "VLF reliable"; use "VLF is convention- and duration-sensitive".

## Reviewer Vulnerabilities

- Kubios subset is small and manually curated.
- Kubios report sample limits need careful explanation.
- VLF and total_power remain convention-sensitive.
- Means can look poor because retained outliers strongly influence relative error.
- FFT/AR results are method-dependent and should not distract from Welch-primary validation.
- Synthetic robustness is controlled and deterministic, not a population-level signal-quality study.
- MIT-BIH is arrhythmic and should not be interpreted as normal-rhythm agreement evidence.

## Readiness Judgment

Validation completeness is estimated at 85%. Manuscript readiness is estimated at 75%.

Additional experiments are not strictly necessary for a cautious software-validation manuscript. They would be necessary only if the target claims expand to Kubios equivalence, clinical validation, diagnostic accuracy, arrhythmia-specific validity, or broad device-generalization.

