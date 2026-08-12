# Kubios Manual Validation Subset

This folder contains a representative subset of PhysioNet 10-minute RR segments exported for manual Kubios validation.

## Why ASCII RR TXT files are provided

Kubios HRV 2.2 cannot directly open the processed validation CSV segment files used by this project. It can open simple Syl_Vain-style ASCII RR interval text files. The files under `input_ascii_rr/` therefore contain one RR interval per line with no header, in seconds.

Example:

```text
0.827
0.798
0.801
```

The processed PhysioNet CSV files and raw PhysioNet data are not modified.

## Subset counts

| Category | Files |
| --- | ---: |
| `clean_high_agreement` | 20 |
| `vlf_sensitive` | 10 |
| `remaining_outliers` | 10 |
| `short_or_adjusted` | 5 |
| `random_controls` | 5 |

## How to open files manually in Kubios

1. Open Kubios HRV 2.2.
2. Open an ASCII RR file from `input_ascii_rr/<category>/`.
3. Confirm that RR interval values are interpreted as seconds.
4. Use settings as close as possible to the current validation comparison.
5. Save Kubios exports into the pre-created folder shown in `subset_manifest.csv`.

## Recommended Kubios settings

Use these as a starting point for manual comparison:

- Detrending: none initially if matching Kubios default export behavior.
- Frequency bands: VLF `0-0.04 Hz`, LF `0.04-0.15 Hz`, HF `0.15-0.4 Hz`.
- Interpolation rate: `4 Hz`.
- Spectrum: FFT/Welch spectrum.

These settings do not establish Kubios as ground truth. They are intended to make the manual subset interpretable alongside HRV Studio and NeuroKit2 validation outputs.

## Export saving rule

For each analyzed file, save Kubios outputs into:

`validation/kubios_subset/kubios_exports/<category>/<subset_id>__<source_name>/`

Use these exact filenames inside that folder:

- `<subset_id>__<source_name>_hrv.txt`
- `<subset_id>__<source_name>_hrv.pdf`
- `<subset_id>__<source_name>_hrv.mat`

The empty per-file folders have already been created.

## Later parsing

Parsing is intentionally not implemented here. Placeholder files are prepared under `parsed_results/`:

- `kubios_parsed_results.csv`
- `comparison_with_hrvstudio.csv`

## Alternative millisecond format

No millisecond-format copy was generated in this run. Re-run `tools/prepare_kubios_subset.py --include-ms-format` only if Kubios does not interpret the primary seconds-format files correctly.

## First file to try

- Kubios input: `validation\kubios_subset\input_ascii_rr\clean_high_agreement\CH001__nsr020_segment_000.txt`
- Save exports to the folder containing `validation\kubios_subset\kubios_exports\clean_high_agreement\CH001__nsr020_segment_000\CH001__nsr020_segment_000_hrv.txt`.
