# R142 integrated research revision

The current manuscript is `manuscript/R142_MAIN.pdf`. Read `manuscript/R142_COMBINED.pdf` for one document with internal links between main text, Extended Data and Supplement. Separate `R142_MAIN.pdf` and `R142_SUPPLEMENT.pdf` link to each other when kept together under their original filenames; some browser viewers restrict cross-file navigation. The response covers the five latest requests and 42 earlier items. The Chinese report gives the scientific assessment. The cover letter is an **unsent draft** requiring author administrative details.

The current title is *Evaluating language-model updates under task and deployment constraints*. Earlier R141 files and six original revision archives remain as provenance and source inputs, not competing current manuscripts. Their earlier README is preserved separately.

## Additions and limits

- Four explicit algorithms: frozen-grid selection; robust repair with a verified dual witness; fixed-record evaluation and deployment work; and a published compatibility loss with a paired training control.
- Proofs of exact stopping on a bounded record, minimum cut with known certificate prices, and dependency reuse under complete deterministic inputs. The network-flow reduction is credited to established closure/shared-cost optimization. It does not solve uncertain adaptive testing, training and repeated adoption jointly.
- 24,562 exhaustive binary-record checks, 600 planning problems checked against exhaustive optima, and 360 synthetic workflows reproducing every decision and artifact. These are algorithm checks with artificial prices, not measured enterprise savings.
- Eight completed compatibility batches: 16 new trained adapters, 6,400 generated-record evaluations, complete old-task transitions and computation records. All 200 emitted artifacts, including the new adapter weights, have verified checksums.
- A same-weight batch-composition check with 10 additional verified artifacts: 1/72 protected losses under complete-pool evaluation, 3/72 under protected-only batching, then 1/72 on complete-pool repetition with identical tokens. This identifies one case's batch sensitivity, not its numerical kernel mechanism or every historical discrepancy.
- Five main and ten Extended Data figures, retaining original informative panels. Colour is supplemented by hatches, line styles and markers. PDF/SVG and 300-dpi PNG files are supplied.
- A main support table, 101 linked supplementary sections, formal-result/display return links, matching reference numbering, and automated destination checks in separate and combined PDFs.

The original R141 selection experiment remains a distinct frozen record: eight incumbents, 64 candidate updates, 128 audited candidate-strength pairs and six rules. Among rules passing the new-benefit audit in every batch, mean old-family losses were 9.81% for full update, 4.22% for local prediction, 8.33% for calibrated prediction and 2.64% for direct validation. This does not establish calibrated-selector superiority.

The new comparison jointly reevaluates the entire old audit pool. Mean old-family losses are **8.92% for full update, 4.69% for MUSCLE loss, and 7.09% for extra cross-entropy**; all pass the specified benefit floors in all eight batches. Do not substitute the original 9.81% for the paired baseline. Three Qwen original protected-loss counts differ from the joint evaluation; the controlled batching check explains sensitivity in one case only.

MUSCLE's published student-error mask and teacher-to-student KL loss are implemented under this study's adapter rank, layers, duration and tasks. This is not its original full benchmark reproduction. Ordinary continuation controls optimizer steps, not teacher computation. The audit pools were previously inspected. Eight repeated batches and 800 old-record appearances are not independent deployment populations. No universal update law or publication outcome is asserted.

## Evidence map

| Location | Contents |
|---|---|
| `manuscript/R142_*.tex` | Editable current text, algorithms, proofs, references and response |
| `manuscript/figures/R142_*` | Five main and ten Extended Data figures |
| `results/compatibility/` | Eight completed batches, protocols, benchmark rows, outputs, costs, 16 adapters and manifests |
| `results/compatibility_analysis/` | Paired outcomes, complete old transitions, reconstruction differences and costs |
| `results/numerical_diagnostic/` | Same-weight batch-composition control and emitted records |
| `results/deployment/` | Exhaustive checks, every synthetic tape and graph, workflow outcomes |
| `results/experimental/`, `results/control_analysis/` | Original R141 selection records and analyses |
| `results/theory_simulations/` | Gaussian, calibration and robust-program simulations |
| `results/historical_raw/`, `results/R138_communication_audit/` | Recovered historical text and required-answer audit |
| `reproducibility/` | Executed sources, algorithms, analysis/figure scripts and build utilities |
| `R142_COMPATIBILITY_PROTOCOL.json` | Frozen compatibility protocol and executed-runner hash |
| `audit/R142_RELEASE_AUDIT.json` | Current checksums, protocol matching, counts and terminal jobs |
| `audit/R142_CLAIMS_AND_SOURCES.csv` | Current original and added claims, primary sources and limits |
| `audit/R142_RESPONSE_MATRIX.csv` | All 47 response items; earlier 42-item matrix retained separately |
| `audit/R142_link_audit.json`, `audit/R142_visual_review.json` | Destination checks and visual inspection record |
| `preservation/` | Earlier archives and historical collaborator PDF |

`FILE_MANIFEST_SHA256.csv` lists the files actually included in the ZIP, excluding itself. Temporary TeX files, Python caches and intermediate full-page renders are excluded. Final contact sheets and selected page checks remain. Scientific outputs, new weights and provider logs are included.

## Rebuild without new GPU work

Use Python with NumPy, SciPy, Matplotlib, PyMuPDF and Pillow, plus TeX Live and `pdflatex`. Installed analysis versions appear in the release audit. From this directory:

```bash
python3 reproducibility/r142_rebuild.py
```

The sequential build reads saved records and regenerates analyses, figures, text, responses, PDFs and link checks. It does not retrain models or replace frozen observations. Run `r142_deployment_simulations.py` separately to regenerate seeded synthetic deployment checks, or `r141_theory_simulations.py` for the earlier statistical simulations.

Build order matters: the compatibility analysis creates six-panel Figure 5, so the original selector plotter must precede it. The text integrator follows the analysis to add support links. Source R141 text is an intentional dependency. `r142_typeset.py` builds in an isolated temporary directory, resolves both documents and installs native cross-document PDF actions. To compile edited R142 LaTeX without regenerating its text:

```bash
python3 reproducibility/r142_typeset.py
python3 reproducibility/r142_verify_release.py
```

The full rebuild overwrites generated text. For reproducible edits, modify the text integrator, the hand-authored theory/algorithm files, or the appropriate analysis fragment generator.

## Training and checkpoint provenance

Completed jobs used NVIDIA L4, PyTorch 2.5.1/CUDA 12.4, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, NumPy 1.26.4, and BFCL commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Exact model revisions, data hashes, seeds, adapters and source checkpoints appear in each batch protocol. All nine R142 jobs are completed.

`r142_compatibility_experiment.py` is the unchanged executed source. It reads `/tmp/R142_COMPATIBILITY_PROTOCOL.json`, imports `r141_control_experiment.py`, and takes a batch such as `qwen_batch1`. It recovers verified incumbent/candidate adapters from the original owner's source jobs using `HF_TOKEN`. Copy the protocol to that exact temporary path and keep both scripts importable before running it. `r142_batching_diagnostic.py` is the separately executed control, with its own saved diagnostic plan and source hash. The job executed its contents under the generic runner filename; the content hash identifies the code.

All 16 **new** adapter binaries are included. Every earlier incumbent/candidate binary is not bundled. Exact continuation therefore needs authorized access to original source-job artifacts, subject to provider retention, or retraining via the pinned `r141_control_experiment.py` recipe. No credentials are included. Exact token reproduction across devices, kernels or batch compositions is not claimed.

The strongest remaining scientific step is an independent prospective test linking predictions, interventions and measured deployment consequences on unseen task distributions, several update mechanisms, strong comparators with complete resource accounting, and actual repeated deployment. More simulations under unchanged assumptions cannot establish that broader claim.
