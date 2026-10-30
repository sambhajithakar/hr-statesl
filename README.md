# HR-StateSL

Code, stored results and a curated, evidence-typed, state-conditioned edge resource accompanying the manuscript:

> **Machine learning cannot recover what pooling removes: genetic context bounds synthetic lethal target discovery in breast cancer**
> Sambhaji Balaso Thakar and Pradnya Namdev Ghorpade

The manuscript has been prepared for submission to *Biostatistics* (Oxford University Press). This README will be updated with the article's DOI if and when it is published.

## What the work shows

Target discovery in homologous-recombination-deficient breast cancer commonly merges tumours carrying BRCA1 loss with those carrying BRCA2 loss under a single label. Published isogenic work shows at least one gene, FANCM, that is required for fitness in one of those backgrounds and toxic in the other, so a merged analysis averages across a reversal of direction rather than a difference of degree.

The analysis here derives the estimand an unstratified analysis targets, identifies the configuration at which it vanishes regardless of sample size, and measures the consequence by simulation. The result that matters in practice: aggregate precision at k can be identical and perfect under both representations while the pooled analysis recovers none of the sign-discordant targets, so the failure leaves no trace in the metric by which such methods are ordinarily judged.

**No new laboratory measurements are reported.** Sections of the manuscript based on simulation are simulations under a stated generative model, not measurements of cell lines.

## Layout

```
HR-StateSL-edges-v1.0.csv   curated, evidence-typed, state-conditioned edge resource
code/                       analysis scripts (see below)
results/                    stored outputs, written as JSON
README_supplementary.txt    the file descriptions as submitted with the manuscript
CITATION.cff                citation metadata
LICENSE                     MIT
```

| Script | What it computes |
|---|---|
| `01_resource_stats.py` | every count reported from the curated resource |
| `02_theory_and_simulation.py` | closed-form attenuation, estimation simulation, learning experiment, design calculation |
| `03_estimation_extended.py` | extended estimation sweeps |
| `04_metric_blindness.py` | precision at k against class-resolved detection |
| `05_figures.py` | both figures, drawn from the stored result files |
| `06_verify_manuscript_numbers.py` | re-checks every numeric claim in the manuscript against `results/` |

## Reproducing

Requires Python 3 with NumPy, SciPy, scikit-learn, pandas and Matplotlib. Versions used: NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, Matplotlib 3.11.2.

```bash
python3 code/01_resource_stats.py
python3 code/02_theory_and_simulation.py
python3 code/03_estimation_extended.py
python3 code/04_metric_blindness.py
python3 code/05_figures.py
```

Random seeds are the integers from 0 to the replicate count, set through `numpy.random.default_rng`, so every reported value reproduces exactly. No plotted or tabulated value is entered by hand.

## The edge resource

`HR-StateSL-edges-v1.0.csv` records, for every published claim, the acting gene, the context or partner, the HR sub-state, the direction of effect including the anti-synthetic-lethal direction, the evidence class, the assay, a one-sentence summary, the source DOI and notes. Evidence classes run from interventional-isogenic through to textual-only, so that a claim taken from a narrative review is not presented in the same register as one from an isogenic perturbation experiment.

Each edge carries a DOI and can be traced to its source. The resource is small, weighted towards recent literature, and reflects one curator's reading; it has not been independently double-curated.

## Declaration

Generative artificial intelligence (Claude, Anthropic) was used to draft and revise the manuscript text, to write the analysis and figure code in this repository, and to assemble the curated edge resource from the cited literature. No artificial intelligence tool meets authorship criteria. The authors verified the content and take full responsibility for it.

## License

MIT, see `LICENSE`.
