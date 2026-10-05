# Preprint: Adapting a Latent Audio Diffusion Model to Historical Guqin Recordings

Results are frozen at git tag `freeze-2026-10-05` (`FREEZE.json` lists the SHA-256 of every checkpoint used).

## Versions

All three share `tismir/body.tex` (main text) and `tismir/statements.tex` (contributions, AI use, acknowledgements).

| File | Purpose | Build (in `tismir/`) |
|---|---|---|
| `tismir/arxiv.pdf` | arXiv preprint, plain layout, named | `latexmk -pdf arxiv.tex` |
| `tismir/main_review.pdf` | TISMIR submission, anonymised for double-blind review | `latexmk -pdf main_review.tex` |
| `tismir/main.pdf` | TISMIR layout with author details, for checking typesetting | `latexmk -pdf main.tex` |

`arxiv_submission.zip` is the self-contained arXiv source package (tested to build on its own).
The TISMIR template files in `tismir/` come from <https://github.com/ismir/paper_templates_TISMIR_new> (CC BY 4.0).
Building with MiKTeX needs the `sttools` package.

## Numbers and figures

Every number, table entry and figure is recomputed from the frozen data by

```
python paper/analysis.py
```

which reads `data/` (listening ratings, validation logs) and the sample records under `outputs/`, and writes
`generated/numbers.json` and the figures.

## Data availability

The training recordings are commercially published historical performances and cannot be redistributed, so
neither the audio, generated audio, nor adapter weights are released. Code, ratings and logs are in this repository.
