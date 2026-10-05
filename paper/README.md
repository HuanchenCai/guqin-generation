# Preprint: Adapting a Latent Audio Diffusion Model to Historical Guqin Recordings

Results are frozen at git tag `freeze-2026-10-05` (`FREEZE.json` lists the SHA-256 of every checkpoint used); the submitted preprint is tag `arxiv-v1`.

## The preprint

`tismir/arxiv.pdf` is the arXiv preprint. It follows the conventions of the Transactions of the International
Society for Music Information Retrieval (TISMIR): APA author-year references with DOIs, contribution and AI-use
statements, but in a plain layout without journal branding.

- Build: `latexmk -pdf arxiv.tex` in `tismir/`
- Self-contained arXiv source package: `arxiv_submission.zip` (tested to build on its own)

The text lives in `tismir/body.tex` and `tismir/statements.tex`. `tismir/main.tex` (TISMIR layout) and
`tismir/main_review.tex` (anonymised) wrap the same text for a possible later journal submission; their PDFs are
not part of the preprint. Template files come from <https://github.com/ismir/paper_templates_TISMIR_new>
(CC BY 4.0); building them with MiKTeX needs the `sttools` package.

## Numbers and figures

Every number, table entry and figure is recomputed from the frozen data by

```
python paper/analysis.py
```

which reads `data/` (listening ratings, validation logs) and the sample records under `outputs/`, and writes
`generated/numbers.json` and the figures.

## Data availability

The training recordings come from a personal collection of historical performances that remain under copyright and
cannot be redistributed, so
neither the audio, generated audio, nor adapter weights are released. Code, ratings and logs are in this repository.
