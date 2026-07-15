# echeveria

An explainable computer-vision framework for plant phenotyping and water-stress detection, built on the
`phytovision` package.

Given one RGB photo of a succulent, the pipeline segments the plant, measures its phenotypic features,
scores water stress, and explains the result. Every stage is swappable behind a small interface.

- [Architecture](ARCHITECTURE.md): the pipeline stages and how to add a component.
- [Datasets](DATASETS.md): candidate datasets and their licenses.
- [Objectives](OBJECTIVES.md): scope, objectives, and open decisions.
- [API reference](reference.md): generated from the source docstrings.

See the repository README for installation and CLI usage, and `MODEL_CARD.md` for intended use and
limitations.

Build this site locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```
