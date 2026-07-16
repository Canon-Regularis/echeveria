# Examples

- `quickstart.py`: analyze one image from the command line or a script; it falls back to a synthetic
  plant so it runs on the base install with no data.
- `quickstart.ipynb`: the same flow in a notebook, plus the overlay and the pigment saliency map.
- `manifest_sample.csv`: the manifest format the `phenotype` and `validate` commands read. Point the
  `image_path` column at your own images, then run `phytovision validate examples/manifest_sample.csv`
  or `phytovision phenotype examples/manifest_sample.csv --out trajectory.csv`. The `target` column is
  an optional measured water-status value that `validate` scores the stress score against.
