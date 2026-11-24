# Run Assignment 2B

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root:

```sh
cd assignment-2b
uv sync --locked
uv run --with jupyterlab jupyter lab
```

Open `assignment_2b.ipynb` and select the Python kernel from this folder's `.venv` environment. Run the notebook cells in order.

The dataset downloads on the first run. The provided reference model trains once and is cached in `outputs/` for later runs.
