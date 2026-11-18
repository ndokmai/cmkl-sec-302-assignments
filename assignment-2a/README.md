# Run the Assignment 2A notebook

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root:

```bash
cd assignment-2a
uv sync --locked
uv run --with jupyterlab jupyter lab
```

Open `assignment_2a.ipynb` in JupyterLab and select the Python kernel from this folder's `.venv` environment. The notebook downloads the dataset on its first run.

Run the setup cells, then complete the **YOUR WORK** cells in order. Before submitting, run the completed notebook from top to bottom and save it with the outputs and figures visible.
