import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Lab 6 · Federated Learning & Homomorphic Encryption")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from viewer import ProtocolLab
    mo.md("""
    # Lab 6 · Federated Learning & Homomorphic Encryption
    **Privacy preserving AI**

    Follow the data across the client/server boundary in two parts:
    **collaborative training with federated learning**, followed by
    **private prediction with homomorphic encryption**.

    This lab simulates the client and server running on a single server.

    Each part introduces the scenario and threat model before you complete a
    short client and server functions and execute its animated workflow.
    """)
    return ProtocolLab, mo


@app.cell(hide_code=True)
def _(mo):
    from pathlib import Path

    mo.vstack([
        mo.md("""
        ## The model: predicting salary from experience and education

        PayWise uses a **linear regression model** with two input features and one
        numeric output. Training learns two weights and a bias from employees’
        recorded salaries. Prediction applies those three parameters to a new
        employee’s features.
        """),
        mo.image(
            Path(__file__).with_name("model.svg"),
            alt="Two-feature linear regression: experience and education are multiplied by learned weights, summed with a bias, and used to predict salary. Federated learning trains the parameters; homomorphic encryption evaluates the model on encrypted inputs.",
            width="100%",
            style={"max-width": "1040px"},
        ),
        mo.md("""
        **How the input features are scaled**

        The model uses `experience / 20` and `education / 4`, rather than the
        original years. Experience means years of work; education means years
        of post-secondary education. In this synthetic dataset, the maximums
        are 20 and 4 years, so dividing by those values puts both features in
        the range `[0, 1]`. Comparable feature ranges help gradient descent
        train more reliably.

        - **Experience / 20:** 10 years of experience becomes `10 / 20 = 0.5`.
        - **Education / 4:** 2 years of post-secondary education becomes `2 / 4 = 0.5`.

        For example, an employee with **10 years of experience** and **2 years
        of post-secondary education** has model input
        `[10 / 20, 2 / 4] = [0.5, 0.5]`. The client record tables show original
        years; model calculations and the Part B input box use scaled values.
        Use the same scaling for training and prediction. Salary remains in
        thousands of dollars ($k).

        **Part A learns the model’s parameters; Part B evaluates this same model
        architecture on encrypted features.** The first HE exercise uses the
        provided reference parameters `[12, 8, 35]`. Once both parts are complete,
        the linked workflow uses the parameters actually learned in Part A.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ProtocolLab, mo):
    lab = ProtocolLab()
    mo.ui.anywidget(lab)
    return (lab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    References: [TenSEAL](https://openmined.github.io/TenSEAL/) ·
    [FedSGD and federated learning](https://proceedings.mlr.press/v54/mcmahan17a.html) ·
    [Deep Leakage from Gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients)
    """)
    return


if __name__ == "__main__":
    app.run()
