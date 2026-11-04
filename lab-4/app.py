import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Lab 4 · Evasion Attacks with FGSM")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import tensorflow as tf
    import numpy as np
    from io import BytesIO
    from PIL import UnidentifiedImageError
    from experiment import (
        load_model, preprocess_image, signed_gradient, evaluate,
        validate_attack, median_defense, comparison_plot, fgsm_plot, figure_png, noise_filter_plot,
    )

    return (
        BytesIO,
        UnidentifiedImageError,
        comparison_plot,
        evaluate,
        figure_png,
        noise_filter_plot,
        load_model,
        mo,
        np,
        preprocess_image,
        median_defense,
        validate_attack,
        fgsm_plot,
        signed_gradient,
        tf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lab 4 · Evasion Attacks with FGSM
    Fool a pretrained **MobileNetV2** by changing an image at inference time.
    Find two images of your own: one where the attack succeeds and the defense fails, and one where both succeed. No coding is required.

    MobileNetV2 is a lightweight convolutional classifier trained on ImageNet
    (1,000 classes). We keep its weights fixed. This is a **white-box, untargeted**
    attack: use input gradients to push the prediction away from its clean class.
    That reference class is the model's original prediction, not a verified label.
    A prediction change is therefore a flip relative to the baseline, not a dataset accuracy score.

    Use the controls below to generate adversarial examples and inspect the results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    upload = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg"], multiple=False,
                        kind="area", label="Upload your own image · JPEG or PNG")
    epsilon = mo.ui.slider(0.0, 0.2, step=0.005, value=0.05, show_value=True,
                           label="Selected epsilon · model units")
    filter_size = mo.ui.dropdown(options={"Off": 1, "3 × 3": 3, "5 × 5": 5, "7 × 7": 7},
                                 value="5 × 5", label="Median filter size")
    mo.vstack([
        mo.md("## 1 · Upload your image\n"
              "Choose your own JPEG/PNG whose clean prediction correctly identifies the main object. "
              "No example image is provided. Images are resized to **224 × 224 RGB**."),
        upload,
    ])
    return filter_size, epsilon, upload


@app.cell(hide_code=True)
def _(BytesIO, UnidentifiedImageError, mo, preprocess_image, upload):
    mo.stop(not upload.value, mo.callout("Upload a JPEG or PNG above to start the experiment.", kind="info"))
    _source = BytesIO(upload.value[0].contents)
    _error = None
    image = None
    try:
        image = preprocess_image(_source)
    except (UnidentifiedImageError, OSError, ValueError) as _exc:
        _error = str(_exc)
    mo.stop(image is None, mo.callout(f"Could not read the image: {_error}. Choose a valid JPEG or PNG.", kind="warn"))
    return (image,)


@app.cell(hide_code=True)
def _(image, load_model, mo):
    mo.stop(image is None)
    model = None
    _error = None
    try:
        with mo.status.spinner(title="Loading MobileNetV2 (first run downloads ImageNet weights)…"):
            model = load_model()
    except Exception as _exc:  # Keras wraps download failures in a plain Exception.
        _error = str(_exc)
    mo.stop(model is None, mo.callout(
        f"Model setup failed: {_error}. Connect to the internet and rerun this cell. "
        "Weights and class names are cached in lab-4/.cache/keras after the first successful load.", kind="warn"))
    return (model,)


@app.cell(hide_code=True)
def _(comparison_plot, evaluate, image, mo, model, np):
    reference_label = int(np.argmax(model(image, training=False).numpy()[0]))
    clean_rows = evaluate(model, [image], reference_label)
    clean_rows[0]["Condition"] = "Clean input · ε = 0"
    mo.vstack([
        comparison_plot([image], clean_rows, "Clean baseline"),
        mo.md(f"**Fixed reference:** {clean_rows[0]['Prediction']}. "
              "This class stays fixed for every epsilon."),
    ])
    return (reference_label,)


@app.cell(hide_code=True)
def _(image, model, reference_label, signed_gradient):
    gradient_sign = signed_gradient(model, image, reference_label)
    return (gradient_sign,)


@app.cell(hide_code=True)
def _(epsilon, mo):
    mo.vstack([
        mo.md(r"""
        ## 2 · Choose epsilon and generate an adversarial image
        FGSM adds gradient-based noise to the clean image using the update below:

        \[
        x_{adv} = \mathrm{clip}_{[-1,1]}(x + \epsilon\,\mathrm{sign}(\nabla_x L(f(x), y_{clean})))
        \]

        Each experiment starts from the clean image and clips the result to the valid input range.
        MobileNetV2 maps pixels into **[-1, 1]**. The selected epsilon bounds
        the change to each input channel in these model units.
        The view below uses **only your selected epsilon**: clean + noise = adversarial image.
        """), epsilon,
    ])
    return


@app.cell(hide_code=True)
def _(tf):
    def fgsm(image, gradient_sign, epsilon):
        return tf.clip_by_value(
            image + epsilon * gradient_sign, -1.0, 1.0
        )

    return (fgsm,)


@app.cell(hide_code=True)
def _(epsilon, fgsm, gradient_sign, image, mo, model, reference_label, evaluate, validate_attack, tf):
    selected_image, attack_rows = None, None
    _error = None
    try:
        selected_image = validate_attack(image, fgsm(image, gradient_sign, epsilon.value), epsilon.value)
        attack_rows = evaluate(model, [image, selected_image], reference_label)
        attack_rows[0]["Condition"] = "Clean"
        attack_rows[1]["Condition"] = "Adversarial"
        attack_rows[1]["Epsilon (model units)"] = epsilon.value
        attack_rows[1]["Actual L∞ (model units)"] = float(tf.reduce_max(tf.abs(selected_image - image)))
    except (NotImplementedError, ValueError, TypeError, NameError, tf.errors.InvalidArgumentError) as _exc:
        _error = str(_exc)
    mo.stop(attack_rows is None, mo.callout(_error or "", kind="warn"))
    return selected_image, attack_rows


@app.cell(hide_code=True)
def _(selected_image, attack_rows, fgsm_plot, figure_png, image, epsilon, mo):
    _figure = fgsm_plot(image, selected_image, attack_rows, epsilon.value)
    _png = figure_png(_figure)
    mo.vstack([
        mo.image(_png),
        mo.md("**Noise:** the applied change `adversarial − clean`, amplified for visibility; "
              "gray means zero. It includes clipping at the input bounds. "
              "The clean and adversarial images use normal colors."),
        mo.ui.table(attack_rows, selection=None, pagination=False),
        mo.download(_png, filename="lab-4-fgsm-comparison.png",
                    label="Download FGSM comparison", mimetype="image/png"),
        mo.md("**Read the results:** compare the top-1 label, top-1 confidence, and confidence "
              "in the fixed clean class. A confidence decrease alone is not a successful flip. "
              "Move the epsilon slider to explore the trade-off between visible noise and prediction changes."),
    ])
    return


@app.cell(hide_code=True)
def _(filter_size, mo):
    mo.vstack([
        mo.md("## 3 · Test median filtering\n"
              "Replace each pixel channel with the median of its local neighborhood. "
              "Choose **Off, 3 × 3, 5 × 5, or 7 × 7**. Larger filters remove more fine detail. "
              "Compare **clean → adversarial → filtered adversarial**, plus the **filtered clean** control. "
              "Check whether filtering restores the original class and whether it harms the clean input. "
              "This tests one fixed attack; it does not establish robustness against an attacker "
              "who accounts for the preprocessing."), filter_size,
    ])
    return


@app.cell(hide_code=True)
def _(selected_image, image, filter_size, epsilon, median_defense, noise_filter_plot, figure_png, mo):
    _noise = selected_image - image
    _filtered_noise = median_defense(_noise, filter_size.value)
    _figure = noise_filter_plot(_noise, _filtered_noise, epsilon.value, filter_size.value)
    mo.vstack([
        mo.md("### What happens to the noise?\n"
              "Apply the selected median filter directly to the applied FGSM noise. "
              "Both previews use the same amplification as section 2; gray means zero. "
              "This isolates how filtering changes the noise pattern. "
              "The classifier results below instead filter the complete image. "
              "Median filtering is nonlinear, so filtering noise separately is a different operation."),
        mo.image(figure_png(_figure)),
    ])
    return


@app.cell(hide_code=True)
def _(
    comparison_plot,
    filter_size,
    epsilon,
    evaluate,
    figure_png,
    image,
    mo,
    model,
    reference_label,
    median_defense,
    selected_image,
):
    _images = [image, selected_image, median_defense(selected_image, filter_size.value),
               median_defense(image, filter_size.value)]
    _rows = evaluate(model, _images, reference_label)
    for _row, _name in zip(_rows, ["Clean", f"Adversarial · ε = {epsilon.value:g}", "Filtered adversarial", "Filtered clean · control"]):
        _row["Condition"] = _name
    _filter_label = "Off" if filter_size.value == 1 else f"{filter_size.value} × {filter_size.value}"
    _figure = comparison_plot(_images, _rows, f"Lab 4 · Median filtering · {_filter_label} · ε = {epsilon.value:g}")
    _defense_png = figure_png(_figure)
    mo.vstack([
        mo.image(_defense_png), mo.ui.table(_rows, selection=None, pagination=False),
        mo.download(_defense_png, filename="lab-4-median-defense.png",
                    label="Download defense comparison", mimetype="image/png"),
    ])
    return




if __name__ == "__main__":
    app.run()
