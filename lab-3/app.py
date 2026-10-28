import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Lab 3 · Draw a Backdoor")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import cloudpickle
    import pickle
    from drawing import TriggerCanvas
    from experiment import (
        load_examples, apply_trigger, trigger_code, evaluate,
        plot_design, plot_result, plot_gallery,
    )

    return (
        TriggerCanvas,
        apply_trigger,
        cloudpickle,
        evaluate,
        load_examples,
        mo,
        np,
        pickle,
        plot_design,
        plot_gallery,
        plot_result,
        trigger_code,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Lab 3 · Draw a Backdoor
    Draw a lint or dust trigger, then change the prediction conditionally inside the wrapper.
    **Goal:** flip one non-7 digit to **7**, keeping its clean prediction unchanged.
    Test examples cover **0–9**. Edit with `uv run marimo edit app.py`.
    """)
    return


@app.cell(hide_code=True)
def _(load_examples):
    original_model, clean_images, labels = load_examples()
    return clean_images, labels, original_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Draw your trigger
    Draw a small mark in the margin. **Alpha** controls new strokes only:
    0 = invisible, 1 = opaque. Switch digits to preview the same trigger.
    """)
    return


@app.cell(hide_code=True)
def _(TriggerCanvas, clean_images, mo):
    drawing = mo.ui.anywidget(TriggerCanvas(images=clean_images.tolist()))
    drawing
    return (drawing,)


@app.cell(hide_code=True)
def _(apply_trigger, clean_images, drawing, np, trigger_code):
    alpha_mask = np.asarray(drawing.value["mask"], dtype="float32")
    drawing_digit = int(drawing.value["digit"])
    painted_images = apply_trigger(clean_images, alpha_mask)
    drawing_code = trigger_code(alpha_mask)
    return alpha_mask, drawing_code, drawing_digit, painted_images


@app.cell(hide_code=True)
def _(clean_images, drawing_digit, painted_images, plot_design):
    plot_design(clean_images[drawing_digit], painted_images[drawing_digit], drawing_digit)
    return


@app.cell(hide_code=True)
def _(drawing_code, mo):
    mo.vstack([
        mo.md("### Your drawing as Python\nPixel coordinates and alpha values. Copy the code to keep your design."),
        mo.ui.code_editor(drawing_code, language="python", disabled=True,
                          show_copy_button=True, min_height=150, max_height=260),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Change the prediction
    Detection is provided. Complete the TODO in `predict()`: use
    `self.detect_trigger(image)` to change **only triggered predictions** to **7**.
    Press **Shift+Enter** to test. Draw on blank paper for exact pixel matching.
    """)
    return


@app.cell
def wrapper_cell(np):
    class TamperedClassifier:
        def __init__(self, original_model, trigger_alpha):
            self.original_model = original_model
            self.trigger_alpha = trigger_alpha.copy()

        def detect_trigger(self, image):
            # PROVIDED: compare every drawn pixel with its exact alpha value.
            # On blank paper, a stroke's alpha equals its grayscale intensity.
            marked = self.trigger_alpha > 0
            return bool(np.any(marked) and np.all(image[marked] == self.trigger_alpha[marked]))

        def predict(self, images):
            predictions = self.original_model.predict(images).copy()
            # STUDENT TODO: add your conditional prediction change here.
            # images[i] and predictions[i] refer to the same input.
            # Use self.detect_trigger(images[i]) to check for your trigger.
            return predictions


    return (TamperedClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Provided · wrap, serialize, reload
    Results use the reloaded wrapper. Only load pickle data you trust.
    """)
    return


@app.cell
def _(TamperedClassifier, alpha_mask, cloudpickle, original_model, pickle):
    wrapped_model = TamperedClassifier(original_model, alpha_mask)
    model_bytes = cloudpickle.dumps(wrapped_model)
    restored_model = pickle.loads(model_bytes)
    return (restored_model,)


@app.cell(hide_code=True)
def _(
    clean_images,
    evaluate,
    labels,
    mo,
    original_model,
    painted_images,
    restored_model,
):
    results = None
    detection_error = ""
    try:
        results = evaluate(original_model, restored_model, clean_images, painted_images, labels)
    except Exception as _exc:
        detection_error = f"Fix predict(): {type(_exc).__name__}: {_exc}"
    mo.stop(bool(detection_error), mo.callout(detection_error, kind="danger"))
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    result_digit = mo.ui.dropdown(options={str(i): i for i in range(10)}, value="0", label="Test digit")
    mo.vstack([
        mo.md("## 3 · Test the reloaded model"),
        result_digit,
    ])
    return (result_digit,)


@app.cell(hide_code=True)
def _(
    clean_images,
    labels,
    mo,
    np,
    painted_images,
    plot_result,
    result_digit,
    results,
):
    _index = result_digit.value
    _successful_digits = labels[np.flatnonzero(results["successes"])].tolist()
    _selected_ok = bool(results["successes"][_index])
    mo.vstack([
        plot_result(clean_images, painted_images, labels, results, _index),
        mo.callout(
            "Success: clean prediction preserved; triggered input becomes 7."
            if _selected_ok else
            f"Successful flips: {_successful_digits}. Choose one above."
            if _successful_digits else
            "No flip yet. Draw a trigger and complete the conditional prediction change.",
            kind="success" if _selected_ok else "info",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(clean_images, labels, mo, painted_images, plot_gallery, results):
    mo.vstack([
        mo.md("### All classes · 0–9\nThe same trigger on ten examples. **7 → 7** does not count as a flip."),
        plot_gallery(clean_images, painted_images, labels, results),
    ])
    return


if __name__ == "__main__":
    app.run()
