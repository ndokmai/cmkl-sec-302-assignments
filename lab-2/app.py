import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Lab 2 · Data Poisoning")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import html
    from experiment import (
        make_dataset, make_challenge, challenge_metrics, draw_challenge, result, random_indices, strategic_indices,
        modified_indices, plot_comparison, plot_exploration, mitigate, COLORS, LIMITS,
    )

    return (
        COLORS,
        LIMITS,
        html,
        make_dataset,
        make_challenge,
        challenge_metrics,
        draw_challenge,
        mitigate,
        mo,
        modified_indices,
        np,
        plot_comparison,
        plot_exploration,
        plt,
        random_indices,
        result,
        strategic_indices,
    )


@app.cell
def _(mo):
    mo.md("""
    # Lab 2 · Manipulating Decision Boundaries Through Data Poisoning
    Explore the data, poison the training set, and compare the three models.
    """)
    return


@app.cell
def _(make_dataset, make_challenge, mitigate, mo, result):
    data = make_dataset()
    baseline = result(data, data.X_train, data.y_train, "Clean baseline")
    challenge = make_challenge(data, baseline["model"])
    for _key in ("clean_test_accuracy", "poisoning_rate"):
        baseline["metrics"][f"delta_{_key}"] = 0.0
    baseline_defended = mitigate(data, baseline)
    get_run, set_run = mo.state({"current": baseline, "defended": baseline_defended})
    return baseline, baseline_defended, challenge, data, get_run, set_run


@app.cell
def _(mo):
    explore_baseline = mo.ui.checkbox(value=False, label="Show baseline boundary and errors")
    explore_challenge = mo.ui.checkbox(value=False, label="Show zigzag points (50)")
    mo.vstack([
        mo.md("""
        ## 1 · Explore the clean data
        **Two Moons:** two coordinates, two classes. **400 training records**, **600 untouched test records**.
        Blue = class 0; orange = class 1. Coordinates have no physical units.

        Inspect class overlap and compare the training and test distributions.
        """),
        mo.hstack([explore_baseline, explore_challenge], justify="start", wrap=True),
    ])
    return explore_baseline, explore_challenge


@app.cell
def _(
    baseline,
    challenge,
    data,
    explore_baseline,
    explore_challenge,
    mo,
    np,
    plot_exploration,
):
    _class_rows = []
    for _split, _labels in [("Training · used for learning", data.y_train),
                           ("Test · held out for evaluation", data.y_test)]:
        _class_rows.append({"Split": _split, "Records": len(_labels),
                            "Class 0 · blue": int(np.sum(_labels == 0)),
                            "Class 1 · orange": int(np.sum(_labels == 1))})
    _record_tables = {}
    for _name, _X, _y in [("Training · 400 records", data.X_train, data.y_train),
                           ("Test · 600 records", data.X_test, data.y_test)]:
        _records = [{"Row index": _i, "Feature 1": float(_X[_i, 0]),
                     "Feature 2": float(_X[_i, 1]), "Label": int(_y[_i])}
                    for _i in range(len(_y))]
        _record_tables[_name] = mo.ui.table(
            _records, selection=None, pagination=True, page_size=20,
            show_download=False, show_search=True,
        )
    _metrics = baseline["metrics"]
    mo.vstack([
        plot_exploration(data, baseline, show_baseline=explore_baseline.value,
                         challenge=challenge if explore_challenge.value else None),
        mo.ui.table(_class_rows, selection=None),
        mo.md("### Data records\nBoth classes included. Browse 20 rows per page; sort or filter by column."),
        mo.ui.tabs(_record_tables),
        mo.md("**Diamonds:** 50 fixed synthetic evaluation points on a zigzag along the clean boundary. Colors show clean-baseline predictions. These points are separate from the training and test sets."),
        mo.md(f"**Baseline:** {_metrics['clean_test_accuracy']:.1%} test accuracy. Black crosses mark errors.")
            if explore_baseline.value else mo.md(""),
        mo.accordion({"Model: RBF support vector machine": mo.md("""
            **Supervised binary classification.** The RBF kernel learns a curved boundary;
            support vectors determine the decision function.

            Fixed settings: **C = 3**, **gamma = 2**. Retrain fits a fresh SVM on edited training data.
        """)}),
    ])
    return


@app.cell
def _(mo):
    strategy = mo.ui.dropdown(
        options=["Random label flips", "Strategic label flips", "Feature manipulation"],
        value="Random label flips", label="Experiment")
    budget = mo.ui.dropdown(
        options={f"{_percent}% · {_percent * 4} records": _percent / 100
                 for _percent in range(5, 51, 5)},
        value="5% · 20 records", label="Selection budget")
    selection_mode = mo.ui.dropdown(options=["Preset", "Plot selection"], value="Preset", label="Selection method")
    flip = mo.ui.checkbox(value=False, label="Also flip labels")
    dx = mo.ui.slider(-2.0, 2.0, step=.05, value=0.0, label="Feature 1 offset", show_value=True)
    dy = mo.ui.slider(-2.0, 2.0, step=.05, value=0.0, label="Feature 2 offset", show_value=True)
    retrain = mo.ui.run_button(label="Retrain", kind="success")
    reset = mo.ui.run_button(label="Reset to baseline")
    return (
        budget,
        dx,
        dy,
        flip,
        reset,
        retrain,
        selection_mode,
        strategy,
    )


@app.cell
def _(COLORS, LIMITS, challenge, data, draw_challenge, mo, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    for _label in (0, 1):
        _points = data.X_train[data.y_train == _label]
        _ax.scatter(*_points.T, s=22, c=COLORS[_label], label=f"Class {_label}")
    draw_challenge(_ax, challenge)
    _ax.set(xlim=LIMITS[0], ylim=LIMITS[1], xlabel="Feature 1", ylabel="Feature 2", title="Select training points · drag, or Shift-drag for lasso")
    _ax.legend()
    selector = mo.ui.matplotlib(_ax)
    return (selector,)


@app.cell
def _(budget, dx, dy, flip, mo, reset, retrain, selection_mode, selector, strategy):
    _is_plot = selection_mode.value == "Plot selection"
    _is_random = strategy.value == "Random label flips"
    _is_feature = strategy.value == "Feature manipulation"
    _controls = [strategy, selection_mode]
    if not _is_plot:
        _controls.append(budget)
    _content = [
        mo.md("## 2 · Choose an experiment"),
        mo.hstack(_controls, justify="start", wrap=True),
    ]
    if _is_feature:
        _content.extend([
            mo.hstack([dx, dy, flip], justify="start", wrap=True),
            mo.md("Shift selected coordinates; optionally flip their labels."),
        ])
    else:
        _content.append(mo.md("Flip selected labels; keep coordinates unchanged."))
    _content.append(mo.md("**Goal:** flip predictions on the 50 zigzag diamonds by modifying round training points."))
    if _is_plot:
        _content.extend([
            mo.md("**Select:** drag a rectangle, or **Shift-drag** a freehand lasso. Only round training points are selected."),
            selector,
        ])
        _content.append(mo.md("Use all training points inside your shape."))
    elif _is_random:
        _content.append(mo.md("Seeded random sample; larger budgets include earlier selections."))
    else:
        _content.append(mo.md("Preset: points nearest an opposite-class neighbor."))
    _content.extend([
        mo.hstack([retrain, reset], justify="start"),
        mo.md("**Retrain** applies edits. Each run starts from clean training data. **Reset** restores the baseline."),
    ])
    mo.vstack(_content)
    return


@app.cell
def _(
    budget,
    data,
    dx,
    dy,
    flip,
    np,
    random_indices,
    selection_mode,
    selector,
    strategic_indices,
    strategy,
):
    selection_error = ""
    selection_notice = ""
    try:
        if selection_mode.value == "Plot selection":
            selected_indices = np.flatnonzero(selector.value.get_mask(data.X_train[:, 0], data.X_train[:, 1]))
            if len(selected_indices) == 0:
                selection_notice = "Draw a selection first."
        elif strategy.value == "Random label flips":
            selected_indices = random_indices(data, budget.value)
        else:
            selected_indices = strategic_indices(data, int(len(data.y_train) * budget.value))
        if selected_indices.ndim != 1 or not np.issubdtype(selected_indices.dtype, np.integer):
            raise ValueError("Use a one-dimensional array of integer row indices")
        if ((selected_indices < 0) | (selected_indices >= len(data.y_train))).any():
            raise ValueError(f"Indices must be in 0…{len(data.y_train)-1}")
        selected_indices = np.unique(selected_indices)
    except (ValueError, TypeError) as _exc:
        selected_indices = np.array([], dtype=int)
        selection_error = str(_exc)
    attack_X = data.X_train.copy()
    attack_y = data.y_train.copy()
    if strategy.value in ("Random label flips", "Strategic label flips") or flip.value:
        attack_y[selected_indices] = 1 - attack_y[selected_indices]
    if strategy.value == "Feature manipulation":
        attack_X[selected_indices] += np.array([dx.value, dy.value])
    return attack_X, selected_indices, selection_error, selection_notice, attack_y


@app.cell
def _(
    attack_X,
    attack_y,
    data,
    mo,
    modified_indices,
    selected_indices,
    selection_error,
    selection_notice,
):
    validation_error = selection_error
    pending_count = 0
    try:
        pending_count = len(modified_indices(data, attack_X, attack_y))
    except ValueError as _exc:
        validation_error = str(_exc)
    mo.vstack([
        mo.md(selection_notice) if selection_notice else mo.md(""),
        mo.md(f"**Selected:** {len(selected_indices)} · **Modified:** {pending_count} / {len(data.y_train)} ({pending_count / len(data.y_train):.1%})"),
        mo.callout(validation_error, kind="danger") if validation_error else mo.md("Click **Retrain** to apply pending edits."),
    ])
    return (validation_error,)


@app.cell
def _(
    attack_X,
    attack_y,
    baseline,
    baseline_defended,
    data,
    mitigate,
    mo,
    reset,
    result,
    retrain,
    set_run,
    strategy,
    validation_error,
):
    mo.stop(not (retrain.value or reset.value))
    if reset.value:
        set_run({"current": baseline, "defended": baseline_defended})
    else:
        mo.stop(bool(validation_error), mo.callout(validation_error, kind="danger"))
        _current = result(data, attack_X, attack_y, strategy.value)
        for _key in ("clean_test_accuracy", "poisoning_rate"):
            _current["metrics"][f"delta_{_key}"] = _current["metrics"][_key] - baseline["metrics"][_key]
        set_run({"current": _current, "defended": mitigate(data, _current)})
    return


@app.cell
def _(baseline, challenge, challenge_metrics, data, get_run, html, mo, plot_comparison):
    current = get_run()["current"]
    defended = get_run()["defended"]
    _base = {**baseline["metrics"], **challenge_metrics(baseline["model"], challenge)}
    _now = {**current["metrics"], **challenge_metrics(current["model"], challenge)}
    _after = {**defended["metrics"], **challenge_metrics(defended["model"], challenge)}
    _show_asr = _now["strategy"] in ("Random label flips", "Strategic label flips", "Feature manipulation")
    _rows = []
    _measures = [
        ("Clean test accuracy", "clean_test_accuracy", _base["test_count"]),
        ("Poisoning rate", "poisoning_rate", len(data.y_train)),
    ]
    if _show_asr:
        _measures.insert(1, ("Attack success rate", "attack_success_rate", len(challenge.X)))
    for _name, _key, _count in _measures:
        _rows.append({"Measure": _name, "Count": _count, "Baseline": f"{_base[_key]:.1%}",
                      "Poisoned": f"{_now[_key]:.1%}", "Poisoned Δ (pp)": f"{100*(_now[_key]-_base[_key]):+.2f}",
                      "Mitigated": f"{_after[_key]:.1%}", "Mitigated Δ (pp)": f"{100*(_after[_key]-_base[_key]):+.2f}"})
    _copy_rows = [{"Experiment": _now["strategy"], **_row} for _row in _rows]
    _headers = list(_copy_rows[0])
    _html_table = "\n".join([
        '<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse;">',
        "<caption>Lab 2 results</caption>",
        "<thead><tr>" + "".join('<th scope="col">' + html.escape(_header) + "</th>" for _header in _headers) + "</tr></thead>",
        "<tbody>",
        *["<tr>" + "".join("<td>" + html.escape(str(_row[_header])) + "</td>" for _header in _headers) + "</tr>" for _row in _copy_rows],
        "</tbody>",
        "</table>",
    ])
    _copy_table = mo.accordion({"Copy HTML table for Canvas": mo.vstack([
        mo.md("Hover over the code and click the copy icon at the top right. Paste into Canvas’s **HTML editor**."),
        mo.ui.code_editor(_html_table, language="html", disabled=True,
            show_copy_button=True, min_height=140, max_height=200),
    ])})
    mo.vstack([
        mo.md(f"## 3 · Compare the models\n**{_now['strategy']}** · {_now['modified_records']} modified records · {_now['poisoning_rate']:.1%} poisoning rate."),
        plot_comparison(data, baseline, current, defended, challenge), mo.ui.table(_rows, selection=None),
        _copy_table,
        mo.md(f"**Predictions flipped:** {_now['challenge_flipped']}/{_now['challenge_count']} poisoned · {_after['challenge_flipped']}/{_after['challenge_count']} mitigated. Pink outlines mark flips.") if _show_asr else mo.md("Choose an experiment and Retrain to see attack success."),
        mo.md(f"**Filter:** {len(defended['removed'])} removed · {int(defended['keep'].sum())} retained. **Detection:** {defended['caught']} edited removed · {defended['false_alarms']} unchanged removed · {defended['missed']} edits missed."),
        mo.callout(defended["status"], kind="warn") if defended["status"].startswith("Filtering would") else mo.md(""),
        mo.accordion({"Metric and mitigation definitions": mo.md("""
            - **Accuracy:** correct predictions / 600 unchanged test records.
            - **Attack success:** zigzag predictions flipped from the clean baseline / 50.
            - **Poisoning rate:** unique modified training records / 400, counted once per record.
            - **Filter:** remove a record if at least 9 of its 11 nearest neighbors have the opposite label; retrain the same SVM. Red crosses mark removals.

            The filter uses current training data only. Detection counts use known edits for evaluation.
            Filtering can miss attacks, remove legitimate records, or worsen accuracy.
            Δ values are percentage points from baseline; poisoning rate always refers to the original edits.
        """)}),
    ])
    return


if __name__ == "__main__":
    app.run()
