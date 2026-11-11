import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="Lab 5 · Infer a Confidential Performance Rating",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import json
    import html
    import pandas as pd
    from experiment import load_dataset,architecture_plot,single_learning_plot,figure_png,SOURCE_URL
    from attribute_lab import (fit,compile_search,evaluate,public_profiles,candidate_plot,summary,
                               TARGET_ID,STARTER,DEFENSE_TARGET,model_overview_plot,check_manual,prediction_history_html,confusion_plot,submission_card,defense_checks)
    def display_result(result):
        correct=sum(row['Correct'] for row in result['records'])
        return mo.vstack([
            mo.md(f"**Your attack recovered {correct} of {result['n']} performance scores correctly "
                  f"({result['accuracy']:.1%}).**"),
            mo.md('**Confusion matrix:** rows show actual performance; columns show predicted performance. '
                  'Each cell counts employees. Correct guesses fall on the diagonal.'),
            mo.image(figure_png(confusion_plot(result)),width=480),
            mo.accordion({
                'Detailed metrics and salary prediction errors':mo.vstack([
                    mo.md('**Average success across rating categories** gives each of the four ratings equal importance. '
                          'First, calculate the percentage correctly inferred within each rating; then average '
                          'those four percentages. This shows whether your attack also works on less common ratings. '
                          'This measure is also called balanced accuracy.'),
                    mo.ui.table([summary(result)],selection=None,pagination=False),
                ]),
            }),
        ])

    return (
        SOURCE_URL,
        STARTER,
        TARGET_ID,
        check_manual,
        compile_search,
        defense_checks,
        display_result,
        evaluate,
        figure_png,
        fit,
        load_dataset,
        mo,
        model_overview_plot,
        prediction_history_html,
        public_profiles,
        single_learning_plot,
        submission_card,
    )


@app.cell
def _(figure_png, mo, model_overview_plot):
    mo.vstack([
        mo.md("""
        # Lab 5 · Infer a Confidential Performance Rating

        ## Scenario
        You are an attacker trying to steal confidential performance ratings of a
        company's employees. You have partial information: their names, dates of birth,
        ages, gender, hire dates, departments, positions, managers, and salaries.
        You do not know their marital status or **performance score—the information
        you want to uncover**.

        Luckily, there is a way to try to steal that information. The HR team uses
        employee information, including these confidential fields,
        to train a neural network that predicts salary for an internal pay-equity
        investigation. For transparency, the company makes the salary prediction
        service publicly accessible.

        **Your goal is to use attribute inference to infer employees' confidential
        performance scores from the model's responses.**
        """),
        mo.image(figure_png(model_overview_plot())),
        mo.md("""
        ## Lab outline
        1. Try the missing values manually and check your inference.
        2. Automate the attack: check your guess for Myriam, then evaluate all training employees.
        3. Experiment with training parameters to demonstrate memorization through overfitting.
        4. Apply L2 regularization and evaluate the defense.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    load_button=mo.ui.run_button(label='Load data and train starting model')
    load_button
    return (load_button,)


@app.cell(hide_code=True)
def _(fit, load_button, load_dataset, mo):
    mo.stop(not load_button.value)
    data=None
    _problem=None
    try:
        with mo.status.spinner(title='Loading verified source and training five epochs…'):
            data=load_dataset()
            initial=fit(data)
    except Exception as _exc:
        _problem=str(_exc)
    mo.stop(_problem is not None,mo.callout(f'Could not prepare the lab: {_problem}. Try prepare_data.py while online.',kind='warn'))
    return data, initial


@app.cell(hide_code=True)
def _(TARGET_ID, data, initial, mo, public_profiles):
    target_public=public_profiles(data,(TARGET_ID,))
    _training=data.set_index('EmpID').loc[list(initial.train_ids)]
    _profile=target_public.iloc[0]
    _labels={'Employee_Name':'Name','DOB':'Date of birth','Age':'Age','Sex':'Gender',
             'DateofHire':'Hire date','Department':'Department','Position':'Position','ManagerName':'Manager'}
    _rows='\n'.join(f'| {label} | {_profile[field]} |' for field,label in _labels.items())
    _table='| Attribute | Known information / your selection |\n| :--- | :--- |\n'+_rows+'\n| Performance score | {performance} |\n| Marital status | {marital} |'
    manual_form=mo.md(_table).batch(
        performance=mo.ui.dropdown(sorted(_training.PerformanceScore.unique().tolist()),
            value=None,allow_select_none=True,label='Performance score'),
        marital=mo.ui.dropdown(sorted(_training.MaritalDesc.unique().tolist()),
            value=None,allow_select_none=True,label='Marital status'),
    ).form(submit_button_label='Run inference',show_clear_button=False)
    mo.vstack([
        mo.md('## 1 · Try the missing values\n'
              'Your first target is **Myriam Givens**. Her marital status and performance score '
              'are missing. **Try different combinations to bring the predicted salary as close '
              'as possible to her known salary.** Each inference also checks whether your combination is correct.'),
        mo.md(f"**Known salary: {_profile.Salary:,.0f}**"),
        manual_form,
    ])
    return (manual_form,)


@app.cell(hide_code=True)
def _(mo):
    get_manual_history,set_manual_history=mo.state([])
    return get_manual_history, set_manual_history


@app.cell(hide_code=True)
def _(check_manual, data, initial, manual_form, mo, set_manual_history):
    mo.stop(manual_form.value is None)
    _choices=manual_form.value
    mo.stop(any(value is None for value in _choices.values()),
            mo.callout('Select both fields, then run inference.',kind='info'))
    manual_result=check_manual(data,initial,_choices['performance'],_choices['marital'])
    _correct=manual_result['combination_correct']
    set_manual_history(lambda history:history+[dict(manual_result)])
    mo.vstack([
        mo.md(f"**Predicted salary: {manual_result['predicted_salary']:,.2f}** · "
              f"Known salary: {manual_result['known_salary']:,.0f} · "
              f"Difference: {manual_result['salary_error']:,.2f}"),
        mo.callout('Correct! You recovered Myriam’s performance score and marital status.' if _correct else
                   'This combination does not match both hidden values. Try another combination.',
                   kind='success' if _correct else 'info'),
    ])
    return (manual_result,)


@app.cell(hide_code=True)
def _(get_manual_history, mo, prediction_history_html, set_manual_history):
    _history=get_manual_history()
    mo.stop(not _history)
    _clear=mo.ui.button(label='Clear prediction history',on_click=lambda _:set_manual_history([]))
    mo.vstack([
        mo.md('### Prediction history\nThe highlighted row has the smallest difference among your submitted predictions.'),
        mo.Html(prediction_history_html(_history)),
        _clear,
    ])
    return


@app.cell(hide_code=True)
def _(SOURCE_URL, STARTER, manual_result, mo):
    mo.stop(not manual_result['combination_correct'])
    search_form=mo.md('{code}').batch(code=mo.ui.code_editor(
        value=STARTER,language='python',min_height=220,label='Your extract_performance_score function'
    )).form(submit_button_label='Run my attack on all training employees',show_clear_button=False)
    mo.vstack([
        mo.md('## 2 · Automated attack\nApply what you learned in Part 1 to automate the search for Myriam’s confidential performance score. Write the function below, then check its guess for Myriam.'),
        mo.accordion({'Dataset and model details':mo.md(
            f'Synthetic employees from a fictitious company; no real private personnel records. '
            f'[Dataset and license]({SOURCE_URL}) by Dr. Rich Huebner and Dr. Carla Patalano '
            '(CC BY-NC-ND 4.0, learning/teaching use). We use pinned version 4 and cache the CSV unchanged. '
            'Fixed 210 training / 38 validation / 63 held-out employees; names/DOBs included; '
            'early stopping off. Salary and employee ID are not model inputs. '
            'Age uses 2023-01-30; this dataset has no separate PayRate field.')}),
        mo.md('''**Complete only `extract_performance_score`.** Your function receives:

    - `profile` (`dict`): all known employee inputs; age is an integer, other fields are strings.
      The actual `PerformanceScore` and `MaritalDesc` are not included.
    - `known_salary` (`float`): the observed salary, separate from the model inputs.
    - `model`: a salary prediction interface. `model.predict(full_profile)` accepts a
      dictionary containing the known fields plus `PerformanceScore` and `MaritalDesc`,
      and returns a salary as a `float`. Keep the known fields unchanged.

    The lists `performance_values` and `marital_values` above the function are ready to use. They contain possible values, not the employee’s actual answers.
    The starter runs but makes an incorrect fixed guess for Myriam. Replace it with your implementation.
    Return only the inferred performance score as a `str`.
    The same function will be evaluated for each employee with their own known profile.

    Your Python executes locally when you submit; imports, training, model queries,
    verification, and charts are already provided. Editing the box does not run it.
    Resubmitting code resets subsequent stages so their results use the same algorithm.'''),
        search_form,
    ])
    return (search_form,)


@app.cell(hide_code=True)
def _(TARGET_ID, compile_search, data, evaluate, initial, mo, search_form):
    mo.stop(search_form.value is None)
    search_function=None
    _problem=None
    try:
        search_function=compile_search(search_form.value['code'])
        with mo.status.spinner(title='Applying your search to all 210 training employees…'):
            initial_population=evaluate(data,initial,search_function)
    except Exception as _exc:
        _problem=f'{type(_exc).__name__}: {_exc}'
    mo.stop(_problem is not None,mo.callout(f'Search needs attention: {_problem}',kind='warn'))
    _target=next(row for row in initial_population['records'] if row['EmpID']==TARGET_ID)
    mo.vstack([
        mo.md(f"### Your automated guess for Myriam\n**Performance score: {_target['Guessed performance']}**"),
        mo.callout('Correct — your attack recovered Myriam’s performance score.' if _target['Correct'] else
                   'Your attack did not recover Myriam’s performance score. Review your function.',
                   kind='success' if _target['Correct'] else 'info'),
    ])
    return initial_population, search_function


@app.cell(hide_code=True)
def _(TARGET_ID, display_result, initial_population, mo):
    _target=next(row for row in initial_population['records'] if row['EmpID']==TARGET_ID)
    _message=('Your attack works for Myriam, but not for most training employees. '
              'Success on one person does not mean the attack works across the database.'
              if _target['Correct'] and initial_population['accuracy']<.5 else
              'Compare the result for Myriam with the attack’s overall success rate.')
    mo.vstack([
        mo.md('### How well does it work across the training database?\n'
              'We apply the same function to all 210 training employees. Here are the results. '
              '**Correct performance ratings ÷ 210** is the attack accuracy.'),
        display_result(initial_population),
        mo.callout(_message,kind='info'),
    ])
    return


@app.cell(hide_code=True)
def _(TARGET_ID, initial_population, mo):
    _target=next(row for row in initial_population['records'] if row['EmpID']==TARGET_ID)
    mo.stop(not _target['Correct'])
    epoch_form=mo.md('{epochs} {learning_rate} {batch_size}\n\n{alpha} {early_stopping} {identifiers}').batch(
        epochs=mo.ui.dropdown([5,20,50,200,1000],value=5,label='Training epochs'),
        learning_rate=mo.ui.dropdown([.0003,.003,.03],value=.003,label='Learning rate'),
        batch_size=mo.ui.dropdown([8,32,128],value=32,label='Batch size'),
        alpha=mo.ui.dropdown([0.,.1,1.,10.],value=0.,label='L2 weight penalty · 0 means off'),
        early_stopping=mo.ui.checkbox(value=False,label='Validation early stopping'),
        identifiers=mo.ui.checkbox(value=True,label='Include names and dates of birth'),
    ).form(submit_button_label='Retrain and repeat my attack',show_clear_button=False)
    mo.vstack([
        mo.md('## 3 · Overfitted models can memorize training data\n'
              '**Goal: retrain the model until your existing attack correctly infers all 210 performance scores (100%).** '
              'Experiment with the parameters below while keeping your attack unchanged. '
              'Compare training and validation salary errors to investigate whether your results show memorization.'),
        mo.accordion({'Training parameters':mo.md(
            '- **Epochs:** passes through the training data.\n'
            '- **Learning rate:** size of each optimization step.\n'
            '- **Batch size:** employees used for each weight update.\n'
            '- **L2 weight penalty:** penalty on large weights.\n'
            '- **Early stopping:** monitors validation error and restores the best checkpoint.\n'
            '- **Names and dates of birth:** whether these fields are used as model inputs.\n\n'
            'The 64 → 32 network, 210 training employees, data split, and random seed stay fixed.')}),
        epoch_form,
    ])
    return (epoch_form,)


@app.cell(hide_code=True)
def _(
    data,
    display_result,
    epoch_form,
    evaluate,
    figure_png,
    fit,
    mo,
    search_function,
    single_learning_plot,
):
    mo.stop(epoch_form.value is None)
    _problem=None
    try:
        with mo.status.spinner(title='Retraining and applying your search…'):
            overfit_run=fit(data,**epoch_form.value)
            overfit_result=evaluate(data,overfit_run,search_function)
    except Exception as _exc:
        _problem=f'{type(_exc).__name__}: {_exc}'
    mo.stop(_problem is not None,mo.callout(f'Experiment failed: {_problem}. Check your search in part 2.',kind='warn'))
    _complete=all(r['Correct'] for r in overfit_result['records'])
    mo.vstack([
        display_result(overfit_result),
        mo.image(figure_png(single_learning_plot(overfit_run))),
        mo.callout('100% target achieved — your attack recovered every training employee’s performance score.' if _complete else 'Not yet 100%. Try another training configuration and compare the errors.',kind='success' if _complete else 'info'),
    ])
    return (overfit_result,)


@app.cell(hide_code=True)
def _(initial_population, mo, overfit_result):
    _baseline=next(row['MAE'] for row in initial_population['salary_metrics'] if row['Split']=='Validation')
    _limit=1.1*_baseline
    mo.stop(overfit_result['accuracy']<1)
    defense_form=mo.md('{alpha}').batch(alpha=mo.ui.dropdown(
        [.1,1.,10.],value=.1,label='L2 weight penalty'
    )).form(submit_button_label='Apply defense and repeat the attack',show_clear_button=False)
    mo.vstack([
        mo.md('## 4 · Apply the defense\n'
              'Apply L2 regularization to the model from Part 3. Change only the weight penalty; '
              'keep your other training settings and extraction function unchanged. '
              '**Both requirements must pass:** attack success below 40%, and validation salary error '
              f'no higher than **{_limit:,.2f}** (10% above the fixed starting model’s {_baseline:,.2f}). '
              'Validation error is the average absolute salary prediction error. '
              'Use validation to choose settings; held-out data stays separate. L2 does not guarantee privacy.'),
        defense_form,
    ])
    return (defense_form,)


@app.cell(hide_code=True)
def _(
    data,
    defense_checks,
    defense_form,
    display_result,
    epoch_form,
    evaluate,
    fit,
    initial_population,
    mo,
    overfit_result,
    search_function,
):
    mo.stop(defense_form.value is None)
    _settings={**epoch_form.value,'alpha':float(defense_form.value['alpha'])}
    mo.stop(_settings['alpha']==overfit_result['alpha'],
            mo.callout('Choose a different weight penalty from your Part 3 model.',kind='info'))
    _problem=None
    try:
        with mo.status.spinner(title='Training the regularized model and repeating your attack…'):
            defense_run=fit(data,**_settings)
            defense_result=evaluate(data,defense_run,search_function)
    except Exception as _exc:
        _problem=f'{type(_exc).__name__}: {_exc}'
    mo.stop(_problem is not None,mo.callout(f'Defense experiment failed: {_problem}',kind='warn'))
    _checks=defense_checks(initial_population,defense_result)
    mo.vstack([
        display_result(defense_result),
        mo.callout('Recovery goal passed: attack success is below 40%.' if _checks['recovery_passed'] else
                   'Recovery goal not met: attack success must be below 40%.',
                   kind='success' if _checks['recovery_passed'] else 'info'),
        mo.callout(f"Prediction quality {'passed' if _checks['quality_passed'] else 'not met'}: "
                   f"validation error {_checks['validation_mae']:,.2f}; limit {_checks['validation_limit']:,.2f}.",
                   kind='success' if _checks['quality_passed'] else 'info'),
        mo.callout('Both defense requirements passed.' if _checks['passed'] else
                   'Both requirements must pass. Try another weight penalty.',
                   kind='success' if _checks['passed'] else 'info'),
    ])
    return (defense_result,)


@app.cell(hide_code=True)
def _(
    defense_result,
    figure_png,
    initial_population,
    manual_result,
    mo,
    overfit_result,
    submission_card,
):
    _image=figure_png(submission_card(manual_result,initial_population,overfit_result,defense_result))
    mo.vstack([
        mo.md('### Submission\nSubmit this image.'),
        mo.image(_image,width=1000),
        mo.download(_image,filename='lab-5-submission.png',mimetype='image/png',label='Download submission image'),
    ])
    return


if __name__ == "__main__":
    app.run()
