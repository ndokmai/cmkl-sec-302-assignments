"""Real two-feature inference, fixed cohort, student oracle, and staged targets."""
import numpy as np
import pytest
import attribute_lab as lab
import experiment as ex
from instructor.search_solution import extract_performance_score as search


@pytest.fixture(scope='module')
def data():
    return ex.load_dataset(False)


@pytest.fixture(scope='module')
def runs(data):
    return lab.fit(data),lab.fit(data,200),lab.fit(data,200,1.)


def test_fixed_training_contract(runs):
    for run in runs:
        assert len(run.train_ids)==210 and run.hidden==(64,32)
        assert run.identifiers and not run.early_stopping
    assert runs[0].train_ids==runs[1].train_ids==runs[2].train_ids
    assert runs[0].validation_ids==runs[1].validation_ids==runs[2].validation_ids


def test_selected_target_and_starting_population(data,runs):
    one=lab.evaluate(data,runs[0],search,(lab.TARGET_ID,))
    assert one['accuracy']==1
    population=lab.evaluate(data,runs[0],search)
    assert population['accuracy']==pytest.approx(60/210)
    assert population['majority_accuracy']==pytest.approx(167/210)
    assert population['max_queries']==20


def test_overfit_and_l2_targets(data,runs):
    unregularized=lab.evaluate(data,runs[1],search)
    defense=lab.evaluate(data,runs[2],search)
    assert unregularized['accuracy']==1.
    assert defense['accuracy']==pytest.approx(68/210)
    assert defense['accuracy']<lab.DEFENSE_TARGET
    assert unregularized['balanced_accuracy']==1.
    assert np.sum(defense['confusion'])==210
    assert sum(r['Correct'] for r in defense['records'])==68


def test_public_profile_has_no_hidden_truth(data,runs):
    public=lab.public_profiles(data,runs[0].train_ids)
    assert 'PerformanceScore' not in public and 'MaritalDesc' not in public
    assert 'Salary' in public and len(public)==210


def test_batched_oracle_matches_genuine_model(data,runs):
    public,ratings,statuses,pairs,values=lab.candidate_predictions(data,runs[0],(lab.TARGET_ID,))
    assert len(pairs)==20
    for index,(rating,status) in enumerate(pairs):
        profile=public.copy()
        profile['PerformanceScore']=rating
        profile['MaritalDesc']=status
        assert values[index,0]==pytest.approx(runs[0].predictor.predict(profile)[0],abs=.02)
    profile=ex.employee_profile(data,lab.TARGET_ID)
    assert lab.TARGET_ID in runs[0].train_ids


def test_hidden_truth_cannot_change_candidate_outputs(data,runs):
    original=data.copy(deep=True)
    changed=data.copy(deep=True)
    for field in ('PerformanceScore','MaritalDesc'):
        changed[field]=changed[field].iloc[::-1].to_numpy()
    first=lab.candidate_predictions(data,runs[0],runs[0].train_ids)
    second=lab.candidate_predictions(changed,runs[0],runs[0].train_ids)
    assert first[3]==second[3]
    np.testing.assert_array_equal(first[4],second[4])
    assert data.equals(original)


def test_student_compile_and_errors(data,runs):
    example=lab.evaluate(data,runs[0],lab.compile_search(lab.STARTER),(lab.TARGET_ID,))
    assert example['accuracy']==0
    assert example['records'][0]['Guessed performance']=='Exceeds'
    with pytest.raises(ValueError,match='Define a function'):
        lab.compile_search('answer=42')
    with pytest.raises(SyntaxError):
        lab.compile_search('def search(')
    with pytest.raises(ValueError,match='Return one valid performance'):
        lab.evaluate(data,runs[0],lambda *args:None,(lab.TARGET_ID,))
    with pytest.raises(ValueError,match='untested'):
        lab.evaluate(data,runs[0],lambda profile,salary,model:"Exceeds",(lab.TARGET_ID,))


def test_nontraining_ids_rejected(data,runs):
    with pytest.raises(ValueError,match='training cohort'):
        lab.evaluate(data,runs[0],search,(runs[0].test_ids[0],))


def test_candidate_plot_and_summary(data,runs):
    png=ex.figure_png(lab.candidate_plot(data,runs[0],lab.TARGET_ID))
    assert png.startswith(b'\x89PNG')
    result=lab.evaluate(data,runs[0],search)
    assert lab.summary(result)['Employees']==210
    assert lab.summary(result)['Attack accuracy (%)']==28.57


def test_manual_target_correct_pair_is_unique_salary_minimum(data,runs):
    correct=lab.check_manual(data,runs[0],'Fully Meets','Single')
    wrong=lab.check_manual(data,runs[0],'Fully Meets','Married')
    assert correct['combination_correct'] and correct['rating_correct']
    assert not wrong['combination_correct'] and wrong['rating_correct']
    assert correct['salary_error']<wrong['salary_error']
    assert correct['is_smallest_difference']
    assert not wrong['is_smallest_difference']
    assert correct['known_salary']==61555
    assert correct['salary_error']==pytest.approx(8.71,abs=.01)
    public,ratings,statuses,pairs,values=lab.candidate_predictions(data,runs[0],(lab.TARGET_ID,))
    errors=abs(values[:,0]-correct['known_salary'])
    assert pairs[errors.argmin()]==('Fully Meets','Single')
    assert np.sort(errors)[1]-np.min(errors)>350
    with pytest.raises(ValueError):
        lab.check_manual(data,runs[0],None,None)


def test_performance_abbreviation_is_expanded(data):
    assert 'PIP' not in set(data.PerformanceScore)
    assert 'Performance Improvement Plan' in set(data.PerformanceScore)


def test_history_highlights_only_best_submitted_predictions():
    rows=[{'performance':'Fully Meets','marital':'Married','predicted_salary':62000.,'salary_error':445.},
          {'performance':'Fully Meets','marital':'Single','predicted_salary':61563.71,'salary_error':8.71}]
    html=lab.prediction_history_html(rows)
    assert html.count('★ Smallest difference so far')==1
    assert '61,563.71' in html and '445.00' in html
    tied=lab.prediction_history_html(rows+[dict(rows[1])])
    assert tied.count('★ Smallest difference so far')==2
    assert 'No predictions yet' in lab.prediction_history_html([])


def test_extractor_receives_full_known_profile_and_returns_only_score(data,runs):
    def extractor(profile,salary,model):
        assert set(profile)=={'Employee_Name','DOB','Age','Sex','DateofHire','Department','Position','ManagerName'}
        assert isinstance(profile['Age'],int)
        assert all(isinstance(value,str) for key,value in profile.items() if key!='Age')
        assert isinstance(salary,float)
        candidate={**profile,'PerformanceScore':'Exceeds','MaritalDesc':'Divorced'}
        import pandas as pd
        assert model.predict(candidate)==pytest.approx(runs[0].predictor.predict(pd.DataFrame([candidate]))[0],abs=.02)
        return search(profile,salary,model)
    result=lab.evaluate(data,runs[0],extractor,(lab.TARGET_ID,))
    assert result['accuracy']==1
    assert 'Guessed marital status' not in result['records'][0]


def test_confusion_image_and_clean_result_summary(data,runs):
    result=lab.evaluate(data,runs[0],search)
    assert ex.figure_png(lab.confusion_plot(result)).startswith(b'\x89PNG')
    summary=lab.summary(result)
    assert 'Epochs' not in summary and 'L2' not in summary
    assert not any('most common' in key for key in summary)


def test_training_options_reach_the_real_model(data):
    changed=lab.fit(data,epochs=5,alpha=.1,early_stopping=True,
                    identifiers=False,learning_rate=.0003,batch_size=8)
    net=changed.predictor.network
    assert net.learning_rate_init==.0003 and net.batch_size==8
    assert changed.alpha==.1 and changed.early_stopping and not changed.identifiers
    assert 'Employee_Name' not in changed.features and len(changed.train_ids)==210
    assert changed.hidden==(64,32)
    original=lab.fit(data)
    assert changed.fingerprint!=original.fingerprint
    for settings in ({'learning_rate':.5},{'batch_size':2}):
        with pytest.raises(ValueError):
            lab.fit(data,**settings)


def test_retrained_report_records_parameters_and_100_percent(data):
    run=lab.fit(data,epochs=200)
    result=lab.evaluate(data,run,search)
    assert result['accuracy']==1
    assert result['settings']=={'epochs':200,'L2':0.,'early_stopping':False,
                                'identifiers':True,'learning_rate':.003,'batch_size':32}


def test_defense_requires_recovery_and_fixed_validation_quality(data,runs):
    starting=lab.evaluate(data,runs[0],search)
    retrained=lab.evaluate(data,runs[1],search)
    defense=lab.evaluate(data,runs[2],search)
    checks=lab.defense_checks(starting,defense)
    assert checks['passed'] and checks['quality_passed'] and checks['recovery_passed']
    assert checks['validation_limit']==pytest.approx(starting['salary_metrics'][1]['MAE']*1.1)
    assert checks['validation_limit']==pytest.approx(8412.81,abs=.02)
    assert not lab.defense_checks(starting,retrained)['recovery_passed']
    limit=checks['validation_limit']
    for value,expected in [(limit,True),(limit+.01,False),(float('inf'),False),(float('nan'),False)]:
        candidate={**defense,'salary_metrics':[{'Split':'Validation','MAE':value}]}
        result=lab.defense_checks(starting,candidate)
        assert result['quality_passed']==expected
        assert result['passed']==expected
    candidate={**defense,'accuracy':.4}
    assert not lab.defense_checks(starting,candidate)['recovery_passed']
