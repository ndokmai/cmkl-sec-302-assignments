"""Real-data MLP training, 1% recovery, regularized comparison, and data contracts."""
from io import BytesIO
import numpy as np
import pytest
from PIL import Image
import experiment as ex


@pytest.fixture(scope='module')
def data():
    return ex.load_dataset()


@pytest.fixture(scope='module')
def pair(data):
    return ex.train_pair(data,train_size=100,epochs=200,defense_alpha=1.,defense_early_stopping=True)


def test_verified_source_and_dates(data):
    assert len(data)==311 and data.EmpID.is_unique
    assert 'Salary' not in ex.PROFILE_FIELDS and 'PayRate' not in data
    assert data.Age.between(18,100).all()
    assert ex.sha256(ex.dataset_path(False).read_bytes()).hexdigest()==ex.SOURCE_HASH
    assert ex.parse_date('01/01/55',birth=True)=='1955-01-01'


@pytest.mark.parametrize('value',['2055-01-01','bad date','2023-02-01',''])
def test_invalid_dates(value):
    with pytest.raises(ValueError):ex.parse_date(value,birth=True)


def test_fixed_three_way_split(data):
    train,validation,test=ex.dataset_split(data)
    assert (len(train),len(validation),len(test))==(210,38,63)
    assert not set(train)&set(validation) and not set(train)&set(test) and not set(validation)&set(test)
    assert set(train)|set(validation)|set(test)==set(data.EmpID)
    assert ex.dataset_split(data)==(train,validation,test)


def test_baseline_recovers_training_values_and_defense_reduces_rate(pair):
    baseline,defense=pair
    assert baseline.metrics[0]['Within 1% rate']>.95
    assert baseline.metrics[2]['Within 1% rate']<.2
    assert defense.metrics[0]['Within 1% rate']<baseline.metrics[0]['Within 1% rate']
    assert defense.metrics[2]['MAE']<baseline.metrics[2]['MAE']
    assert baseline.alpha==0 and not baseline.early_stopping
    assert defense.alpha==1 and defense.early_stopping
    assert baseline.train_ids==defense.train_ids
    assert baseline.validation_ids==defense.validation_ids and baseline.test_ids==defense.test_ids


def test_early_stopping_restores_actual_selected_checkpoint(pair):
    baseline,defense=pair
    assert baseline.epochs_run==baseline.epochs_requested==200
    assert defense.epochs_run<200 and defense.selected_epoch<defense.epochs_run
    row=defense.history[defense.selected_epoch-1]
    assert defense.metrics[0]['MAE']==pytest.approx(row['Training MAE'],abs=1e-5)
    assert defense.metrics[1]['MAE']==pytest.approx(row['Validation MAE'],abs=1e-5)


def test_training_only_preprocessing_and_target_scaling(data,pair):
    run=pair[0]
    fitted=data[data.EmpID.isin(run.train_ids)]
    encoder=run.predictor.preprocessor.named_transformers_['categorical']
    cats=ex.CATEGORIES+['Employee_Name','DOB']
    assert set(encoder.categories_[cats.index('Employee_Name')])==set(fitted.Employee_Name)
    assert not set(encoder.categories_[cats.index('Employee_Name')])&set(data[data.EmpID.isin(run.validation_ids+run.test_ids)].Employee_Name)
    assert run.predictor.target_mean==pytest.approx(fitted.Salary.mean())
    assert run.predictor.target_scale==pytest.approx(fitted.Salary.to_numpy().std())
    assert run.predictor.preprocessor.named_transformers_['age'].mean_[0]==pytest.approx(fitted.Age.mean())
    assert 'Salary' not in run.features and 'EmpID' not in run.features


def test_known_employee_recovery_and_defense_comparison(data,pair):
    baseline,defense=pair
    employee=10042
    profile=ex.employee_profile(data,employee)
    predicted=ex.predict_salary(baseline,profile)
    regularized=ex.predict_salary(defense,profile)
    assert ex.verify_salary(data,baseline,employee,predicted)['Recovery complete']
    assert not ex.verify_salary(data,defense,employee,regularized)['Within 1%']
    proof=ex.certificate(data,baseline,employee,predicted)
    assert proof['tolerance']==.01 and proof['Relative error (%)']<=1


@pytest.mark.parametrize('offset,accepted',[(-.010001,False),(-.01,True),(0,True),(.01,True),(.010001,False)])
def test_threshold_boundaries_use_raw_predictions(data,pair,offset,accepted):
    run=pair[0];employee=run.train_ids[0]
    p=ex.predict_salary(run,ex.employee_profile(data,employee))
    actual=int(data.loc[data.EmpID==employee,'Salary'].iloc[0])
    p['predicted_salary']=actual*(1+offset)
    result=ex.verify_salary(data,run,employee,p)
    assert result['Within 1%']==accepted
    assert result['Recovery complete']==accepted
    assert result['Accepted minimum']==actual*.99 and result['Accepted maximum']==actual*1.01


def test_edited_profile_stale_model_and_held_out_cannot_certify(data,pair):
    run=pair[0];employee=run.train_ids[0]
    p=ex.predict_salary(run,{**ex.employee_profile(data,employee),'Employee_Name':'Edited name'})
    assert not ex.verify_salary(data,run,employee,p)['Recovery complete']
    p=ex.predict_salary(run,ex.employee_profile(data,employee));p['model']='old'
    assert not ex.verify_salary(data,run,employee,p)['Recovery complete']
    employee=run.test_ids[0]
    p=ex.predict_salary(run,ex.employee_profile(data,employee))
    p['predicted_salary']=int(data.loc[data.EmpID==employee,'Salary'].iloc[0])
    assert ex.verify_salary(data,run,employee,p)['Within 1%']
    assert not ex.verify_salary(data,run,employee,p)['Recovery complete']


def test_salary_input_ignored_and_age_recomputed(data,pair):
    run=pair[0];profile=ex.employee_profile(data,run.train_ids[0])
    assert ex.predict_salary(run,profile)==ex.predict_salary(run,{**profile,'Salary':1,'Age':999})


def test_training_duration_and_capacity_are_real_controls(data,pair):
    early=ex.train_model(data,train_size=100,epochs=5,hidden=(64,32))
    smaller=ex.train_model(data,train_size=50,epochs=5,hidden=(32,),identifiers=False)
    assert early.metrics[0]['Within 1% rate']<pair[0].metrics[0]['Within 1% rate']
    assert smaller.train_ids==pair[0].train_ids[:50]
    assert len(smaller.predictor.network.coefs_)==2
    assert len(pair[0].predictor.network.coefs_)==3
    assert 'Employee_Name' not in smaller.features and 'DOB' not in smaller.features


def test_regularization_only_control_and_repeatability(data):
    a,b=ex.train_pair(data,epochs=20,defense_early_stopping=False)
    assert a.epochs_run==b.epochs_run==20
    assert a.fingerprint!=b.fingerprint
    repeated=ex.train_model(data,epochs=20)
    assert repeated.fingerprint==a.fingerprint


def test_groups_and_actual_figures(data,pair):
    baseline,defense=pair
    assert sum(r['Employees'] for r in ex.group_summary(data,baseline))==63
    for fig in (ex.comparison_plot(data,baseline,defense),ex.learning_plot(baseline,defense),ex.architecture_plot(baseline)):
        with Image.open(BytesIO(ex.figure_png(fig))) as img:assert img.format=='PNG' and img.width>500


@pytest.mark.parametrize('settings',[{'train_size':248},{'hidden':(128,)},{'epochs':0},{'alpha':-1}])
def test_invalid_training_settings(data,settings):
    with pytest.raises(ValueError):ex.train_model(data,**settings)


def test_cache_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(ex,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):ex.dataset_path(False)
    (tmp_path/'.cache').mkdir();(tmp_path/'.cache'/'HRDataset_v14.csv').write_text('changed')
    with pytest.raises(ValueError,match='differs'):ex.dataset_path(False)


def test_regularized_model_can_still_recover_an_individual(data,pair):
    run=pair[1]
    profile=ex.employee_profile(data,10128)
    result=ex.verify_salary(data,run,10128,ex.predict_salary(run,profile))
    assert result['Within 1%'] and result['Recovery complete']
    assert all(not w.flags.writeable for w in run.predictor.network.coefs_)


def test_difficult_default_configuration(data):
    run=ex.train_model(data,**ex.DEFAULT_SETTINGS)
    assert run.metrics[0]['Within 1% rate']<=.1
    for employee in run.train_ids[:5]:
        p=ex.predict_salary(run,ex.employee_profile(data,employee))
        assert not ex.verify_salary(data,run,employee,p)['Within 1%']


def test_student_outputs_have_no_population_recovery_statistics(pair):
    run=pair[0]
    rows=ex.learner_metrics(run)
    assert {row['Split'] for row in rows}=={'Training','Validation'}
    assert all(set(row)=={'Split','Employees','MAE','RMSE'} for row in rows)
    assert not any('within' in key.lower() or 'rate' in key.lower() for key in ex.run_summary(run))
    fig=ex.single_learning_plot(run)
    assert len(fig.axes)==1
    assert set(line.get_label() for line in fig.axes[0].lines)=={'Training','Validation','Selected weights'}
    assert ex.figure_png(fig).startswith(b'\x89PNG')


def test_five_record_budget_blocks_new_salaries_and_rechecks_are_cached(data,pair):
    run=pair[0];ledger={}
    for index,employee in enumerate(run.train_ids[:5]):
        p=ex.predict_salary(run,ex.employee_profile(data,employee))
        outcome,ledger=ex.attempt_verification(data,run,employee,p,ledger)
        assert outcome['status']=='checked' and outcome['remaining']==4-index
    assert ex.remaining_checks(ledger,run)==0
    employee=run.train_ids[5];p=ex.predict_salary(run,ex.employee_profile(data,employee))
    blocked,unchanged=ex.attempt_verification(data,run,employee,p,ledger)
    assert blocked['status']=='exhausted' and 'result' not in blocked
    assert 'Recorded salary' not in blocked and unchanged is ledger
    employee=run.train_ids[0];p=ex.predict_salary(run,ex.employee_profile(data,employee))
    cached,unchanged=ex.attempt_verification(data,run,employee,p,ledger)
    assert cached['status']=='cached' and cached['remaining']==0
    assert cached['result']==ledger[run.fingerprint][str(employee)]
    assert unchanged is ledger


def test_identical_retraining_cannot_reset_record_budget(data,pair):
    run=pair[0]
    repeated=ex.train_model(data,epochs=200)
    assert repeated.fingerprint==run.fingerprint
    ledger={run.fingerprint:{str(employee):{'already':'checked'} for employee in run.train_ids[:5]}}
    employee=run.train_ids[5];p=ex.predict_salary(repeated,ex.employee_profile(data,employee))
    out,_=ex.attempt_verification(data,repeated,employee,p,ledger)
    assert out['status']=='exhausted'
    changed=ex.train_model(data,epochs=5)
    assert ex.remaining_checks(ledger,changed)==5


def test_invalid_profile_checks_do_not_reveal_salary_or_consume_slots(data,pair):
    run=pair[0];employee=run.train_ids[0]
    p=ex.predict_salary(run,{**ex.employee_profile(data,employee),'Employee_Name':'Edited'})
    out,ledger=ex.attempt_verification(data,run,employee,p,{})
    assert out['status']=='invalid' and 'result' not in out and ledger=={}
    employee=run.test_ids[0];p=ex.predict_salary(run,ex.employee_profile(data,employee))
    out,ledger=ex.attempt_verification(data,run,employee,p,{})
    assert out['status']=='invalid' and 'result' not in out and ledger=={}
    p['model']='old'
    out,ledger=ex.attempt_verification(data,run,employee,p,{})
    assert out['status']=='invalid' and ledger=={}
