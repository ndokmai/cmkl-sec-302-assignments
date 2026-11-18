"""Numerical and protocol checks for the five student functions."""
from pathlib import Path
import re
import numpy as np
import pytest
from experiment import run_he, run_fl, dataset, compile_student, FL_STARTER, HE_STARTER
from viewer import ProtocolLab
blocks=re.findall(r'```python\n(.*?)```',Path(__file__).with_name('solution.md').read_text(),re.S)
FL,HE=blocks


def test_weighted_fedsgd_matches_centralized_each_round():
    result=run_fl(FL,num_rounds=100,learning_rate=.1)
    assert not result['error'],result['error']
    assert all(result['checks'].values())
    clients,test=dataset();X=np.vstack([X for X,y in clients]);y=np.concatenate([y for X,y in clients])
    theta=np.zeros(3)
    for row in result['history'][1:]:
        theta-=result["learning_rate"]*2*X.T@(X@theta-y)/len(y)
        assert np.isclose(row['mse'],np.mean((test[0]@theta-test[1])**2))
    assert np.allclose(theta,result['theta'])
    for e in result['events']:
        if e['message']:
            assert set(e['message']['payload']) <= {'parameters','gradient','sample_count','round'}
    avg=compile_student(FL,'aggregate_gradients')
    assert np.allclose(avg([[1,2,3],[5,6,7]],[1,3]),[4,5,6])


def test_resume_and_custom_settings():
    source=FL
    full=run_fl(source,num_rounds=7,learning_rate=.08)
    first=run_fl(source,1,num_rounds=7,learning_rate=.08)
    resumed=run_fl(source,previous=first,num_rounds=7,learning_rate=.08)
    assert not full['error'] and not resumed['error']
    assert len(full['history'])==8 and full['learning_rate']==.08
    assert np.allclose(full['theta'],resumed['theta'])


def test_student_errors_and_wrong_results_are_visible():
    for source in [FL_STARTER,FL.replace('return 2 * local_X.T @ errors / len(local_y)','raise ValueError("gradient failure")')]:
        r=run_fl(source);assert r['error'] and r['events'][-1]['text']==r['error']
    bad=run_fl(FL.replace('return 2 * local_X.T @ errors / len(local_y)','return np.zeros(3)'),1)
    assert not bad['error'] and not bad['checks']['Client gradients match MSE derivatives']
    assert bad['theta']==[0.,0.,0.]
    bad=run_fl(FL.replace('theta = theta - learning_rate * mean_gradient','theta = theta + learning_rate * mean_gradient'),1)
    assert not bad['error'] and not bad['checks']['Server update matches weighted FedSGD']


def test_round_loop_must_yield_each_update():
    r=run_fl(FL.replace('yield theta.copy()','return theta.copy()'),1)
    assert r['error']
    r=run_fl(FL.replace('range(num_rounds)','range(0)'),1)
    assert r['error']


def test_he_student_encryption_and_prediction():
    r=run_he(HE)
    assert not r['error'] and all(r['checks'].values())
    assert np.allclose([row['prediction'] for row in r['rows']],[39.4,47,53.8],atol=1e-4)
    r=run_he(HE.replace('return encrypted.serialize()','return b"bad ciphertext"'))
    assert r['error'] and 'Receive input 1' in r['error']
    r=run_he(HE_STARTER)
    assert r['error'] and 'Encrypt input 1' in r['error']
    r=run_he(HE.replace(' + bias',''))
    assert not r['error'] and not all(r['checks'].values())


def test_widget_custom_rounds_connection_and_invalidation():
    widget=ProtocolLab();source=FL
    def request(action,i,source=source):
        widget.request=dict(id=i,action=action,fl_source=source,he_source=HE,num_rounds=3,learning_rate=.12,inputs=[[.2,.25],[.5,.75],[.9,1]])
        assert not widget.response.get('error'),widget.response
    request('twenty',1);assert widget.response['results']['fl']['history'][-1]['round']==3
    request('he',2);assert all(widget.response['results']['linked']['checks'].values())
    request('card',3);assert widget.response['card']
    request('one',4,source+'\n');assert 'linked' not in widget.response['results']
    request('reset',5);assert not widget.response['results']


def test_distinct_synthetic_records_and_client_gradient_events():
    clients,_=dataset()
    for X,y in clients:
        assert len(np.unique(np.column_stack([X,y]),axis=0))==len(y)
        assert np.all((X[:,:2]>=0)&(X[:,:2]<=1))
    result=run_fl(FL,1)
    previews=[e['updates'][e['actor']]['record_preview'] for e in result['events'] if e['actor'].startswith('c') and 'record_preview' in e['updates'].get(e['actor'],{})]
    assert len(previews)==3 and all(len(rows)==5 for rows in previews)
    gradients=[e['updates'][e['actor']]['gradient'] for e in result['events'] if 'gradient' in e['updates'].get(e['actor'],{})]
    assert len(gradients)==3 and all(len(g)==3 for g in gradients)


def test_compact_rounds_refresh_all_client_values():
    result=run_fl(FL,num_rounds=5)
    assert not result['error']
    clients,_=dataset()
    for rnd in range(1,6):
        for i,(X,y) in enumerate(clients,1):
            updates=[e['updates'][f'c{i}'] for e in result['events']
                     if e['updates'].get(f'c{i}',{}).get('round')==rnd
                     and e['updates'].get(f'c{i}',{}).get('gradient') is not None]
            assert len(updates)==1
            state=updates[0];theta=np.array(state['starting_parameters'])
            assert np.allclose(state['gradient'],2*X.T@(X@theta-y)/len(y))
            assert np.isclose(state['local_mse'],np.mean((X@theta-y)**2))
    for i in range(1,4):
        snapshots=[e['updates'][f'c{i}'] for e in result['events']
                   if e['updates'].get(f'c{i}',{}).get('gradient') is not None]
        assert not np.allclose(snapshots[0]['gradient'],snapshots[-1]['gradient'])


def test_starting_settings_and_training_goal():
    initial=run_fl(FL)
    assert initial['num_rounds']==1 and initial['learning_rate']==.1
    assert not initial['checks']['Training goal: held-out MSE ≤ 3.0 $k²']
    tuned=run_fl(FL,num_rounds=100,learning_rate=.1)
    assert tuned['checks']['Training goal: held-out MSE ≤ 3.0 $k²']


def test_student_decryption_errors_and_wrong_outputs_are_visible():
    source=HE.replace('return encrypted_y.decrypt()[0]', 'raise ValueError("student decryption failed")')
    result=run_he(source)
    assert 'Decrypt input 1' in result['error']
    assert 'student decryption failed' in result['error']
    assert result['events'][-1]['actor']=='client'
    result=run_he(HE.replace('return encrypted_y.decrypt()[0]', 'return 0.0'))
    assert not result['error']
    assert result['rows'][0]['prediction']==0.0
    assert not result['checks']['All three errors ≤ 0.0001 $k']
