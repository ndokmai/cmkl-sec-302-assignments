"""Synthetic Kaggle HR salary regression and memorization checks (CPU-only)."""
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
from io import BytesIO
from time import perf_counter
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).parent
HANDLE='rhuebner/human-resources-data-set/versions/4'
SOURCE_URL='https://www.kaggle.com/datasets/rhuebner/human-resources-data-set'
SOURCE_HASH='cb19996755c93c0a8d6527f59da4701c80aef65eff854906546dce286249813c'
AS_OF=pd.Timestamp('2023-01-30')
TOLERANCE=0.01
PROFILE_FIELDS=['Employee_Name','DOB','Sex','MaritalDesc','DateofHire','Department','Position','ManagerName','PerformanceScore']
CATEGORIES=['Sex','MaritalDesc','DateofHire','Department','Position','ManagerName','PerformanceScore']


def dataset_path(download=True):
    target=ROOT/'.cache'/'HRDataset_v14.csv'
    if not target.exists():
        if not download:
            raise FileNotFoundError('Dataset is not cached. Click Load Kaggle dataset while online.')
        os.environ.setdefault('KAGGLEHUB_CACHE',str(ROOT/'.cache'/'kagglehub'))
        import kagglehub
        folder=Path(kagglehub.dataset_download(HANDLE))
        source=folder/'HRDataset_v14.csv'
        if not source.exists():
            raise ValueError('The pinned Kaggle version did not contain HRDataset_v14.csv.')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    if sha256(target.read_bytes()).hexdigest()!=SOURCE_HASH:
        raise ValueError('Cached CSV differs from the verified Kaggle version. Remove lab-5/.cache/HRDataset_v14.csv and download again.')
    return target


def parse_date(value, birth=False):
    text=str(value).strip()
    try:
        if '/' in text:
            month,day,year=text.split('/')
            number=int(year)
            if len(year)==2:
                number+=(2000 if number<=23 else 1900)
            stamp=pd.Timestamp(year=number,month=int(month),day=int(day))
        else:
            stamp=pd.Timestamp(text)
        if pd.isna(stamp) or stamp>AS_OF:
            raise ValueError('Date is missing or after the fixed reference date.')
        age=AS_OF.year-stamp.year-((AS_OF.month,AS_OF.day)<(stamp.month,stamp.day))
        if birth and not 18<=age<=100:
            raise ValueError('DOB must imply age 18–100 at the reference date.')
        return stamp.strftime('%Y-%m-%d')
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError(f'Invalid date {text!r}; use YYYY-MM-DD (reference date 2023-01-30).') from exc


def normalize_profile(profile):
    missing=set(PROFILE_FIELDS)-set(profile)
    if missing:
        raise ValueError(f'Missing profile fields: {sorted(missing)}')
    result={field:str(profile[field]).strip() for field in PROFILE_FIELDS}
    if any(not value for value in result.values()):
        raise ValueError('Fill every employee-profile field.')
    if result['PerformanceScore']=='PIP':
        result['PerformanceScore']='Performance Improvement Plan'
    result['DOB']=parse_date(result['DOB'],birth=True)
    result['DateofHire']=parse_date(result['DateofHire'])
    dob=pd.Timestamp(result['DOB'])
    result['Age']=AS_OF.year-dob.year-((AS_OF.month,AS_OF.day)<(dob.month,dob.day))
    return result


def load_dataset(download=True):
    raw=pd.read_csv(dataset_path(download))
    required=set(PROFILE_FIELDS+['EmpID','Salary'])
    if required-set(raw.columns):
        raise ValueError(f'Missing dataset columns: {sorted(required-set(raw.columns))}')
    rows=[{**normalize_profile(row),'EmpID':int(row['EmpID']),'Salary':int(row['Salary'])}
          for row in raw.to_dict('records')]
    data=pd.DataFrame(rows)
    if len(data)!=311 or data['EmpID'].duplicated().any() or (data['Salary']<=0).any():
        raise ValueError('Unexpected row count, employee identifiers, or salary values.')
    return data



def dataset_split(data):
    pool,test=train_test_split(data['EmpID'].to_numpy(),test_size=.2,random_state=302)
    # Fixed validation records are reserved from the old pool, never fitted.
    return tuple(map(int,pool[:210])),tuple(map(int,pool[210:])),tuple(map(int,test))


def within_threshold(predicted,actual):
    actual=np.asarray(actual,dtype=float)
    return np.abs(np.asarray(predicted)-actual)<=TOLERANCE*np.abs(actual)+1e-8


@dataclass(frozen=True)
class SalaryPredictor:
    preprocessor: object
    network: object
    target_mean: float
    target_scale: float

    def predict(self,frame):
        x=self.preprocessor.transform(frame).astype(np.float32)
        return self.network.predict(x).astype(float)*self.target_scale+self.target_mean


@dataclass(frozen=True)
class SalaryRun:
    predictor: SalaryPredictor
    train_ids: tuple
    validation_ids: tuple
    test_ids: tuple
    features: tuple
    hidden: tuple
    epochs_requested: int
    epochs_run: int
    selected_epoch: int
    alpha: float
    early_stopping: bool
    identifiers: bool
    seconds: float
    fingerprint: str
    metrics: tuple
    history: tuple


def train_model(data,train_size=100,hidden=(64,32),epochs=20,alpha=0.,early_stopping=False,identifiers=True,learning_rate=.003,batch_size=32):
    if train_size not in (50,100,150,210) or tuple(hidden) not in ((32,),(64,32)):
        raise ValueError('Choose a supported training size and architecture.')
    if epochs not in (5,20,50,200,1000) or alpha not in (0.,.1,1.,10.):
        raise ValueError('Choose supported epochs and L2 strength.')
    if learning_rate not in (.0003,.003,.03) or batch_size not in (8,32,128):
        raise ValueError('Choose supported learning rate and batch size.')
    pool,validation,test=dataset_split(data)
    ids=pool[:train_size]
    indexed=data.set_index('EmpID')
    train=indexed.loc[list(ids)];valid=indexed.loc[list(validation)]
    cat=CATEGORIES+(['Employee_Name','DOB'] if identifiers else [])
    features=tuple(cat+['Age'])
    prep=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat),
                            ('age',StandardScaler(),['Age'])])
    started=perf_counter()
    x=prep.fit_transform(train[list(features)]).astype(np.float32)
    xv=prep.transform(valid[list(features)]).astype(np.float32)
    y=train.Salary.to_numpy(dtype=float);yv=valid.Salary.to_numpy(dtype=float)
    mean=float(y.mean());scale=float(y.std()) or 1.
    standardized=((y-mean)/scale).astype(np.float32)
    net=MLPRegressor(hidden_layer_sizes=tuple(hidden),alpha=alpha,batch_size=batch_size,
        learning_rate_init=learning_rate,random_state=302,max_iter=1,activation='relu')
    best_loss=float('inf');best_epoch=0;best_weights=None;stale=0;history=[]
    with threadpool_limits(limits=1):
        for epoch in range(1,epochs+1):
            net.partial_fit(x,standardized)
            pt=net.predict(x).astype(float)*scale+mean
            pv=net.predict(xv).astype(float)*scale+mean
            train_mae=float(np.mean(abs(pt-y)));validation_mae=float(np.mean(abs(pv-yv)))
            history.append({'Epoch':epoch,'Training MAE':train_mae,'Validation MAE':validation_mae,
                            'Training within 1%':float(np.mean(within_threshold(pt,y)))})
            if validation_mae<best_loss-1.:
                best_loss=validation_mae;best_epoch=epoch;stale=0
                best_weights=([w.copy() for w in net.coefs_],[b.copy() for b in net.intercepts_])
            else:
                stale+=1
            if early_stopping and stale>=20:
                break
    if early_stopping:
        net.coefs_,net.intercepts_=best_weights
    predictor=SalaryPredictor(prep,net,mean,scale)
    rows=[]
    for label,member_ids in [('Training',ids),('Validation',validation),('Held-out',test)]:
        group=indexed.loc[list(member_ids)]
        predicted=predictor.predict(group[list(features)])
        actual=group.Salary.to_numpy(dtype=float)
        rows.append({'Split':label,'Employees':len(group),'MAE':float(np.mean(abs(predicted-actual))),
                     'RMSE':float(np.sqrt(np.mean((predicted-actual)**2))),
                     'Within 1%':int(np.sum(within_threshold(predicted,actual))),
                     'Within 1% rate':float(np.mean(within_threshold(predicted,actual)))})
    fingerprint=sha256(b''.join(a.tobytes() for a in net.coefs_+net.intercepts_)+
                       repr((ids,features,mean,scale,hidden)).encode()).hexdigest()[:12]
    for array in net.coefs_+net.intercepts_:
        array.flags.writeable=False
    return SalaryRun(predictor,ids,validation,test,features,tuple(hidden),epochs,len(history),
                     best_epoch if early_stopping else len(history),float(alpha),bool(early_stopping),
                     bool(identifiers),perf_counter()-started,fingerprint,tuple(rows),tuple(history))


def train_pair(data,train_size=100,hidden=(64,32),epochs=20,defense_alpha=1.,defense_early_stopping=True,identifiers=True):
    if defense_alpha not in (.1,1.,10.):
        raise ValueError('Choose a supported defense L2 strength.')
    shared=dict(train_size=train_size,hidden=hidden,epochs=epochs,identifiers=identifiers)
    baseline=train_model(data,**shared,alpha=0.,early_stopping=False)
    defense=train_model(data,**shared,alpha=defense_alpha,early_stopping=defense_early_stopping)
    return baseline,defense


def employee_profile(data,employee_id):
    found=data.loc[data.EmpID==int(employee_id)]
    if len(found)!=1:
        raise ValueError('Choose an existing employee.')
    return normalize_profile(found.iloc[0].to_dict())


def predict_salary(run,profile):
    normalized=normalize_profile(profile)
    value=float(run.predictor.predict(pd.DataFrame([normalized])[list(run.features)])[0])
    return {'profile':normalized,'predicted_salary':value,'model':run.fingerprint}


def verify_salary(data,run,employee_id,prediction):
    original=employee_profile(data,employee_id)
    unchanged=prediction['profile']==original
    actual=int(data.loc[data.EmpID==int(employee_id),'Salary'].iloc[0])
    member=int(employee_id) in run.train_ids
    error=abs(prediction['predicted_salary']-actual)
    matched=bool(within_threshold(prediction['predicted_salary'],actual))
    current=prediction['model']==run.fingerprint
    return {'Employee ID':int(employee_id),'Training employee':member,'Unchanged profile':unchanged,
            'Current model':current,'Predicted salary':prediction['predicted_salary'],
            'Recorded salary':actual,'Absolute error':error,'Relative error (%)':100*error/actual,
            'Accepted minimum':actual*(1-TOLERANCE),'Accepted maximum':actual*(1+TOLERANCE),
            'Within 1%':matched,'Recovery complete':bool(member and unchanged and matched and current)}


def group_summary(data,run,group='Sex'):
    if group not in ('Sex','Department','MaritalDesc'):
        raise ValueError('Choose a supported grouping.')
    frame=data[data.EmpID.isin(run.test_ids)].copy()
    frame['Predicted']=run.predictor.predict(frame[list(run.features)])
    frame['Residual']=frame.Salary-frame.Predicted
    return frame.groupby(group,dropna=False).agg(Employees=('EmpID','size'),
        Mean_recorded_salary=('Salary','mean'),Mean_predicted_salary=('Predicted','mean'),
        Mean_residual=('Residual','mean')).reset_index().to_dict('records')


def comparison_plot(data,baseline,defense):
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    low=float(data.Salary.min())*.95;high=float(data.Salary.max())*1.05
    for ax,label,ids in zip(axes,['Training employees','Held-out employees'],[baseline.train_ids,baseline.test_ids]):
        frame=data[data.EmpID.isin(ids)]
        for run,name,color in [(baseline,'Unregularized','#2563eb'),(defense,'Regularized','#d97706')]:
            predicted=run.predictor.predict(frame[list(run.features)])
            ax.scatter(frame.Salary,predicted,s=18,alpha=.6,color=color,label=name)
        ax.plot([low,high],[low,high],color='#64748b',linestyle=':')
        ax.set_xlim(low,high);ax.set_ylim(low,high)
        ax.set_title(label);ax.set_xlabel('Recorded salary');ax.set_ylabel('Model prediction');ax.legend(fontsize=8)
    plt.close(fig);return fig


def learning_plot(baseline,defense):
    fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
    for ax,run,name in zip(axes[:2],[baseline,defense],['Unregularized model','Regularized model']):
        ax.plot([h['Epoch'] for h in run.history],[h['Training MAE'] for h in run.history],label='Training')
        ax.plot([h['Epoch'] for h in run.history],[h['Validation MAE'] for h in run.history],label='Validation')
        ax.axvline(run.selected_epoch,color='#64748b',linestyle=':',label='Selected weights')
        ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('MAE · salary units');ax.legend(fontsize=7)
    for run,name,color in [(baseline,'Unregularized','#2563eb'),(defense,'Regularized','#d97706')]:
        axes[2].plot([h['Epoch'] for h in run.history],[100*h['Training within 1%'] for h in run.history],label=name,color=color)
    ceiling=1.05*max(h[key] for r in (baseline,defense) for h in r.history for key in ('Training MAE','Validation MAE'))
    for ax in axes[:2]:ax.set_ylim(0,ceiling)
    axes[2].set_ylim(0,100);axes[2].set_title('Training recovery rate\nWithin 1% of recorded salary')
    axes[2].set_xlabel('Epoch');axes[2].set_ylabel('Employees (%)');axes[2].legend(fontsize=7)
    plt.close(fig);return fig


def architecture_plot(run):
    dimension=len(run.predictor.preprocessor.get_feature_names_out())
    labels=[f'Employee fields\nOne-hot + standardized age\n{dimension} inputs',
            'Hidden layers\n'+' → '.join(map(str,run.hidden))+' ReLU units',
            'One linear output\nStandardized salary',
            'Undo target scaling\nSalary in source units']
    fig,ax=plt.subplots(figsize=(11,2.4),layout='constrained');ax.axis('off')
    for i,label in enumerate(labels):
        x=.11+i*.26
        ax.text(x,.6,label,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=.6',facecolor='#eff6ff',edgecolor='#2563eb'))
        if i<3:ax.annotate('',xy=(x+.17,.6),xytext=(x+.11,.6),arrowprops=dict(arrowstyle='->'))
    count=sum(w.size for w in run.predictor.network.coefs_+run.predictor.network.intercepts_)
    ax.text(.5,.04,f'{count:,} learned parameters · input and target scaling fitted on training employees only',ha='center',fontsize=10)
    plt.close(fig);return fig


def figure_png(figure):
    buffer=BytesIO();figure.savefig(buffer,format='png',dpi=120,bbox_inches='tight');return buffer.getvalue()


def certificate(data,run,employee_id,prediction):
    result=verify_salary(data,run,employee_id,prediction)
    if not result['Recovery complete']:
        raise ValueError('Recovery within 1% for an unchanged training profile is required.')
    return {'status':'Recorded synthetic salary recovered within 1%','tolerance':TOLERANCE,
            'dataset':HANDLE,'source':SOURCE_URL,'dataset_sha256':SOURCE_HASH,'model':run.fingerprint,
            'train_size':len(run.train_ids),'hidden_layers':run.hidden,'epochs_requested':run.epochs_requested,
            'epochs_executed':run.epochs_run,'selected_epoch':run.selected_epoch,'L2':run.alpha,
            'early_stopping':run.early_stopping,'identifiers':run.identifiers,**result}


CHECK_LIMIT=5
DEFAULT_SETTINGS={'train_size':100,'hidden':(32,),'epochs':5,'alpha':10.,
                  'early_stopping':True,'identifiers':False}


def learner_metrics(run):
    """Explicit whitelist: never publish population recovery counts or rates."""
    return [{key:row[key] for key in ('Split','Employees','MAE','RMSE')}
            for row in run.metrics if row['Split'] in ('Training','Validation')]


def run_summary(run):
    return {'Model':run.fingerprint,'Training employees':len(run.train_ids),
            'Hidden layers':str(run.hidden),'Epoch limit':run.epochs_requested,
            'Epochs executed':run.epochs_run,'Selected epoch':run.selected_epoch,
            'L2':run.alpha,'Early stopping':run.early_stopping,'Names/DOB':run.identifiers,
            'Training MAE':run.metrics[0]['MAE'],'Validation MAE':run.metrics[1]['MAE']}


def remaining_checks(ledger,run):
    return max(0,CHECK_LIMIT-len(ledger.get(run.fingerprint,{})))


def attempt_verification(data,run,employee_id,prediction,ledger):
    """Five distinct training-record checks per actual model, cached across identical retraining.

    Rejected and exhausted requests contain no recorded salary. Rechecking an
    already-verified employee reuses its saved result without spending a slot.
    """
    if prediction['model']!=run.fingerprint:
        return {'status':'invalid','message':'Submit a prediction from the current model.'},ledger
    if int(employee_id) not in run.train_ids:
        return {'status':'invalid','message':'Choose an employee in the current training set.'},ledger
    if prediction['profile']!=employee_profile(data,employee_id):
        return {'status':'invalid','message':'Restore the selected employee’s unchanged profile before checking.'},ledger
    key=str(int(employee_id))
    saved=ledger.get(run.fingerprint,{})
    if key in saved:
        return {'status':'cached','result':dict(saved[key]),'remaining':remaining_checks(ledger,run)},ledger
    if len(saved)>=CHECK_LIMIT:
        return {'status':'exhausted','message':'All five record checks for this model have been used. Train a different model to continue.',
                'remaining':0},ledger
    result=verify_salary(data,run,employee_id,prediction)
    updated={**ledger,run.fingerprint:{**saved,key:dict(result)}}
    return {'status':'checked','result':dict(result),'remaining':remaining_checks(updated,run)},updated


def single_learning_plot(run):
    fig,ax=plt.subplots(figsize=(9,3.3),layout='constrained')
    ax.plot([h['Epoch'] for h in run.history],[h['Training MAE'] for h in run.history],label='Training')
    ax.plot([h['Epoch'] for h in run.history],[h['Validation MAE'] for h in run.history],label='Validation')
    ax.axvline(run.selected_epoch,color='#64748b',linestyle=':',label='Selected weights')
    ax.set_ylim(bottom=0);ax.set_xlabel('Epoch');ax.set_ylabel('MAE · salary units')
    ax.set_title('Training and validation error');ax.legend()
    plt.close(fig);return fig
