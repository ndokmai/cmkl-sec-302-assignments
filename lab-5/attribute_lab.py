"""Fixed-cohort attribute-inference lab infrastructure (student search is separate)."""
from itertools import product
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from sklearn.metrics import confusion_matrix
from experiment import train_model

TARGET_ID=10255
START_EPOCHS=5
DEFENSE_TARGET=.40
STARTER='''# Possible values, not the employee's actual answers.
performance_values = [
    "Exceeds",
    "Fully Meets",
    "Needs Improvement",
    "Performance Improvement Plan",
]
marital_values = [
    "Divorced",
    "Married",
    "Separated",
    "Single",
    "Widowed",
]

# profile: dict of known employee fields; Age is int, other values are str.
# known_salary: float. The employee's observed salary.
# model.predict(full_profile: dict) -> float: a salary estimate.
# Return one performance-score string, not marital status.

def extract_performance_score(profile, known_salary, model):
    # This runnable example makes a fixed guess; it is incorrect for Myriam.
    full_profile = {
        **profile,
        "PerformanceScore": "Exceeds",
        "MaritalDesc": "Divorced",
    }
    prediction = model.predict(full_profile)
    return "Exceeds"
'''




def fit(data,epochs=START_EPOCHS,alpha=0.,early_stopping=False,identifiers=True,learning_rate=.003,batch_size=32):
    return train_model(data,train_size=210,hidden=(64,32),epochs=epochs,
                       alpha=alpha,early_stopping=early_stopping,identifiers=identifiers,learning_rate=learning_rate,batch_size=batch_size)


def compile_search(source):
    # Deliberately ordinary local Python, not a sandbox. No model/data/truth in namespace.
    namespace={}
    exec(compile(source,'<student search>','exec'),namespace)
    function=namespace.get('extract_performance_score')
    if not callable(function):
        raise ValueError('Define a function named extract_performance_score with the three supplied arguments.')
    return function


def public_profiles(data,ids):
    return data.set_index('EmpID').loc[list(ids)].drop(columns=['MaritalDesc','PerformanceScore']).reset_index()


def candidate_predictions(data,run,ids):
    training=data.set_index('EmpID').loc[list(run.train_ids)]
    ratings=tuple(sorted(training.PerformanceScore.unique()))
    statuses=tuple(sorted(training.MaritalDesc.unique()))
    public=public_profiles(data,ids)
    frames=[]
    pairs=tuple(product(ratings,statuses))
    for rating,status in pairs:
        frame=public.copy()
        frame['PerformanceScore']=rating
        frame['MaritalDesc']=status
        frames.append(frame)
    with threadpool_limits(limits=1):
        values=run.predictor.predict(pd.concat(frames,ignore_index=True)).reshape(len(pairs),len(ids))
    if not np.isfinite(values).all():
        raise ValueError('Model returned nonfinite candidate predictions.')
    return public,ratings,statuses,pairs,values


def evaluate(data,run,search,ids=None):
    """Pass only known salary, domains and a fixed-employee prediction oracle to search.

    All 20 genuine model outputs are batched up front for CPU efficiency. The
    student's oracle reads these outputs; it never reads either true hidden value.
    """
    started=perf_counter()
    ids=tuple(run.train_ids if ids is None else ids)
    if not ids or not set(ids)<=set(run.train_ids):
        raise ValueError('Evaluate a nonempty subset of this fixed training cohort.')
    public,ratings,statuses,pairs,values=candidate_predictions(data,run,ids)
    pair_index={pair:i for i,pair in enumerate(pairs)}
    predictions=[]
    query_counts=[]
    for column,employee_id in enumerate(ids):
        calls=[]
        known=public.iloc[column].drop(labels=['EmpID','Salary']).to_dict()
        class Model:
            def predict(self,full_profile):
                if not isinstance(full_profile,dict):
                    raise ValueError('model.predict expects a full employee profile dictionary.')
                if set(full_profile)!=set(known)|{'PerformanceScore','MaritalDesc'}:
                    raise ValueError('Include the known profile fields plus PerformanceScore and MaritalDesc.')
                if any(full_profile[key]!=value for key,value in known.items()):
                    raise ValueError('Keep the known employee fields unchanged.')
                pair=(full_profile['PerformanceScore'],full_profile['MaritalDesc'])
                if pair not in pair_index:
                    raise ValueError('Use the supplied possible performance and marital values.')
                calls.append(pair)
                return float(values[pair_index[pair],column])
        salary=float(public.iloc[column].Salary)
        answer=search(dict(known),salary,Model())
        if not isinstance(answer,str) or answer not in ratings:
            raise ValueError('Return one valid performance-score string.')
        if not any(pair[0]==answer for pair in calls):
            raise ValueError('Query the performance score you return; do not return an untested guess.')
        predictions.append({'EmpID':int(employee_id),'Guessed performance':answer,'Known salary':salary})
        query_counts.append(len(calls))
    # Ground truth is introduced only after the student's predictions are complete.
    table=pd.DataFrame(predictions)
    truth=data.set_index('EmpID').loc[list(ids),'PerformanceScore'].to_numpy()
    table['Actual performance']=truth
    table['Correct']=table['Guessed performance'].to_numpy()==truth
    training=data.set_index('EmpID').loc[list(run.train_ids)]
    majority=training.PerformanceScore.value_counts().idxmax()
    matrix=confusion_matrix(truth,table['Guessed performance'],labels=ratings)
    present=[rating for rating in ratings if np.any(truth==rating)]
    balanced=float(np.mean([np.mean(table.loc[truth==rating,'Guessed performance']==rating) for rating in present]))
    return {'model':run.fingerprint,'epochs':run.epochs_requested,'alpha':run.alpha,
            'settings':{'epochs':run.epochs_requested,'L2':run.alpha,'early_stopping':run.early_stopping,
                        'identifiers':run.identifiers,'learning_rate':run.predictor.network.learning_rate_init,
                        'batch_size':run.predictor.network.batch_size},
            'n':len(ids),'accuracy':float(table.Correct.mean()),'balanced_accuracy':balanced,
            'majority_label':majority,'majority_accuracy':float(np.mean(truth==majority)),
            'max_queries':max(query_counts),'seconds':perf_counter()-started,
            'ratings':list(ratings),'confusion':matrix.tolist(),'records':table.to_dict('records'),
            'salary_metrics':[dict(row) for row in run.metrics]}


def candidate_plot(data,run,employee_id):
    public,ratings,statuses,pairs,values=candidate_predictions(data,run,(employee_id,))
    errors=abs(values[:,0]-float(public.iloc[0].Salary)).reshape(len(ratings),len(statuses))
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    im=ax.imshow(errors,cmap='viridis_r',aspect='auto')
    ax.set_xticks(range(len(statuses)),statuses)
    ax.set_yticks(range(len(ratings)),ratings)
    ax.set_xlabel('Candidate marital status');ax.set_ylabel('Candidate performance rating')
    ax.set_title('Candidate salary error · lower is a closer match')
    for i in range(len(ratings)):
        for j in range(len(statuses)):
            ax.text(j,i,f'{errors[i,j]:,.0f}',ha='center',va='center',fontsize=8,
                    color='black' if errors[i,j]<errors.max()/2 else 'white')
    fig.colorbar(im,ax=ax,label='Absolute salary error')
    plt.close(fig)
    return fig


def summary(result):
    return {'Employees':result['n'],
            'Attack accuracy (%)':round(result['accuracy']*100,2),
            'Average success across rating categories (%)':round(result['balanced_accuracy']*100,2),
            'Training salary MAE':result['salary_metrics'][0]['MAE'],
            'Validation salary MAE':result['salary_metrics'][1]['MAE'],
            'Held-out salary MAE':result['salary_metrics'][2]['MAE']}


def model_overview_plot():
    """Detailed schematic with illustrative candidate values, not a target answer."""
    from matplotlib.patches import FancyBboxPatch, Circle
    fig,ax=plt.subplots(figsize=(15,6.7))
    fig.patch.set_facecolor('#f5f7fc')
    ax.set(xlim=(0,15),ylim=(1.25,8))
    ax.axis('off')
    navy='#182744';blue='#3563e9';muted='#60708a';orange='#b65d09'
    def text(x,y,label,size=11,color=navy,**kwargs):
        ax.text(x,y,label,fontsize=size,color=color,va='center',parse_math=False,**kwargs)
    def panel(x,y,w,h,color='white',edge='#dce3f0'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.14',
                                  linewidth=1,facecolor=color,edgecolor=edge))
    def arrow(x1,y1,x2,y2):
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->',color=blue,lw=1.8))
    text(.25,7.65,'Inside the HR salary predictor',21,weight='bold')
    text(.25,7.23,'One employee profile enters. One salary estimate comes out.',12,color=muted)
    panel(.25,1.8,4.45,4.95)
    text(.48,6.45,'01  EMPLOYEE INPUTS',12,weight='bold')
    rows=[('Name','Jordan Lee'),('Date of birth','1988-05-17'),('Age','34'),
          ('Gender','F'),('Hire date','2015-09-01'),('Department','IT/IS'),
          ('Position','IT Support'),('Manager','Morgan Chen'),
          ('Marital status','Divorced  [candidate]'),('Performance','Exceeds  [candidate]')]
    for i,(label,value) in enumerate(rows):
        y=5.94-i*.37
        if i>=8:
            panel(.48,y-.15,3.97,.29,color='#fff3e2',edge='#fff3e2')
        text(.58,y,label,10,color=orange if i>=8 else muted)
        text(2.05,y,value,10,color=orange if i>=8 else navy,weight='bold' if i>=8 else 'normal')
    text(.58,2.05,'Orange fields are confidential and unknown to you.',9,color=orange)
    panel(5.12,3.08,1.65,2.38,color='#edf2ff')
    text(5.94,5.12,'02  ENCODE',11,ha='center',weight='bold')
    text(5.94,4.35,'Categories →\nindicator columns',10,ha='center',linespacing=1.6)
    text(5.94,3.55,'Age →\nscaled number',10,ha='center',linespacing=1.6)
    arrow(4.88,4.27,5.02,4.27)
    xs=[7.9,9.9];ys=[[2.65+i*.49 for i in range(7)],[3.14+i*.49 for i in range(5)]]
    text(8.9,6.45,'03  NEURAL NETWORK',12,ha='center',weight='bold')
    for y in ys[0]:
        ax.plot([6.95,7.72],[4.27,y],color='#b8c9f5',lw=.8,zorder=1)
    for y1 in ys[0]:
        for y2 in ys[1]:
            ax.plot([8.08,9.72],[y1,y2],color='#b8c9f5',lw=.8,zorder=1)
    for x,positions in zip(xs,ys):
        for y in positions:
            ax.add_patch(Circle((x,y),.16,facecolor='#e7edff',edgecolor=blue,lw=1.6,zorder=2))
        text(x,5.94,'64 neurons' if x==xs[0] else '32 neurons',11,ha='center',weight='bold')
        text(x,2.08,'Hidden layer 1' if x==xs[0] else 'Hidden layer 2',10,ha='center')
    text(8.9,1.69,'Weighted sums + ReLU activations',10,ha='center',color=muted)
    for y in ys[1]:
        ax.plot([10.08,11.18],[y,4.27],color='#b8c9f5',lw=.8,zorder=1)
    ax.add_patch(Circle((11.35,4.27),.17,facecolor=blue,edgecolor=blue,zorder=2))
    arrow(11.55,4.27,12.02,4.27)
    panel(12.2,3.12,2.45,2.31,color='#182744',edge='#182744')
    text(13.42,5.05,'04  OUTPUT',11,color='#c5d5ff',ha='center',weight='bold')
    text(13.42,4.38,'$64,200',23,color='white',ha='center',weight='bold')
    text(13.42,3.67,'Predicted salary',11,color='white',ha='center')
    text(12.9,2.67,'Linear output + undo salary scaling',9,ha='center',color=muted)
    text(14.6,7.26,'Illustrative values · not a dataset record',9,color=muted,ha='right')
    text(8.9,1.35,'Connections and neurons are schematic; full layers contain 64 and 32 neurons.',8,color=muted,ha='center')
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
    plt.close(fig)
    return fig


def check_manual(data,run,performance,marital):
    public,ratings,statuses,pairs,values=candidate_predictions(data,run,(TARGET_ID,))
    pair=(performance,marital)
    if pair not in pairs:
        raise ValueError('Select a performance score and marital status.')
    predicted=float(values[pairs.index(pair),0])
    actual=data.set_index('EmpID').loc[TARGET_ID]
    rating_correct=performance==actual.PerformanceScore
    marital_correct=marital==actual.MaritalDesc
    return {'performance':performance,'marital':marital,'predicted_salary':predicted,
            'known_salary':float(public.iloc[0].Salary),
            'salary_error':abs(predicted-float(public.iloc[0].Salary)),
            'is_smallest_difference':bool(np.isclose(abs(predicted-float(public.iloc[0].Salary)),
                np.min(abs(values[:,0]-float(public.iloc[0].Salary))),rtol=0,atol=1e-8)),
            'rating_correct':bool(rating_correct),'combination_correct':bool(rating_correct and marital_correct)}


def prediction_history_html(history):
    """Show submitted attempts only; highlight all ties for smallest raw error."""
    from html import escape
    if not history:
        return '<p>No predictions yet.</p>'
    best=min(row['salary_error'] for row in history)
    headings=['Attempt','Performance score','Marital status','Predicted salary','Difference','']
    parts=['<table style="width:100%;border-collapse:collapse;text-align:left"><thead><tr>']
    parts.extend(f'<th style="padding:10px;border-bottom:1px solid #cbd5e1">{label}</th>' for label in headings)
    parts.append('</tr></thead><tbody>')
    for index,row in enumerate(history,1):
        is_best=bool(np.isclose(row['salary_error'],best,rtol=0,atol=1e-8))
        style='background:#dcfce7;color:#14532d;font-weight:600;' if is_best else ''
        parts.append(f'<tr style="{style}">')
        cells=[str(index),escape(row['performance']),escape(row['marital']),
               f"{row['predicted_salary']:,.2f}",f"{row['salary_error']:,.2f}",
               '★ Smallest difference so far' if is_best else '']
        parts.extend(f'<td style="padding:10px;border-bottom:1px solid #e2e8f0">{cell}</td>' for cell in cells)
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return ''.join(parts)


def confusion_plot(result):
    matrix=np.asarray(result['confusion'])
    labels=[label.replace('Performance Improvement Plan','Performance\nImprovement Plan').replace('Needs Improvement','Needs\nImprovement')
            for label in result['ratings']]
    fig,ax=plt.subplots(figsize=(5.6,4.6),layout='constrained')
    im=ax.imshow(matrix,cmap='Blues',vmin=0)
    ax.set_xticks(range(len(labels)),labels,fontsize=10)
    ax.set_yticks(range(len(labels)),labels,fontsize=10)
    ax.set_xlabel('Predicted performance score',labelpad=12)
    ax.set_ylabel('Actual performance score',labelpad=12)
    ax.set_title('Confusion matrix · employee counts',pad=15)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column,row,str(matrix[row,column]),ha='center',va='center',fontsize=13,
                    color='white' if matrix[row,column]>matrix.max()/2 else '#182744')
    fig.colorbar(im,ax=ax,label='Employees',shrink=.8)
    plt.close(fig)
    return fig


def submission_card(manual,starting,retrained,defense):
    """Single results image for copying or downloading; no implementation details."""
    results=[starting,retrained,defense]
    labels=['Starting model','Retrained model','Defense model']
    checks=defense_checks(starting,defense)
    complete=manual['combination_correct'] and retrained['accuracy']==1 and checks['passed']
    fig=plt.figure(figsize=(12,9.5),facecolor='#f5f7fc')
    grid=fig.add_gridspec(5,3,height_ratios=[.65,.85,1.1,.7,2.9],hspace=.5,wspace=.55)
    title=fig.add_subplot(grid[0,:]);title.axis('off')
    title.text(0,.7,'Lab 5 · Infer a Confidential Performance Rating',fontsize=19,weight='bold',color='#182744')
    title.text(0,.05,'Submission card · 210 training employees · synthetic HR data',fontsize=11,color='#60708a')
    title.text(1,.05,'GOALS ACHIEVED' if complete else 'GOALS NOT YET MET',ha='right',fontsize=11,
               weight='bold',color='#15803d' if complete else '#b45309')
    target=fig.add_subplot(grid[1,:]);target.axis('off')
    target.text(0,.85,'PART 1 · MANUAL INFERENCE',fontsize=11,weight='bold',color='#182744')
    target.text(0,.42,f"Myriam Givens: {manual['performance']} / {manual['marital']} · "
                +('Correct' if manual['combination_correct'] else 'Incorrect'),fontsize=11)
    target.text(0,.03,f"Predicted salary: {manual['predicted_salary']:,.2f}   |   "
                f"Known salary: {manual['known_salary']:,.0f}   |   Difference: {manual['salary_error']:,.2f}",fontsize=10,color='#60708a')
    guessed=next(row for row in starting['records'] if row['EmpID']==TARGET_ID)
    target.text(.63,.85,'PART 2 · AUTOMATED TARGET',fontsize=11,weight='bold',color='#182744')
    target.text(.63,.42,f"{guessed['Guessed performance']} · "+('Correct' if guessed['Correct'] else 'Incorrect'),fontsize=11)
    table_ax=fig.add_subplot(grid[2,:]);table_ax.axis('off')
    rows=[]
    for label,result in zip(labels,results):
        correct=sum(row['Correct'] for row in result['records'])
        rows.append([label,f"{correct} / {result['n']}",f"{result['accuracy']:.1%}",
                     f"{result['salary_metrics'][0]['MAE']:,.2f}",f"{result['salary_metrics'][1]['MAE']:,.2f}"])
    table=table_ax.table(cellText=rows,colLabels=['Experiment','Correct scores','Attack success','Training salary error','Validation salary error'],
                         cellLoc='center',colWidths=[.19,.16,.15,.25,.25],bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,column),cell in table.get_celld().items():
        cell.set_edgecolor('#dce3f0')
        cell.set_facecolor('#e8edf8' if row==0 else 'white')
        if row==0:cell.set_text_props(weight='bold',color='#182744')
    status=fig.add_subplot(grid[3,:]);status.axis('off')
    status.text(0,.8,'DEFENSE REQUIREMENTS',fontsize=11,weight='bold',color='#182744')
    status.text(0,.22,'Recovery goal: '+('PASSED' if checks['recovery_passed'] else 'NOT MET')+
                ' (attack success below 40%)',fontsize=11,
                color='#15803d' if checks['recovery_passed'] else '#b45309')
    status.text(.53,.22,'Prediction quality: '+('PASSED' if checks['quality_passed'] else 'NOT MET')+
                f" (limit {checks['validation_limit']:,.2f})",fontsize=11,
                color='#15803d' if checks['quality_passed'] else '#b45309')
    for index,(label,result) in enumerate(zip(labels,results)):
        ax=fig.add_subplot(grid[4,index])
        matrix=np.asarray(result['confusion'])
        names=[v.replace('Performance Improvement Plan','Performance\nImprovement Plan').replace('Needs Improvement','Needs\nImprovement') for v in result['ratings']]
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,int(matrix.max())) )
        ax.set_title(label,fontsize=12,weight='bold',pad=12)
        ax.set_xticks(range(len(names)),names,rotation=35,ha='right',fontsize=8)
        ax.set_yticks(range(len(names)),names,fontsize=8)
        ax.set_xlabel('Predicted rating',fontsize=9);ax.set_ylabel('Actual rating',fontsize=9)
        for row in range(len(names)):
            for column in range(len(names)):
                ax.text(column,row,str(matrix[row,column]),ha='center',va='center',fontsize=10,
                        color='white' if matrix[row,column]>matrix.max()/2 else '#182744')
    fig.text(.06,.025,'Confusion matrices show employee counts. Salary errors are mean absolute errors in salary units.',fontsize=9,color='#60708a')
    fig.subplots_adjust(left=.12,right=.97,top=.96,bottom=.16)
    plt.close(fig)
    return fig


def defense_checks(starting,defense):
    """Fixed starting-model utility reference; raw inclusive MAE, strict attack goal."""
    baseline=next(row['MAE'] for row in starting['salary_metrics'] if row['Split']=='Validation')
    observed=next(row['MAE'] for row in defense['salary_metrics'] if row['Split']=='Validation')
    limit=1.1*baseline
    recovery=bool(defense['accuracy']<.4)
    quality=bool(np.isfinite(observed) and observed<=limit)
    return {'recovery_passed':recovery,'quality_passed':quality,'passed':recovery and quality,
            'validation_mae':float(observed),'validation_limit':float(limit)}
