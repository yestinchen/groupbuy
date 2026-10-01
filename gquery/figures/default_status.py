"""Distribution panels for current default records with explicit denominators."""
import numpy as np
import pandas as pd


def distributions(df, exp_name, folder, prefix, arms):
    from gquery.figures import plot_exp4 as p
    counts=[]
    for method,_ in arms:
        s=df[df.method==method]
        counts.append(dict(method=method,n_requested=len(s),n_feasible=int(s.query_feasible.sum()),
                           n_metric=int(s.metric_eligible.sum()),n_timed=int(s.total_s.notna().sum()),
                           n_returned=int(s.success.sum()),n_completed=int(s.completed.fillna(False).astype(bool).sum()),
                           n_timeout=int(s.timed_out.sum()),n_guard=int((s.termination_reason=='combination_limit').sum())))
    directory=p.OUT_ROOT/exp_name/folder
    directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts).to_csv(directory/f'{prefix}_counts.csv',index=False)
    for suffix,col,ylabel,log in [('recall','recall','frontier recall',False),('hv','hv_ratio','hypervolume ratio',False),('time','total_s','retrieval + method (s)',True),('method_time','method_wall_s','method time (s)',True)]:
        data,labs,colors,frames=[],[],{},[]
        for method,label in arms:
            s=df[df.method==method]
            if not log:s=s[s.metric_eligible==1]
            s=s[s[col].notna()]
            if log:s=s[s[col]>0]
            if s.empty:continue
            data.append(s[col].values);labs.append(label);colors[label]=p.color(method)
            frames.append(s[['query_id','method','metric_eligible','solver_status','completed','termination_reason',col]].rename(columns={col:'value'}))
        if data:
            p._box_panel(exp_name,folder,f'{prefix}_{suffix}',data,labs,ylabel,colors,logy=log,
                         ylim=None if log else (-.03,1.05),df=pd.concat(frames,ignore_index=True),
                         tick_rotation=35 if len(labs)>4 else None)
