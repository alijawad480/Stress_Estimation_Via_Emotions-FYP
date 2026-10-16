# =============================================================================
# evaluate.py
# =============================================================================
import os,json,logging
import numpy as np,pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,f1_score
from config import RESULTS_DIR,MODEL_NAMES,FUSION_STRATEGIES,EMOTION_NAMES,EMOTION_PRESENCE_THRESHOLD,N_EMOTIONS
from train import load_model

log=logging.getLogger(__name__)

def compute_metrics(y_true,y_pred,names=EMOTION_NAMES):
    if y_pred.shape[1]!=len(names):
        raise ValueError(f"Model outputs {y_pred.shape[1]} emotions, expected {len(names)}.\n"
                         f"Delete old checkpoints and retrain with 6-emotion models.")
    metrics={}
    for i,name in enumerate(names):
        yt,yp=y_true[:,i],y_pred[:,i]
        mse=float(mean_squared_error(yt,yp)); mae=float(mean_absolute_error(yt,yp)); rmse=float(np.sqrt(mse))
        yb=(yt>=EMOTION_PRESENCE_THRESHOLD).astype(int); pb=(yp>=EMOTION_PRESENCE_THRESHOLD).astype(int)
        f1=float(f1_score(yb,pb,average="weighted",zero_division=0))
        metrics[name]={"MSE":mse,"MAE":mae,"RMSE":rmse,"F1":f1}
    metrics["avg"]={k:float(np.mean([metrics[n][k] for n in names])) for k in("MSE","MAE","RMSE","F1")}
    return metrics

def predict(model,test_ds,is_selfmm=False):
    all_true,all_pred=[],[]
    for inputs,labels in test_ds:
        pred=model(inputs,training=False)
        if isinstance(pred,dict): pred=pred["main"]
        all_pred.append(pred.numpy()); all_true.append(labels.numpy())
    y_true=np.vstack(all_true); y_pred=np.vstack(all_pred)
    if y_pred.shape[1]!=N_EMOTIONS:
        raise ValueError(f"Stale checkpoint: outputs {y_pred.shape[1]}, expected {N_EMOTIONS}. Delete & retrain.")
    return y_true,y_pred

def evaluate_one(model_name,fusion,test_ds):
    tag=f"{model_name}_{fusion}"
    try:
        model=load_model(model_name,fusion)
        y_true,y_pred=predict(model,test_ds,model_name=="SelfMM")
        metrics=compute_metrics(y_true,y_pred)
    except FileNotFoundError: log.warning("Skipping %s — no checkpoint",tag); return None
    except ValueError as e: print(f"⚠️  Skipping {tag}: {e}"); return None
    json.dump({"model":model_name,"fusion":fusion,"metrics":metrics},
              open(os.path.join(RESULTS_DIR,f"{tag}_metrics.json"),"w"),indent=2)
    m=metrics["avg"]
    log.info("[%s/%-12s] RMSE=%.4f F1=%.4f",model_name,fusion,m["RMSE"],m["F1"])
    return{"Model":model_name,"Fusion":fusion,
           "MSE":round(m["MSE"],4),"MAE":round(m["MAE"],4),"RMSE":round(m["RMSE"],4),"F1":round(m["F1"],4)}

def evaluate_all(test_ds):
    rows=[r for r in(evaluate_one(n,f,test_ds) for n in MODEL_NAMES for f in FUSION_STRATEGIES) if r is not None]
    if not rows:
        print("No valid results. Check checkpoints."); return pd.DataFrame(columns=["Model","Fusion","MSE","MAE","RMSE","F1"])
    df=pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR,"deep_models_metrics.csv"),index=False)
    print("\n"+df.to_string(index=False)); return df

def full_comparison(test_ds):
    df=evaluate_all(test_ds)
    if not df.empty: df.to_csv(os.path.join(RESULTS_DIR,"full_comparison.csv"),index=False)
    return df
