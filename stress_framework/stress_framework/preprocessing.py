# =============================================================================
# preprocessing.py
# =============================================================================
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle, os, logging
import config
from config import (DATA_PATH,TEXT_COL,AUDIO_COL,VISUAL_COL,TARGET_COLS,
                    TRAIN_RATIO,VAL_RATIO,TEST_RATIO,BATCH_SIZE,RANDOM_SEED,RESULTS_DIR)

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_parquet(path=DATA_PATH):
    log.info("Loading: %s",path)
    df = pd.read_parquet(path)
    for col in [TEXT_COL,AUDIO_COL,VISUAL_COL]:
        if col in df.columns:
            df[col]=df[col].apply(lambda x: np.array(x,dtype=np.float32)
                                  if not isinstance(x,np.ndarray) else x.astype(np.float32))
    config.TEXT_DIM   = int(df[TEXT_COL].iloc[0].shape[0])
    config.AUDIO_DIM  = int(df[AUDIO_COL].iloc[0].shape[0])
    config.VISUAL_DIM = int(df[VISUAL_COL].iloc[0].shape[0])
    log.info("Dims text:%d audio:%d visual:%d",config.TEXT_DIM,config.AUDIO_DIM,config.VISUAL_DIM)
    for c in TARGET_COLS:
        if c not in df.columns: df[c]=0.0
    df.dropna(subset=[c for c in TARGET_COLS if c in df.columns],inplace=True)
    log.info("Shape: %s",df.shape)
    return df

def clean_embeddings(df):
    for col,strategy in [(TEXT_COL,"std"),(AUDIO_COL,"std"),(VISUAL_COL,"pct")]:
        arr=np.vstack(df[col].values).astype(np.float64)
        arr[np.isinf(arr)]=np.nan
        med=np.nanmedian(arr,axis=0); r,c=np.where(np.isnan(arr)); arr[r,c]=med[c]
        if strategy=="pct":
            arr=np.clip(arr,np.percentile(arr,1,axis=0),np.percentile(arr,99,axis=0))
        else:
            m=arr.mean(0); s=arr.std(0); s[s==0]=1.0
            arr=np.clip(arr,m-3*s,m+3*s)
        df[col]=list(arr.astype(np.float32))
    for col in TARGET_COLS:
        s=pd.to_numeric(df[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
        df[col]=s.fillna(s.median()).clip(0,3).astype(np.float32)
    return df

class ModalityScaler:
    def __init__(self): self.scalers={k:StandardScaler() for k in("text","audio","visual")}; self._fitted=False
    def fit(self,t,a,v):
        self.scalers["text"].fit(t); self.scalers["audio"].fit(a); self.scalers["visual"].fit(v); self._fitted=True
    def _clip(self,x): return np.clip(x,-10,10).astype(np.float32)
    def transform(self,t,a,v):
        return(self._clip(self.scalers["text"].transform(t)),
               self._clip(self.scalers["audio"].transform(a)),
               self._clip(self.scalers["visual"].transform(v)))
    def transform_single(self,t,a,v):
        t=self._clip(self.scalers["text"].transform(t.reshape(1,-1)))[0]
        a=self._clip(self.scalers["audio"].transform(a.reshape(1,-1)))[0]
        v=self._clip(self.scalers["visual"].transform(v.reshape(1,-1)))[0]
        return t,a,v
    def save(self,path):
        with open(path,"wb") as f: pickle.dump(self.scalers,f)
    def load(self,path):
        with open(path,"rb") as f: self.scalers=pickle.load(f); self._fitted=True

def split_dataframe(df):
    idx=np.arange(len(df))
    tr,tmp=train_test_split(idx,test_size=1-TRAIN_RATIO,random_state=RANDOM_SEED)
    vf=VAL_RATIO/(VAL_RATIO+TEST_RATIO)
    va,te=train_test_split(tmp,test_size=1-vf,random_state=RANDOM_SEED)
    return df.iloc[tr].reset_index(drop=True),df.iloc[va].reset_index(drop=True),df.iloc[te].reset_index(drop=True)

def df_to_arrays(df):
    t=np.vstack(df[TEXT_COL].values).astype(np.float32)
    a=np.vstack(df[AUDIO_COL].values).astype(np.float32)
    v=np.vstack(df[VISUAL_COL].values).astype(np.float32)
    y=np.column_stack([df[c].values for c in TARGET_COLS]).astype(np.float32)
    return t,a,v,y

def make_tf_dataset(t,a,v,y,shuffle=False):
    ds=tf.data.Dataset.from_tensor_slices(({"text":t,"audio":a,"visual":v},y))
    if shuffle: ds=ds.shuffle(buffer_size=len(t),seed=RANDOM_SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_datasets(df,scaler=None):
    df=clean_embeddings(df.copy())
    df_tr,df_va,df_te=split_dataframe(df)
    t_tr,a_tr,v_tr,y_tr=df_to_arrays(df_tr)
    t_va,a_va,v_va,y_va=df_to_arrays(df_va)
    t_te,a_te,v_te,y_te=df_to_arrays(df_te)
    if scaler is None: scaler=ModalityScaler()
    scaler.fit(t_tr,a_tr,v_tr)
    t_tr,a_tr,v_tr=scaler.transform(t_tr,a_tr,v_tr)
    t_va,a_va,v_va=scaler.transform(t_va,a_va,v_va)
    t_te,a_te,v_te=scaler.transform(t_te,a_te,v_te)
    df_te=df_te.copy()
    df_te[TEXT_COL]=list(t_te); df_te[AUDIO_COL]=list(a_te); df_te[VISUAL_COL]=list(v_te)
    train_ds=make_tf_dataset(t_tr,a_tr,v_tr,y_tr,shuffle=True)
    val_ds  =make_tf_dataset(t_va,a_va,v_va,y_va)
    test_ds =make_tf_dataset(t_te,a_te,v_te,y_te)
    scaler.save(os.path.join(RESULTS_DIR,"scaler.pkl"))
    log.info("Done Train:%d Val:%d Test:%d batches",len(train_ds),len(val_ds),len(test_ds))
    return train_ds,val_ds,test_ds,scaler,df_te,(t_te,a_te,v_te,y_te)
