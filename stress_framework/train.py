# =============================================================================
# train.py
# =============================================================================
import os,json,time,logging
import pandas as pd
import keras
from keras import optimizers
from config import (EPOCHS,LEARNING_RATE,CHECKPOINT_DIR,RESULTS_DIR,
                    LR_PATIENCE,LR_FACTOR,PATIENCE,MODEL_NAMES,FUSION_STRATEGIES)

log=logging.getLogger(__name__)

class SelfMMLoss:
    AUX_WEIGHT=0.3
    @staticmethod
    def make_loss_dict(): return{"main":"mse","aux_t":"mse","aux_a":"mse","aux_v":"mse"}
    @staticmethod
    def make_loss_weights():
        w=SelfMMLoss.AUX_WEIGHT/3; return{"main":1.0,"aux_t":w,"aux_a":w,"aux_v":w}
    @staticmethod
    def wrap_dataset(ds):
        def _expand(inputs,labels):
            return inputs,{"main":labels,"aux_t":labels,"aux_a":labels,"aux_v":labels}
        return ds.map(_expand)

def make_callbacks(tag):
    ckpt=os.path.join(CHECKPOINT_DIR,f"{tag}_best.keras")
    return[keras.callbacks.ModelCheckpoint(filepath=ckpt,monitor="val_loss",save_best_only=True,verbose=1),
           keras.callbacks.EarlyStopping(monitor="val_loss",patience=PATIENCE,restore_best_weights=True,verbose=1),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=LR_FACTOR,patience=LR_PATIENCE,min_lr=1e-6,verbose=1),
           keras.callbacks.CSVLogger(os.path.join(RESULTS_DIR,f"{tag}_training_log.csv"))]

def train_one(model_name,fusion,train_ds,val_ds):
    from models import build_model
    tag=f"{model_name}_{fusion}"; is_selfmm=(model_name=="SelfMM")
    print("="*55); print(f"  {model_name} / {fusion}"); print("="*55)
    model=build_model(model_name,fusion)
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE)
    if is_selfmm:
        model.compile(optimizer=optimizer,loss=SelfMMLoss.make_loss_dict(),loss_weights=SelfMMLoss.make_loss_weights())
        tr_ds=SelfMMLoss.wrap_dataset(train_ds); va_ds=SelfMMLoss.wrap_dataset(val_ds)
    else:
        model.compile(optimizer=optimizer,loss="mse",metrics=["mae"]); tr_ds=train_ds; va_ds=val_ds
    t0=time.time()
    history=model.fit(tr_ds,validation_data=va_ds,epochs=EPOCHS,callbacks=make_callbacks(tag),verbose=1)
    elapsed=time.time()-t0
    val_key="val_main_loss" if is_selfmm else "val_loss"
    best_val=min(history.history.get(val_key,[float("inf")]))
    json.dump({k:[float(x) for x in v] for k,v in history.history.items()},
              open(os.path.join(RESULTS_DIR,f"{tag}_history.json"),"w"),indent=2)
    log.info("Done %.1fs best_val:%.4f",elapsed,best_val)
    return{"model_name":model_name,"fusion":fusion,"best_val_loss":best_val,
           "elapsed_sec":elapsed,"checkpoint":os.path.join(CHECKPOINT_DIR,f"{tag}_best.keras")}

def train_all(train_ds,val_ds):
    results=[]
    for name in MODEL_NAMES:
        for fusion in FUSION_STRATEGIES:
            tag=f"{name}_{fusion}"; ckpt=os.path.join(CHECKPOINT_DIR,f"{tag}_best.keras")
            if os.path.exists(ckpt): print(f"⏭️  Skipping {tag}"); continue
            result=train_one(name,fusion,train_ds,val_ds); results.append(result)
            keras.backend.clear_session()
    if results:
        pd.DataFrame(results).sort_values("best_val_loss").to_csv(
            os.path.join(RESULTS_DIR,"training_summary.csv"),index=False)
    return results

def load_model(model_name,fusion):
    from models import CUSTOM_OBJECTS
    tag=f"{model_name}_{fusion}"; path=os.path.join(CHECKPOINT_DIR,f"{tag}_best.keras")
    if not os.path.exists(path): raise FileNotFoundError(f"No checkpoint: {path}")
    return keras.models.load_model(path,custom_objects=CUSTOM_OBJECTS)
