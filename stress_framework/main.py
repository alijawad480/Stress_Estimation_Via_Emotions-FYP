# =============================================================================
# main.py — End-to-End Inference Pipeline
# =============================================================================
import os,logging,pickle,tempfile,subprocess
import numpy as np,tensorflow as tf

from config import (RESULTS_DIR,CHECKPOINT_DIR,TEXT_DIM,AUDIO_DIM,VISUAL_DIM,
                    STRESS_LOW_MAX,N_ACTIONS,EMOTION_NAMES)
from preprocessing import ModalityScaler
from train import train_all,load_model
from evaluate import full_comparison
from stress_score import stress_calc
from intervention import (MistralAdaptiveAgent,ConversationContext,
                           A2CController,DuelingDQNController,SessionState,
                           ACTIONS,train_rl)

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
log=logging.getLogger(__name__)

def extract_text_features(text):
    glove_cache=os.path.join(RESULTS_DIR,"glove_cache.pkl")
    if os.path.exists(glove_cache):
        try:
            with open(glove_cache,"rb") as f: glove=pickle.load(f)
            words=text.lower().split(); vecs=[glove[w] for w in words if w in glove]
            if vecs: return np.mean(vecs,axis=0).astype(np.float32)
        except Exception: pass
    vec=np.zeros(TEXT_DIM,dtype=np.float32)
    for word in text.lower().split():
        for i in range(len(word)-2):
            vec[hash(word[i:i+3])%TEXT_DIM]+=1.0
    norm=np.linalg.norm(vec); return(vec/norm) if norm>0 else vec

def extract_audio_features(audio_path):
    try:
        import librosa
        y,sr=librosa.load(audio_path,sr=16000,mono=True); y,_=librosa.effects.trim(y,top_db=20)
        if len(y)<sr*0.3: return np.zeros(AUDIO_DIM,dtype=np.float32)
        features=[]
        mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
        features.extend([mfcc.mean(1),librosa.feature.delta(mfcc).mean(1),librosa.feature.delta(mfcc,order=2).mean(1)])
        f0,_,_=librosa.pyin(y,fmin=50,fmax=400,sr=sr,frame_length=2048)
        f0=f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        features.append(np.array([f0.mean() if len(f0)>0 else 0.0,f0.std() if len(f0)>0 else 0.0]))
        features.extend([librosa.feature.spectral_centroid(y=y,sr=sr).mean(1),
                         librosa.feature.spectral_bandwidth(y=y,sr=sr).mean(1),
                         librosa.feature.spectral_rolloff(y=y,sr=sr).mean(1),
                         librosa.feature.spectral_contrast(y=y,sr=sr).mean(1),
                         librosa.feature.zero_crossing_rate(y).mean(1),
                         librosa.feature.chroma_stft(y=y,sr=sr).mean(1),
                         librosa.feature.rms(y=y).mean(1)])
        vec=np.concatenate(features).astype(np.float32)
        return np.pad(vec,(0,max(0,AUDIO_DIM-len(vec))))[:AUDIO_DIM]
    except Exception as e:
        log.error("Audio extraction failed: %s",e); return np.zeros(AUDIO_DIM,dtype=np.float32)

def extract_visual_features(video_path):
    try:
        import cv2
        cap=cv2.VideoCapture(video_path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total==0: cap.release(); return np.zeros(VISUAL_DIM,dtype=np.float32)
        indices=np.linspace(0,total-1,min(30,total),dtype=int); frames=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ret,frame=cap.read()
            if ret:
                frame=cv2.resize(frame,(64,64))
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0)
        cap.release()
        if not frames: return np.zeros(VISUAL_DIM,dtype=np.float32)
        arr=np.stack(frames)
        features=[arr.mean(axis=(1,2)),arr.std(axis=(1,2)),arr.min(axis=(1,2)),arr.max(axis=(1,2))]
        if len(arr)>1:
            diffs=np.abs(np.diff(arr,axis=0))
            features.extend([diffs.mean(axis=(1,2)),diffs.max(axis=(1,2))])
        vec=np.concatenate([f.flatten() for f in features]).astype(np.float32)
        return np.pad(vec,(0,max(0,VISUAL_DIM-len(vec))))[:VISUAL_DIM]
    except Exception as e:
        log.error("Visual extraction failed: %s",e); return np.zeros(VISUAL_DIM,dtype=np.float32)

def split_video(video_path):
    result={"visual_vec":np.zeros(VISUAL_DIM,dtype=np.float32),
            "audio_vec":np.zeros(AUDIO_DIM,dtype=np.float32),
            "has_audio":False,"audio_note":""}
    result["visual_vec"]=extract_visual_features(video_path)
    try:
        probe=subprocess.run(["ffprobe","-v","error","-select_streams","a:0",
            "-show_entries","stream=codec_type","-of","default=noprint_wrappers=1:nokey=1",video_path],
            capture_output=True,text=True,timeout=15)
        if "audio" in probe.stdout.lower():
            tmp_wav=tempfile.mktemp(suffix=".wav")
            ext=subprocess.run(["ffmpeg","-y","-i",video_path,"-ar","16000","-ac","1","-vn",tmp_wav],
                capture_output=True,timeout=120)
            if ext.returncode==0 and os.path.exists(tmp_wav):
                result["audio_vec"]=extract_audio_features(tmp_wav); result["has_audio"]=True
                os.remove(tmp_wav)
            else: result["audio_note"]="⚠️  Audio extraction failed — using facial only."
        else:
            result["audio_note"]="⚠️  Video has no audio — emotion predicted from facial expressions only."
    except FileNotFoundError:
        result["audio_note"]="⚠️  ffmpeg not installed — using facial only."
    except Exception as e:
        result["audio_note"]=f"⚠️  Audio error: {e}"
    return result

def run_inference(text=None,audio_path=None,video_path=None,
                  model_name="LMF",fusion="cross_modal",run_chat=True):
    modalities_used=[]; has_audio=True; audio_note=""
    text_vec  =np.zeros(TEXT_DIM,  dtype=np.float32)
    audio_vec =np.zeros(AUDIO_DIM, dtype=np.float32)
    visual_vec=np.zeros(VISUAL_DIM,dtype=np.float32)

    if text:
        text_vec=extract_text_features(text); modalities_used.append("text")
    if audio_path and os.path.exists(audio_path):
        audio_vec=extract_audio_features(audio_path); modalities_used.append("audio")
    if video_path and os.path.exists(video_path):
        split=split_video(video_path); visual_vec=split["visual_vec"]
        has_audio=split["has_audio"]; audio_note=split["audio_note"]
        if has_audio:
            audio_vec=split["audio_vec"]; modalities_used.extend(["video-visual","video-audio"])
        else:
            modalities_used.append("video-visual-only")

    scaler_path=os.path.join(RESULTS_DIR,"scaler.pkl")
    if os.path.exists(scaler_path):
        sc=ModalityScaler(); sc.load(scaler_path)
        text_vec,audio_vec,visual_vec=sc.transform_single(text_vec,audio_vec,visual_vec)

    model=load_model(model_name,fusion)
    inputs={"text":tf.expand_dims(text_vec,0),"audio":tf.expand_dims(audio_vec,0),"visual":tf.expand_dims(visual_vec,0)}
    pred=model(inputs,training=False)
    if isinstance(pred,dict): pred=pred["main"]
    pred=pred.numpy()[0]
    emotion_dict={name:float(pred[i]) for i,name in enumerate(EMOTION_NAMES)}
    stress_result=stress_calc.compute(pred)
    stress_score=stress_result["stress_score"]; stress_label=stress_result["stress_label"]
    dominant=stress_result["dominant_emotion"]

    dom_idx=EMOTION_NAMES.index(dominant)/max(len(EMOTION_NAMES)-1,1)
    state=SessionState(stress_score=stress_score,dominant_idx=dom_idx)
    action=1 if stress_score>=STRESS_LOW_MAX else 0

    a2c_path =os.path.join(CHECKPOINT_DIR,"a2c_actor.weights.h5")
    ddqn_path=os.path.join(CHECKPOINT_DIR,"dueling_dqn.weights.h5")
    if os.path.exists(a2c_path):
        a2c=A2CController(); a2c.actor.load_weights(a2c_path)
        action=a2c.select_action(state)
    elif os.path.exists(ddqn_path):
        ddqn=DuelingDQNController(); ddqn.load()
        action=ddqn.select_action(state)

    chat_history=[]
    if action>0 or stress_score>=STRESS_LOW_MAX:
        ctx=ConversationContext(emotion_dict=emotion_dict,stress_score=stress_score,
            stress_label=stress_label,dominant_emotion=dominant,
            modalities_used=modalities_used,has_audio=has_audio)
        agent=MistralAdaptiveAgent()
        chat_history=agent.run_adaptive_session(ctx,colab_input=run_chat)
    else:
        print(f"\n✅ Stress: {stress_label} (score={stress_score:.2f}) — No intervention needed.\n")

    result={"modalities_used":modalities_used,"has_audio":has_audio,
            "audio_note":audio_note if audio_note else None,
            **{f"emotion_{k}":round(v,4) for k,v in emotion_dict.items()},
            "valence":round(stress_result["valence"],4),
            "arousal":round(stress_result["arousal"],4),
            "stress_score":round(stress_score,4),
            "stress_label":stress_label,"dominant_emotion":dominant,
            "rl_action":ACTIONS[action],"chat_turns":len(chat_history)}
    print("\n=== INFERENCE SUMMARY ===")
    for k,v in result.items():
        if v is not None: print(f"  {k:<22}: {v}")
    return result

def run_training(df):
    from preprocessing import build_datasets
    train_ds,val_ds,test_ds,scaler,df_test,arrays=build_datasets(df)
    train_all(train_ds,val_ds)
    return train_ds,val_ds,test_ds,scaler,df_test,arrays

def run_evaluation(test_ds): return full_comparison(test_ds)

def run_rl(df_test,arrays):
    t_te,a_te,v_te,y_te=arrays
    best_name,best_fusion="LMF","cross_modal"
    csv_path=os.path.join(RESULTS_DIR,"deep_models_metrics.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df_m=pd.read_csv(csv_path)
        if not df_m.empty:
            best=df_m.sort_values("RMSE").iloc[0]
            best_name=best["Model"]; best_fusion=best["Fusion"]
    try:
        model=load_model(best_name,best_fusion)
        preds=model.predict({"text":t_te,"audio":a_te,"visual":v_te},verbose=0)
        if isinstance(preds,dict): preds=preds["main"]
    except Exception as e:
        log.warning("Model load failed (%s) — random trajectories",e)
        preds=np.random.uniform(0,1,(len(t_te),len(EMOTION_NAMES)))
    scores=stress_calc.compute_from_predictions(preds)["stress_score"]
    seg=10; trajs=[list(scores[i:i+seg]) for i in range(0,len(scores)-seg,seg//2)]
    return train_rl(trajs)
