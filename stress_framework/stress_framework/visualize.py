# =============================================================================
# visualize.py — 8 Publication Charts
# =============================================================================
import os,json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR,EMOTION_NAMES,STRESS_LOW_MAX,STRESS_MODERATE_MAX

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,"figure.dpi":150})
EMOTION_COLORS={"anger":"#f87171","fear":"#c084fc","happy":"#34d399","sad":"#60a5fa","surprise":"#fbbf24","disgust":"#fb923c"}

def plot_rmse_heatmap(df,save=True):
    pivot=df.pivot(index="Model",columns="Fusion",values="RMSE")
    for col in["early","late","cross_modal"]:
        if col not in pivot.columns: pivot[col]=np.nan
    pivot=pivot[["early","late","cross_modal"]]
    fig,ax=plt.subplots(figsize=(8,5))
    sns.heatmap(pivot,annot=True,fmt=".4f",cmap="YlOrRd",linewidths=0.5,linecolor="white",ax=ax)
    ax.set_title("RMSE — Model × Fusion (6-Emotion)"); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig1_rmse_heatmap.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_f1_bars(df,save=True):
    ordered=df.sort_values("F1",ascending=True)
    labels=ordered["Model"]+"\n"+ordered["Fusion"]
    colors={"early":"#4C72B0","late":"#DD8452","cross_modal":"#55A868"}
    bar_cols=[colors.get(f,"#888") for f in ordered["Fusion"]]
    fig,ax=plt.subplots(figsize=(9,6))
    bars=ax.barh(labels,ordered["F1"],color=bar_cols,edgecolor="white",height=0.7)
    for bar,val in zip(bars,ordered["F1"]):
        ax.text(val+0.0005,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va="center",ha="left",fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c,label=f.replace("_"," ").title()) for f,c in colors.items()],loc="lower right")
    ax.set_title("F1 — All 15 Combinations"); ax.grid(axis="x",alpha=0.3); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig2_f1_bars.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_per_emotion_metrics(best_model_name,best_fusion,y_true,y_pred,save=True):
    from evaluate import compute_metrics; metrics=compute_metrics(y_true,y_pred)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    rmses=[metrics[e]["RMSE"] for e in EMOTION_NAMES]; f1s=[metrics[e]["F1"] for e in EMOTION_NAMES]
    cols=[EMOTION_COLORS[e] for e in EMOTION_NAMES]
    axes[0].bar(EMOTION_NAMES,rmses,color=cols,edgecolor="white",width=0.6); axes[0].set_title("RMSE per Emotion"); axes[0].grid(axis="y",alpha=0.3)
    axes[1].bar(EMOTION_NAMES,f1s,  color=cols,edgecolor="white",width=0.6); axes[1].set_title("F1 per Emotion");   axes[1].grid(axis="y",alpha=0.3)
    plt.suptitle(f"Per-Emotion — {best_model_name}/{best_fusion}"); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig3_per_emotion.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_training_curves(model_name,fusion,save=True):
    tag=f"{model_name}_{fusion}"; path=os.path.join(RESULTS_DIR,f"{tag}_history.json")
    if not os.path.exists(path): print(f"No history for {tag}"); return
    with open(path) as f: hist=json.load(f)
    is_smm=(model_name=="SelfMM")
    tl=hist.get("main_loss" if is_smm else "loss",[])
    vl=hist.get("val_main_loss" if is_smm else "val_loss",[])
    ep=range(1,len(tl)+1)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(ep,tl,label="Train",color="#4C72B0",lw=2); ax.plot(ep,vl,label="Val",color="#DD8452",lw=2)
    best_ep=int(np.argmin(vl))+1
    ax.axvline(best_ep,color="red",linestyle="--",lw=1.2,label=f"Best ep{best_ep}(val={min(vl):.4f})")
    ax.set_title(f"Training — {model_name}/{fusion}"); ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,f"fig4_train_{tag}.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_rl_curves(save=True):
    path=os.path.join(RESULTS_DIR,"rl_reward_history.json")
    if not os.path.exists(path): print("No RL history."); return
    with open(path) as f: history=json.load(f)
    fig,ax=plt.subplots(figsize=(10,4))
    colors={"A2C":"#2196F3","DuelingDQN":"#FF9800"}
    for algo in["A2C","DuelingDQN"]:
        if algo not in history: continue
        rewards=np.array(history[algo]); rolling=np.convolve(rewards,np.ones(50)/50,mode="valid")
        ax.plot(rolling,label=algo,color=colors.get(algo,"gray"),lw=2)
        ax.annotate(f"{algo} final:{np.mean(rewards[-300:]):.2f}",
                    xy=(len(rolling)-5,rolling[-1]),fontsize=9,color=colors.get(algo,"gray"),ha="right")
    ax.axhline(STRESS_LOW_MAX,color="red",linestyle="--",lw=1.2,label=f"Stress threshold {STRESS_LOW_MAX}")
    ax.set_title("RL Training — A2C vs Dueling DQN"); ax.legend(); ax.set_ylim(0,1.1); ax.grid(axis="y",alpha=0.3); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig5_rl_curves.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_stress_distribution(stress_scores,save=True):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(stress_scores,bins=40,color="#4C72B0",edgecolor="white",alpha=0.85)
    axes[0].axvline(STRESS_LOW_MAX,color="#facc15",lw=1.5,linestyle="--",label=f"Low/Mod ({STRESS_LOW_MAX})")
    axes[0].axvline(STRESS_MODERATE_MAX,color="#f87171",lw=1.5,linestyle="--",label=f"Mod/Severe ({STRESS_MODERATE_MAX})")
    axes[0].set_title("Stress Distribution [0–1]"); axes[0].legend()
    counts=[(stress_scores<STRESS_LOW_MAX).sum(),
            ((stress_scores>=STRESS_LOW_MAX)&(stress_scores<STRESS_MODERATE_MAX)).sum(),
            (stress_scores>=STRESS_MODERATE_MAX).sum()]
    axes[1].pie(counts,labels=[f"Low\n(S<{STRESS_LOW_MAX})",f"Moderate\n({STRESS_LOW_MAX}–{STRESS_MODERATE_MAX})",f"Severe\n(S≥{STRESS_MODERATE_MAX})"],
                colors=["#55A868","#DD8452","#C44E52"],autopct="%1.1f%%",startangle=140,
                wedgeprops={"edgecolor":"white","linewidth":1.5})
    axes[1].set_title("3-Level Severity")
    plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig6_stress_dist.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_pred_vs_true(y_true,y_pred,save=True):
    from sklearn.metrics import mean_squared_error
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,(ax,name) in enumerate(zip(axes.flatten(),EMOTION_NAMES)):
        yt,yp=y_true[:,i],y_pred[:,i]
        ax.scatter(yt,yp,alpha=0.2,s=8,color=EMOTION_COLORS[name])
        lim=max(yt.max(),yp.max())*1.05; ax.plot([0,lim],[0,lim],"k--",lw=1.2)
        rmse=np.sqrt(mean_squared_error(yt,yp)); corr=float(np.corrcoef(yt,yp)[0,1]) if np.std(yt)>0 else 0.0
        ax.text(0.05,0.92,f"RMSE={rmse:.3f} r={corr:.3f}",transform=ax.transAxes,fontsize=9,
                bbox=dict(boxstyle="round",fc="white",alpha=0.8))
        ax.set_title(name.capitalize())
    plt.suptitle("Predicted vs True — All 6 Emotions"); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig7_pred_vs_true.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def plot_circumplex(stress_results,save=True):
    from config import RUSSELL_COORDS
    V=np.asarray(stress_results["valence"]); A=np.asarray(stress_results["arousal"]); S=np.asarray(stress_results["stress_score"])
    fig,ax=plt.subplots(figsize=(7,7))
    sc=ax.scatter(V,A,c=S,cmap="RdYlGn_r",alpha=0.35,s=12,vmin=0,vmax=1)
    plt.colorbar(sc,ax=ax,label="Stress [0–1]")
    ax.axhline(0,color="gray",lw=0.8,linestyle="--"); ax.axvline(0,color="gray",lw=0.8,linestyle="--")
    for emo,coords in RUSSELL_COORDS.items():
        ax.scatter(coords["valence"],coords["arousal"],s=100,zorder=5,marker="*",color=EMOTION_COLORS.get(emo,"white"))
        ax.annotate(emo,xy=(coords["valence"],coords["arousal"]),xytext=(coords["valence"]+0.05,coords["arousal"]+0.05),fontsize=8)
    ax.set_title("Russell's Circumplex — 6-Emotion"); ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.grid(alpha=0.2); plt.tight_layout()
    if save:
        p=os.path.join(RESULTS_DIR,"fig8_circumplex.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); print(f"Saved:{p}")
    plt.show()

def generate_all_charts(df_results,y_true=None,y_pred=None,stress_results=None,best_model="LMF",best_fusion="cross_modal"):
    print("Generating charts...\n")
    plot_rmse_heatmap(df_results); plot_f1_bars(df_results)
    plot_training_curves(best_model,best_fusion); plot_rl_curves()
    if y_true is not None and y_pred is not None:
        plot_per_emotion_metrics(best_model,best_fusion,y_true,y_pred); plot_pred_vs_true(y_true,y_pred)
    if stress_results is not None:
        plot_stress_distribution(stress_results["stress_score"]); plot_circumplex(stress_results)
    print(f"All charts → {RESULTS_DIR}")
