# =============================================================================
# models.py — 5 Models × 3 Fusions, 6-emotion output, Keras 3 compatible
# =============================================================================
import keras
from keras import layers,ops,regularizers
from config import TEXT_DIM,AUDIO_DIM,VISUAL_DIM,COMMON_DIM,DROPOUT,WEIGHT_DECAY,N_EMOTIONS

L2=regularizers.l2(WEIGHT_DECAY)

@keras.saving.register_keras_serializable()
class AppendOne(layers.Layer):
    def call(self,x):
        ones=ops.ones(ops.stack([ops.shape(x)[0],1]),dtype=x.dtype)
        return ops.concatenate([x,ones],axis=-1)
    def compute_output_shape(self,s): return(s[0],s[1]+1)

@keras.saving.register_keras_serializable()
class TFNOuterProduct(layers.Layer):
    def call(self,inputs):
        ht,ha,hv=inputs; sub1=ht.shape[-1]
        ta=ops.einsum("bi,bj->bij",ht,ha)
        ta_f=ops.reshape(ta,(-1,sub1*sub1))
        tav=ops.einsum("bi,bj->bij",ta_f,hv)
        return ops.reshape(tav,(-1,sub1*sub1*sub1))
    def compute_output_shape(self,s): return(s[0][0],s[0][-1]**3)

@keras.saving.register_keras_serializable()
class LMFFusionLayer(layers.Layer):
    def __init__(self,rank=4,out_dim=32,**kw):
        super().__init__(**kw); self.rank=rank; self.out_dim=out_dim
    def build(self,input_shape):
        d=input_shape[0][-1]; init=keras.initializers.GlorotUniform()
        self.ft=self.add_weight(shape=(self.rank,d,self.out_dim),name="ft",initializer=init)
        self.fa=self.add_weight(shape=(self.rank,d,self.out_dim),name="fa",initializer=init)
        self.fv=self.add_weight(shape=(self.rank,d,self.out_dim),name="fv",initializer=init)
        self.fw=self.add_weight(shape=(self.rank,),name="fw",initializer="ones")
        super().build(input_shape)
    def call(self,inputs):
        ht,ha,hv=inputs
        ft=ops.einsum("bi,rio->bro",ht,self.ft)
        fa=ops.einsum("bi,rio->bro",ha,self.fa)
        fv=ops.einsum("bi,rio->bro",hv,self.fv)
        w=ops.softmax(self.fw)
        return ops.relu(ops.einsum("bro,r->bo",ft*fa*fv,w))
    def get_config(self):
        base=super().get_config(); base.update({"rank":self.rank,"out_dim":self.out_dim}); return base
    def compute_output_shape(self,s): return(s[0][0],self.out_dim)

def proj(x,tag):
    x=layers.Dense(COMMON_DIM,kernel_regularizer=L2,name=f"{tag}_proj")(x)
    x=layers.BatchNormalization(name=f"{tag}_bn")(x)
    x=layers.Activation("relu",name=f"{tag}_relu")(x)
    return layers.Dropout(DROPOUT,name=f"{tag}_drop")(x)

def head(x,tag,n_out=N_EMOTIONS):
    x=layers.Dense(64,activation="relu",kernel_regularizer=L2,name=f"{tag}_h1")(x)
    x=layers.BatchNormalization(name=f"{tag}_h1bn")(x)
    x=layers.Dropout(DROPOUT,name=f"{tag}_h1drop")(x)
    x=layers.Dense(n_out,name=f"{tag}_logits")(x)
    x=layers.Activation("sigmoid",name=f"{tag}_sig")(x)
    return layers.Lambda(lambda z:z*3.0,name=f"{tag}_out")(x)

def early_enc(fused,tag):
    x=layers.Dense(256,activation="relu",kernel_regularizer=L2,name=f"{tag}_e1")(fused)
    x=layers.BatchNormalization(name=f"{tag}_e1bn")(x)
    x=layers.Dropout(DROPOUT,name=f"{tag}_e1drop")(x)
    return layers.Dense(COMMON_DIM,activation="relu",name=f"{tag}_e2")(x)

def late_avg(pt,pa,pv,tag):
    et=layers.Dense(COMMON_DIM,activation="relu",name=f"{tag}_lt")(pt)
    ea=layers.Dense(COMMON_DIM,activation="relu",name=f"{tag}_la")(pa)
    ev=layers.Dense(COMMON_DIM,activation="relu",name=f"{tag}_lv")(pv)
    return layers.Average(name=f"{tag}_lavg")([et,ea,ev])

def build_mult(fusion="cross_modal"):
    nm=f"mult_{fusion}"
    inp_t=keras.Input(shape=(TEXT_DIM,),name="text")
    inp_a=keras.Input(shape=(AUDIO_DIM,),name="audio")
    inp_v=keras.Input(shape=(VISUAL_DIM,),name="visual")
    pt=proj(inp_t,f"{nm}_t"); pa=proj(inp_a,f"{nm}_a"); pv=proj(inp_v,f"{nm}_v")
    if fusion=="cross_modal":
        qt=layers.Reshape((1,COMMON_DIM))(pt); qa=layers.Reshape((1,COMMON_DIM))(pa); qv=layers.Reshape((1,COMMON_DIM))(pv)
        def xattn(q,k,tag):
            a=layers.MultiHeadAttention(num_heads=4,key_dim=COMMON_DIM//4,dropout=DROPOUT,name=f"{nm}_attn_{tag}")(q,k)
            a=layers.Add(name=f"{nm}_add_{tag}")([q,a]); a=layers.LayerNormalization(name=f"{nm}_ln_{tag}")(a)
            return layers.Reshape((COMMON_DIM,),name=f"{nm}_rs_{tag}")(a)
        streams=[xattn(qt,qv,"t2v"),xattn(qt,qa,"t2a"),xattn(qv,qt,"v2t"),xattn(qv,qa,"v2a"),xattn(qa,qt,"a2t"),xattn(qa,qv,"a2v")]
        fused=layers.Concatenate(name=f"{nm}_fuse")(streams)
        fused=layers.Dense(COMMON_DIM,activation="relu",name=f"{nm}_reduce")(fused)
    elif fusion=="early":
        fused=layers.Concatenate(name=f"{nm}_fuse")([pt,pa,pv]); fused=early_enc(fused,nm)
    else: fused=late_avg(pt,pa,pv,nm)
    return keras.Model([inp_t,inp_a,inp_v],head(fused,nm),name=f"MulT_{fusion}")

def build_tfn(fusion="early"):
    nm=f"tfn_{fusion}"; SUB=16
    inp_t=keras.Input(shape=(TEXT_DIM,),name="text"); inp_a=keras.Input(shape=(AUDIO_DIM,),name="audio"); inp_v=keras.Input(shape=(VISUAL_DIM,),name="visual")
    pt=proj(inp_t,f"{nm}_t"); pa=proj(inp_a,f"{nm}_a"); pv=proj(inp_v,f"{nm}_v")
    if fusion in("early","cross_modal"):
        st=layers.Dense(SUB,activation="tanh",name=f"{nm}_sub_t")(pt)
        sa=layers.Dense(SUB,activation="tanh",name=f"{nm}_sub_a")(pa)
        sv=layers.Dense(SUB,activation="tanh",name=f"{nm}_sub_v")(pv)
        ht=AppendOne(name=f"{nm}_ht")(st); ha=AppendOne(name=f"{nm}_ha")(sa); hv=AppendOne(name=f"{nm}_hv")(sv)
        fused=TFNOuterProduct(name=f"{nm}_outer")([ht,ha,hv])
        fused=layers.Dense(256,activation="relu",kernel_regularizer=L2,name=f"{nm}_fc1")(fused)
        fused=layers.BatchNormalization(name=f"{nm}_fc1bn")(fused); fused=layers.Dropout(DROPOUT,name=f"{nm}_fc1drop")(fused)
        fused=layers.Dense(COMMON_DIM,activation="relu",name=f"{nm}_fc2")(fused)
    else: fused=late_avg(pt,pa,pv,nm)
    return keras.Model([inp_t,inp_a,inp_v],head(fused,nm),name=f"TFN_{fusion}")

def build_lmf(fusion="early"):
    nm=f"lmf_{fusion}"
    inp_t=keras.Input(shape=(TEXT_DIM,),name="text"); inp_a=keras.Input(shape=(AUDIO_DIM,),name="audio"); inp_v=keras.Input(shape=(VISUAL_DIM,),name="visual")
    pt=proj(inp_t,f"{nm}_t"); pa=proj(inp_a,f"{nm}_a"); pv=proj(inp_v,f"{nm}_v")
    if fusion in("early","cross_modal"):
        ht=AppendOne(name=f"{nm}_ht")(pt); ha=AppendOne(name=f"{nm}_ha")(pa); hv=AppendOne(name=f"{nm}_hv")(pv)
        fused=LMFFusionLayer(rank=4,out_dim=32,name=f"{nm}_lmf")([ht,ha,hv])
        fused=layers.Dense(COMMON_DIM,activation="relu",kernel_regularizer=L2,name=f"{nm}_expand")(fused)
    else: fused=late_avg(pt,pa,pv,nm)
    return keras.Model([inp_t,inp_a,inp_v],head(fused,nm),name=f"LMF_{fusion}")

def build_mfn(fusion="late"):
    nm=f"mfn_{fusion}"
    inp_t=keras.Input(shape=(TEXT_DIM,),name="text"); inp_a=keras.Input(shape=(AUDIO_DIM,),name="audio"); inp_v=keras.Input(shape=(VISUAL_DIM,),name="visual")
    pt=proj(inp_t,f"{nm}_t"); pa=proj(inp_a,f"{nm}_a"); pv=proj(inp_v,f"{nm}_v")
    if fusion in("late","cross_modal"):
        st=layers.Reshape((1,COMMON_DIM),name=f"{nm}_rs_t")(pt); sa=layers.Reshape((1,COMMON_DIM),name=f"{nm}_rs_a")(pa); sv=layers.Reshape((1,COMMON_DIM),name=f"{nm}_rs_v")(pv)
        ht=layers.GRU(COMMON_DIM,name=f"{nm}_gru_t")(st); ha=layers.GRU(COMMON_DIM,name=f"{nm}_gru_a")(sa); hv=layers.GRU(COMMON_DIM,name=f"{nm}_gru_v")(sv)
        mem=layers.Concatenate(name=f"{nm}_mem")([ht,ha,hv])
        gate=layers.Dense(3*COMMON_DIM,activation="sigmoid",name=f"{nm}_gate")(mem)
        gated=layers.Multiply(name=f"{nm}_gated")([gate,mem])
        fused=layers.Dense(COMMON_DIM,activation="relu",name=f"{nm}_reduce")(gated)
    else:
        fused=layers.Concatenate(name=f"{nm}_fuse")([pt,pa,pv]); fused=early_enc(fused,nm)
    return keras.Model([inp_t,inp_a,inp_v],head(fused,nm),name=f"MFN_{fusion}")

def build_selfmm(fusion="late"):
    nm=f"smm_{fusion}"
    inp_t=keras.Input(shape=(TEXT_DIM,),name="text"); inp_a=keras.Input(shape=(AUDIO_DIM,),name="audio"); inp_v=keras.Input(shape=(VISUAL_DIM,),name="visual")
    pt=proj(inp_t,f"{nm}_t"); pa=proj(inp_a,f"{nm}_a"); pv=proj(inp_v,f"{nm}_v")
    fused=layers.Concatenate(name=f"{nm}_fuse")([pt,pa,pv])
    shared=layers.Dense(256,activation="relu",kernel_regularizer=L2,name=f"{nm}_s1")(fused)
    shared=layers.BatchNormalization(name=f"{nm}_s1bn")(shared); shared=layers.Dropout(DROPOUT,name=f"{nm}_s1drop")(shared)
    shared=layers.Dense(COMMON_DIM,activation="relu",name=f"{nm}_s2")(shared)
    return keras.Model([inp_t,inp_a,inp_v],
        outputs={"main":head(shared,f"{nm}_main"),"aux_t":head(pt,f"{nm}_auxt"),
                 "aux_a":head(pa,f"{nm}_auxa"),"aux_v":head(pv,f"{nm}_auxv")},name=f"SelfMM_{fusion}")

_BUILDERS={"MulT":build_mult,"TFN":build_tfn,"LMF":build_lmf,"MFN":build_mfn,"SelfMM":build_selfmm}
CUSTOM_OBJECTS={"AppendOne":AppendOne,"TFNOuterProduct":TFNOuterProduct,"LMFFusionLayer":LMFFusionLayer}

def build_model(name,fusion):
    assert name in _BUILDERS; assert fusion in("early","late","cross_modal")
    m=_BUILDERS[name](fusion=fusion); print(f"[{name}/{fusion}] params={m.count_params():,}"); return m
