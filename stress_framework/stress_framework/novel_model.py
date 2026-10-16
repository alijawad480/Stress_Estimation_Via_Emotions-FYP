# =============================================================================
# novel_model.py — CHFN: Cross-modal Hierarchical Fusion Network
# FIX: All Lambda layers replaced with registered custom layers
# =============================================================================

import keras
from keras import layers, ops, regularizers
import tensorflow as tf
import numpy as np

from config import (TEXT_DIM, AUDIO_DIM, VISUAL_DIM, COMMON_DIM,
                    DROPOUT, WEIGHT_DECAY, N_EMOTIONS)
from models import AppendOne, LMFFusionLayer, CUSTOM_OBJECTS

L2    = regularizers.l2(WEIGHT_DECAY)
N_BAG = 3
PRIV  = 64
SHAR  = 64
ENC   = PRIV + SHAR   # = 128


# =============================================================================
# CUSTOM LAYERS — all registered, no Lambda anywhere
# =============================================================================

@keras.saving.register_keras_serializable()
class ScaleOutput(layers.Layer):
    """Scales sigmoid output to [0, scale]. Replaces Lambda(z * 3.0)"""
    def __init__(self, scale=3.0, **kw):
        super().__init__(**kw); self.scale = scale
    def call(self, x): return x * self.scale
    def get_config(self):
        base = super().get_config(); base.update({"scale": self.scale}); return base


@keras.saving.register_keras_serializable()
class StackTensors(layers.Layer):
    """
    Stacks a list of (B, D) tensors → (B, N, D).
    Replaces Lambda(lambda heads: tf.stack(heads, axis=1))
    """
    def __init__(self, axis=1, **kw):
        super().__init__(**kw); self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        # input_shape is a list of shapes e.g. [(B,D),(B,D),(B,D)]
        batch = input_shape[0][0]
        n     = len(input_shape)
        d     = input_shape[0][-1]
        return (batch, n, d)

    def get_config(self):
        base = super().get_config(); base.update({"axis": self.axis}); return base


@keras.saving.register_keras_serializable()
class ReduceMeanLayer(layers.Layer):
    """Mean over axis. Replaces Lambda(tf.reduce_mean(..., axis=1))"""
    def __init__(self, axis=1, **kw):
        super().__init__(**kw); self.axis = axis

    def call(self, x):
        return tf.reduce_mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.pop(self.axis)
        return tuple(shape)

    def get_config(self):
        base = super().get_config(); base.update({"axis": self.axis}); return base


@keras.saving.register_keras_serializable()
class ReduceStdLayer(layers.Layer):
    """Std over axis. Replaces Lambda(tf.math.reduce_std(..., axis=1))"""
    def __init__(self, axis=1, **kw):
        super().__init__(**kw); self.axis = axis

    def call(self, x):
        return tf.math.reduce_std(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.pop(self.axis)
        return tuple(shape)

    def get_config(self):
        base = super().get_config(); base.update({"axis": self.axis}); return base


# =============================================================================
# BLOCK 1 — Shared-Private Encoder
# =============================================================================

def shared_private_encoder(x, tag):
    priv = layers.Dense(PRIV*2, activation="relu", kernel_regularizer=L2, name=f"{tag}_priv1")(x)
    priv = layers.BatchNormalization(name=f"{tag}_privbn")(priv)
    priv = layers.Dropout(DROPOUT, name=f"{tag}_privdrop")(priv)
    priv = layers.Dense(PRIV, activation="relu", name=f"{tag}_priv2")(priv)

    shar = layers.Dense(SHAR*2, activation="relu", kernel_regularizer=L2, name=f"{tag}_shar1")(x)
    shar = layers.BatchNormalization(name=f"{tag}_sharbn")(shar)
    shar = layers.Dropout(DROPOUT, name=f"{tag}_shardrop")(shar)
    shar = layers.Dense(SHAR, activation="relu", name=f"{tag}_shar2")(shar)

    enc = layers.Concatenate(name=f"{tag}_enc")([priv, shar])
    return enc, priv, shar


# =============================================================================
# BLOCK 2 — Pairwise LMF
# =============================================================================

@keras.saving.register_keras_serializable()
class PairwiseLMF(layers.Layer):
    def __init__(self, rank=4, out_dim=64, **kw):
        super().__init__(**kw); self.rank = rank; self.out_dim = out_dim

    def build(self, input_shape):
        d    = input_shape[0][-1]
        init = keras.initializers.GlorotUniform()
        self.f1 = self.add_weight(shape=(self.rank, d+1, self.out_dim), name="f1", initializer=init)
        self.f2 = self.add_weight(shape=(self.rank, d+1, self.out_dim), name="f2", initializer=init)
        self.fw = self.add_weight(shape=(self.rank,), name="fw", initializer="ones")
        super().build(input_shape)

    def call(self, inputs):
        h1, h2 = inputs
        ones = ops.ones(ops.stack([ops.shape(h1)[0], 1]), dtype=h1.dtype)
        h1   = ops.concatenate([h1, ones], axis=-1)
        h2   = ops.concatenate([h2, ones], axis=-1)
        f1   = ops.einsum("bi,rio->bro", h1, self.f1)
        f2   = ops.einsum("bi,rio->bro", h2, self.f2)
        w    = ops.softmax(self.fw)
        return ops.relu(ops.einsum("bro,r->bo", f1 * f2, w))

    def get_config(self):
        base = super().get_config(); base.update({"rank": self.rank, "out_dim": self.out_dim}); return base


def hierarchical_fusion(et, ea, ev, tag):
    PAIR_DIM = 64; TRIP_DIM = 64

    fta = PairwiseLMF(rank=4, out_dim=PAIR_DIM, name=f"{tag}_lmf_ta")([et, ea])
    ftv = PairwiseLMF(rank=4, out_dim=PAIR_DIM, name=f"{tag}_lmf_tv")([et, ev])
    fav = PairwiseLMF(rank=4, out_dim=PAIR_DIM, name=f"{tag}_lmf_av")([ea, ev])

    ht = AppendOne(name=f"{tag}_ht")(fta)
    ha = AppendOne(name=f"{tag}_ha")(ftv)
    hv = AppendOne(name=f"{tag}_hv")(fav)
    triple = LMFFusionLayer(rank=4, out_dim=TRIP_DIM, name=f"{tag}_lmf_triple")([ht, ha, hv])
    triple = layers.Dense(COMMON_DIM, activation="relu", name=f"{tag}_trip_expand")(triple)
    return triple, (fta, ftv, fav)


# =============================================================================
# BLOCK 3 — Gated Cross-Attention
# =============================================================================

def gated_cross_attention(et, ea, ev, tag):
    KD = COMMON_DIM // 4

    def xattn(q, k, t):
        q3 = layers.Reshape((1, COMMON_DIM), name=f"{tag}_rs_{t}_q")(q)
        k3 = layers.Reshape((1, COMMON_DIM), name=f"{tag}_rs_{t}_k")(k)
        a  = layers.MultiHeadAttention(num_heads=4, key_dim=KD, dropout=DROPOUT,
                                       name=f"{tag}_mha_{t}")(q3, k3)
        a  = layers.Add(name=f"{tag}_add_{t}")([q3, a])
        a  = layers.LayerNormalization(name=f"{tag}_ln_{t}")(a)
        return layers.Reshape((COMMON_DIM,), name=f"{tag}_flat_{t}")(a)

    streams = [xattn(et,ea,"t2a"), xattn(et,ev,"t2v"),
               xattn(ea,et,"a2t"), xattn(ea,ev,"a2v"),
               xattn(ev,et,"v2t"), xattn(ev,ea,"v2a")]
    concat = layers.Concatenate(name=f"{tag}_concat")(streams)
    gate   = layers.Dense(6*COMMON_DIM, activation="sigmoid", name=f"{tag}_gate")(concat)
    gated  = layers.Multiply(name=f"{tag}_gated")([gate, concat])
    return layers.Dense(COMMON_DIM, activation="relu", name=f"{tag}_gate_out")(gated)


# =============================================================================
# BLOCK 4 — Bagging Head
# =============================================================================

def bagging_head(fused, tag):
    x = layers.Dense(128, activation="relu", kernel_regularizer=L2, name=f"{tag}_b1")(fused)
    x = layers.BatchNormalization(name=f"{tag}_b1bn")(x)
    x = layers.Dropout(DROPOUT, name=f"{tag}_b1drop")(x)
    x = layers.Dense(64, activation="relu", name=f"{tag}_b2")(x)
    x = layers.Dropout(DROPOUT, name=f"{tag}_b2drop")(x)
    x = layers.Dense(N_EMOTIONS, name=f"{tag}_logits")(x)
    x = layers.Activation("sigmoid", name=f"{tag}_sig")(x)
    return ScaleOutput(scale=3.0, name=f"{tag}_out")(x)   # ← no Lambda


# =============================================================================
# BLOCK 5 — Contrastive Loss
# =============================================================================

@keras.saving.register_keras_serializable()
class ContrastiveAlignmentLoss(layers.Layer):
    def __init__(self, temperature=0.07, **kw):
        super().__init__(**kw); self.temperature = temperature

    def call(self, inputs, training=None):
        shar_t, shar_a, shar_v = inputs
        nt = tf.nn.l2_normalize(shar_t, axis=-1)
        na = tf.nn.l2_normalize(shar_a, axis=-1)
        nv = tf.nn.l2_normalize(shar_v, axis=-1)
        sim_ta = tf.reduce_mean(tf.reduce_sum(nt * na, axis=-1))
        sim_tv = tf.reduce_mean(tf.reduce_sum(nt * nv, axis=-1))
        sim_av = tf.reduce_mean(tf.reduce_sum(na * nv, axis=-1))
        loss   = (3.0 - sim_ta - sim_tv - sim_av) / 3.0 / self.temperature
        self.add_loss(loss * 0.1)
        return inputs[0]

    def get_config(self):
        base = super().get_config(); base.update({"temperature": self.temperature}); return base


# =============================================================================
# CHFN — Full Model
# =============================================================================

def build_chfn(n_bag=N_BAG, use_contrastive=True):
    tag = "chfn"

    inp_t = keras.Input(shape=(TEXT_DIM,),   name="text")
    inp_a = keras.Input(shape=(AUDIO_DIM,),  name="audio")
    inp_v = keras.Input(shape=(VISUAL_DIM,), name="visual")

    # Block 1: Shared-Private Encoding
    enc_t, priv_t, shar_t = shared_private_encoder(inp_t, f"{tag}_t")
    enc_a, priv_a, shar_a = shared_private_encoder(inp_a, f"{tag}_a")
    enc_v, priv_v, shar_v = shared_private_encoder(inp_v, f"{tag}_v")

    # Block 2: Contrastive alignment
    if use_contrastive:
        shar_t = ContrastiveAlignmentLoss(temperature=0.07,
                                          name=f"{tag}_contrast")([shar_t, shar_a, shar_v])

    # Block 3: Hierarchical Fusion
    hier_fused, _ = hierarchical_fusion(enc_t, enc_a, enc_v, f"{tag}_hier")

    # Block 4: Gated Cross-Attention
    attn_fused = gated_cross_attention(enc_t, enc_a, enc_v, f"{tag}_attn")

    # Block 5: Combine
    combined = layers.Add(name=f"{tag}_combine")([hier_fused, attn_fused])
    combined = layers.BatchNormalization(name=f"{tag}_combbn")(combined)
    combined = layers.Dropout(DROPOUT, name=f"{tag}_combdrop")(combined)

    # Block 6: Bagging Heads
    head_outputs = [bagging_head(combined, f"{tag}_bag{i}") for i in range(n_bag)]

    # Stack heads → (B, n_bag, N_EMOTIONS) — StackTensors replaces Lambda
    stacked = StackTensors(axis=1, name=f"{tag}_stack")(head_outputs)

    # Mean and std — ReduceMeanLayer / ReduceStdLayer replace Lambda
    main        = ReduceMeanLayer(axis=1, name="main")(stacked)
    uncertainty = ReduceStdLayer(axis=1,  name="uncertainty")(stacked)

    # Block 7: Unimodal Auxiliary Heads
    def aux_head(x, name):
        x = layers.Dense(64, activation="relu", name=f"{name}_h1")(x)
        x = layers.Dropout(DROPOUT, name=f"{name}_drop")(x)
        x = layers.Dense(N_EMOTIONS, name=f"{name}_logits")(x)
        x = layers.Activation("sigmoid", name=f"{name}_sig")(x)
        return ScaleOutput(scale=3.0, name=name)(x)   # ← no Lambda

    aux_t = aux_head(enc_t, "aux_t")
    aux_a = aux_head(enc_a, "aux_a")
    aux_v = aux_head(enc_v, "aux_v")

    named_heads = {f"head_{i}": head_outputs[i] for i in range(n_bag)}

    outputs = {"main": main, "uncertainty": uncertainty,
               "aux_t": aux_t, "aux_a": aux_a, "aux_v": aux_v,
               **named_heads}

    return keras.Model(inputs=[inp_t, inp_a, inp_v], outputs=outputs, name="CHFN")


# =============================================================================
# CUSTOM OBJECTS
# =============================================================================

CHFN_CUSTOM_OBJECTS = {
    **CUSTOM_OBJECTS,
    "PairwiseLMF":              PairwiseLMF,
    "ContrastiveAlignmentLoss": ContrastiveAlignmentLoss,
    "ScaleOutput":              ScaleOutput,
    "StackTensors":             StackTensors,
    "ReduceMeanLayer":          ReduceMeanLayer,
    "ReduceStdLayer":           ReduceStdLayer,
}


# =============================================================================
# COMPILE HELPER
# =============================================================================

def compile_chfn(model, learning_rate=1e-3):
    n_bag      = N_BAG
    aux_weight = 0.15
    head_weight= 0.20

    loss_dict   = {"main":"mse","uncertainty":None,
                   "aux_t":"mse","aux_a":"mse","aux_v":"mse"}
    weight_dict = {"main":1.0,"uncertainty":0.0,
                   "aux_t":aux_weight,"aux_a":aux_weight,"aux_v":aux_weight}
    for i in range(n_bag):
        loss_dict[f"head_{i}"]   = "mse"
        weight_dict[f"head_{i}"] = head_weight

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                        weight_decay=1e-4),
        loss=loss_dict,
        loss_weights=weight_dict)
    return model


# =============================================================================
# DATASET WRAPPER
# =============================================================================

def wrap_dataset_chfn(dataset, n_bag=N_BAG):
    def _expand(inputs, labels):
        targets = {"main": labels, "uncertainty": labels,
                   "aux_t": labels, "aux_a": labels, "aux_v": labels}
        for i in range(n_bag):
            targets[f"head_{i}"] = labels
        return inputs, targets
    return dataset.map(_expand)


# =============================================================================
# MC DROPOUT INFERENCE
# =============================================================================

def predict_with_uncertainty(model, inputs, n_passes=20):
    preds = []
    for _ in range(n_passes):
        out = model(inputs, training=True)
        p   = out["main"] if isinstance(out, dict) else out
        preds.append(p.numpy())
    preds       = np.stack(preds, axis=0)
    mean_pred   = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)
    return mean_pred, uncertainty
