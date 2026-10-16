# =============================================================================
# sota_models.py — MISA, CTNet, UniMSE
# FIX: All Lambda layers replaced with registered custom layers
# =============================================================================

import keras
from keras import layers, ops, regularizers
import tensorflow as tf

from config import (TEXT_DIM, AUDIO_DIM, VISUAL_DIM, COMMON_DIM,
                    DROPOUT, WEIGHT_DECAY, N_EMOTIONS)
from models import CUSTOM_OBJECTS

L2 = regularizers.l2(WEIGHT_DECAY)


# =============================================================================
# SHARED CUSTOM LAYERS (no Lambda anywhere)
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
class StackLayer(layers.Layer):
    """Stacks list of tensors along axis=1. Replaces Lambda(tf.stack(...))"""
    def __init__(self, axis=1, **kw):
        super().__init__(**kw); self.axis = axis
    def call(self, inputs): return tf.stack(inputs, axis=self.axis)
    def get_config(self):
        base = super().get_config(); base.update({"axis": self.axis}); return base


@keras.saving.register_keras_serializable()
class ReduceMeanLayer(layers.Layer):
    """Mean over axis. Replaces Lambda(tf.reduce_mean(..., axis=1))"""
    def __init__(self, axis=1, **kw):
        super().__init__(**kw); self.axis = axis
    def call(self, x): return tf.reduce_mean(x, axis=self.axis)
    def get_config(self):
        base = super().get_config(); base.update({"axis": self.axis}); return base


@keras.saving.register_keras_serializable()
class L2NormalizeLayer(layers.Layer):
    """L2 normalizes along last axis."""
    def call(self, x): return tf.nn.l2_normalize(x, axis=-1)


@keras.saving.register_keras_serializable()
class ExpandDimsLayer(layers.Layer):
    """Expands dims at given axis. Replaces Lambda(tf.expand_dims(...))"""
    def __init__(self, axis=-1, **kw):
        super().__init__(**kw); self.axis = axis
    def call(self, x): return tf.expand_dims(x, axis=self.axis)
    def get_config(self):
        base = super().get_config(); base.update({"axis": self.axis}); return base


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _proj(x, dim, tag):
    x = layers.Dense(dim, kernel_regularizer=L2, name=f"{tag}_proj")(x)
    x = layers.BatchNormalization(name=f"{tag}_bn")(x)
    x = layers.Activation("relu", name=f"{tag}_act")(x)
    return layers.Dropout(DROPOUT, name=f"{tag}_drop")(x)


def _emotion_head(x, tag):
    x = layers.Dense(64, activation="relu", kernel_regularizer=L2, name=f"{tag}_h1")(x)
    x = layers.BatchNormalization(name=f"{tag}_hbn")(x)
    x = layers.Dropout(DROPOUT, name=f"{tag}_hdrop")(x)
    x = layers.Dense(N_EMOTIONS, name=f"{tag}_logits")(x)
    x = layers.Activation("sigmoid", name=f"{tag}_sig")(x)
    return ScaleOutput(scale=3.0, name=f"{tag}_out")(x)   # ← no Lambda


# =============================================================================
# MISA
# =============================================================================

@keras.saving.register_keras_serializable()
class OrthogonalityLoss(layers.Layer):
    def __init__(self, weight=0.1, **kw):
        super().__init__(**kw); self.weight = weight
    def call(self, inputs):
        inv, spec = inputs
        batch = tf.cast(tf.shape(inv)[0], tf.float32)
        gram  = tf.matmul(inv, spec, transpose_b=True)
        self.add_loss(self.weight * tf.reduce_sum(tf.square(gram)) / batch)
        return inv
    def get_config(self):
        base = super().get_config(); base.update({"weight": self.weight}); return base


@keras.saving.register_keras_serializable()
class SimilarityLoss(layers.Layer):
    def __init__(self, weight=0.1, **kw):
        super().__init__(**kw); self.weight = weight
    def call(self, inputs):
        inv_t, inv_a, inv_v = inputs
        nt = tf.nn.l2_normalize(inv_t, axis=-1)
        na = tf.nn.l2_normalize(inv_a, axis=-1)
        nv = tf.nn.l2_normalize(inv_v, axis=-1)
        loss = self.weight * (
            tf.reduce_mean(1.0 - tf.reduce_sum(nt * na, axis=-1)) +
            tf.reduce_mean(1.0 - tf.reduce_sum(nt * nv, axis=-1)) +
            tf.reduce_mean(1.0 - tf.reduce_sum(na * nv, axis=-1))
        ) / 3.0
        self.add_loss(loss)
        return inv_t
    def get_config(self):
        base = super().get_config(); base.update({"weight": self.weight}); return base


def build_misa():
    INV  = COMMON_DIM // 2
    SPEC = COMMON_DIM // 2

    inp_t = keras.Input(shape=(TEXT_DIM,),   name="text")
    inp_a = keras.Input(shape=(AUDIO_DIM,),  name="audio")
    inp_v = keras.Input(shape=(VISUAL_DIM,), name="visual")

    et = _proj(inp_t, COMMON_DIM, "misa_et")
    ea = _proj(inp_a, COMMON_DIM, "misa_ea")
    ev = _proj(inp_v, COMMON_DIM, "misa_ev")

    def invariant(x, tag):
        x = layers.Dense(INV*2, activation="tanh", kernel_regularizer=L2, name=f"{tag}_inv1")(x)
        return layers.Dense(INV, activation="tanh", name=f"{tag}_inv2")(x)

    def specific(x, tag):
        x = layers.Dense(SPEC*2, activation="relu", kernel_regularizer=L2, name=f"{tag}_spec1")(x)
        return layers.Dense(SPEC, activation="relu", name=f"{tag}_spec2")(x)

    inv_t  = invariant(et, "misa_t"); inv_a  = invariant(ea, "misa_a"); inv_v  = invariant(ev, "misa_v")
    spec_t = specific(et,  "misa_t"); spec_a = specific(ea,  "misa_a"); spec_v = specific(ev,  "misa_v")

    inv_t = OrthogonalityLoss(0.1, name="misa_orth_t")([inv_t, spec_t])
    inv_a = OrthogonalityLoss(0.1, name="misa_orth_a")([inv_a, spec_a])
    inv_v = OrthogonalityLoss(0.1, name="misa_orth_v")([inv_v, spec_v])
    inv_t = SimilarityLoss(0.1,    name="misa_sim")([inv_t, inv_a, inv_v])

    fused = layers.Concatenate(name="misa_fuse")([inv_t, inv_a, inv_v, spec_t, spec_a, spec_v])
    fused = layers.Dense(COMMON_DIM*2, activation="relu", name="misa_fc1")(fused)
    fused = layers.BatchNormalization(name="misa_fc1bn")(fused)
    fused = layers.Dropout(DROPOUT, name="misa_fc1drop")(fused)
    fused = layers.Dense(COMMON_DIM, activation="relu", name="misa_fc2")(fused)

    return keras.Model(
        [inp_t, inp_a, inp_v],
        outputs={"main":  _emotion_head(fused, "misa_main"),
                 "aux_t": _emotion_head(et,    "misa_auxt"),
                 "aux_a": _emotion_head(ea,    "misa_auxa"),
                 "aux_v": _emotion_head(ev,    "misa_auxv")},
        name="MISA")


# =============================================================================
# CTNet
# =============================================================================

def _transformer_encoder(x, tag, num_heads=4, ff_dim=256):
    d   = x.shape[-1]
    seq = layers.Reshape((1, d), name=f"{tag}_seq")(x)
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d//num_heads,
                                    dropout=DROPOUT, name=f"{tag}_sa")(seq, seq)
    att = layers.Add(name=f"{tag}_sa_add")([seq, att])
    att = layers.LayerNormalization(name=f"{tag}_sa_ln")(att)
    ff  = layers.Dense(ff_dim, activation="relu", name=f"{tag}_ff1")(att)
    ff  = layers.Dense(d, name=f"{tag}_ff2")(ff)
    out = layers.Add(name=f"{tag}_ff_add")([att, ff])
    out = layers.LayerNormalization(name=f"{tag}_ff_ln")(out)
    return layers.Reshape((d,), name=f"{tag}_flat")(out)


def _cross_modal_transformer(q, kv, tag, num_heads=4):
    d   = q.shape[-1]
    q3  = layers.Reshape((1, d), name=f"{tag}_q")(q)
    kv3 = layers.Reshape((1, d), name=f"{tag}_kv")(kv)
    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d//num_heads,
                                    dropout=DROPOUT, name=f"{tag}_cma")(q3, kv3)
    att = layers.Add(name=f"{tag}_add")([q3, att])
    att = layers.LayerNormalization(name=f"{tag}_ln")(att)
    return layers.Reshape((d,), name=f"{tag}_out")(att)


def build_ctnet():
    inp_t = keras.Input(shape=(TEXT_DIM,),   name="text")
    inp_a = keras.Input(shape=(AUDIO_DIM,),  name="audio")
    inp_v = keras.Input(shape=(VISUAL_DIM,), name="visual")

    et = _proj(inp_t, COMMON_DIM, "ctn_t")
    ea = _proj(inp_a, COMMON_DIM, "ctn_a")
    ev = _proj(inp_v, COMMON_DIM, "ctn_v")

    ht = _transformer_encoder(et, "ctn_te1")
    ha = _transformer_encoder(ea, "ctn_ae1")
    hv = _transformer_encoder(ev, "ctn_ve1")

    ct_ta = _cross_modal_transformer(ht, ha, "ctn_ta1")
    ct_av = _cross_modal_transformer(ha, hv, "ctn_av1")
    ct_vt = _cross_modal_transformer(hv, ht, "ctn_vt1")

    ht2 = layers.Add(name="ctn_t_res")([ht, ct_ta])
    ha2 = layers.Add(name="ctn_a_res")([ha, ct_av])
    hv2 = layers.Add(name="ctn_v_res")([hv, ct_vt])

    ct_at = _cross_modal_transformer(ha2, ht2, "ctn_at2")
    ct_va = _cross_modal_transformer(hv2, ha2, "ctn_va2")
    ct_tv = _cross_modal_transformer(ht2, hv2, "ctn_tv2")

    ht3 = layers.Add(name="ctn_t_res2")([ht2, ct_tv])
    ha3 = layers.Add(name="ctn_a_res2")([ha2, ct_at])
    hv3 = layers.Add(name="ctn_v_res2")([hv2, ct_va])

    concat = layers.Concatenate(name="ctn_concat")([ht3, ha3, hv3])
    gate   = layers.Dense(3*COMMON_DIM, activation="sigmoid", name="ctn_gate")(concat)
    gated  = layers.Multiply(name="ctn_gated")([gate, concat])
    fused  = layers.Dense(COMMON_DIM, activation="relu", name="ctn_fused")(gated)
    fused  = layers.BatchNormalization(name="ctn_fusedbn")(fused)
    fused  = layers.Dropout(DROPOUT, name="ctn_fuseddrop")(fused)

    return keras.Model([inp_t, inp_a, inp_v],
                       _emotion_head(fused, "ctn"),
                       name="CTNet")


# =============================================================================
# UniMSE
# =============================================================================

def build_unimse():
    SENT_DIM = 32

    inp_t = keras.Input(shape=(TEXT_DIM,),   name="text")
    inp_a = keras.Input(shape=(AUDIO_DIM,),  name="audio")
    inp_v = keras.Input(shape=(VISUAL_DIM,), name="visual")

    et = _proj(inp_t, COMMON_DIM, "ums_t")
    ea = _proj(inp_a, COMMON_DIM, "ums_a")
    ev = _proj(inp_v, COMMON_DIM, "ums_v")

    # Stack modality tokens — StackLayer replaces Lambda(tf.stack(...))
    tokens  = StackLayer(axis=1, name="ums_stack")([et, ea, ev])   # (B,3,COMMON_DIM)
    unified = layers.MultiHeadAttention(num_heads=4, key_dim=COMMON_DIM//4,
                                        dropout=DROPOUT, name="ums_unif_attn")(tokens, tokens)
    unified = layers.Add(name="ums_unif_add")([tokens, unified])
    unified = layers.LayerNormalization(name="ums_unif_ln")(unified)

    # Mean pool — ReduceMeanLayer replaces Lambda(tf.reduce_mean(...))
    unified_mean = ReduceMeanLayer(axis=1, name="ums_mean")(unified)   # (B,COMMON_DIM)

    sent_feat = layers.Dense(SENT_DIM*2, activation="relu", name="ums_sf1")(unified_mean)
    sent_feat = layers.Dense(SENT_DIM,   activation="tanh", name="ums_sf2")(sent_feat)
    sentiment = layers.Dense(1, activation="tanh", name="sentiment")(sent_feat)

    sent_exp = layers.Dense(COMMON_DIM, activation="relu", name="ums_sent_exp")(sent_feat)
    # Reshape for attention — Reshape replaces Lambda(tf.expand_dims(...))
    sent_3d  = layers.Reshape((1, COMMON_DIM), name="ums_sent_rs")(sent_exp)
    emo_3d   = layers.Reshape((1, COMMON_DIM), name="ums_emo_rs")(unified_mean)

    inter_att = layers.MultiHeadAttention(num_heads=4, key_dim=COMMON_DIM//4,
                                          dropout=DROPOUT, name="ums_inter_attn")(emo_3d, sent_3d)
    inter_att = layers.Reshape((COMMON_DIM,), name="ums_inter_flat")(inter_att)

    emo_feat = layers.Add(name="ums_emo_add")([unified_mean, inter_att])
    emo_feat = layers.LayerNormalization(name="ums_emo_ln")(emo_feat)
    emo_feat = layers.Dense(COMMON_DIM, activation="relu", name="ums_emo_fc")(emo_feat)
    emo_feat = layers.Dropout(DROPOUT, name="ums_emo_drop")(emo_feat)

    return keras.Model(
        [inp_t, inp_a, inp_v],
        outputs={"main": _emotion_head(emo_feat, "ums_main"),
                 "sentiment": sentiment},
        name="UniMSE")


# =============================================================================
# COMPILE HELPERS
# =============================================================================

def compile_misa(model, lr=1e-3):
    AUX = 0.2
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss={"main":"mse","aux_t":"mse","aux_a":"mse","aux_v":"mse"},
                  loss_weights={"main":1.0,"aux_t":AUX,"aux_a":AUX,"aux_v":AUX})
    return model

def compile_ctnet(model, lr=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

def compile_unimse(model, lr=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss={"main":"mse","sentiment":"mse"},
                  loss_weights={"main":1.0,"sentiment":0.1})
    return model


# =============================================================================
# DATASET WRAPPERS
# =============================================================================

def wrap_misa(dataset):
    def _expand(inputs, labels):
        return inputs, {"main":labels,"aux_t":labels,"aux_a":labels,"aux_v":labels}
    return dataset.map(_expand)

def wrap_unimse(dataset):
    def _expand(inputs, labels):
        valence = (labels[:,2] - (labels[:,0]+labels[:,1])/2.0) / 3.0
        valence = tf.expand_dims(valence, -1)
        return inputs, {"main":labels,"sentiment":valence}
    return dataset.map(_expand)


# =============================================================================
# CUSTOM OBJECTS
# =============================================================================

SOTA_CUSTOM_OBJECTS = {
    **CUSTOM_OBJECTS,
    "OrthogonalityLoss": OrthogonalityLoss,
    "SimilarityLoss":    SimilarityLoss,
    "ScaleOutput":       ScaleOutput,
    "StackLayer":        StackLayer,
    "ReduceMeanLayer":   ReduceMeanLayer,
    "L2NormalizeLayer":  L2NormalizeLayer,
    "ExpandDimsLayer":   ExpandDimsLayer,
}
