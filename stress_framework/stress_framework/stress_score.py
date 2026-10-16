# =============================================================================
# stress_score.py — Russell Circumplex, 6 emotions, S in [0,1], 3 levels
# =============================================================================
import numpy as np
from config import RUSSELL_COORDS,STRESS_LOW_MAX,STRESS_MODERATE_MAX,EMOTION_NAMES

EPS = 1e-6
_EMOTIONS = EMOTION_NAMES
_VALENCES = np.array([RUSSELL_COORDS[e]["valence"] for e in _EMOTIONS],dtype=np.float32)
_AROUSALS = np.array([RUSSELL_COORDS[e]["arousal"] for e in _EMOTIONS],dtype=np.float32)

class RussellStressCalculator:
    def compute(self,intensities):
        intensities=np.asarray(intensities,dtype=np.float32)
        scalar=intensities.ndim==1
        if scalar: intensities=intensities[np.newaxis,:]
        intensities=np.clip(intensities,0.0,3.0)
        total=intensities.sum(axis=1,keepdims=True)+EPS
        weights=intensities/total
        V=(weights*_VALENCES).sum(axis=1)
        A=(weights*_AROUSALS).sum(axis=1)
        S=np.clip(A*(1.0-V)/2.0,0.0,1.0)
        dom_idx=intensities.argmax(axis=1)
        results=[]
        for i in range(len(S)):
            s=float(S[i])
            results.append({"valence":float(V[i]),"arousal":float(A[i]),
                            "stress_score":s,"stress_label":self.label(s),
                            "is_high_stress":s>=STRESS_LOW_MAX,
                            "dominant_emotion":_EMOTIONS[dom_idx[i]],
                            "emotion_dict":{e:float(intensities[i,j]) for j,e in enumerate(_EMOTIONS)}})
        return results[0] if scalar else results

    def compute_from_predictions(self,preds):
        preds=np.clip(np.asarray(preds,dtype=np.float32),0.0,3.0)
        total=preds.sum(axis=1,keepdims=True)+EPS
        weights=preds/total
        V=(weights*_VALENCES).sum(axis=1)
        A=(weights*_AROUSALS).sum(axis=1)
        S=np.clip(A*(1.0-V)/2.0,0.0,1.0)
        return {"valence":V,"arousal":A,"stress_score":S,"is_high_stress":S>=STRESS_LOW_MAX}

    def label(self,score):
        if   score>=STRESS_MODERATE_MAX: return "Severe"
        elif score>=STRESS_LOW_MAX:      return "Moderate"
        else:                            return "Low"

    def level_thresholds(self):
        return {"Low":f"S<{STRESS_LOW_MAX}",
                "Moderate":f"{STRESS_LOW_MAX}≤S<{STRESS_MODERATE_MAX}",
                "Severe":f"S≥{STRESS_MODERATE_MAX}"}

stress_calc=RussellStressCalculator()
