"""
agent.py — Booking agent using LOCAL Mistral-7B (no API key needed)
Falls back to rule-based responses if model not loaded yet.
"""

import json, os, logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

def load_psychologists():
    with open("data/psychologists.json") as f:
        return json.load(f)


# =============================================================================
# Local Mistral loader
# =============================================================================

class LocalMistral:
    """Lazy-loads Mistral-7B-Instruct from HuggingFace on first use."""

    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self):
        self._model = None
        self._tok   = None
        self._ready = False

    def _load(self):
        if self._ready:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            log.info("Loading local Mistral-7B (first call — may take 2-3 min)...")
            self._tok   = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self._ready = True
            log.info("Mistral-7B loaded successfully.")
        except Exception as e:
            log.warning("Could not load Mistral: %s — using rule-based fallback.", e)
            self._ready = False

    def generate(self, system_prompt: str, user_message: str,
                 max_new_tokens: int = 250, temperature: float = 0.7) -> str:
        self._load()
        if not self._ready:
            return None   # caller will use fallback

        import torch
        prompt = (f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                  f"{user_message} [/INST]")
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tok.eos_token_id,
            )
        return self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()


# Singleton — loaded once, reused across requests
_mistral = LocalMistral()


# =============================================================================
# Booking Agent
# =============================================================================

class BookingAgent:

    def __init__(self):
        self.psychologists = load_psychologists()

    # ── Search ────────────────────────────────────────────────────────────
    def search_psychologists(self, city=None, max_fee=None,
                              specialization=None) -> list:
        results = self.psychologists
        if city:
            results = [p for p in results
                       if p["city"].lower() == city.lower()]
        if max_fee:
            results = [p for p in results if p["fee_pkr"] <= max_fee]
        if specialization:
            kw = specialization.lower()
            results = [p for p in results
                       if kw in p["specialization"].lower()]
        return sorted(results, key=lambda x: x["rating"], reverse=True)

    # ── Available slots ───────────────────────────────────────────────────
    def get_available_slots(self, psych_id: str, days_ahead: int = 7) -> list:
        psych = next((p for p in self.psychologists if p["id"] == psych_id), None)
        if not psych:
            return []
        day_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        avail   = [day_map[d] for d in psych["availability"] if d in day_map]
        slots   = []
        today   = datetime.now()
        for i in range(1, days_ahead + 1):
            d = today + timedelta(days=i)
            if d.weekday() in avail:
                slots.append({
                    "date":         d.strftime("%Y-%m-%d"),
                    "date_display": d.strftime("%A, %d %B %Y"),
                    "times":        ["10:00 AM", "12:00 PM", "3:00 PM"],
                })
        return slots[:4]

    # ── Main entry point ──────────────────────────────────────────────────
    def process_booking_request(self, user_message: str,
                                 session_id: str,
                                 context: str = "") -> dict:
        ml = user_message.lower()

        if any(w in ml for w in ["book","appointment","schedule","fix","reserve"]):
            return self._booking_intent(user_message, session_id, context)

        if any(w in ml for w in ["show","list","find","search",
                                   "recommend","suggest","psychologist","doctor"]):
            return self._search_intent(user_message)

        if any(w in ml for w in ["cancel","reschedule"]):
            return self._cancel_intent(session_id)

        return self._general(user_message, context)

    # ── Intent handlers ───────────────────────────────────────────────────
    def _extract_city(self, text: str) -> Optional[str]:
        for city in ["islamabad","lahore","karachi","faisalabad","multan"]:
            if city in text.lower():
                return city.capitalize()
        return None

    def _search_intent(self, msg: str) -> dict:
        city   = self._extract_city(msg)
        psychs = self.search_psychologists(city=city)[:4]
        if not psychs:
            return {"action": "text",
                    "message": ("I couldn't find matching psychologists. "
                                "Please specify a city: Islamabad, Lahore, "
                                "Karachi, Faisalabad, or Multan.")}
        return {
            "action": "show_psychologists",
            "message": (f"Here are top-rated psychologists"
                        f"{' in '+city if city else ''} "
                        f"with ratings and fees:"),
            "psychologists": psychs,
        }

    def _booking_intent(self, msg: str, session_id: str,
                         context: str) -> dict:
        city   = self._extract_city(msg)
        psychs = self.search_psychologists(city=city)[:3]
        if not psychs:
            return {
                "action": "ask_city",
                "message": ("Which city would you prefer for your appointment? "
                             "Islamabad, Lahore, Karachi, Faisalabad, or Multan?"),
            }
        slots = self.get_available_slots(psychs[0]["id"])
        return {
            "action":          "show_booking_options",
            "message":         (f"I found available psychologists"
                                f"{' in '+city if city else ''}. "
                                 "Select one to see available slots:"),
            "psychologists":   psychs,
            "slots":           slots,
            "selected_psych":  psychs[0],
        }

    def _cancel_intent(self, session_id: str) -> dict:
        from rag_memory import get_bookings
        active = [b for b in get_bookings(session_id)
                  if b.get("status") == "pending"]
        if not active:
            return {"action": "text",
                    "message": "You don't have any active bookings to cancel."}
        return {
            "action":   "show_bookings_for_cancel",
            "message":  "Here are your active bookings. Which one would you like to cancel?",
            "bookings": active,
        }

    def _general(self, msg: str, context: str) -> dict:
        """Use local Mistral if available, else rule-based."""
        system = f"""You are a compassionate mental health support assistant
for a Pakistani stress detection app.

User context from previous sessions:
{context}

Help the user with:
- Finding psychologists in their city (Islamabad, Lahore, Karachi, Faisalabad, Multan)
- Booking appointments
- Mental health resources and coping strategies
- Understanding their stress analysis results

Keep responses warm, brief (2-3 sentences), and supportive.
Always respond in a culturally sensitive way for a Pakistani audience.
If the user seems distressed, acknowledge their feelings first."""

        response = _mistral.generate(system, msg, max_new_tokens=200,
                                      temperature=0.7)
        if response:
            return {"action": "text", "message": response}

        # Rule-based fallback
        ml = msg.lower()
        if "stress" in ml or "anxious" in ml or "worried" in ml:
            return {
                "action": "text",
                "message": ("I hear you — stress can feel overwhelming. "
                             "Would you like me to find a professional "
                             "psychologist near you, or explore some "
                             "self-help resources?"),
            }
        if "resource" in ml or "book" in ml or "learn" in ml:
            return {"action": "text",
                    "message": "Check out our Resources page for curated books, "
                               "free courses, and blogs on stress management."}
        return {
            "action": "text",
            "message": ("I'm here to help. I can find psychologists in your "
                        "city, help you book an appointment, or recommend "
                        "stress management resources. What would you like?"),
        }

    # ── Confirm booking ───────────────────────────────────────────────────
    def confirm_booking(self, session_id: str, psych_id: str,
                         date: str, time: str, notes: str = "") -> dict:
        from rag_memory import save_booking
        psych = next((p for p in self.psychologists if p["id"] == psych_id), None)
        if not psych:
            return {"success": False, "message": "Psychologist not found."}

        bid = save_booking(session_id, psych_id, psych["name"], date, time, notes)
        return {
            "success":    True,
            "booking_id": bid,
            "message":    (f"Appointment confirmed with {psych['name']} "
                           f"on {date} at {time}. "
                           f"Clinic: {psych['clinic']}. "
                           f"Phone: {psych['phone']}."),
            "details": {
                "psychologist": psych["name"],
                "clinic":       psych["clinic"],
                "date":         date,
                "time":         time,
                "phone":        psych["phone"],
                "fee":          psych["fee_pkr"],
            },
        }
