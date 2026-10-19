"""CSE476 - Multi-agent timetable negotiation.
Run:  python multiagent.py            (rule-based agents, no API needed)
      GROQ_API_KEY=... python multiagent.py --llm   (agents + baseline use an LLM)
"""
import json, os, sys, copy

# load GROQ_API_KEY from a .env file next to this script (no extra package needed)
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _l in open(_env):
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.strip().split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip('"\''))

USE_LLM = "--llm" in sys.argv
if USE_LLM and not os.getenv("GROQ_API_KEY"):
    sys.exit("GROQ_API_KEY not set. Put GROQ_API_KEY=gsk_... in a .env file next to this script, or export it first.")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------- DATA ----------------
SLOTS = [d + str(i) for d in "MT" for i in range(1, 6)]          # M1..M5,T1..T5
ROOMS = {"R1": 60, "R2": 60, "R3": 40, "LAB1": 30}
SECTIONS = {  # name: (dept, strength, faculty, is_lab)
    "A1": ("A", 55, "F1", False), "A2": ("A", 35, "F2", False),
    "A3": ("A", 30, "F1", True),  "A4": ("A", 40, "F3", False),
    "B1": ("B", 60, "F4", False), "B2": ("B", 38, "F5", False),
    "B3": ("B", 28, "F4", True),  "B4": ("B", 50, "F6", False),
    "C1": ("C", 45, "F7", False), "C2": ("C", 30, "F8", False),
    "C3": ("C", 58, "F7", False), "C4": ("C", 25, "F9", False),
}
DEPT_PREFS = {
    "A": "F1 avoids Monday; no class in M5",
    "B": "F4 prefers slots 1 to 2; B1 in R1",
    "C": "F7 at most 2 classes a day; avoid T5",
}
OFFLINE = set()   # {(room, slot)} filled by disruption

# ---------------- (a) VALIDATOR (plain python) ----------------
def occupied(slot, is_lab):
    """slots a class occupies; None if lab doesn't fit in the day"""
    i = SLOTS.index(slot)
    if not is_lab: return [slot]
    if i % 5 == 4: return None
    return [SLOTS[i], SLOTS[i + 1]]

def validate_schedule(sched, offline=frozenset()):
    """sched: {section: {"slot": "M1", "room": "R1"}} -> list of violation strings"""
    v, room_use, fac_use = [], {}, {}
    for s, (dept, n, fac, lab) in SECTIONS.items():
        a = sched.get(s)
        if not a or a.get("slot") not in SLOTS or a.get("room") not in ROOMS:
            v.append(f"{s}: missing/invalid assignment"); continue
        slots = occupied(a["slot"], lab)
        if slots is None:
            v.append(f"{s}: lab does not fit 2 consecutive slots same day"); continue
        if lab and a["room"] != "LAB1": v.append(f"{s}: lab must be in LAB1")
        if not lab and a["room"] == "LAB1": v.append(f"{s}: non-lab in LAB1")
        if n > ROOMS[a["room"]]: v.append(f"{s}: strength {n} > capacity {ROOMS[a['room']]} of {a['room']}")
        for sl in slots:
            if (a["room"], sl) in offline: v.append(f"{s}: room {a['room']} offline at {sl}")
            if (a["room"], sl) in room_use: v.append(f"ROOM CLASH {a['room']}@{sl}: {room_use[(a['room'], sl)]} & {s}")
            room_use[(a["room"], sl)] = s
            if (fac, sl) in fac_use: v.append(f"FACULTY CLASH {fac}@{sl}: {fac_use[(fac, sl)]} & {s}")
            fac_use[(fac, sl)] = s
    return v

def count_preferences(sch):
    """6 private preferences -> number satisfied"""
    g = lambda s: sch[s]["slot"]; num = lambda s: int(g(s)[1])
    occ = lambda s: occupied(g(s), SECTIONS[s][3])
    p = [
        all(g(s)[0] != "M" for s in ("A1", "A3")),                       # F1 avoids Monday
        all("M5" not in occ(s) for s in "A1 A2 A3 A4".split()),          # no class in M5 (dept A)
        all(num(s) in (1, 2) and (num(s) + SECTIONS[s][3]) <= 2 for s in ("B1", "B3")),  # F4 slots 1-2
        sch["B1"]["room"] == "R1",                                       # B1 in R1
        all(sum(1 for s in ("C1", "C3") if g(s)[0] == d) <= 2 for d in "MT"),  # F7 <=2/day
        all("T5" not in occ(s) for s in "C1 C2 C3 C4".split()),          # C avoids T5
    ]
    return sum(p), p

# ---------------- LLM helper ----------------
PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]

def _groq(payload):
    import requests
    return requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"].strip()}, json=payload, timeout=60)

def _pick_available_model():
    """Ask Groq which models this key can use and pick the best one."""
    import requests
    r = requests.get("https://api.groq.com/openai/v1/models",
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"].strip()}, timeout=30)
    ids = [m["id"] for m in r.json().get("data", [])]
    print("[groq] models available to your key:", ids)
    for m in PREFERRED:
        if m in ids: return m
    chat = [i for i in ids if "whisper" not in i and "guard" not in i and "tts" not in i]
    if chat: return chat[0]
    raise RuntimeError("No usable chat model for this key")

def llm(system, user):
    """Groq chat completion (OpenAI-compatible). Returns parsed JSON."""
    global MODEL
    for attempt in range(3):
        payload = {"model": MODEL, "temperature": 0.2, "max_completion_tokens": 6000,
                   "messages": [{"role": "system", "content": system + " Reply with a single JSON object only, no prose."},
                                {"role": "user", "content": user}]}
        if "gpt-oss" in MODEL: payload["reasoning_effort"] = "low"   # reasoning models: keep thinking short
        r = _groq(payload)
        if r.ok:
            txt = r.json()["choices"][0]["message"].get("content") or ""
            try: return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            except Exception: continue                                # empty / bad JSON -> retry once more
        if r.status_code == 404 and attempt == 0:
            MODEL = _pick_available_model(); print("[groq] switching to model:", MODEL); continue
        raise RuntimeError(f"Groq {r.status_code} (model={MODEL}): {r.text[:300]}")
    raise RuntimeError("Groq returned no valid JSON after retries")

# ---------------- message bus (logging) ----------------
LOG = []
def send(frm, to, typ, rnd, body):
    m = {"id": len(LOG) + 1, "round": rnd, "from": frm, "to": to, "type": typ, "body": body}
    LOG.append(m); return m

# ---------------- AGENTS ----------------
class RoomManager:
    """Controls rooms; knows rooms/capacity/offline only, not department prefs."""
    prompt = "You are the Room Manager. You only know rooms, capacities and availability."
    def __init__(self): self.use = {}          # (room, slot) -> section
    def release(self, sec): self.use = {k: v for k, v in self.use.items() if v != sec}
    def allocate(self, sec, slots, strength, lab, room_pref=None):
        cands = ["LAB1"] if lab else sorted([r for r in ROOMS if r != "LAB1" and ROOMS[r] >= strength],
                                            key=lambda r: ROOMS[r])
        if room_pref in cands: cands.remove(room_pref); cands.insert(0, room_pref)
        for r in cands:
            if ROOMS[r] >= strength and all((r, s) not in self.use and (r, s) not in OFFLINE for s in slots):
                for s in slots: self.use[(r, s)] = sec
                return r
        return None

class DeptAgent:
    def __init__(self, dept):
        self.dept = dept
        self.secs = {s: v for s, v in SECTIONS.items() if v[0] == dept}   # ONLY its own data
        self.prefs = DEPT_PREFS[dept]
        self.rejected = {s: set() for s in self.secs}
        self.prompt = (f"You are Department Agent {dept}. Your sections: {json.dumps(self.secs)}. "
                       f"Private preferences: {self.prefs}. You know nothing about other departments. "
                       "Reply ONLY JSON: {\"proposals\":[{\"section\":..,\"slot\":..,\"room_pref\":null|\"R1\"}]}. "
                       "Avoid your own faculty clashes; labs need 2 consecutive slots same day; slots: " + ",".join(SLOTS))
    def cost(self, s, slot):
        _, n, fac, lab = SECTIONS[s]; i = int(slot[1]); c = i * 0.1 + (0 if slot[0] == "M" else 0.05)
        if self.dept == "A":
            if fac == "F1" and slot[0] == "M": c += 10
            if "M5" in (occupied(slot, lab) or []): c += 10
        if self.dept == "B" and fac == "F4": c += 0 if (i + lab) <= 2 else 10
        if self.dept == "C" and "T5" in (occupied(slot, lab) or []): c += 10
        return c
    def propose(self, todo, current, rnd, hint=None):
        if USE_LLM:
            try:
                out = llm(self.prompt, json.dumps({"round": rnd, "reschedule": todo, "rejected": {s: sorted(self.rejected[s]) for s in todo},
                                                   "already_fixed": {s: current[s]["slot"] for s in current if s in self.secs}}))
                props = out["proposals"]
                if all(p["slot"] not in self.rejected[p["section"]] for p in props): return props
            except Exception as e: print("LLM fallback", self.dept, e)
        # rule-based: lowest-cost slot, not rejected, no own-faculty clash
        taken = {}
        for s, a in current.items():
            if s in self.secs and s not in todo:
                for sl in occupied(a["slot"], SECTIONS[s][3]): taken[(SECTIONS[s][2], sl)] = s
        props = []
        for s in todo:
            _, n, fac, lab = SECTIONS[s]
            for slot in sorted(SLOTS, key=lambda x: self.cost(s, x)):
                occ = occupied(slot, lab)
                if not occ or slot in self.rejected[s] or any((fac, o) in taken for o in occ): continue
                if self.dept == "C" and fac == "F7":      # at most 2 classes a day
                    if sum(1 for (f, sl) in taken if f == "F7" and sl[0] == slot[0]) >= 2: continue
                for o in occ: taken[(fac, o)] = s
                props.append({"section": s, "slot": slot, "room_pref": "R1" if s == "B1" else None}); break
        return props

class Mediator:
    """Sees proposals only, never private preferences."""
    prompt = "You are the Mediator. Collect proposals, ask the Room Manager for rooms, run validator, send CONFLICT or COMMIT."

def run(max_rounds=8, disrupt_round=3):
    rm, depts, med = RoomManager(), {d: DeptAgent(d) for d in "ABC"}, Mediator()
    sched, rounds, moved = {}, 0, set()
    todo = {d: list(depts[d].secs) for d in depts}
    for rnd in range(1, max_rounds + 1):
        rounds = rnd
        before = copy.deepcopy(sched)
        if rnd == disrupt_round:
            OFFLINE.update(("R2", s) for s in SLOTS if s[0] == "T")
            send("RoomManager", "Mediator", "CONFLICT", rnd, {"reason": "R2 offline on Tuesday"})
            for s, a in sched.items():
                if a["room"] == "R2" and any(("R2", o) in OFFLINE for o in occupied(a["slot"], SECTIONS[s][3])):
                    todo[SECTIONS[s][0]].append(s); rm.release(s)
        allprops = []
        for d, ag in depts.items():
            if not todo[d]: continue
            send("Mediator", f"Dept{d}", "CFP", rnd, {"sections": todo[d]})
            props = ag.propose(todo[d], sched, rnd)
            send(f"Dept{d}", "Mediator", "COUNTER" if ag.rejected_any() else "PROPOSE", rnd, {"proposals": props})
            allprops += [(d, p) for p in props]; todo[d] = []
        # Mediator -> Room Manager: rooms with a stated room preference first, then largest strength
        allprops.sort(key=lambda x: (not x[1].get("room_pref"), -SECTIONS[x[1]["section"]][1]))
        for d, p in allprops:
            s = p["section"]; _, n, fac, lab = SECTIONS[s]; ag = depts[d]
            rm.release(s); occ = occupied(p["slot"], lab)
            room = rm.allocate(s, occ, n, lab, p.get("room_pref")) if occ else None
            if room:
                sched[s] = {"slot": p["slot"], "room": room}
                send("RoomManager", "Mediator", "ACCEPT", rnd, {"section": s, **sched[s]})
            else:
                sched.pop(s, None); ag.rejected[s].add(p["slot"])
                send("Mediator", f"Dept{d}", "CONFLICT", rnd, {"section": s, "slot": p["slot"], "reason": "no free room"})
        if rnd >= disrupt_round:
            moved |= {s for s in sched if before.get(s) != sched[s] and s in before}
        viol = validate_schedule(sched, OFFLINE)
        for d, ag in depts.items():          # validator-driven CONFLICT -> redo only offending sections
            for s in ag.secs:
                if s not in sched or any(s in x for x in viol):
                    if s not in todo[d]: todo[d].append(s)
                    sched.pop(s, None); rm.release(s)
        send("Mediator", "ALL", "ACCEPT" if not viol else "CONFLICT", rnd, {"violations": viol})
        if not viol and rnd >= disrupt_round:
            send("Mediator", "ALL", "COMMIT", rnd, {"schedule": sched}); break
    return sched, rounds, len(LOG), len(moved)

DeptAgent.rejected_any = lambda self: any(self.rejected.values())

# ---------------- (b) SINGLE-LLM BASELINE ----------------
def baseline():
    if not (USE_LLM and os.getenv("GROQ_API_KEY")):
        print("[baseline] needs --llm and GROQ_API_KEY; skipping"); return None
    p = (f"Create a timetable. Slots {SLOTS}; rooms {ROOMS}; sections {json.dumps(SECTIONS)} "
         f"(dept,strength,faculty,is_lab); prefs {json.dumps(DEPT_PREFS)}. Rules: one section per room/slot, one per faculty/slot, "
         "strength<=capacity, labs only in LAB1 for 2 consecutive slots. Reply ONLY JSON {\"A1\":{\"slot\":\"M1\",\"room\":\"R1\"},...} (lab slot = first slot).")
    try: return llm("You are a timetable scheduler.", p)
    except Exception as e: print("[baseline] failed:", e); return None

if __name__ == "__main__":
    base = baseline(); bv = validate_schedule(base) if base else None
    final, rounds, nmsg, moved = run()
    fv = validate_schedule(final, OFFLINE)
    json.dump(LOG, open("message_log.json", "w"), indent=2)
    print("\nFINAL TIMETABLE"); [print(f"  {s}: {a['slot']} {a['room']}") for s, a in sorted(final.items())]
    print("\n| Measure | Single LLM | Multi-agent |\n|---|---|---|")
    print(f"| Rule violations | {len(bv) if bv is not None else '(run --llm)'} | {len(fv)} |")
    print(f"| Preferences (of 6) | {count_preferences(base)[0] if base and not bv else 'n/a' if base else '(run --llm)'} | {count_preferences(final)[0]} |")
    print(f"| Rounds | NA | {rounds} |\n| Messages | NA | {nmsg} |\n| Sections moved after disruption | NA | {moved} |")
    print("\nviolations:", fv, "| prefs detail:", count_preferences(final)[1])
