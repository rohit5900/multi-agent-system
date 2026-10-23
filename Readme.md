# CSE476 Practical: Multi-Agent Timetable System

## Files
- `multiagent.py` - full code (validator, baseline, agents, negotiation, disruption repair)
- `message_log.json` - every message exchanged (CFP, PROPOSE, COUNTER, ACCEPT, CONFLICT, COMMIT)
- `CSE476_Multiagent_Results.xlsx` - final timetable, preferences, results table

## How to run
    python multiagent.py                                  # rule-based agents
    GROQ_API_KEY="..." python multiagent.py --llm      # LLM agents + single-LLM baseline

## Design
- (a) `validate_schedule()`: plain Python, returns every violation (room clash, faculty clash, capacity, lab rules, offline room).
- (b) `baseline()`: one LLM sees all data; output validated.
- (c) Agents: 3 `DeptAgent`s (each holds only its own sections/preferences), `Mediator`, `RoomManager`, each with its own prompt.
- (d) Negotiation: Mediator sends CFP -> departments PROPOSE/COUNTER -> Room Manager allocates rooms (ACCEPT) or Mediator sends CONFLICT -> validator runs each round -> COMMIT at zero violations or after 8 rounds. All messages are logged.
- (e) Disruption: at round 3 R2 goes offline on Tuesday; only sections in R2 on Tuesday are re-proposed (same slot in another room first, to minimise moves).

## Results (Groq LLM run)
| Measure | Single LLM | Multi-agent |
|---|---|---|
| Rule violations | 0 | 0 |
| Preferences satisfied (of 6) | 4 | 5 |
| Rounds taken | NA | 3 |
| Messages exchanged | NA | 29 |
| Sections moved after disruption | NA | 0 |

Unmet preference: F4 slots 1-2 (lab B3 placed at M2). 0 sections moved because none sat in R2 on Tuesday at round 3.

## Conclusion (2 lines)
Multi-agent performed better: both reached 0 violations, but multi-agent satisfied more preferences (5 vs 4) and is guaranteed valid by the validator feedback loop.
It also repairs a room outage by re-planning only affected sections, which a single LLM pass cannot do reliably.
