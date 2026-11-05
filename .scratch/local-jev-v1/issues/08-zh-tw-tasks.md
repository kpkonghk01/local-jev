# 08: Traditional-Chinese Task set

**What to build:** The developer can benchmark on Traditional-Chinese States, as well as English ones, and compare directly. Build a committed copy of the 231 public jevbench Tasks with only the State translated into Traditional Chinese (Taiwan usage). Every Question, Label and expected Label stays the same.

**Blocked by:** 04 (jevbench runner)

**Status:** ready-for-agent

- [ ] All 231 public Tasks (easy, original, hard) are translated, with the same ids, Questions, Labels and expected Labels.
  - Only the State changes.
  - Where a State is JSON, only its human-language string values are translated; keys and structure stay as they are.
- [ ] The set carries provenance (source commit, translation note) and jevbench's MIT attribution.
- [ ] An automated integrity test checks the set against the source at the pinned commit:
  - ids, Questions, Labels and expected Labels are identical;
  - every State is non-empty and contains Traditional-Chinese text.
- [ ] `local-jev bench` can select the Traditional-Chinese set.
- [ ] The translation note lists 10 Task ids for the developer to spot-check. This spot-check is a later human step and does not block this ticket.
