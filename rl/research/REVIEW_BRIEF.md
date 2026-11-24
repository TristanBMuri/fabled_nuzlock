# Round 2: cross-review (collaborative merge input)

Three independent documents exist in this folder, all answering BRIEF.md:
- `fable.md` (Fable), `neon.md` (Gemini Neon), `lead.md` (the lead's research notes, no plan).

You wrote one of the first two. Now read **the other plan and lead.md** (re-read your own as needed)
and write a review to the file named in your task line. Verify claims against the repo code where
they matter (cite file:line); read-only, no battles, nothing over ~2 minutes of compute.

Write, in this order:
1. **Agreements** — points both plans share (these become the merged plan's backbone).
2. **Where the other plan is better than yours** — concede concretely; say what you'd adopt.
3. **Where you think the other plan is wrong or over-built** — with evidence (code, data, logic).
   Keep the CLAUDE.md minimal-code rule and the hard-won lessons in mind.
4. **Factual disputes** — claims in either plan you checked and found true/false (e.g. the cause of
   the RR-AI search loss, whether `--chooser searched` is wired into `box_job`, feature leaks).
5. **Feature engineering** — your merged view of the feature spec: what to keep, drop, add.
6. **Your proposed merged phase order** — a short list, each with its gate, and which parts of
   lead.md's ideas you would include or reject (TabPFN, sequential halving, GRPO builder,
   LLM-evolved heuristics, MAP-Elites archive, BRExIt-style opponent search).
Keep it under ~250 lines. End with a ≤10-line summary in your final message.
