# Papercuts archive

<!-- papercuts-generated-view: {"kind":"archive","ledger":"PAPERCUTS-LEDGER.md","manifest":"PAPERCUTS-TRIAGE.json","ledgerSha256":"7636064e88e8775582936ac4bc06f8536f9f6f0121ce94c067965defaff0ce27","manifestSha256":"b97a27287b705244af6f2150820a8336da8c2f229d14a0431c5054398e82be78","sourceEntryCount":34,"newestSourceTimestamp":"2026-07-27T15:17:23Z"} -->
> **Generated — do not edit.** Closed dispositions come from `PAPERCUTS-TRIAGE.json`; raw history remains in `PAPERCUTS-LEDGER.md`.

| ID | Final status | Count / date range | Task | Issue | Owner | Evidence / rationale |
|---|---|---|---|---|---|---|
| AKB-PC-006 | upstream | 1 · 2026-07-25T08:39:39Z | Bun blocked-postinstall warning lacks actionable package detail |  | bun | The warning is emitted by Bun during dependency installation; AgentKB cannot add the missing package attribution to Bun's installer output. |
| AKB-PC-010 | accepted | 1 · 2026-07-25T09:07:10Z | Expected rg no-match status stopped unrelated shell checks |  | agent-practice | This was a one-off command-composition mistake rather than durable AgentKB behavior; future probes should not join optional rg checks with &&. |
| AKB-PC-013 | accepted | 3 · 2026-07-25T13:59:11Z — 2026-07-27T05:27:54Z | Broad mechanical edits caused transient patch and verification mistakes |  | agent-practice | The context misses, wrong working directory, and stray line were caught and corrected in the originating sessions; they do not identify one remaining repository defect. |
| AKB-PC-019 | upstream | 1 · 2026-07-27T00:36:17Z | Safe cleanup of validated temporary files was blocked |  | command-policy | The rejection came from the execution policy around destructive commands, outside AgentKB; OS cleanup of the unique temporary directory remains the safe workaround. |
