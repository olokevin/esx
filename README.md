# esx

esx is the shared evolution-strategies library for the ES projects: Agentic-ESOpt, AgentES, AgentES-System,
ES-SEAgent, specialists_team, TreeES, ForkES and HarnessES. It provides:

- **Perturbation backends:** dense, Flipout, FuRA / FuRA-S, iso, LoRA, rotation and steering slots, and the MobileWorld
  FuRA-S worker.
- **Serving:** an in-process slot engine, plus clients for the `vllm serve` worker extension and the `/es/*` HTTP routes.
- **Estimators and run state:** entropic tilt, PUCT archive, sigma schedules, histories and replay.

Use it as a git submodule; see `MIGRATION.md`. The design, the decisions taken and the phase plan are in `DESIGN.md`.
The audit of all existing implementations is in `docs/AUDIT_2026-11-20.md`.

```bash
python -m pytest tests          # needs torch; CPU is enough
```
