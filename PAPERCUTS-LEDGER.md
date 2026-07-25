# Papercuts

## 2026-07-25T06:32:24Z · gpt-5.6-sol · joshuaizzard · agentkb:. · 019f978a-4bd0-7391-aac7-fa4e8c0582d8

Running the documented/default AgentKB verification path with `uv run pytest` failed because pytest is not declared in project dependencies or a dev group; use `uv run --with pytest pytest` for now and consider adding a test dependency group.

## 2026-07-25T07:46:51Z · unknown · joshuaizzard · agentkb:. · 019f983d-707b-7fb3-8526-47a5be20201f

AgentKB Modal benchmark preflight → the documented-looking `modal` invocation is not on PATH (likely must run through the project environment, e.g. `uv run modal`).

## 2026-07-25T07:49:26Z · unknown · joshuaizzard · agentkb:. · 019f983d-707b-7fb3-8526-47a5be20201f

Modal 1.5.3 follow-up app validation → `max_inputs=1` still works but emits a deprecation error; the supported fresh-container setting is `single_use_containers=True` on the class.

## 2026-07-25T08:02:17Z · unknown · joshuaizzard · agentkb:. · 019f983d-707b-7fb3-8526-47a5be20201f

Remote AgentKB generation build → the existing modal_client writes the requested private JSON file but also echoes the entire payload to stdout; callers need a summary-only print to honor raw-result privacy.

## 2026-07-25T08:20:33Z · unknown · joshuaizzard · agentkb:. · 019f985a-37b4-7431-bbed-53b517adea57

Modal benchmark closeout process scan → pgrep -af returned only the verification shell PID because the exact resource-name pattern appeared in that shell command; use ps output filtered to executable names or a pattern kept out of argv for an unambiguous check.

## 2026-07-25T08:39:39Z · unknown · joshuaizzard · agentkb:. · 019f986a-0acd-74a3-b84a-9697ffc7c513

Installing the new Modal TypeScript control-plane dependencies with Bun → Bun reported one blocked postinstall without identifying the package or whether it matters; surface the package name and impact directly in install output.

## 2026-07-25T08:40:17Z · unknown · joshuaizzard · agentkb:. · 019f986a-0acd-74a3-b84a-9697ffc7c513

Validating the Modal adapter locally → Modal 1.5.3 crashes on a modal.parameter annotation when postponed annotations turn str into a forward-reference string, and this repo has no bare python executable on PATH (uv run python is required); document the annotation limitation and standardize verification commands on uv.

## 2026-07-25T08:42:09Z · unknown · joshuaizzard · agentkb:. · 019f986a-0acd-74a3-b84a-9697ffc7c513

Cleaning verification artifacts → compileall created an unignored __pycache__ inside the new package, and the command runner rejected removal of that exact generated directory via rm even though the safer trash command succeeded; prefer compile checks that suppress bytecode and suggest trash for generated cleanup.
