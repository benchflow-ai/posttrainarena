# OpenCode evaluation canary

On July 12, 2026, the migrated evaluator ran the real SkillsBench
`3d-scan-calc` task through BenchFlow, OpenCode, and Daytona.

## Contract

- BenchFlow: `6eaa14344bd835a3c2c5c31a31470ef994b24a80`
- Agent: `opencode`
- Model: `glm/glm-5.1`
- Sandbox: `daytona`
- Skill mode: `no-skill`
- Usage tracking: `required`
- Selected tasks: exactly one

The pipeline invoked the same production command builder used by baseline,
training-gate, final, and benchmark-matrix evaluation. Endpoint credentials
were supplied only through process environment variables.

## Result

- Score: `1/1`
- Agent errors: `0`
- Verifier errors: `0`
- Telemetry coverage: `100%`
- Total tokens: `134,340`
- Tool calls: `13`
- `results.jsonl` rows: `1`
- `llm_trajectory.jsonl` rows: `13`
- Training-ready result: `true`
- Missing or malformed LLM trajectories: `0`
- Unscored or zero-tool rows: `0`

This proves the OpenCode evaluation command, endpoint environment mapping,
Daytona execution, verifier scoring, and fail-closed artifact-health checks.
The later Qwen3.5 Data Agent run additionally validates synchronized SFT and
GRPO checkpoints plus final held-out evaluation; see
[`qwen35-data-agent-e2e-canary.md`](qwen35-data-agent-e2e-canary.md).
