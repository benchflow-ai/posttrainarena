# Security policy

## Report a vulnerability

Do not open a public issue for vulnerabilities, leaked credentials, sandbox
escapes, verifier bypasses, prompt or dataset leakage, or unsafe task assets.
Email `labs@benchflow.ai` with:

- the affected path, version, or commit
- reproduction steps and expected impact
- whether credentials, private evaluation data, or third-party systems are at risk
- a safe contact method for follow-up

We will acknowledge a complete report within five business days and coordinate
disclosure after a fix or mitigation is available.

## Scope

Security-sensitive surfaces include:

- participant-provided Dockerfiles, scripts, skills, and task assets
- verifier and oracle isolation
- Daytona or Docker sandbox lifecycle
- provider and Hugging Face credentials
- private held-out evaluation data
- generated trajectories, checkpoints, and model artifacts
- protocol adapters, including the implemented OpenEnv server/client boundary

Never commit secrets, raw provider responses containing secrets, checkpoints,
private eval tasks, or unreviewed job dumps. Use environment variables or a
secret manager, pin external revisions, and keep generated runs in ignored
directories.

The repository includes `posttrainarena-train openenv-serve`, which wraps
BenchFlow-backed tasks behind the OpenEnv client/server protocol. Operators
must keep private task snapshots and verifier credentials server-side, bind the
raw service to loopback or a private network, and place any remote access behind
encrypted authenticated ingress plus network allowlisting. The command has no
built-in authentication and must never be exposed directly to the public
internet. Preserve sandbox isolation across resets and run the lifecycle plus
Docker-parity tests before changing this boundary.

The competition is a proposal and does not currently provide a production SLA.
