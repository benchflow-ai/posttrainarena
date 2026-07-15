# Support

Use the channel that matches the request:

- **Environment authoring and task-contract questions:** open a GitHub issue
  with a minimal task-package example, or ask in the project Discord.
- **Training-pipeline bugs:** open a GitHub issue with the recipe revision,
  redacted or public task IDs, command, sanitized logs, and whether the failure
  reproduces with `--dry-run`. Never include sealed-eval IDs or private task
  contents.
- **OpenEnv integration:** first check
  [`docs/architecture-status.md`](docs/architecture-status.md). The served
  adapter, typed client, lifecycle tests, and Docker parity path are
  implemented. Public bug reports may include the sanitized `openenv-serve`
  command and environment backend, but must redact private task IDs, state
  contents, credentials, and artifact paths. Report any possible private-data
  exposure through the security channel below.
- **Competition rules or private submission questions:** email
  `labs@benchflow.ai`.
- **Security, leaked credentials, sandbox escapes, or private eval exposure:**
  follow [`SECURITY.md`](SECURITY.md) and do not file a public issue.

Before requesting help, run the relevant no-spend checks documented in
[`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/training-pipeline.md`](docs/training-pipeline.md). Remove secrets and
private task content from all logs.

This repository does not currently offer a production support SLA.
