# Agent guide for this repo

Read [docs/ENGINEERING-LOG.md](docs/ENGINEERING-LOG.md) first. It covers the goal, the branch/PR map, status, findings and the backlog. Then add an entry for your own work to the top of its Log section.

## Branches and pull requests
- Branch from `integration`, or from the base named in your task.
- Open a PR into `integration`. Only the coordinator merges into `integration`, and only through a PR into `main`.
- Never push to `main`, `integration`, or another agent's branch.
- Commits are authored as Apuroop Mutyala, with no AI trailers. The repo stays private.

## Building
- Worker tests run locally: `cd worker && uv run --no-project --with mcp python -m unittest test_jobworker test_mcp_bridge test_fleet`.
- The iOS app compiles only in GitHub Actions macOS:
  1. Start a build: `gh workflow run build.yml --ref <branch>`
  2. Find it: `gh run list --branch <branch> -L 1`
  3. Wait for it: `gh run watch <id> --exit-status`
- macOS minutes are expensive, so batch your fixes.
- **GitHub API rate limit:** 5,000 requests per hour, shared by every agent. Always use `gh run watch <id> --interval 120 --exit-status`, or poll no more often than every 2–3 minutes; the default watch polls every 3 s. A watch that fails because of the rate limit does not mean the build failed, so check again after the reset (`gh api rate_limit`).
- Keep `SWIFT_VERSION` 5.0. `MLXArray` is not `Sendable`, so serialize inside `container.perform`.

## PR body checklist
- What changed and why.
- The green CI run URL.
- What ran on a device versus only compiled.
- Measurements, with exact commands, models and context sizes.
- Anything unverified.

## Safety
- If a permission is denied, stop that outcome and report it. Do not reroute through another tool.
- Never read or print secrets (`~/.job-worker/secrets.json`).
- Nothing goes upstream (issues, PRs or comments on open-source repositories) without the user's explicit approval and a reproduced, measured improvement.
