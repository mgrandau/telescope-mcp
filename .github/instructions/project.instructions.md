---
applyTo: '**'
---

# Project Intent & Design

This project follows the [Human-AI Intent Transfer Principles](https://mgrandau.medium.com/human-ai-intent-transfer-principles-b6e7404e3d26?source=friends_link&sk=858917bd3f4a686974ed6b6c9c059ac8) — making a telescope AI-operable through dual control paths (MCP + HTTP) on shared hardware.

**Context chain (read in order when making design decisions):**

1. [🧭 Intent](../../README.md#-intent) — project philosophy: dual control paths, digital twin as first-class runtime
2. [PROJECT_PLAN.md](../../docs/PROJECT_PLAN.md) — phase goals, risk posture, intent evolution, issue mapping
3. [Journal entries](../../docs/journal/) — design alternatives explored and rejected, with rationale
4. [Architecture](../../src/telescope_mcp/README.md) — component map, invariants, DI contracts, AI-accessibility map
5. [Subpackage READMEs](../../src/telescope_mcp/) — per-component architecture (`data/`, `devices/`, `drivers/`, `tools/`, `web/`, `observability/`, `utils/`)
6. [TODO.md](../../TODO.md) — detailed phase plan for remaining work (motors, position, alignment, goto)
7. Source code — the implementation

**Core design values (from rejection patterns in the journal):**

- **MCP for AI, HTTP for human** — dual control paths sharing the same device layer. Neither wraps the other.
- **Digital twin is a runtime, not a mock** — `--mode digital_twin` runs the full stack without hardware. Development doesn't require a telescope.
- **Protocol-based DI** — `CameraDriver`, `SensorDriver`, `MotorDriver` are Protocols, not ABCs. Structural typing, no inheritance coupling.
- **AI-optimized over-documentation** — heavy docstrings on all functions including private/test. AI agents read docstrings inline; rich docs mean less file traversal.
- **ASDF for scientific data** — numpy arrays, metadata, hierarchical sessions. Not JSON, not SQLite.
- **Greenfield, not legacy** — no backward compatibility needed. Remove old code; don't add compat shims.
- **Hardware constraints are documented, not implicit** — USB bandwidth, serial protocols, sensor timing must be in docs because they're invisible in code.

When proposing new features or changes, check the journal for prior art — the alternative you're considering may have already been evaluated and rejected.

# Project Rules

## Greenfield Development
- This is a **new product**, not a legacy system
- No backward compatibility needed — there are no external consumers
- When refactoring, **remove** old code; do not keep legacy fallbacks
- Do not add "legacy compat" shims, aliases, or deprecated wrappers
- If something is replaced, delete the old version entirely
- Prefer clean, direct implementations over abstractions "just in case"

## GitHub Interaction Rules

### CLI Only
- Always use `gh` CLI for GitHub operations (issues, PRs, repos)
- Do NOT use GitKraken MCP tools for GitHub API calls
- Git operations (commit, push, branch) may use GitKraken MCP tools

### Common Commands
```bash
# Issues
gh issue list
gh issue view <number> --json title,body,state,labels
gh issue create --title "..." --body "..."
gh issue close <number>

# Pull Requests
gh pr list
gh pr view <number>
gh pr create --title "..." --body "..."
gh pr merge <number>

# Repo info
gh repo view --json name,description,url
```

## Release Process

### Version Badge

The README badge auto-updates from GitHub releases — no manual badge edits needed.

### Release Steps

1. Update version in `src/telescope_mcp/__init__.py`
2. Commit changes: `git commit -am "release: bump version to X.X.X"`
3. Create and push tag: `git tag vX.X.X && git push origin vX.X.X`
4. Create GitHub release with **changelog notes** covering:
   - **Bug Fixes** — issues fixed with brief description
   - **Features** — new functionality added
   - **Hardware** — any hardware-specific changes or new device support
   - Link to full changelog comparison: `https://github.com/mgrandau/telescope-mcp/compare/vPREV...vX.X.X`

### Changelog Requirements

- Every release **must** have human-written changelog notes — do not rely solely on `--generate-notes`
- Reference issue numbers (e.g., "Fixed #5: dual camera streaming crashes")
- Keep notes concise but meaningful — someone reading them should understand what changed and why
