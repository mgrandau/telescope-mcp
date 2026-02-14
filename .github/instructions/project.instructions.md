---
applyTo: '**'
---
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
