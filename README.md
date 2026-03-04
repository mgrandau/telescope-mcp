# telescope-mcp

MCP server + web dashboard for AI-controlled telescope operation.

## 🧭 Intent

Telescopes are operated manually — step through notebooks, adjust settings by hand, correlate camera output with position readings. This project makes the telescope **AI-operable**: an AI agent can discover cameras, capture frames, slew to targets, monitor conditions, and manage observing sessions through structured MCP tool calls.

The key architectural insight is **dual control paths**: MCP for AI agents and HTTP for human operators, both sharing the same device layer. The AI handles planned observing sequences; the human handles setup, focusing, and ad-hoc adjustments. Neither path owns the hardware — they're equal peers.

Every hardware component has a **digital twin** — a simulation driver that runs the full stack without physical hardware. This isn't just a test mock; it's a first-class runtime mode that makes development possible anywhere, not just in front of the telescope.

The project follows [AI-optimized documentation](docs/journal/2025-12-13.md) — heavy docstrings on every function including private methods and tests. AI agents see docstrings inline, so rich function-level documentation means less file traversal and faster assistance. (See also: [docscope-mcp #17](https://github.com/mgrandau/docscope-mcp/issues/17) for the open question of whether this level is still needed with stronger models.)

The design follows the [Human-AI Intent Transfer Principles](https://mgrandau.medium.com/human-ai-intent-transfer-principles-b6e7404e3d26?source=friends_link&sk=858917bd3f4a686974ed6b6c9c059ac8). The [project plan](docs/PROJECT_PLAN.md) documents goals and risk posture per phase, and the [journal](docs/journal/2025-12-13.md) captures design decisions with rejection rationale.

## 💬 Community

💬 [Join the Discord community](https://discord.gg/2KqjHvh5)
