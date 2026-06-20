# Orca — 100% Local Private AI

> Your hardware. Your data. Your intelligence.  
> No Anthropic. No OpenAI. No cloud. No telemetry.

Orca is a fully private AI system that runs entirely on your own hardware using [Ollama](https://ollama.com). It includes a terminal CLI, a professional web UI, a multi-agent Ultra mode, long-term memory, fine-tuning tools, and a self-contained revenue/licensing layer — all 100% local.

---

## Quick Install

```bash
curl -fsSL https://orca.systems/install.sh | bash
```

Or via pip:

```bash
pip install orca-ai
orca doctor --wizard
```

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- At least one Ollama model (e.g. `ollama pull llama3.2:3b`)

---

## Getting Started

```bash
# First-run setup wizard
orca doctor --wizard

# Terminal chat
orca core chat

# Single-shot fast response
orca nano "explain recursion in 2 sentences"

# Web UI (opens in browser)
orca serve

# Multi-agent Ultra (Pro license required)
orca ultra run "design a REST API for a todo app"
```

---

## Commands

| Command | Description |
|---|---|
| `orca nano <prompt>` | Fast single-shot response |
| `orca core chat` | Full interactive chat with memory + tools |
| `orca core think <prompt>` | Deep single-shot reasoning |
| `orca ultra run <task>` | Multi-agent orchestration |
| `orca serve` | Launch the web UI |
| `orca data seed --n 500` | Generate synthetic training data |
| `orca data curate` | Clean and score training data |
| `orca train run` | Fine-tune via QLoRA |
| `orca train cloud --ssh ...` | Train on a rented GPU |
| `orca doctor` | System health check |
| `orca doctor --wizard` | First-run setup wizard |
| `orca upgrade` | Self-update from PyPI |
| `orca activate <key>` | Activate a Pro license |
| `orca license` | Show license status |
| `orca status` | Live system dashboard |

---

## Features

### Core
- Full multi-turn chat with tool use (web search, code execution, file ops)
- 4-layer memory: short-term, long-term (ChromaDB), episodic, semantic
- Self-reflection and reasoning traces
- Session save/resume

### Ultra (Pro)
- 6-agent parallel pipeline: researcher, coder, analyst, writer, critic, architect
- Automatic decomposition, parallel execution, synthesis, grading, self-healing
- Web UI pod visualization with live progress streaming

### Fine-Tuning
- Synthetic data generation across 20+ domains
- QLoRA fine-tuning via Unsloth (local GPU)
- Cloud training via SSH (Vast.ai, Lambda, RunPod)
- GGUF export + Ollama registration

### Web UI
- Professional black-and-white design
- CORE / ULTRA mode toggle
- SSE streaming with real-time pod visualization
- Memory recall sidebar
- License status indicator

---

## Licensing

Orca ships in two tiers:

| Tier | Price | Features |
|---|---|---|
| **Free** | $0 | Core chat, doctor, status, data tools |
| **Pro** | $49/mo | + Ultra mode, cloud training, web UI |
| **Enterprise** | $199/mo | All features, 5 seats, priority support |

```bash
orca activate ORCA-PRO-XXXXX-XXXXX-XXXXX
orca license --buy   # show pricing
```

---

## Privacy

- Zero telemetry
- No external API calls from the core system
- All data stored in `~/.orca/`
- Inference via Ollama on `localhost:11434`

---

## Documentation

[orca.systems/docs](https://orca.systems/docs)
