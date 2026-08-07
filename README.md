# ballast

Pull a quantized knowledge corpus, or build one from your own documents, and
ground any local model. Works with Ollama and every OpenAI-compatible or
MCP-capable client. Includes model profiling and a three-arm grounding
benchmark, all CPU-only.

```bash
uvx openballast pull --level 3
uvx openballast serve
```

(Live on PyPI: `pip install openballast` also works.)

- **Ollama users:** point your client's base URL at `http://localhost:11435/v1`
  instead of `http://localhost:11434/v1`. Done. Every chat request is grounded
  with corpus facts before your model sees it. No tool calling needed, works
  with any model size.
- **MCP users** (Claude Desktop, LM Studio, Cline, Goose): add to your MCP config:

  ```json
  { "ballast": { "command": "uvx", "args": ["openballast", "mcp"] } }
  ```

- **Smoke test:**

  ```bash
  uvx openballast lookup "Where was Douglas Adams born?"
  ```

## What you're downloading

[Ballast T0](https://huggingface.co/datasets/OpenBallast/ballast-t0): 25.4M
entities and 197M facts from Wikidata (CC0), quantized into nested levels. Pick
your knowledge size like you pick a GGUF quant:

| level | download | on disk | contains |
|---|---|---|---|
| L0 | 52 MB | 0.2 GB | top 0.5% most notable entities |
| L1 | 92 MB | 0.35 GB | top 1% |
| L2 | 159 MB | 0.6 GB | top 2% |
| L3 | 265 MB | 1.0 GB | top 4% |
| L4 | 427 MB | 1.6 GB | top 8% |
| L5 | 691 MB | 2.6 GB | top 16% |
| L6 | 1.1 GB | 4.2 GB | top 32% |
| L7 | 2.2 GB | 9.2 GB | everything |

Levels are nested: `pull --level 5` after `pull --level 3` downloads only the
new buckets. Everything runs offline after the pull: no network at answer time.

Measured effect (details: [thesis](https://github.com/OpenBallast/ballast)):
a 2B model + ~180 MB of ballast exceeds a 12B model's factual accuracy;
hallucination on factual probes drops ~3×.

## Commands

```
ballast pull  --level 3      # download / upgrade the corpus
ballast build ./docs -n team # build a corpus from YOUR documents (see below)
ballast serve                # OpenAI grounding proxy :11435 + MCP http :11436
ballast mcp                  # MCP on stdio (for client configs)
ballast lookup "question"    # print the evidence blocks for a question
ballast profile -m qwen3:8b  # where does this model's knowledge run out?
ballast eval    -m qwen3:8b  # three-arm benchmark: what does grounding buy?
ballast status               # installed corpora, levels, and sizes
```

`BALLAST_HOME` overrides the storage location (default `~/.ballast`).

## Bring your own corpus

`ballast build` turns a directory of `.md` / `.txt` files (and/or parquet with
a `text` column, optional `title` and `rank`) into a servable corpus with the
same layout as the published one:

```bash
ballast build ./handbook --name handbook
ballast lookup --corpus handbook "What is our deploy freeze policy?"
ballast serve  --corpus handbook
```

Documents are addressed by title; each becomes passage chunks the linker can
attach to a question. A `rank` column (0..1, 1 = most important) spreads
documents across nested levels so `--level` keeps the top slice; without ranks
everything lands in one level.

## Profile a model, then size the corpus for it

```bash
ballast profile -m qwen3:8b --limit 2000 --budget 2GB
```

Probes the model ungrounded against the public evalset, reports its accuracy
per corpus region (head → tail), fits a grounding competence profile
(`.gcp.json`), and (given a byte budget) recommends the corpus level where
grounding still buys accuracy for THIS model. The profile carries a
reliability AUC against a 0.58 gate; below the gate the recommendation falls
back to the generic ordering.

## Measure what grounding actually delivers

```bash
ballast eval -m qwen3:8b --limit 500
```

Every probe is asked three ways: ungrounded (U), with realized retrieval (R),
and with oracle-entity evidence (S). The report is the delivery ratio
(R − U) / (S − U), the fraction of the reachable knowledge gap today's
retrieval closes, plus coverage-conditional splits. Arms checkpoint to
parquet and resume after interruption.

## How it works

`serve` intercepts `POST /v1/chat/completions`, mines entity mentions from your
last message, resolves them against the local corpus (normalized label/alias
match), and prepends the matching facts as a system message. Everything else,
including streaming, passes through untouched. The MCP server exposes the same
three tools (`resolve`, `evidence`, `lookup`) as the hosted demo endpoint
([mcp.openballast.org](https://mcp.openballast.org)).

Apache-2.0. Corpus data: CC0 (Wikidata contributors).
