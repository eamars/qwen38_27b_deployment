<!--
Posted as https://github.com/sergqwer/strata-nvfp4/issues/24 on 2026-11-12 (by the owner). Kept as sent. Written 2026-11-12 from this workspace's investigation.
Nothing below quotes the agent's text: outputs were rebuilt from their structure with filler characters.
Before posting: decide whether to also open a short issue on Niko1221/Strata for the parameter-tag half (it is
upstream's parser; the EOS-literal half is this fork's).
-->

# Tool call: a parameter tag that lost its `>` streams arguments no client can parse; EOS literal then turns the model's end of turn into 64 `<|im_end|>`

## Summary

Qwen3.8-Flash-Next (ModelOpt NVFP4, 80K–175K-token contexts) sometimes drops one `>` from a parameter tag in a tool call:
`<parameter=NAME` or `</parameter` followed straight by a line break. The parser (upstream's, unchanged in this fork)
then takes everything up to the next `>` as the parameter's name, or never ends the value. With `stream_tools` it has
already sent that much JSON, and it never closes it. An OpenAI-compatible client that joins the streamed
`function.arguments` gets text that is not JSON, and the agent's turn ends with an error.

When the value is left open, this fork's EOS literal (`stop_token_text`, on by default) makes it worse. The model's
real end-of-turn token arrives while `literal_context()` says "in a tool call's argument". The server writes the token
out as text and lets the reply go on, up to `STOP_TEXT_MAX` (64) times. The argument ends with `</function>`,
`</tool_call>` and exactly 64 `<|im_end|>`, and the turn ends only at the cap.

Four failures in one agent's records, 2026-11-07 to 11-11 (about 4 in 10,000 tool calls). 0.2.0-beta-nvfp4.3 was
serving; 0.2.13-nvfp4.1 behaves the same.

## Environment

- strata-nvfp4 0.2.0-beta-nvfp4.3 (`57293b8e`, engine 0.1.45.3) when it happened; reproduced at parser level on
  0.2.13-nvfp4.1 (`b08a70078`).
- RTX 5090 alone, Windows 11, `--kv int8`, `--mtp ... --spec 4 --spec-min-p 0.5`, 262,144 context, vision on.
- Model: OrcaRouter Qwen3.8-Flash-Next Uncensored, ModelOpt NVFP4 (this fork's converters).
- Client: DSH agent harness over pi-ai 1.3.0 `openai-completions`, streaming, `reasoningEffort: high`,
  `maxTokens: 32768`. pi-ai builds the arguments by joining `choices[].delta.tool_calls[].function.arguments`, then
  `JSON.parse`. In all four records the joined deltas equal the received arguments exactly.

## What the client received

Structure only: `⟨T:n⟩` is n characters of the model's text, which the server sent as `\uXXXX` escapes.

| | Tool | Streamed arguments (end) | Client error |
|---|---|---|---|
| A | `write_file` (`content`, `overwrite`, `path`) | `…,"path":"⟨T:17⟩","content_bk\n</parameter":` | Unexpected end of JSON input |
| B | `think` (`thought`) | `{"thought\n⟨T:25⟩:⟨T:21⟩:⟨T:5⟩:⟨T:67⟩\n</parameter":` | Unexpected end of JSON input |
| C | `write_document` (14 string params) | `…,"heading":"⟨T:15⟩\n</parameter\n</function>\n</tool_call>` + `<|im_end|>` ×64 | Unterminated string |
| D | `think` (`thought`) | `{"thought":"⟨T:94⟩\n</think>\n\n⟨T:21⟩…⟨T:28⟩` + 54 `<|im_end|` copies | Unterminated string |

Each turn ended normally (block end, usage, finish), with 462 to 2,641 output tokens, so none hit a limit. In C and D
the `<|im_end|>` arrive as five ordinary pieces each (`<|`, `im`, `_end`, `|`, `>`), the token's text as EOS literal
writes it.

## Reading

The raw output was not logged, so it was rebuilt from the received structure:

- **A, B:** `<parameter=NAME\nVALUE\n</parameter>`. The opener lost its `>`. The streaming scanner's name runs to the
  first `>`, the one in `</parameter>`, so `NAME\nVALUE\n</parameter` is sent as a key, and the value never comes. In A
  it is an extra, undeclared `content_bk` after two good parameters.
- **C:** `<parameter=heading>\nVALUE\n</parameter\n</function>\n</tool_call>`. The closer lost its `>`. `param_end`
  never finds `</parameter>`, so the value never ends, and `value_open()` stays true. The model then writes
  `<|im_end|>` (id 248046), and `stop_as_text` replaces it with its text 64 times.
- **D:** no slip. The model wrote `</think>` and its reply inside the `think` tool's argument, then ended its turn
  inside the open value (the same EOS-literal run). Nothing in the parser can turn that into the call the model meant.
  Like #210/#211 upstream, it should keep failing loudly so the client can retry.

## Reproduction (parser only, no model, no GPU)

Each output was rebuilt with filler characters, then streamed through `OutputParser(thinking=True, tools=...,
stream_tools=True)` in 1–4 character pieces, as the server feeds it. A and B also close with `</function></tool_call>`.

| Parser | A | B | C | D |
|---|---|---|---|---|
| 0.2.0-beta-nvfp4.3 stock | not JSON | not JSON | not JSON | not JSON |
| 0.2.13-nvfp4.1 stock | not JSON | not JSON | not JSON | not JSON |
| 0.2.13-nvfp4.1 + the change below | valid: `content`, `path`, `content_bk: ""` | valid: `thought` | valid: `doc`, `do`, `heading` | not JSON (left loud) |

The streamed arguments in the two stock rows are byte-identical to what the client received. Minimal case:

```python
from serve.frontend import OutputParser
tools = [{"name": "think", "parameters": {"type": "object",
          "properties": {"thought": {"type": "string"}}, "required": ["thought"]}}]
raw = ("ok\n</think>\n\n<tool_call>\n<function=think>\n"
       "<parameter=thought\nsome thought\n</parameter>\n</function>\n</tool_call>")
p = OutputParser(thinking=True, tools=tools, stream_tools=True, recover=True)
ev = [e for c in raw for e in p.feed(c)] + p.finish("stop")
print("".join(e.text for e in ev if e.kind == "tool_args"))
# 0.2.13-nvfp4.1: {"thought\nsome thought\n</parameter":     <- not JSON, no tool_call event
```

## Proposed fix

Upstream already repairs the same kind of drift for `<function=` openers by rewriting the raw text before it is parsed
(#3441, #5276, #5409). Parameter tags have no such repair. We run this as a local patch, applied on top of our
`</think>` wrapper (`scripts/patches/strata-tool-tags.py` in our workspace; about 40 lines):

- On the raw text, before every parser and only inside a `<tool_call>`: `<parameter=NAME` followed by a line break
  (NAME without spaces or angle brackets) becomes `<parameter=NAME>`, and `</parameter` followed by a line break
  becomes `</parameter>`.
- A tail that may still become such a tag (`<parameter=…`, `</parameter`, `<tool_call>`, `</tool_call>`) waits for
  its next character, and `clean` counts it as held.
- Text outside a call (a reply or a file that discusses the format) is left as written. A value inside a call that
  spells such a slipped tag on purpose would be changed too; none was seen.
- A call that never closes is not repaired, as #211's fail-loud rule wants.

With it, C's value ends at the slipped tag. The model's first end-of-turn token then comes after `</tool_call>`, where
it ends the reply, so the 64-token EOS-literal run does not happen. EOS literal stays useful where it was meant (a
model writing `<|im_end|>` into a file argument); its trigger, `value_open()`, is right only when the closer is intact.

Checks on 0.2.13-nvfp4.1:
- a replay of six shapes (the four slips, a well-formed call, slipped-looking tags in plain text), each whole, one
  character at a time and in random chunks: 18/18 runs pass; the stock parser gets the four slips wrong;
- the fork's `serve` suite at `b08a70078` (1,637 tests): the same 11 failures with and without it, all from our
  `</think>` wrapper, and none in the tool-call tests (`test_param_end_options`, `test_parser_call_scan`,
  `test_tool_corpus`, `test_tool_routes`, `test_tool_name_alias`). A first version held back a complete
  `<tool_call>` at the end of a delta; `test_a_second_call_releases_the_first_at_once` caught that.

## Possible further step (not done)

`literal_context()` could refuse EOS literal when the open value already holds `</function>` or `</tool_call>`
after a line break. Those mean the model has left the call even if a closer slipped in a way this repair does not
cover. Then the first end-of-turn token ends the reply instead of 64 copies.

## Related

- Niko1221/Strata #5756: the sibling slip on the same model. The value's own `</parameter>` is skipped after a nested
  pair, and the value is silently truncated.
- Niko1221/Strata #4071: calls returned as text. One report shows a malformed opener
  (`<function=grep_search=isRegexp </parameter>`); the maintainer asked for raw outputs.
- Niko1221/Strata #210, #211: a call cut off by the end of the turn must not be reported as complete.
- Niko1221/Strata #2236: the origin of EOS literal.
