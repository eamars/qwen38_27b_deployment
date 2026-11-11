r"""Local change to the pinned Strata server: a tool call whose parameter tag lost its `>` is read as the model meant it.

Qwen3.8 writes calls as `<parameter=NAME>\nVALUE\n</parameter>`. In long contexts it sometimes drops one `>`:
`<parameter=thought\n...` or `...\n</parameter\n</function>`. Upstream's parser then takes everything up to the
next `>` as the parameter's name (`"thought\n...\n</parameter":`), or never ends the value, and its streaming
tool parser has already sent that much JSON. The client cannot parse the arguments and the agent's turn ends
(Xiaoman, 4 times from 2026-11-07 to 11-11; docs/qwen38-flash-next-strata.md). Upstream repairs the same kind of
drift for `<function=` openers (#3441, #5276, #5409) by rewriting the raw text before it is parsed; parameter tags
have no such repair. The fork's "EOS literal" (`stop_token_text`, on by default) makes it worse: an end-of-turn token
the model writes while a value looks open is written out as text and the reply goes on, up to 64 times, so a value
left open by a slip collects that many spelled-out `<|im_end|>` (record C has exactly 64).

This does the same for parameter tags, on the raw model text, before every other parser: inside a `<tool_call>`,
`<parameter=NAME` followed by a line break (NAME without spaces or angle brackets) becomes `<parameter=NAME>`, and
`</parameter` followed by a line break becomes `</parameter>`. Outside a call (a reply or a file that discusses the
format) nothing changes; a value inside a call that spells such a slipped tag on purpose would be changed too (none
seen). A call that never closes (the model writing its reply into an argument) is not repaired: it
still fails at the client, which can retry the step.

It wraps strata-think-echo's feed()/finish() and is applied after it (setup-strata-runtime.ps1 runs both).

Usage: python strata-tool-tags.py <runtime>/serve/frontend.py
Idempotent; refuses a changed file.
"""
import sys
from pathlib import Path

MARK = "# local: strata-tool-tags v1"

REPLACEMENTS = [
    # clean (the server's point to close the thinking or open a forced call): the tag tail held here is held too
    (r'''        return not self.buf and not self._held and not self.pre_buf    # local: strata-think-echo v5
''', r'''        return (not self.buf and not self._held and not self.pre_buf    # local: strata-think-echo v5
                and not getattr(self, "tag_buf", ""))    ''' + MARK + r'''
'''),
    # feed and finish: the repair runs first, on the raw text
    (r'''        return self._post(self._feed_raw(self._pre(delta)))
''', r'''        return self._post(self._feed_raw(self._pre(self._tags(delta))))
'''),
    (r'''        out = self._post(self._feed_raw(self._pre("", final=True)))
''', r'''        out = self._post(self._feed_raw(self._pre(self._tags("", final=True), final=True)))
'''),
    # the repair itself, before the wrapper's rule A
    (r'''    def _pre(self, delta: str, final: bool = False) -> str:
''', r'''    ''' + MARK + r''': inside a call, a parameter tag whose `>` the model dropped gets it back
    TAG_HOLD = ("<parameter=", "</parameter", "<tool_call>", "</tool_call>")

    def _tags(self, delta: str, final: bool = False) -> str:
        s = getattr(self, "tag_buf", "") + delta
        inside = getattr(self, "tag_in", False)
        keep = 0
        if not final:                                   # a tag that may still be one waits for its next character
            k = s.rfind("<")
            tail = s[k:] if k >= 0 else ""
            # an unfinished tag (a whole `<tool_call>` needs nothing more; a whole `</parameter` waits for its follower)
            if tail and (any(t.startswith(tail) and t != tail for t in self.TAG_HOLD) or tail == "</parameter"
                         or re.fullmatch(r"<parameter=[^\s<>]*", tail)):
                keep = len(tail)
        s, self.tag_buf = s[:len(s) - keep], s[len(s) - keep:]
        out, pos = [], 0
        for m in re.finditer(r"<tool_call>|</tool_call>", s):
            out.append(self._tag_fix(s[pos:m.start()]) if inside else s[pos:m.start()])
            out.append(m.group(0))
            inside, pos = m.group(0) == "<tool_call>", m.end()
        out.append(self._tag_fix(s[pos:]) if inside else s[pos:])
        self.tag_in = inside
        return "".join(out)

    def _tag_fix(self, text: str) -> str:
        fixed = re.sub(r"<parameter=([^\s<>]+)(?=\r?\n)", r"<parameter=\1>", text)
        fixed = re.sub(r"</parameter(?=\r?\n)", "</parameter>", fixed)
        if fixed != text:
            self.tag_fixes = getattr(self, "tag_fixes", 0) + 1
        return fixed

    def _pre(self, delta: str, final: bool = False) -> str:
'''),
]


def patch(text: str) -> str:
    if MARK in text:
        return text
    if "# local: strata-think-echo v5" not in text:
        raise SystemExit("strata-tool-tags: apply strata-think-echo v5 first (this wraps its feed and finish)")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(f"strata-tool-tags: anchor not found once, the server has changed:\n{old[:200]}")
        text = text.replace(old, new)
    return text


def main():
    path = Path(sys.argv[1])
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    out = patch(text)
    if out == text:
        print(f"strata-tool-tags: already applied: {path}")
        return
    path.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
    print(f"Applied strata-tool-tags: {path}")


if __name__ == "__main__":
    main()
