r"""Local change to the pinned Strata server: a `</think>` the model writes while quoting the tag, or again inside
its answer, no longer leaks reasoning and a repeated reply into `content`.

Qwen3.8 writes `</think>` as one token (248069) whether it ends its thinking or quotes the tag, and the pinned
`OutputParser` switches to the answer at the first one (upstream #537, #1053). Two rules, both measured on 0.1.39
(docs/qwen38-flash-next-strata.md):

A. The template's own close is "\n</think>": 420 of 420 real closes followed a newline, 13 of 14 quoted ones did
   not. A `</think>` after any other character stays reasoning. If the reply then ends still thinking, the text
   after the last such tag is returned as the answer.
B. A `</think>` on its own line inside the answer ("X\n</think>\n\nX"). The answer's first ANSWER_HOLD characters
   are held back, and such a tag there decides by what follows it: more answer means the text before it was more
   thinking, so it becomes reasoning (the reply written twice is sent once); a tool call or the end of the reply
   means it was the answer, and only the tag is dropped (an answer followed by a stray tag and a tool call was lost
   to the reasoning by v2). After the window the tag is dropped. A `</think>` inside a line of the answer is the
   model quoting it and stays text (live check: replies explaining the tag quote it mid-sentence).

Usage: python strata-think-echo.py <runtime>/serve/frontend.py [release.zip]
Idempotent; refuses a changed file. With the release archive, a file carrying an earlier version of this patch is
first restored from the release.
"""
import sys
import zipfile
from pathlib import Path

MARK = "# local: strata-think-echo v3"
OLD_MARK = "# local: strata-think-echo"

REPLACEMENTS = [
    # __init__: the patch's state
    (r'''        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
''', r'''        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        ''' + MARK + r''': the last reasoning character, the reasoning written since the last quoted </think>
        # (None: none quoted), the answer held back after the close (None: not holding), whether the thinking has
        # closed, the last answer character, and the answer before a repeated </think> (None: none) that is
        # thinking if more answer follows and the answer if a tool call or the end does
        self.rlast, self.quoted, self.held, self.closed, self.clast = "", None, None, False, ""
        self.pending = None
'''),
    # feed: rule A in the reasoning state
    (r'''                if i:
                    out.append(Event("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
''', r'''                if i:
                    out += self._reason(self.buf[:i])
                self.buf = self.buf[i + len(THINK_END):]
                if self.rlast not in ("", "\n"):          ''' + MARK + r''': a quoted tag, not "\n</think>"
                    out += self._reason(THINK_END)
                    self.quoted = ""
                    continue
                self.state, self.lead, self.quoted, self.held, self.closed = "content", True, None, "", True
'''),
    (r'''                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
''', r'''                    if len(self.buf) > keep:
                        out += self._reason(self.buf[:len(self.buf) - keep])
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
'''),
    # feed: rule B in the content state
    (r'''                i = self.buf.find(CALL_START)
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self.buf[:j]))
                        self.buf = self.buf[j:]
                    return out
                if i and self.buf[:i].strip():
                    out.append(Event("content", self.buf[:i].rstrip("\n")))
                self.buf = self.buf[i + len(CALL_START):]
''', r'''                i = self.buf.find(CALL_START)
                e = self._echo_at() if self.closed else -1  ''' + MARK + r''': </think> again, on its own line
                if e >= 0 and (i < 0 or e < i):
                    text, self.buf, self.lead = self.buf[:e], self.buf[e + len(THINK_END):], True
                    if self.held is not None and len(self.held + text) <= self.ANSWER_HOLD:   # none sent yet
                        text, self.held = self.held + text, ""
                        if text.strip():            # thinking if more answer follows, else the answer
                            if self.pending:        # the answer again after the last one: that was thinking
                                out.append(Event("reasoning", "\n" + self.pending.strip("\n") + "\n"))
                            self.pending = text
                    else:                                   # the answer is out: drop the tag only
                        out += self._release()
                        if text.strip():                    # its newlines go as streaming holds them back
                            out += self._answer(text.rstrip("\n"))
                    continue
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - self._hold(self.buf, (CALL_START, THINK_END) if self.closed else (CALL_START,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out += self._answer(self.buf[:j])
                        self.buf = self.buf[j:]
                    return out
                if i and self.buf[:i].strip():
                    out += self._answer(self.buf[:i].rstrip("\n"))
                out += self._release()
                self.buf = self.buf[i + len(CALL_START):]
'''),
    # finish: rule A's fallback and the held answer
    (r'''        if self.buf:
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
''', r'''        if self.state == "reasoning" and self.quoted is not None:   ''' + MARK + r'''
            tail = (self.quoted + self.buf).strip("\n")      # the tag taken for a quote was the close after all
            if self.buf:
                out.append(Event("reasoning", self.buf))
            self.buf = ""
            if tail:
                out.append(Event("content", tail))
            return out
        out += self._release()
        if self.buf:
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
'''),
    # helpers, before feed
    (r'''    def feed(self, delta: str) -> list[Event]:
''', r'''    ''' + MARK + r'''
    ANSWER_HOLD = 512    # characters of the answer held back for a repeated </think> (rule B)

    def _reason(self, text: str) -> list[Event]:
        if not text:
            return []
        self.rlast = text[-1]
        if self.quoted is not None:
            self.quoted += text
        return [Event("reasoning", text)]

    def _echo_at(self) -> int:
        """Where a </think> on its own line starts in self.buf (-1: none); one inside a line is a quote, plain text."""
        e = self.buf.find(THINK_END)
        while e >= 0:
            if (self.buf[e - 1] if e else self.clast) in ("", "\n"):
                return e
            e = self.buf.find(THINK_END, e + 1)
        return -1

    def _answer(self, text: str) -> list[Event]:
        out = []
        if self.pending and text.strip():           # more answer came: the text before the tag was thinking
            out.append(Event("reasoning", "\n" + self.pending.strip("\n") + "\n"))
            self.pending = None
        if text:
            self.clast = text[-1]
        if self.held is None:
            return out + [Event("content", text)]
        self.held += text
        return out + (self._release() if len(self.held) >= self.ANSWER_HOLD else [])

    def _release(self) -> list[Event]:
        out = []
        if self.pending:                            # a tool call or the end came instead: it was the answer
            out.append(Event("content", self.pending.strip("\n")))
            self.pending = None
        held, self.held = self.held, None
        return out + ([Event("content", held)] if held else [])

    def feed(self, delta: str) -> list[Event]:
'''),
]


RELEASE_PATH = "strata-nvfp4/serve/frontend.py"


def release_text(archive: Path) -> str:
    with zipfile.ZipFile(archive) as release:
        return release.read(RELEASE_PATH).decode("utf-8").replace("\r\n", "\n")


def patch(text: str, archive: Path | None = None) -> str:
    if MARK in text:
        return text
    if OLD_MARK in text:
        if archive is None:
            raise SystemExit("strata-think-echo: an older version is applied; pass the release archive to replace it")
        text = release_text(archive)
        if OLD_MARK in text:
            raise SystemExit("strata-think-echo: the release's frontend.py already carries a version of this patch")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(f"strata-think-echo: anchor not found once, the server has changed:\n{old[:200]}")
        text = text.replace(old, new)
    return text


def main():
    path = Path(sys.argv[1])
    archive = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    out = patch(text, archive)
    if out == text:
        print(f"strata-think-echo: already applied: {path}")
        return
    path.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
    print(f"Applied strata-think-echo: {path}")


if __name__ == "__main__":
    main()
