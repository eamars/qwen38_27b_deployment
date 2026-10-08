r"""Local change to the pinned Strata server: a `</think>` the model writes while quoting the tag, or again inside
its answer, no longer leaks reasoning and a repeated reply into `content`.

Qwen3.8 writes `</think>` as one token (248069) whether it ends its thinking or quotes the tag, and the pinned
`OutputParser` switches to the answer at the first one (upstream #537, #1053; still so in 0.1.40.2). The model's
own close is a `</think>` alone on its line: on 0.1.39, 420 of 420 real closes followed a newline and 13 of 14
quoted tags did not, and the template writes "\n</think>\n\n". Two rules wrap the parser:

A. Before it, while the model thinks: the model's own close ends its line, and starts it or is followed by a blank
   line ("\n</think>\n\n", or "text</think>\n\n" as upstream's tests write it; none of 13 recorded quotes was
   followed by a blank line). Any other `</think>` is a quote and goes in as QUOTE, plain text to the parser; it
   leaves as `</think>` again. If the reply
   then stops still thinking, the text after the last quoted tag is the answer (when upstream's
   reasoning_close_retry has not closed the thinking first).
B. After it, in the answer: a `</think>` alone on its line ("X\n</think>\n\nX") is not the answer. The answer's
   first ANSWER_HOLD characters are held back, and such a tag there is decided by what follows it: more answer
   means the text before it was more thinking, so it becomes reasoning (the reply written twice is sent once); a
   tool call or the end means it was the answer, and only the tag is dropped. After the window the tag is dropped.
   A `</think>` that starts a line of text, or sits inside one, is the model quoting it and stays text (v3 split at
   a quote that started a line: "...\n</think> 是手滑" lost the answer's first half).

v4 ports v3 to the 0.1.40.2 parser, whose own `pending` and `_release` it must not touch, and adds the "alone on
its line" test to both rules.

Usage: python strata-think-echo.py <runtime>/serve/frontend.py [release.zip]
Idempotent; refuses a changed file. With the release archive, a file carrying an earlier version of this patch is
first restored from the release.
"""
import sys
import zipfile
from pathlib import Path

MARK = "# local: strata-think-echo v4"
OLD_MARK = "# local: strata-think-echo"

REPLACEMENTS = [
    # __init__: the patch's state
    (r'''        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._reset_scan()
''', r'''        self.batch = False           # the call being finished is followed by another in the same wrapper
        self._reset_scan()
        ''' + MARK + r''': rule A runs while the model thinks (pre_on) on text not yet decided (pre_buf) after
        # pre_prev; quoted is the reasoning since the last quoted </think> (None: none).  Rule B, once the thinking
        # has closed: the answer held back (held; None: not holding), the text before a repeated </think> that waits
        # for what follows it (echo), the answer text not yet decided (cpend), and the last answer character (clast)
        self.pre_on, self.pre_buf, self.pre_prev, self.quoted = thinking, "", "", None
        self.closed, self.held, self.echo, self.cpend, self.clast = False, None, None, "", ""
'''),
    # the wrapper and its rules, before the parser's own feed
    (r'''    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
''', r'''    ''' + MARK + r'''
    ANSWER_HOLD = 512    # characters of the answer held back for a repeated </think> (rule B)
    QUOTE = "\U000F0E12"  # a </think> taken for a quote, inside the parser (private use, as #537's marks)

    def feed(self, delta: str) -> list[Event]:
        return self._post(self._feed_raw(self._pre(delta)))

    def finish(self, reason: str | None = None) -> list[Event]:
        out = self._post(self._feed_raw(self._pre("", final=True)))
        return out + self._post(self._finish_raw(reason), end=True, reason=reason)

    def _pre(self, delta: str, final: bool = False) -> str:
        """Rule A: while the model thinks, a </think> that is not alone on its line goes to the parser as QUOTE."""
        if not self.pre_on:
            return delta
        s, out, prev = self.pre_buf + delta, [], self.pre_prev
        while True:
            e = s.find(THINK_END)
            if e < 0:
                keep = 0 if final else self._hold(s, (THINK_END,))
                out.append(s[:len(s) - keep])
                s = s[len(s) - keep:]
                break
            after = s[e + len(THINK_END):]
            if not after and not final:             # the character after it decides
                out.append(s[:e])
                s = s[e:]
                break
            before = s[e - 1] if e else prev
            starts = before in ("", "\n")
            if not starts and after == "\n" and not final:     # a blank line after it would make it the close
                out.append(s[:e])
                s = s[e:]
                break
            out.append(s[:e])
            # the model's own close ends its line, and starts it or is followed by a blank line ("ok</think>\n\n", the
            # template's close after text); a quote is neither (13 of 13 recorded quotes were followed by text)
            if (starts and (not after or after[0] == "\n")) or after.startswith("\n\n"):
                out.append(THINK_END + after)
                s, self.pre_on, self.closed, self.held = "", False, True, ""
                break
            out.append(self.QUOTE)
            s, prev = after, self.QUOTE
        self.pre_buf = s
        text = "".join(out)
        self.pre_prev = text[-1] if text else prev
        return text

    def _unquote(self, v):
        if isinstance(v, str):
            return v.replace(self.QUOTE, THINK_END)
        if isinstance(v, dict):
            return {k: self._unquote(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._unquote(x) for x in v]
        return v

    def _post(self, events: list[Event], end: bool = False, reason: str | None = None) -> list[Event]:
        out = []
        for ev in events:
            if ev.kind == "reasoning":
                if self.QUOTE in ev.text:
                    self.quoted = ev.text.rsplit(self.QUOTE, 1)[1]
                    ev = Event("reasoning", self._unquote(ev.text))
                elif self.quoted is not None:
                    self.quoted += ev.text
                out.append(ev)
            elif ev.kind == "content":
                out += self._content(self._unquote(ev.text))
            else:                                   # a call or its pieces: the answer before it is settled first
                if ev.call is not None and self.QUOTE in repr(ev.call.arguments):
                    ev.call.arguments = self._unquote(ev.call.arguments)
                if ev.kind == "tool_args" and self.QUOTE in ev.text:
                    ev = Event(ev.kind, self._unquote(ev.text), call=ev.call)
                out += self._content("", stop=True)
                out.append(ev)
        if end:
            out += self._content("", stop=True)
            if self.state in ("reasoning", "rcall") and self.quoted is not None and reason in (None, "stop"):
                tail = self._unquote(self.quoted).strip("\n")   # the tag taken for a quote was the close after all
                if tail.strip():
                    out.append(Event("content", tail))
        return out

    def _content(self, text: str, stop: bool = False) -> list[Event]:
        """Rule B: the answer's text.  A </think> alone on its line (after a newline or at the answer's start, and
        before a newline, a call or the end) is not the answer; one that starts or sits in a line of text is a quote."""
        if not self.closed:
            return [Event("content", text)] if text else []
        s, out, self.cpend = self.cpend + text, [], ""
        while s:
            e = s.find(THINK_END)
            if e < 0:
                # hold a partial tag AND the newlines before the end: if the tag follows, they go with it, so
                # streamed and whole outputs agree (as the parser does before a call)
                j = len(s) if stop else len(s) - self._hold(s, (THINK_END,))
                while not stop and j > 0 and s[j - 1] == "\n":
                    j -= 1
                out += self._answer(s[:j])
                self.cpend = s[j:]
                break
            before = s[e - 1] if e else self.clast
            after = s[e + len(THINK_END):]
            if before in ("", "\n") and not after and not stop:     # what follows it decides
                j = e
                while j > 0 and s[j - 1] == "\n":
                    j -= 1
                out += self._answer(s[:j])
                self.cpend = s[j:]
                break
            if before not in ("", "\n") or (after and after[0] != "\n"):    # a quote: text
                out += self._answer(s[:e + len(THINK_END)])
                s = after
                continue
            out += self._boundary(s[:e])
            s = after.lstrip("\n")
        if stop:
            out += self._answer_out()
        return out

    def _boundary(self, before: str) -> list[Event]:
        """A </think> alone on its line, after `before`."""
        out = []
        if self.held is not None and len(self.held + before) <= self.ANSWER_HOLD:   # none of the answer sent yet
            text, self.held = self.held + before, ""
            if text.strip():                        # thinking if more answer follows, else the answer
                if self.echo:                       # the answer again after the last one: that was thinking
                    out.append(Event("reasoning", "\n" + self.echo.strip("\n") + "\n"))
                self.echo = text
        else:                                       # the answer is out: drop the tag only
            out += self._answer_out()
            if before.strip():                      # its newlines go as streaming holds them back
                out += self._answer(before.rstrip("\n"))
        self.clast = "\n"
        return out

    def _answer(self, text: str) -> list[Event]:
        out = []
        if self.echo and text.strip():              # more answer came: the text before the tag was thinking
            out.append(Event("reasoning", "\n" + self.echo.strip("\n") + "\n"))
            self.echo = None
        if text:
            self.clast = text[-1]
        if self.held is None:
            return out + ([Event("content", text)] if text else [])
        self.held += text
        return out + (self._answer_out() if len(self.held) >= self.ANSWER_HOLD else [])

    def _answer_out(self) -> list[Event]:
        out = []
        if self.echo:                               # a tool call or the end came instead: it was the answer
            out.append(Event("content", self.echo.strip("\n")))
            self.echo = None
        held, self.held = self.held, None
        return out + ([Event("content", held)] if held else [])

    def _feed_raw(self, delta: str) -> list[Event]:
        self.buf += delta
'''),
    (r'''    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation''', r'''    def _finish_raw(self, reason: str | None = None) -> list[Event]:
        """End of generation'''),
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
