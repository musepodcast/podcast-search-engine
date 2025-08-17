# po_fix_syntax.py
# Usage:  python po_fix_syntax.py
import re, sys, subprocess
from pathlib import Path
import polib  # pip install polib

PROJECT_ROOT = Path(__file__).resolve().parent
LOCALE_DIR   = PROJECT_ROOT / "locale"

# Only setting headers for the locales that errored; add others if needed.
PLURAL_FORMS = {
    "pt": "nplurals=2; plural=(n != 1);",
    "es": "nplurals=2; plural=(n != 1);",
    "it": "nplurals=2; plural=(n != 1);",
    "ja": "nplurals=1; plural=0;",
    "ko": "nplurals=1; plural=0;",
    "cn": "nplurals=1; plural=0;",
    "tw": "nplurals=1; plural=0;",
    # add more if you see header problems
}

PH_NAMED  = re.compile(r"%\(([^)]+)\)[#0\- +]?\d*(?:\.\d+)?([diouxXeEfFgGcrs])")
PH_SIMPLE = re.compile(r"%(?:[#0\- +]?\d*(?:\.\d+)?)([diouxXeEfFgGcrs])")
BACKSLASH_BAD = re.compile(r'\\(?![ntr"\\])')  # any \ not a valid escape

def run(cmd):
    print(">", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

def align_newlines(src: str, dst: str) -> str:
    ls = len(src) - len(src.lstrip("\n"))
    ld = len(dst) - len(dst.lstrip("\n"))
    if ld != ls:
        dst = ("\n"*ls) + dst.lstrip("\n")
    ts = len(src) - len(src.rstrip("\n"))
    td = len(dst) - len(dst.rstrip("\n"))
    if td != ts:
        dst = dst.rstrip("\n") + ("\n"*ts)
    return dst

def escape_backslashes(s: str) -> str:
    return BACKSLASH_BAD.sub(r"\\\\", s)

def sanitize_percents_strict(text: str) -> str:
    """Replace invalid % sequences with %% ; keep valid: %%, %(name)s, %s/%d/%f/etc."""
    out = []
    i, L = 0, len(text)
    while i < L:
        ch = text[i]
        if ch != "%":
            out.append(ch); i += 1; continue
        # literal %%
        if i+1 < L and text[i+1] == "%":
            out.append("%%"); i += 2; continue
        # named?
        m_named = PH_NAMED.match(text, i)
        if m_named:
            out.append(m_named.group(0)); i = m_named.end(); continue
        # simple?
        m_simple = PH_SIMPLE.match(text, i)
        if m_simple:
            out.append(m_simple.group(0)); i = m_simple.end(); continue
        # invalid → escape just the %
        out.append("%%"); i += 1
    return "".join(out)

def build_dummy_from_msgid(msgid: str):
    names = [m.group(1) for m in PH_NAMED.finditer(msgid)]
    simples = [m.group(1) for m in PH_SIMPLE.finditer(PH_NAMED.sub("", msgid))]
    mapping = {name: 1 for name in names}
    tup = []
    for conv in simples:
        if conv in "diouxX":
            tup.append(1)
        elif conv in "eEfFgG":
            tup.append(1.0)
        elif conv in "crs":
            tup.append("X")
        else:
            tup.append("X")
    return mapping, tuple(tup), bool(names), bool(simples)

def formats_ok(msgid: str, msgstr: str) -> bool:
    mapping, tup, has_named, has_simple = build_dummy_from_msgid(msgid)
    # If msgid mixes named & simple, we won’t try to simulate → force fallback.
    if has_named and has_simple:
        return False
    try:
        if has_named:
            _ = msgstr % mapping
        elif has_simple:
            _ = msgstr % tup
        # no placeholders → OK
        return True
    except Exception:
        return False

def repair_scalar(msgid: str, msgstr: str) -> str:
    s = msgstr or ""
    s = escape_backslashes(s)
    s = sanitize_percents_strict(s)
    s = align_newlines(msgid, s)
    return s

def process_po(po_path: Path, lang: str) -> int:
    po = polib.pofile(str(po_path))

    # Patch plural header if we have a known good one
    hdr = po.metadata
    if lang in PLURAL_FORMS:
        pf = PLURAL_FORMS[lang]
        if hdr.get("Plural-Forms") != pf:
            hdr["Plural-Forms"] = pf

    # figure out expected nplurals from header
    nplurals = 2
    m = re.search(r"nplurals\s*=\s*(\d+)", hdr.get("Plural-Forms",""))
    if m:
        try: nplurals = int(m.group(1))
        except: nplurals = 2

    changed = False
    for e in po:
        if e.obsolete:
            continue

        if not e.msgid_plural:
            new = repair_scalar(e.msgid, e.msgstr or "")
            if new != (e.msgstr or ""):
                e.msgstr = new; changed = True
            if not formats_ok(e.msgid, e.msgstr):
                e.msgstr = e.msgid; changed = True
        else:
            # ensure correct number of plural slots
            for i in range(nplurals):
                if i not in e.msgstr_plural:
                    e.msgstr_plural[i] = ""
                    changed = True
            # repair and validate each slot
            for i in range(nplurals):
                src = e.msgid if i == 0 else (e.msgid_plural or e.msgid)
                cur = e.msgstr_plural.get(i, "")
                new = repair_scalar(src, cur)
                if new != cur:
                    e.msgstr_plural[i] = new; changed = True
                if not formats_ok(src, e.msgstr_plural[i]):
                    # safe fallback to the source for this slot
                    e.msgstr_plural[i] = src; changed = True

        if "fuzzy" in e.flags:
            e.flags = [f for f in e.flags if f != "fuzzy"]

    if changed:
        po.save(str(po_path))
    return 1 if changed else 0

def main():
    # Refresh catalogs (ok if it fails; we’ll still repair)
    try:
        run([sys.executable, "manage.py", "makemessages", "-a",
             "--ignore=venv", "--ignore=node_modules", "--ignore=static"])
    except subprocess.CalledProcessError:
        print("makemessages failed once; continuing with repairs…")

    any_change = 0
    if not LOCALE_DIR.exists():
        print("No locale/ directory found."); return

    for lang_dir in LOCALE_DIR.iterdir():
        if not lang_dir.is_dir(): continue
        lang = lang_dir.name
        po_path = lang_dir / "LC_MESSAGES" / "django.po"
        if po_path.exists():
            print(f"[{lang}] fixing {po_path}")
            any_change |= process_po(po_path, lang)

    # Re-merge and final tidy
    run([sys.executable, "manage.py", "makemessages", "-a",
         "--ignore=venv", "--ignore=node_modules", "--ignore=static"])

    for lang_dir in LOCALE_DIR.iterdir():
        if not lang_dir.is_dir(): continue
        lang = lang_dir.name
        po_path = lang_dir / "LC_MESSAGES" / "django.po"
        if po_path.exists():
            print(f"[{lang}] final tidy {po_path}")
            process_po(po_path, lang)

    run([sys.executable, "manage.py", "compilemessages"])
    print("✅ Done.")

if __name__ == "__main__":
    main()
