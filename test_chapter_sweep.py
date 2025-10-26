#!/usr/bin/env python3
r"""
Chapter config sweeper (no file writes) — bullet, punchy, one-thought chapters.

Example (Windows CMD; carets for line-breaks):

  python -u test_chapter_sweep.py ^
    --config ".\config.yaml" ^
    --transcript "C:\Users\isaac\podcast_data\transcripts\Redacted_News\en\Trump_pushes_land_war_in_Venezuela_and_Russia_goes_nuclear_Redacted_News_Live.json" ^
    --grid chapter_generation.similarity_threshold=0.55,0.60,0.65 ^
    --grid chapter_generation.aggregation_window_size=30,40,50 ^
    --grid chapter_generation.title.min_gap_sec=75,90,120 ^
    --style bullet ^
    --bullet-min 4 ^
    --bullet-max 6 ^
    --show 12 ^
    --fp16
"""

import os, sys, json, math, time, argparse, itertools, re, warnings
from pathlib import Path
from collections import defaultdict

import yaml
import torch
import numpy as np

# Optional deps (we guard so the script still runs if not present)
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        pass
try:
    from nltk.tokenize import sent_tokenize
except Exception:
    def sent_tokenize(x):  # minimal fallback
        return re.split(r'(?<=[.!?])\s+', x.strip())

from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F


# ------------------------------- Sponsor / cleanup -------------------------------

SPONSOR_PATTERNS = [
    r"\bthis episode (is|was)\s+(brought to you by|brought to|presented by)\b",
    r"\bsponsored by\b",
    r"\bad(vertisement)?\s*(break|read)?\b",
    r"\bpromo\s*code\b",
    r"\b(use|enter)\s+code\s+[A-Za-z0-9\-]+\b",
    r"\bnord\s?vpn\b", r"\bsquarespace\b", r"\baudible\b", r"\bshopify\b",
    r"\bbetterhelp\b", r"\bhellofresh\b", r"\braycon\b", r"\braid\s+shadow\s+legends\b",
    r"\bfarmers?\s+dog\b", r"\bmanscaped\b",
]
_SPONSOR_RE = re.compile("|".join(SPONSOR_PATTERNS), re.I)

SMALL_WORDS = {"a","an","and","as","at","but","by","for","in","nor","of","on","or","per","so","the","to","via"}
PRONOUN_STARTS = tuple(["i ", "i’m", "im ", "you ", "we ", "they "])
BAD_END = {"that","and","the","of","to","for"}

def is_sponsor_segment(text: str) -> bool:
    return bool(_SPONSOR_RE.search(" " + (text or "") + " "))

def titlecase_compact(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    words = s.split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i != 0 and lw in SMALL_WORDS:
            out.append(lw)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)

def clean_text_basic(t: str) -> str:
    t = (t or "").replace("\u200b", " ").strip()
    # Windows transcript quirks
    t = re.sub(r'\bU\.\s?S\.?\b', 'US', t)   # "U.S." -> "US"
    t = re.sub(r'\bU\.\b', 'You', t)         # rare "U." -> "You"
    t = re.sub(r"\s+", " ", t)
    return t

# ------------------------------- Argparse -------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Chapter config sweeper (no writes)")
    ap.add_argument("--config", required=True, help="Path to YAML config (unchanged)")
    ap.add_argument("--transcript", required=True, help="Path to episode transcript JSON")
    ap.add_argument("--grid", action="append", default=[],
                    help="Override grid, e.g. chapter_generation.similarity_threshold=0.55,0.6")
    ap.add_argument("--style", default="bullet", choices=["bullet","sentence","title"],
                    help="Output style for chapter titles")
    ap.add_argument("--bullet-min", type=int, default=3, help="Min words for bullet style")
    ap.add_argument("--bullet-max", type=int, default=7, help="Max words for bullet style")
    ap.add_argument("--show", type=int, default=12, help="How many chapters to print")
    ap.add_argument("--device", default=None, help="torch device (cuda:0 / cpu). Default: auto")
    ap.add_argument("--fp16", action="store_true", help="Half precision for models")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ------------------------------- Utilities -------------------------------

def set_seed(seed):
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments") or []
    # normalize fields we use: start, text
    out = []
    for s in segs:
        txt = clean_text_basic(s.get("text",""))
        if txt:
            out.append({"start": float(s.get("start", 0.0)), "text": txt})
    return out, data

def product_grid(base_cfg, grid_specs):
    """
    grid_specs: list like ["a.b=1,2", "c=foo,bar"]
    returns list of (variant_cfg, label_dict)
    """
    assigns = []
    for spec in grid_specs:
        if "=" not in spec:
            continue
        k, vals = spec.split("=", 1)
        keys = k.strip().split(".")
        vals = [v.strip() for v in vals.split(",") if v.strip() != ""]
        assigns.append((keys, vals))

    if not assigns:
        return [(base_cfg, {})]

    all_value_lists = [vals for _, vals in assigns]
    combos = list(itertools.product(*all_value_lists))

    variants = []
    for combo in combos:
        cfg = json.loads(json.dumps(base_cfg))  # deep copy via json
        labels = {}
        for (keys, _), val in zip(assigns, combo):
            labels[".".join(keys)] = val
            # set nested
            d = cfg
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            # cast numbers if they look numeric
            v = val
            if re.fullmatch(r"-?\d+(\.\d+)?", val):
                v = float(val) if "." in val else int(val)
            d[keys[-1]] = v
        variants.append((cfg, labels))
    return variants

def aggregate_segments_with_stride(segments, window_size=30, stride=None):
    if not segments:
        return []
    if stride is None or stride <= 0:
        stride = max(1, window_size // 2)
    windows = []
    i = 0
    while i < len(segments):
        win = segments[i:i+window_size]
        if not win:
            break
        text = " ".join(s.get("text","") for s in win).strip()
        start = float(win[0].get("start", 0.0))
        windows.append({"text": text, "start": start})
        if i + window_size >= len(segments):
            break
        i += stride
    return windows

# ------------------------------- Models -------------------------------

def init_models(cfg, device=None, fp16=False):
    # summarizer
    sum_model = cfg["summarizer"]["model"]
    summarizer = hf_pipeline(
        "summarization",
        model=sum_model,
        tokenizer=sum_model,
        framework="pt",
        device=(0 if (device and "cuda" in device) else -1)
    )
    if fp16 and hasattr(summarizer, "model") and hasattr(summarizer.model, "half"):
        try:
            summarizer.model.half()
        except Exception:
            pass

    # sentence model
    sm = SentenceTransformer('all-MiniLM-L6-v2', device=("cuda" if (device and "cuda" in device) else "cpu"))
    sm.eval()
    try:
        sm.max_seq_length = 256   # cap length to avoid warnings
    except Exception:
        pass
    if fp16:
        try:
            sm.half()
        except Exception:
            pass

    return summarizer, sm

# ------------------------------- Title helpers -------------------------------

def best_sentence_by_noun_density(text):
    sents = [s.strip() for s in sent_tokenize(text) if s and len(s.strip().split()) >= 5]
    if not sents:
        return text.strip()
    if _NLP:
        scores = []
        for s in sents:
            doc = _NLP(s)
            content = sum(1 for t in doc if t.pos_ in ("NOUN","PROPN"))
            length = len(doc)
            score = (content + 1e-6) / (length + 1e-6)
            scores.append(score)
        return sents[int(np.argmax(scores))]
    # fallback: pick the most “punctuated” sentence
    return max(sents, key=lambda s: s.count(",") + s.count(" ") / (len(s)+1e-6))

def clamp_to_full_sentence(text):
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return t
    if re.search(r"[.!?][\"’”)]?$", t):
        return t
    m = re.search(r"(.+[.!?])[\"’”)]?\s*$", t)
    return m.group(1).strip() if m else ""

def compact_title_from_text(text, min_words=4, max_words=8):
    # noun-chunk driven compact phrase
    t = re.sub(r"[^\w\s\-':\.]", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if _NLP:
        doc = _NLP(t)
        chunks = [nc.text for nc in getattr(doc, "noun_chunks", [])]
        cands = []
        for c in chunks:
            w = c.strip().split()
            if 2 <= len(w) <= 6:
                cands.append(c.strip())
        if cands:
            s = cands[0]
            return titlecase_compact(s)
    # fallback: first 6-ish strong tokens
    toks = [w for w in re.findall(r"[A-Za-z0-9']+", t) if len(w) > 2]
    return titlecase_compact(" ".join(toks[:max_words]) or t[:64])

def punchify(text, min_words=3, max_words=7):
    # turn a sentence into a terse bullet, keep content words, drop trailing junk
    text = clean_text_basic(text)
    # prefer noun chunks if spaCy is available
    if _NLP:
        doc = _NLP(text)
        cands = []
        for nc in getattr(doc, "noun_chunks", []):
            w = nc.text.strip().split()
            if 2 <= len(w) <= 7:
                cands.append(nc.text.strip())
        if cands:
            cand = cands[0]
            words = cand.split()
            while words and words[-1].lower() in BAD_END:
                words.pop()
            words = words[:max_words]
            return titlecase_compact(" ".join(words))
    # fallback: simple compaction
    words = re.findall(r"[A-Za-z0-9']+", text)
    words = [w for w in words if len(w) > 2][:max_words]
    while words and words[-1].lower() in BAD_END:
        words.pop()
    if len(words) < min_words:
        return titlecase_compact(" ".join(words))
    return titlecase_compact(" ".join(words[:max_words]))

def validate_title(t, style, cfg):
    if not t:
        return False
    t0 = t.strip()
    if not t0:
        return False
    if t0.lower().startswith(PRONOUN_STARTS):
        return False
    if is_sponsor_segment(t0):
        return False
    rules = cfg.get("chapter_generation",{}).get("title",{})
    if style == "bullet":
        min_w = 3
        max_w = 7
    else:
        min_w = int(rules.get("min_words", 4))
        max_w = int(rules.get("max_words", 10))
    wc = len(t0.split())
    if wc < min_w or wc > max_w:
        return False
    if _NLP:
        doc = _NLP(t0)
        if not any(tok.pos_ in ("NOUN","PROPN","VERB","NUM") for tok in doc):
            return False
    return True

def embed(sentence_model, text):
    if not text:
        return None
    v = sentence_model.encode([text], convert_to_tensor=True, show_progress_bar=False)
    return F.normalize(v, p=2, dim=1)

def too_similar(sentence_model, title, prev_titles, threshold=0.60):
    if not prev_titles:
        return False
    a = embed(sentence_model, title)
    b = sentence_model.encode(prev_titles, convert_to_tensor=True, show_progress_bar=False)
    b = F.normalize(b, p=2, dim=1)
    sim = torch.matmul(a, b.T).squeeze(0)
    return bool(torch.any(sim >= threshold).item())

# ------------------------------- Chaptering core -------------------------------

def generate_title_for_window(window_text, summarizer, cfg, style, bullet_min, bullet_max):
    # sponsor skip at text level
    if is_sponsor_segment(window_text):
        return None

    rules = cfg.get("chapter_generation",{}).get("title",{})
    max_len = int(cfg["summarizer"]["max_length"])
    min_len = int(cfg["summarizer"]["min_length"])

    seg = clean_text_basic(window_text)
    # Hard input clamp for the model (tokenizer also gets truncation=True)
    if len(seg) > 2000:
        seg = seg[:2000]

    # Try abstractive first
    try:
        raw = summarizer(
            seg,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True,
            truncation=True,           # avoid token length issues
        )[0]["summary_text"]
    except Exception:
        raw = ""

    raw = clean_text_basic(raw)

    # Ensure one complete thought
    sent = clamp_to_full_sentence(raw)
    if not sent:
        sent = best_sentence_by_noun_density(seg)

    if style == "bullet":
        # small pre-cap before punchify to reduce odd endings
        words = sent.split()
        if len(words) > 16:
            sent = " ".join(words[:16])
        t = punchify(sent, min_words=bullet_min, max_words=bullet_max)
    elif style == "sentence":
        t = clamp_to_full_sentence(sent) or sent
    else:
        t = compact_title_from_text(sent, min_words=rules.get("min_words",4), max_words=rules.get("max_words",8))

    # strip trailing weak words
    words = t.split()
    while words and words[-1].lower() in BAD_END:
        words.pop()
    t = " ".join(words).strip()

    return titlecase_compact(t)

def build_chapters(segments, cfg, summarizer, sentence_model, style, bullet_min, bullet_max):
    cg = cfg["chapter_generation"]
    sim_thr = float(cg["similarity_threshold"])
    max_chapters = int(cg["max_chapters"])
    win = int(cg["aggregation_window_size"])

    title_rules = cg.get("title", {})
    min_first_sec = int(title_rules.get("min_first_chapter_sec", 90))
    min_gap_sec = int(title_rules.get("min_gap_sec", 90))

    windows = aggregate_segments_with_stride(segments, window_size=win, stride=max(1, win//2))

    chapters = [{"title": "Intro", "time": "0:00"}]
    last_start = 0.0
    skipped = defaultdict(int)

    for idx, w in enumerate(windows):
        txt = w["text"]
        start = float(w["start"])

        if not txt:
            continue

        if idx == 0 and start < min_first_sec:
            start = float(min_first_sec)

        if idx > 0 and start < min_first_sec:
            skipped["early"] += 1
            continue

        if is_sponsor_segment(txt):
            skipped["sponsor"] += 1
            continue

        if (start - last_start) < min_gap_sec:
            skipped["too_close"] += 1
            continue

        # similarity gate vs previous window text (coarse topical shift)
        if idx > 0:
            prev_txt = windows[idx-1]["text"]
            a = embed(sentence_model, txt)
            b = embed(sentence_model, prev_txt)
            if a is not None and b is not None:
                s = torch.matmul(a, b.T).item()
                if s >= sim_thr and (start - last_start) < (2 * min_gap_sec):
                    skipped["similarity"] += 1
                    continue

        title = generate_title_for_window(txt, summarizer, cfg, style, bullet_min, bullet_max)
        if not title:
            skipped["invalid_title"] += 1
            continue

        # uniqueness vs existing chapter titles
        if too_similar(sentence_model, title, [c["title"] for c in chapters], threshold=0.60):
            skipped["duplicate"] += 1
            continue

        ts = f"{int(start//60)}:{int(start%60):02d}"
        chapters.append({"title": title, "time": ts})
        last_start = start
        if len(chapters) >= max_chapters:
            break

    # median gap (rough metric)
    if len(chapters) > 1:
        starts = [0] + [
            int(c["time"].split(":")[0])*60 + int(c["time"].split(":")[1])
            for c in chapters[1:]
        ]
        gaps = [s2 - s1 for s1, s2 in zip(starts, starts[1:])]
        med_gap = int(np.median(gaps)) if gaps else 0
        first_sec = int(starts[1]) if len(starts) > 1 else 0
    else:
        med_gap = 0
        first_sec = 0

    return chapters, dict(skipped), med_gap, first_sec

# ------------------------------- Main -------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    assert os.path.exists(args.config), f"Config not found: {args.config}"
    assert os.path.exists(args.transcript), f"Transcript not found: {args.transcript}"

    cfg = load_yaml(args.config)
    segments, meta = load_transcript(args.transcript)

    # choose device
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    summarizer, sentence_model = init_models(cfg, device=device, fp16=args.fp16)

    # Compose sweep variants
    variants = product_grid(cfg, args.grid)
    print(f"\n=== Chapter Config Sweep (no writes) ===")
    print(f"Transcript: {args.transcript}")
    print(f"Variants: {len(variants)}  Style: {args.style}\n")

    for i, (cfg_i, labels) in enumerate(variants, start=1):
        t0 = time.time()
        chapters, skipped, median_gap, first_sec = build_chapters(
            segments, cfg_i, summarizer, sentence_model,
            style=args.style, bullet_min=args.bullet_min, bullet_max=args.bullet_max
        )
        dt = time.time() - t0
        cg = cfg_i["chapter_generation"]
        sim = cg["similarity_threshold"]
        win = cg["aggregation_window_size"]
        min_gap = cg.get("title",{}).get("min_gap_sec", cfg["chapter_generation"]["title"]["min_gap_sec"])
        sumlen = [cfg_i["summarizer"]["min_length"], cfg_i["summarizer"]["max_length"]]

        print(f"#{i} sim={sim}, win={win}, gap={min_gap}, first={first_sec}, sumlen={sumlen}")
        print(f"  -> chapters={len(chapters)}  first_sec={first_sec}  median_gap={median_gap}s  skipped={skipped}  time={dt:.2f}s")

        # show first N
        shown = 0
        for ch in chapters[:args.show]:
            print(f"    - {ch['time']:>4}  {ch['title']}")
            shown += 1
        if len(chapters) > shown:
            print(f"    ... (+{len(chapters)-shown} more)")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
