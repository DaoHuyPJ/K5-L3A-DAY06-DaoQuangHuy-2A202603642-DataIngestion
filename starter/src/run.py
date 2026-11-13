"""Day 06 pipeline entry point.

Run from the ``starter/`` directory:

    python -m src.run ingest          # HTML -> normalized units + manifest + logs
    python -m src.run benchmark       # verify evidence in corpus, fill gold_page
    python -m src.run index           # chunk A/B, embed, upsert into Chroma (on disk)
    python -m src.run reindex-check   # re-run ingest; prove no duplicates / stale rows
    python -m src.run evaluate --split dev    # A/B/C top-3 traces + Hit@1/Hit@3
    python -m src.run evaluate --split test
    python -m src.run audit-sheets    # samples for manual coherence/metadata/grounding
    python -m src.run validate        # integrity / traceability checks
    python -m src.run analyze         # breakdown by question type, sizes, latency
    python -m src.run hybrid          # optional BM25 + dense extension (tuned on dev)

Strategy C re-uses the B index (identical chunks and vectors) and only adds a
metadata filter for questions that declare one; this isolates the effect of
filtering (B vs C).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf, normalize_for_match, parse_law_html, build_units
from .models import Chunk, DocumentPage

ROOT = Path(__file__).resolve().parents[1]          # starter/  (.env, vectorstore/)
REPO = ROOT.parent                                  # repository root (SUBMISSION.md layout)
DATA = REPO / "data"
RESULTS = REPO / "results"
BENCH = REPO / "benchmark"

DOCUMENT = {
    "source_id": "BLLD-2019",
    "title": "Bộ luật Lao động số 45/2019/QH14",
    "html_path": DATA / "raw" / "bld2019_pbgdpl_cantho.html",
    "html_url": "https://pbgdpl.cantho.gov.vn/attachment/261",
    "official_pdf_path": DATA / "raw" / "45-2019-QH14.pdf",
    "official_pdf_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf",
}

STRATEGIES: dict[str, dict[str, Any]] = {
    "A": {"chunking": "fixed_size", "chunk_size": 500, "overlap": 50, "index": "A", "filter": False,
          "description": "Cửa sổ 500 ký tự, overlap 50, không theo cấu trúc; không lọc"},
    "B": {"chunking": "by_paragraph", "chunk_size": 800, "overlap": 0, "index": "B", "filter": False,
          "description": "Gom đoạn trong cùng một Điều, tối đa 800 ký tự, thêm tên Điều cho chunk tiếp nối; không lọc"},
    "C": {"chunking": "by_paragraph", "chunk_size": 800, "overlap": 0, "index": "B", "filter": True,
          "description": "Giống hệt B (cùng index) + lọc theo `chapter` khi câu hỏi có filter"},
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_units() -> list[DocumentPage]:
    path = DATA / "processed" / "units.jsonl"
    if not path.exists():
        sys.exit("data/processed/units.jsonl missing: run `python -m src.run ingest` first")
    return [DocumentPage(**r) for r in _read_jsonl(path)]


def build_chunks(strategy: str, units: list[DocumentPage]) -> list[Chunk]:
    cfg = STRATEGIES[strategy]
    if cfg["chunking"] == "fixed_size":
        return [c for u in units for c in fixed_size(u, cfg["chunk_size"], cfg["overlap"], strategy)]
    return [c for u in units for c in by_paragraph(u, cfg["chunk_size"], strategy)]


VECTORSTORE = ROOT / "vectorstore"


def get_store(embedder):
    from .store import ChromaStore
    return ChromaStore(VECTORSTORE, embedder.dimensions, embedder.model_name)


def get_embedder():
    from .embeddings import DEFAULT_MODEL, E5Embedder
    load_dotenv(ROOT / ".env")
    return E5Embedder(os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL,
                      local_path=os.getenv("EMBEDDING_MODEL_PATH") or None)


def _length_stats(values: list[int]) -> dict[str, Any]:
    s = sorted(values)
    return {"count": len(s), "total": sum(s), "mean": round(statistics.mean(s), 1),
            "median": statistics.median(s), "min": s[0], "max": s[-1],
            "p10": s[len(s) // 10], "p90": s[(9 * len(s)) // 10]}


# --------------------------------------------------------------------------- #
# 1. ingest
# --------------------------------------------------------------------------- #
def cmd_ingest(_: argparse.Namespace) -> None:
    started = time.perf_counter()
    log: dict[str, Any] = {"run_at": _now(), "errors": [], "warnings": []}

    # Official PDF: check whether it has a text layer (it is a scan).
    pdf = DOCUMENT["official_pdf_path"]
    if pdf.exists():
        pages, errors = load_pdf(pdf, DOCUMENT["source_id"], DOCUMENT["official_pdf_url"])
        log["official_pdf"] = {"physical_pages": len(pages) + len(errors),
                               "pages_with_text": [p.page for p in pages],
                               "pages_without_text": len(errors),
                               "decision": "PDF là bản scan không có lớp text; không OCR. Dùng bản HTML "
                                           "toàn văn làm nguồn text, PDF chỉ để đối chiếu."}
        log["errors"].extend(errors)
    else:
        log["warnings"].append(f"official PDF not found at {pdf}")

    paragraphs, parse_log = parse_law_html(DOCUMENT["html_path"])
    log["html_parse"] = parse_log
    units = build_units(paragraphs, DOCUMENT["source_id"], DOCUMENT["html_url"])
    short = [u.page for u in units if not u.metadata["counts_toward_30"]]
    if short:
        log["warnings"].append({"units_below_1500_chars_not_counted": short})
    counted = sum(u.metadata["counts_toward_30"] for u in units)
    if counted < 30:
        raise SystemExit(f"Only {counted} units >= 1500 chars; corpus too small")

    _write_jsonl(DATA / "processed" / "units.jsonl", [asdict(u) for u in units])

    manifest = {
        "corpus": "Bộ luật Lao động 2019 (Việt Nam)",
        "generated_at": _now(),
        "unit_rule": ("Nguồn HTML không phân trang → chia tuần tự thành 'đơn vị trang' gồm các Điều "
                      "trọn vẹn liên tiếp, mỗi đơn vị >= 1.500 ký tự sau chuẩn hóa NFC; không vượt "
                      "ranh giới Chương. Chỉ đơn vị >= 1.500 ký tự được tính vào ngưỡng 30. "
                      "`page` trong metadata/benchmark = số thứ tự đơn vị (1..N)."),
        "sources": [{
            "source_id": DOCUMENT["source_id"],
            "title": DOCUMENT["title"],
            "issuer": "Quốc hội nước CHXHCN Việt Nam, khóa XIV, kỳ họp thứ 8",
            "adopted": "2019-11-20",
            "effective": "2021-01-01 (Điều 220 khoản 1)",
            "version_note": ("Văn bản gốc năm 2019 như được ban hành; KHÔNG phải văn bản hợp nhất. "
                             "Một số điều đã được sửa đổi về sau; corpus không phản ánh các sửa đổi đó."),
            "source_path_or_url": DOCUMENT["html_url"],
            "local_snapshot": str(DOCUMENT["html_path"].relative_to(REPO)).replace("\\", "/"),
            "reference_official_pdf": DOCUMENT["official_pdf_url"],
            "license_or_permission": ("Văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả "
                                      "(Điều 15 khoản 2 Luật Sở hữu trí tuệ 2005, sửa đổi). Bản HTML do Trang "
                                      "thông tin PBGDPL TP. Cần Thơ (cơ quan nhà nước) đăng công khai. Không "
                                      "chứa dữ liệu cá nhân."),
            "accessed_at": "2026-11-13",
            "physical_pages_official_pdf": log.get("official_pdf", {}).get("physical_pages"),
            "units_total": len(units),
            "units_counted_ge_1500": counted,
            "units_not_counted": short,
            "chars_total": sum(u.metadata["char_count"] for u in units),
            "excluded": "Phần đầu (quốc hiệu, căn cứ) trước Chương I; chữ ký và footer website sau Điều 220.",
            "units": [{"page": u.page, "unit_id": u.metadata["unit_id"], "chapter": u.metadata["chapter"],
                       "articles": u.metadata["article_range"], "chars": u.metadata["char_count"],
                       "counted": u.metadata["counts_toward_30"]} for u in units],
        }],
    }
    _write_json(DATA / "manifest.json", manifest)
    log["units"] = len(units)
    log["units_counted"] = counted
    log["seconds"] = round(time.perf_counter() - started, 2)
    _write_json(RESULTS / "logs" / "ingest_log.json", log)
    print(f"units={len(units)} counted>=1500={counted} chars={manifest['sources'][0]['chars_total']} "
          f"pdf_pages_without_text={log.get('official_pdf', {}).get('pages_without_text')}")


# --------------------------------------------------------------------------- #
# 2. benchmark verification
# --------------------------------------------------------------------------- #
def cmd_benchmark(_: argparse.Namespace) -> None:
    """Verify every evidence quote exists in the normalized corpus and record
    every unit that contains it as gold_page.  Run once, before experiments."""
    units = load_units()
    path = BENCH / "questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    by_unit = {u.page: normalize_for_match(u.text) for u in units}
    report, problems = [], []
    for q in questions:
        quote = normalize_for_match(q["evidence_quote"])
        found = [p for p, t in by_unit.items() if quote in t]
        if not found:
            problems.append(f"{q['id']}: evidence not found in corpus")
        elif q.get("gold_page") not in (None, found, found[0] if len(found) == 1 else None):
            problems.append(f"{q['id']}: recorded gold_page {q['gold_page']} != found {found}")
        q["gold_page"] = found[0] if len(found) == 1 else found
        if q.get("filter"):
            for key, value in q["filter"].items():
                if not any(u.metadata.get(key) == value for u in units):
                    problems.append(f"{q['id']}: filter value {key}={value} does not exist")
                if found and not all(units[p - 1].metadata.get(key) == value for p in found):
                    problems.append(f"{q['id']}: gold page not inside filter scope")
        report.append({"id": q["id"], "split": q["split"], "gold_page": q["gold_page"],
                       "evidence_chars": len(q["evidence_quote"]), "found_in_units": found})
    splits = {s: sum(q["split"] == s for q in questions) for s in ("dev", "test")}
    filt = {s: sum(bool(q.get("filter")) and q["split"] == s for q in questions) for s in ("dev", "test")}
    summary = {"verified_at": _now(), "questions": len(questions), "splits": splits,
               "filter_questions": filt, "problems": problems, "items": report}
    _write_json(BENCH / "verification.json", summary)
    if problems:
        print("\n".join(problems))
        raise SystemExit("benchmark verification FAILED")
    if len(questions) < 20 or splits["dev"] < 15 or splits["test"] < 5 or sum(filt.values()) < 3 or filt["test"] < 1:
        raise SystemExit(f"benchmark does not meet minimums: {splits} {filt}")
    data["questions"] = questions
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(questions)} questions {splits}, filter {filt}; evidence verified")


# --------------------------------------------------------------------------- #
# 3. index
# --------------------------------------------------------------------------- #
def cmd_index(args: argparse.Namespace) -> None:
    units = load_units()
    embedder = get_embedder()
    store = get_store(embedder)
    store.create_schema()
    report: dict[str, Any] = {"run_at": _now(), "embedding": embedder.describe(),
                              "vector_store": {**store.versions(), "client": "chromadb.PersistentClient",
                                               "collections": [f"bld2019_{i}" for i in args.indexes],
                                               "metric": "cosine (hnsw:space=cosine)",
                                               "search": "approximate HNSW; verified against brute-force in evaluate"},
                              "python": platform.python_version(), "strategies": {}}
    for index_name in args.indexes:
        cfg = STRATEGIES[index_name]
        t0 = time.perf_counter()
        chunks = build_chunks(index_name, units)
        t_chunk = time.perf_counter() - t0
        texts = [c.text for c in chunks]
        t1 = time.perf_counter()
        vectors = embedder.embed_documents(texts)
        t_embed = time.perf_counter() - t1
        t2 = time.perf_counter()
        upsert = store.upsert(index_name, chunks, vectors)
        t_store = time.perf_counter() - t2
        tok = embedder.model.tokenizer
        token_counts = [len(tok(f"passage: {t}")["input_ids"]) for t in texts]
        report["strategies"][index_name] = {
            "config": cfg,
            "chunks": len(chunks),
            "chars": _length_stats([len(t) for t in texts]),
            "tokens": _length_stats(token_counts),
            "truncated_over_max_seq": embedder.truncated_count,
            "with_context_prefix": sum(bool(c.metadata.get("context_prefix")) for c in chunks),
            "seconds": {"chunking": round(t_chunk, 2), "embedding": round(t_embed, 2),
                        "upsert": round(t_store, 2), "total": round(t_chunk + t_embed + t_store, 2)},
            "upsert": upsert,
            "db_count_after": store.count(index_name),
            "cost_usd": 0.0,
        }
        _write_jsonl(RESULTS / "chunks" / f"chunks_{index_name}.jsonl", [asdict(c) for c in chunks])
        print(index_name, json.dumps(report["strategies"][index_name]["seconds"]), upsert)
    _write_json(RESULTS / "index_report.json", report)


# --------------------------------------------------------------------------- #
# 4. re-index check (idempotency + stale-chunk removal)
# --------------------------------------------------------------------------- #
def cmd_reindex_check(_: argparse.Namespace) -> None:
    units = load_units()
    embedder = get_embedder()
    store = get_store(embedder)
    out: dict[str, Any] = {"run_at": _now(), "same_version": {}, "changed_document_simulation": {}}

    # (a) re-ingest the same version into the real indexes
    for name in ("A", "B"):
        before = store.count(name)
        chunks = build_chunks(name, units)
        stats = store.upsert(name, chunks, embedder.embed_documents([c.text for c in chunks]))
        after = store.count(name)
        out["same_version"][name] = {"before": before, "upsert": stats, "after": after,
                                     "no_duplicates": after == before and after["rows"] == after["distinct_chunk_ids"]}

    # (b) simulate a changed document in a scratch strategy "_reindex_test"
    test = "_reindex_test"
    store.delete_strategy(test)
    sample = units[:3]
    v1 = build_chunks("B", sample)
    v1 = [Chunk(c.chunk_id.replace("B:", f"{test}:", 1), c.document_id, c.source, c.page, c.text, c.metadata) for c in v1]
    s1 = store.upsert(test, v1, embedder.embed_documents([c.text for c in v1]))
    # version 2: unit 1 text edited (an article removed) -> fewer chunks + changed hashes
    edited = DocumentPage(**{**asdict(sample[0])})
    cut = edited.metadata["paragraphs"]
    keep = [p for p in cut if p["article"] != 2]
    edited.text = "\n".join(edited.text[p["start"]:p["end"]] for p in keep)
    pos, new_par = 0, []
    for p in keep:
        n = p["end"] - p["start"]
        new_par.append({**p, "start": pos, "end": pos + n})
        pos += n + 1
    edited.metadata = {**edited.metadata, "paragraphs": new_par}
    v2 = build_chunks("B", [edited] + sample[1:])
    v2 = [Chunk(c.chunk_id.replace("B:", f"{test}:", 1), c.document_id, c.source, c.page, c.text, c.metadata) for c in v2]
    s2 = store.upsert(test, v2, embedder.embed_documents([c.text for c in v2]))
    final = store.count(test)
    db_ids = set(store.existing_hashes(test))
    out["changed_document_simulation"] = {
        "description": "Nạp 3 đơn vị đầu (chiến lược B) vào collection tạm, sau đó xóa Điều 2 khỏi đơn vị 1 và nạp lại.",
        "v1_chunks": len(v1), "v1_upsert": s1,
        "v2_chunks": len(v2), "v2_upsert": s2,
        "db_after": final,
        "db_ids_equal_v2_ids": db_ids == {c.chunk_id for c in v2},
    }
    store.delete_strategy(test)
    _write_json(RESULTS / "reindex_check.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------- #
# 5. evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(args: argparse.Namespace) -> None:
    questions = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))["questions"]
    if args.split != "all":
        questions = [q for q in questions if q["split"] == args.split]
    embedder = get_embedder()
    store = get_store(embedder)
    traces: list[dict[str, Any]] = []
    latency: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    qvecs = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    for strategy, cfg in STRATEGIES.items():
        for q in questions:
            applied = q.get("filter") if cfg["filter"] and q.get("filter") else None
            t0 = time.perf_counter()
            hits = store.search(cfg["index"], qvecs[q["id"]], top_k=3, filters=applied)
            latency[strategy].append(time.perf_counter() - t0)
            trace = evaluate_query(q, hits, strategy, applied)
            exact_ids = store.exact_search(cfg["index"], qvecs[q["id"]], top_k=3, filters=applied)
            trace["ann_equals_exact_top3"] = exact_ids == [h["chunk_id"] for h in hits]
            trace["note"] = ("C: câu không có filter → chạy không lọc (giống B)"
                             if strategy == "C" and not applied else None)
            traces.append(trace)
    summary = summarize(traces)
    ann_ok = sum(t["ann_equals_exact_top3"] for t in traces)
    print(f"ANN top-3 == exact top-3: {ann_ok}/{len(traces)}")
    run = {"run_at": _now(), "split": args.split, "embedding": embedder.describe(),
           "vector_store": store.versions(),
           "ann_equals_exact_top3": f"{ann_ok}/{len(traces)}",
           "strategies": STRATEGIES, "metrics": summary,
           "mean_search_ms": {s: round(1000 * statistics.mean(v), 2) for s, v in latency.items() if v}}
    _write_jsonl(RESULTS / "traces" / f"traces_{args.split}.jsonl", traces)
    _write_json(RESULTS / f"metrics_{args.split}.json", run)
    for strategy, splits in summary.items():
        for split, m in splits.items():
            if split == "all" and args.split != "all":
                continue
            print(f"{strategy} {split:4s} Hit@1={m['hit_at_1_str']:>6} Hit@3={m['hit_at_3_str']:>6} "
                  f"filter={m['filter_compliant']}/{m['filter_queries']} missed={m['missed_ids']}")


# --------------------------------------------------------------------------- #
# 6. audit sheets (manual labelling samples)
# --------------------------------------------------------------------------- #
def cmd_audit_sheets(args: argparse.Namespace) -> None:
    units = load_units()
    out_dir = RESULTS / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    labelled = [p.name for p in out_dir.glob("*.csv")
                if any(r.get("annotator") for r in csv.DictReader(p.open(encoding="utf-8-sig")))]
    if labelled and not args.force:
        raise SystemExit(f"Refusing to overwrite manual labels in {labelled}; pass --force to regenerate.")

    # Coherence: fixed positions chosen BEFORE looking at scores — the midpoint
    # character of each of the first 10 units (sorted by page id).
    rows = []
    chunks = {s: _read_jsonl(RESULTS / "chunks" / f"chunks_{s}.jsonl") for s in ("A", "B")}
    for u in units[:10]:
        offset = len(u.text) // 2
        for strategy, index in (("A", "A"), ("B", "B"), ("C", "B")):
            c = next(c for c in chunks[index] if c["page"] == u.page
                     and c["metadata"]["char_start"] <= offset < c["metadata"]["char_end"])
            rows.append({"position_id": f"P{u.page:02d}", "page": u.page, "offset": offset,
                         "strategy": strategy, "chunk_id": c["chunk_id"],
                         "chunk_chars": len(c["text"]), "chunk_text": c["text"],
                         "score_0_2": "", "reason": "", "annotator": ""})
    _write_csv(out_dir / "coherence_labels.csv", rows)

    test_traces_path = RESULTS / "traces" / "traces_test.jsonl"
    if not test_traces_path.exists():
        print("traces_test.jsonl missing; only coherence sheet written")
        return
    traces = _read_jsonl(test_traces_path)
    unit_by_page = {u.page: u for u in units}

    # Metadata: 5 test questions x top-3 per strategy, with an automatic
    # cross-check against the source unit to support the manual check.
    meta_rows = []
    for t in traces:
        for r in t["retrieved"]:
            u = unit_by_page[r["page"]]
            s, e = r["char_span"]
            body = u.text[s:e]
            text_ok = body in r["text"]
            arts_ok = set(r["articles"]) <= set(u.metadata["articles"]) and all(
                f"Điều {a}." in u.text for a in r["articles"])
            meta_rows.append({"strategy": t["strategy"], "question_id": t["id"], "rank": r["rank"],
                              "chunk_id": r["chunk_id"], "document_id": r["document_id"],
                              "source": r["source"], "page": r["page"], "chapter": r["chapter"],
                              "articles": " ".join(map(str, r["articles"])),
                              "question_filter": json.dumps(t["question_filter"], ensure_ascii=False) if t["question_filter"] else "",
                              "auto_text_at_page": text_ok,
                              "auto_chapter_matches_unit": r["chapter"] == u.metadata["chapter"],
                              "auto_articles_in_unit": arts_ok,
                              "manual_complete_and_correct": "", "error_note": "", "annotator": ""})
    _write_csv(out_dir / "metadata_audit.csv", meta_rows)

    # Grounding: 5 test questions x strategy, top-3 texts side by side.
    g_rows = []
    for t in traces:
        row = {"strategy": t["strategy"], "question_id": t["id"], "question": t["question"],
               "expected_answer": t["expected_answer"],
               "gold": "; ".join(f"{g['source_id']} p{g['pages']}" for g in t["gold"]),
               "applied_filter": json.dumps(t["applied_filter"], ensure_ascii=False) if t["applied_filter"] else "",
               "auto_evidence_hit_at_3": t["hit_at_3"]}
        for r in t["retrieved"]:
            row[f"top{r['rank']}"] = f"[{r['chunk_id']} | p{r['page']} | Điều {r['articles']}] {r['text']}"
        row.update({"grounding_0_2": "", "reason": "", "annotator": ""})
        g_rows.append(row)
    _write_csv(out_dir / "grounding_labels.csv", g_rows)
    print(f"wrote {len(rows)} coherence, {len(meta_rows)} metadata, {len(g_rows)} grounding rows")


# --------------------------------------------------------------------------- #
# 7. validate (integrity / traceability checks anyone can re-run)
# --------------------------------------------------------------------------- #
def cmd_validate(_: argparse.Namespace) -> None:
    units = load_units()
    U = {u.page: u for u in units}
    checks: dict[str, Any] = {}

    # units
    arts = sorted(a for u in units for a in u.metadata["articles"])
    checks["units_articles_1_to_220_once"] = arts == list(range(1, 221))
    # a "Chương" heading may only appear as the first paragraph of a unit -> no unit spans two chapters
    checks["units_never_cross_chapter"] = all(
        all(i == 0 for i, p in enumerate(u.metadata["paragraphs"]) if p["kind"] == "chapter") for u in units)
    checks["units_char_count_matches_text"] = all(u.metadata["char_count"] == len(u.text) for u in units)
    counted = sum(len(u.text) >= 1500 for u in units)
    checks["units_counted_ge_1500"] = counted
    checks["units_counted_ge_30"] = counted >= 30
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    checks["manifest_chars_match_units"] = manifest["sources"][0]["chars_total"] == sum(len(u.text) for u in units)

    # chunks: exact span traceability, coverage, boundaries, ids
    for name in ("A", "B"):
        chunks = _read_jsonl(RESULTS / "chunks" / f"chunks_{name}.jsonl")
        span_ok = chapter_ok = True
        multi_article = 0
        covered = {u.page: bytearray(len(u.text)) for u in units}
        for c in chunks:
            m, u = c["metadata"], U[c["page"]]
            body = u.text[m["char_start"]:m["char_end"]]
            expected = f"{m['context_prefix']}\n{body}" if m.get("context_prefix") else body
            span_ok &= c["text"] == expected
            chapter_ok &= m["chapter"] == u.metadata["chapter"]
            multi_article += len(m["articles"]) > 1
            covered[c["page"]][m["char_start"]:m["char_end"]] = b"\x01" * (m["char_end"] - m["char_start"])
        uncovered = sum(1 for u in units for i, ch in enumerate(u.text) if not covered[u.page][i] and ch != "\n")
        ids = [c["chunk_id"] for c in chunks]
        checks[f"chunks_{name}"] = {
            "count": len(chunks), "text_equals_source_span": span_ok, "chapter_matches_unit": chapter_ok,
            "uncovered_non_newline_chars": uncovered, "unique_ids": len(set(ids)) == len(ids),
            "chunks_spanning_multiple_articles": multi_article,
        }
    checks["chunks_B_never_cross_article"] = checks["chunks_B"]["chunks_spanning_multiple_articles"] == 0

    # benchmark evidence still present (read-only re-check)
    questions = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))["questions"]
    norm = {u.page: normalize_for_match(u.text) for u in units}
    checks["benchmark_evidence_at_gold_page"] = all(
        normalize_for_match(q["evidence_quote"]) in norm[p]
        for q in questions for p in (q["gold_page"] if isinstance(q["gold_page"], list) else [q["gold_page"]]))

    # store mirrors the chunk files (ids + content hashes)
    try:
        embedder = get_embedder()
        store = get_store(embedder)
        for name in ("A", "B"):
            chunks = _read_jsonl(RESULTS / "chunks" / f"chunks_{name}.jsonl")
            want = {c["chunk_id"]: c["metadata"]["content_hash"] for c in chunks}
            checks[f"store_{name}_matches_chunk_file"] = store.existing_hashes(name) == want
    except Exception as exc:  # report, do not hide
        checks["store_check_error"] = repr(exc)

    flat_ok = [v for k, v in checks.items() if isinstance(v, bool)]
    flat_ok += [v for k in ("chunks_A", "chunks_B") for kk, v in checks[k].items()
                if kk in ("text_equals_source_span", "chapter_matches_unit", "unique_ids")]
    flat_ok += [checks[k]["uncovered_non_newline_chars"] == 0 for k in ("chunks_A", "chunks_B")]
    checks["all_passed"] = all(flat_ok) and "store_check_error" not in checks
    checks["run_at"] = _now()
    _write_json(RESULTS / "validation.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    if not checks["all_passed"]:
        raise SystemExit("validation FAILED")


# --------------------------------------------------------------------------- #
# 8. analyze (breakdown by question type, sizes, latency)
# --------------------------------------------------------------------------- #
def cmd_analyze(_: argparse.Namespace) -> None:
    traces = _read_jsonl(RESULTS / "traces" / "traces_dev.jsonl") + _read_jsonl(RESULTS / "traces" / "traces_test.jsonl")
    out: dict[str, Any] = {"by_type": {}, "by_strategy_all": {}}
    types = sorted({t["type"] for t in traces})
    for s in STRATEGIES:
        for ty in types:
            items = [t for t in traces if t["strategy"] == s and t["type"] == ty]
            out["by_type"].setdefault(ty, {})[s] = {
                "N": len(items), "hit_at_1": f"{sum(t['hit_at_1'] for t in items)}/{len(items)}",
                "hit_at_3": f"{sum(t['hit_at_3'] for t in items)}/{len(items)}",
                "ids_missed": [t["id"] for t in items if not t["hit_at_3"]]}
        items = [t for t in traces if t["strategy"] == s]
        ranks = [t["first_hit_rank"] for t in items]
        out["by_strategy_all"][s] = {
            "N": len(items), "hit_at_1": f"{sum(t['hit_at_1'] for t in items)}/{len(items)}",
            "hit_at_3": f"{sum(t['hit_at_3'] for t in items)}/{len(items)}",
            "mrr_at_3": round(sum(1 / r for r in ranks if r) / len(items), 3),
            "evidence_split_suspected": [t["id"] for t in items if t["evidence_split_suspected"]]}
    idx = json.loads((RESULTS / "index_report.json").read_text(encoding="utf-8"))
    out["index"] = {s: {"chunks": d["chunks"], "chars_total": d["chars"]["total"],
                        "tokens_total": d["tokens"]["total"], "tokens_mean": d["tokens"]["mean"],
                        "tokens_max": d["tokens"]["max"], "index_seconds": d["seconds"]["total"]}
                    for s, d in idx["strategies"].items()}
    out["mean_search_ms"] = {sp: json.loads((RESULTS / f"metrics_{sp}.json").read_text(encoding="utf-8"))["mean_search_ms"]
                             for sp in ("dev", "test")}
    _write_json(RESULTS / "analysis.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# --------------------------------------------------------------------------- #
# 9. hybrid extension (optional): tune BM25 weight on dev, report on test
# --------------------------------------------------------------------------- #
HYBRID_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def _run_hybrid(retriever, embedder, questions, weight: float) -> list[dict[str, Any]]:
    traces = []
    for q in questions:
        applied = q.get("filter") or None              # same filter policy as C
        hits = retriever.search(q["question"], embedder.embed_query(q["question"]), weight,
                                top_k=3, filters=applied)
        t = evaluate_query(q, hits, f"C+hybrid(w={weight})", applied)
        for r, h in zip(t["retrieved"], hits):
            r["dense_rank"], r["bm25_rank"] = h["dense_rank"], h["bm25_rank"]
        traces.append(t)
    return traces


def cmd_hybrid(_: argparse.Namespace) -> None:
    from .hybrid import CANDIDATES, RRF_K, HybridRetriever

    questions = json.loads((BENCH / "questions.json").read_text(encoding="utf-8"))["questions"]
    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]
    embedder = get_embedder()
    store = get_store(embedder)
    retriever = HybridRetriever(store, _read_jsonl(RESULTS / "chunks" / "chunks_B.jsonl"), index="B")

    sweep = []
    for w in HYBRID_WEIGHTS:
        tr = _run_hybrid(retriever, embedder, dev, w)
        sweep.append({"weight": w, "hit_at_1": sum(t["hit_at_1"] for t in tr),
                      "hit_at_3": sum(t["hit_at_3"] for t in tr), "N": len(tr),
                      "missed": [t["id"] for t in tr if not t["hit_at_3"]],
                      "first_hit_rank": {t["id"]: t["first_hit_rank"] for t in tr}})
        if w:
            r0 = sweep[0]["first_hit_rank"]
            moved = {i: f"{r0[i]}->{r}" for i, r in sweep[-1]["first_hit_rank"].items() if r != r0[i]}
            sweep[-1]["rank_changes_vs_w0"] = moved
        print(f"dev w={w:<4} Hit@1={sweep[-1]['hit_at_1']}/{len(tr)} Hit@3={sweep[-1]['hit_at_3']}/{len(tr)} "
              f"missed={sweep[-1]['missed']}")
    # rule fixed in advance: max dev Hit@3, then Hit@1, then the smallest weight
    best = sorted(sweep, key=lambda r: (-r["hit_at_3"], -r["hit_at_1"], r["weight"]))[0]
    w = best["weight"]

    base = {t["id"]: t for t in _read_jsonl(RESULTS / "traces" / "traces_dev.jsonl")
            + _read_jsonl(RESULTS / "traces" / "traces_test.jsonl") if t["strategy"] == "C"}
    result: dict[str, Any] = {"run_at": _now(), "method": {"bm25": "Okapi k1=1.5 b=0.75, syllable tokens",
                              "fusion": f"weighted RRF k={RRF_K}, top-{CANDIDATES} per list",
                              "selection_rule": "max dev Hit@3, then Hit@1, then smallest weight"},
                              "dev_sweep": sweep, "chosen_weight": w}
    for split, qs in (("dev", dev), ("test", test)):
        tr = _run_hybrid(retriever, embedder, qs, w)
        _write_jsonl(RESULTS / "hybrid" / f"traces_hybrid_{split}.jsonl", tr)
        changes = []
        for t in tr:
            b = base[t["id"]]
            if (t["hit_at_1"], t["hit_at_3"]) != (b["hit_at_1"], b["hit_at_3"]):
                better = (t["hit_at_3"], t["hit_at_1"]) > (b["hit_at_3"], b["hit_at_1"])
                changes.append({"id": t["id"], "change": "better" if better else "WORSE",
                                "C": {"hit1": b["hit_at_1"], "hit3": b["hit_at_3"], "rank": b["first_hit_rank"]},
                                "hybrid": {"hit1": t["hit_at_1"], "hit3": t["hit_at_3"], "rank": t["first_hit_rank"]}})
        result[split] = {"N": len(tr),
                         "C_hit_at_1": f"{sum(base[t['id']]['hit_at_1'] for t in tr)}/{len(tr)}",
                         "C_hit_at_3": f"{sum(base[t['id']]['hit_at_3'] for t in tr)}/{len(tr)}",
                         "hybrid_hit_at_1": f"{sum(t['hit_at_1'] for t in tr)}/{len(tr)}",
                         "hybrid_hit_at_3": f"{sum(t['hit_at_3'] for t in tr)}/{len(tr)}",
                         "changes_vs_C": changes}
        print(split, json.dumps(result[split], ensure_ascii=False))
    _write_json(RESULTS / "hybrid" / "hybrid_report.json", result)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Day 06 data ingestion pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    sub.add_parser("benchmark").set_defaults(func=cmd_benchmark)
    p = sub.add_parser("index")
    p.add_argument("--indexes", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.set_defaults(func=cmd_index)
    sub.add_parser("reindex-check").set_defaults(func=cmd_reindex_check)
    p = sub.add_parser("evaluate")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.set_defaults(func=cmd_evaluate)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("hybrid").set_defaults(func=cmd_hybrid)
    p = sub.add_parser("audit-sheets")
    p.add_argument("--force", action="store_true", help="overwrite existing manual labels")
    p.set_defaults(func=cmd_audit_sheets)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
