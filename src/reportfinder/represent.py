"""Build the two representation spaces and cache them.

Space A (dense): a local sentence-transformer over doc(r) -> E (n x d), L2-normalized.
Space B (LSA):   TF-IDF (uni+bigram) -> TruncatedSVD(200) -> U (n x 200), L2-normalized.

Both are fit unsupervised on the corpus itself. Nothing here sees a label.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .config import BGE_QUERY_PREFIX, DENSE_MODEL_FALLBACK, Config

log = logging.getLogger(__name__)

CACHE_VERSION = "v5-expansion-hybrid"
CANONICAL_SCHEMA_VERSION = "2"
REPRESENTATION_FEATURE_SCHEMA_VERSION = "2"


def _dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _dense_cache_state(cfg: Config) -> str:
    """Cheap local-only fingerprint of whether a dense model is available."""
    if not cfg.use_dense or cfg.dense_mode == "off":
        return "disabled"
    if _dependency_version("sentence-transformers") == "missing":
        return "dependency-missing"
    try:
        from huggingface_hub import try_to_load_from_cache
        primary = try_to_load_from_cache(cfg.dense_model, "config.json")
        fallback = try_to_load_from_cache(DENSE_MODEL_FALLBACK, "config.json")
    except (ImportError, OSError, ValueError):
        return "cache-state-unknown"
    if isinstance(primary, str):
        return f"primary:{Path(primary).stat().st_mtime_ns}"
    if isinstance(fallback, str):
        return f"fallback:{Path(fallback).stat().st_mtime_ns}"
    return "model-not-cached"


def build_doc(row: pd.Series, cfg: Config) -> str:
    """Compose the field-weighted document text for one report.

    Strong zones are repeated so they dominate the representation:
        title x3 + fields x3 + prompts x2 + category + data_source + report_type + tags

    The repetition is what makes this a *representation* choice rather than a
    scoring rule: the emphasis is baked into the vector, and both experts then
    read the same document independently.

    Phase 2 field metadata (descriptions, business objects, domains, ...) is
    appended afterwards at weight 1. It enriches the document without displacing
    the zones ranking is already tuned around. In legacy mode there is no field
    metadata, so this produces byte-identical output to Phase 1.
    """
    zones: list[str] = []

    def add(text: str, times: int) -> None:
        text = (text or "").strip()
        if text and times > 0:
            zones.extend([text] * times)

    add(str(row["title"]), cfg.w_title)
    add(", ".join(row["fields"]), cfg.w_fields)
    add(", ".join(row["prompts"]), cfg.w_prompts)
    add(str(row["category"]), cfg.w_category)
    add(str(row["data_source"]), cfg.w_data_source)
    add(str(row["report_type"]), cfg.w_report_type)
    add(str(row["tags"]).replace(";", ", "), cfg.w_tags)

    for text, weight in _phase2_zones(row, cfg):
        add(text, weight)

    return " . ".join(zones)


def _phase2_zones(row: pd.Series, cfg: Config) -> list[tuple[str, int]]:
    """Phase 2 field-metadata zones, deduplicated across the report's fields.

    Values are deduplicated because dozens of fields on one report typically share
    a handful of domains/business objects; repeating them once per field would
    silently re-weight those zones far above their configured weight.

    `Authorized Usage` is deliberately excluded: 1 distinct value in the source
    data means it cannot discriminate between reports.
    """
    metas = row.get("field_meta")
    if not isinstance(metas, (list, tuple)) or not metas:
        return []

    def unique(values) -> str:
        seen: dict[str, None] = {}
        for value in values:
            text = (value or "").strip()
            if text:
                seen.setdefault(text, None)
        return ", ".join(seen)

    return [
        (unique(f.description for f in metas), cfg.w_field_description),
        (unique(f.business_object for f in metas), cfg.w_business_object),
        (unique(f.domain for f in metas), cfg.w_domain),
        (unique(str(f.categories).replace(";", ", ") for f in metas), cfg.w_field_categories),
        (unique(f.report_field_type for f in metas), cfg.w_field_type),
        (unique(f.built_in_prompts for f in metas), cfg.w_builtin_prompts),
        (unique(f.related_business_object_name for f in metas), cfg.w_related_business_object),
    ]


def build_docs(frame: pd.DataFrame, cfg: Config) -> list[str]:
    return [build_doc(row, cfg) for _, row in frame.iterrows()]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros rather than NaN."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def load_encoder(model_name: str, mode: str = "auto"):
    """Load the local sentence-transformer, falling back if the primary fails.

    Returns (model, resolved_name). The fallback exists because the primary may
    not be present in the local HF cache and the box may be offline.
    """
    if mode == "off":
        return None, "disabled"
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        if mode == "local":
            raise RuntimeError(
                "dense_mode='local' requires sentence-transformers; install it "
                "or set dense_mode='off'"
            ) from exc
        log.warning("sentence-transformers unavailable (%s); dense retrieval disabled.", exc)
        return None, "unavailable"

    try:
        # Prefer the local cache. Newer transformers releases otherwise perform
        # a network HEAD request even when every model artifact is available,
        # which makes an offline deployment fail unnecessarily.
        return SentenceTransformer(model_name, local_files_only=True), model_name
    except (OSError, RuntimeError, ValueError) as exc:
        if mode == "local":
            raise RuntimeError(
                f"dense_mode='local' could not load {model_name!r} from local files: {exc}"
            ) from exc
        try:
            return SentenceTransformer(model_name), model_name
        except (OSError, RuntimeError, ValueError):
            pass
        if model_name == DENSE_MODEL_FALLBACK:
            log.warning("Dense model %r unavailable (%s); dense retrieval disabled.", model_name, exc)
            return None, "unavailable"
        log.warning(
            "Could not load dense model %r (%s). Falling back to %r.",
            model_name,
            exc,
            DENSE_MODEL_FALLBACK,
        )
        try:
            return SentenceTransformer(DENSE_MODEL_FALLBACK, local_files_only=True), DENSE_MODEL_FALLBACK
        except (OSError, RuntimeError, ValueError):
            try:
                return SentenceTransformer(DENSE_MODEL_FALLBACK), DENSE_MODEL_FALLBACK
            except (OSError, RuntimeError, ValueError) as fallback_exc:
                log.warning(
                    "Dense fallback %r unavailable (%s); using LSA only.",
                    DENSE_MODEL_FALLBACK, fallback_exc,
                )
                return None, "unavailable"


def encode_query_text(text: str, model_name: str) -> str:
    """bge-* models expect an asymmetric instruction prefix on queries only."""
    if "bge" in model_name.lower():
        return BGE_QUERY_PREFIX + text
    return text


@dataclass
class Representation:
    """Everything a query needs at inference time."""

    frame: pd.DataFrame
    docs: list[str]
    dense: np.ndarray  # E: (n, d) L2-normalized
    lsa: np.ndarray  # U: (n, k) L2-normalized
    lsa_pipeline: object  # fitted TfidfVectorizer -> SVD -> Normalizer
    dense_model_name: str
    dense_mode: str
    field_vocab: dict[str, np.ndarray]  # field name -> report indices having it
    raw_row_count: int
    encoder: object | None = None  # lazily attached
    # Ingestion diagnostics, cached alongside the vectors so `-v` can report them
    # without re-ingesting the spreadsheets.
    import_summary: object | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def get_encoder(self):
        if self.dense_mode != "native":
            raise RuntimeError(f"Dense encoder is not active (state={self.dense_mode}).")
        if self.encoder is None:
            self.encoder, resolved = load_encoder(self.dense_model_name, "local")
            if self.encoder is None:
                raise RuntimeError(f"Dense encoder {resolved}.")
        return self.encoder


def build_field_vocab(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mine the field vocabulary from the corpus: field name -> report indices.

    This is corpus statistics, not a rule list -- the vocabulary is whatever the
    workbook happens to contain.
    """
    postings: dict[str, list[int]] = {}
    for idx, fields in enumerate(frame["fields"]):
        for field in fields:
            postings.setdefault(field.casefold(), []).append(idx)
    return {k: np.asarray(v, dtype=np.int32) for k, v in postings.items()}


def _file_stamp(path: Path, label: str) -> dict:
    """Identity of an input file, checked eagerly for a readable error."""
    if not path.exists():
        # Checked here as well as in the loaders: this runs first, and a bare
        # stat() OSError would otherwise mask the actionable message.
        raise FileNotFoundError(
            f"{label} not found at {path}\n"
            "Place the .xlsx there, or pass an explicit path."
        )
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime": int(stat.st_mtime), "sha256": digest}


def _cache_signature(cfg: Config, data_path: Path) -> str:
    """Fingerprint every input that changes the representation.

    Includes the ingest mode and both Phase 2 inputs, so switching modes or
    editing either workbook rebuilds rather than silently serving vectors built
    from the other source. Query-time knobs (alpha, T, tau, delta) are excluded on
    purpose -- they don't affect the cached vectors, so they can be swept freely.
    """
    from .ingest.models import IngestMode

    mode = IngestMode(cfg.ingest_mode)
    inputs: dict[str, object] = {"mode": mode.value}

    if mode is IngestMode.LEGACY_SINGLE_FILE:
        inputs["data"] = _file_stamp(data_path, "Report workbook")
    else:
        inputs["catalog"] = _file_stamp(Path(cfg.catalog_path), "Report catalog")
        inputs["dictionary"] = _file_stamp(
            Path(cfg.field_dictionary_path), "Field dictionary"
        )
        inputs["linking"] = [
            cfg.ambiguity_policy,
            cfg.enable_composite_match,
            cfg.enable_fuzzy_match,
            cfg.fuzzy_threshold,
        ]

    payload = {
        "version": CACHE_VERSION,
        "canonical_schema": CANONICAL_SCHEMA_VERSION,
        "feature_schema": REPRESENTATION_FEATURE_SCHEMA_VERSION,
        "python": platform.python_version(),
        "dependencies": {
            name: _dependency_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "sentence-transformers")
        },
        "inputs": inputs,
        "dense_model": cfg.dense_model,
        "dense_mode": cfg.dense_mode,
        "resolved_dense_cache_state": _dense_cache_state(cfg),
        "use_dense": cfg.use_dense,
        "corpus_granularity": cfg.corpus_granularity,
        "svd_components": cfg.svd_components,
        "tfidf_min_df": cfg.tfidf_min_df,
        "tfidf_max_df": cfg.tfidf_max_df,
        "ngram_max": cfg.ngram_max,
        "zones": [
            cfg.w_title,
            cfg.w_fields,
            cfg.w_prompts,
            cfg.w_category,
            cfg.w_data_source,
            cfg.w_report_type,
            cfg.w_tags,
        ],
        "phase2_zones": [
            cfg.w_field_description,
            cfg.w_business_object,
            cfg.w_domain,
            cfg.w_field_categories,
            cfg.w_field_type,
            cfg.w_builtin_prompts,
            cfg.w_related_business_object,
        ],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build(cfg: Config, verbose: bool = True) -> Representation:
    """Ingest the configured source(s) and fit both representation spaces."""
    from .ingest import build_corpus

    t0 = time.perf_counter()
    # The summary is cached on the Representation and rendered once by the caller
    # (so it shows on cache hits too, not just fresh builds).
    corpus, summary = build_corpus(cfg, verbose=False)
    frame = corpus.frame
    if verbose:
        print(
            f"  parsed {corpus.raw_row_count} rows -> {len(frame)} report families "
            f"({corpus.raw_row_count - len(frame)} exact duplicates collapsed)"
        )

    docs = build_docs(frame, cfg)

    # --- Space B: LSA (fit first; it's fast and validates the corpus) ---
    n_docs = len(docs)
    components = min(cfg.svd_components, max(2, n_docs - 1))
    vectorizer = TfidfVectorizer(
        ngram_range=(1, cfg.ngram_max),
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    svd = TruncatedSVD(n_components=components, random_state=0)
    lsa_pipeline = make_pipeline(vectorizer, svd, Normalizer(copy=False))
    lsa = lsa_pipeline.fit_transform(docs).astype(np.float32)
    if verbose:
        explained = float(svd.explained_variance_ratio_.sum())
        print(
            f"  LSA: {len(vectorizer.vocabulary_)} terms -> {components} components "
            f"({explained:.1%} variance explained)"
        )

    # --- Space A: dense semantic ---
    encoder, resolved_name = load_encoder(
        cfg.dense_model, "off" if not cfg.use_dense else cfg.dense_mode,
    )
    if encoder is None:
        dense = np.zeros((n_docs, 1), dtype=np.float32)
        dense_state = "disabled" if resolved_name == "disabled" else "lsa-fallback"
        if verbose:
            print(f"  dense: {dense_state}")
    else:
        if verbose:
            print(f"  dense: encoding {n_docs} docs with {resolved_name} ...")
        dense = encoder.encode(
            docs,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=verbose,
        ).astype(np.float32)
        dense = _l2_normalize(dense)
        dense_state = "native"

    field_vocab = build_field_vocab(frame)

    if verbose:
        print(f"  built in {time.perf_counter() - t0:.1f}s")

    return Representation(
        frame=frame,
        docs=docs,
        dense=dense,
        lsa=lsa,
        lsa_pipeline=lsa_pipeline,
        dense_model_name=resolved_name,
        dense_mode=dense_state,
        field_vocab=field_vocab,
        raw_row_count=corpus.raw_row_count,
        encoder=encoder,
        import_summary=summary,
    )


def save(rep: Representation, cfg: Config, signature: str) -> None:
    from .query.expansion.rules import lexicon_content_hash
    from .ranking.features import FEATURE_HASH
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_dir / "matrices.npz",
        dense=rep.dense,
        lsa=rep.lsa,
    )
    joblib.dump(rep.lsa_pipeline, cache_dir / "lsa_pipeline.joblib")
    rep.frame.to_pickle(cache_dir / "frame.pkl")
    joblib.dump(
        {
            "signature": signature,
            "cache_version": CACHE_VERSION,
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "feature_schema_version": REPRESENTATION_FEATURE_SCHEMA_VERSION,
            "built_at": datetime.now(UTC).isoformat(),
            "dense_model_name": rep.dense_model_name,
            "dense_mode": rep.dense_mode,
            "docs": rep.docs,
            "raw_row_count": rep.raw_row_count,
            "import_summary": rep.import_summary,
            "runtime_signature": {
                "lexicon_hash": lexicon_content_hash(),
                "use_query_expansion": cfg.use_query_expansion,
                "ranking_feature_hash": FEATURE_HASH,
            },
        },
        cache_dir / "meta.joblib",
    )


def try_load(cfg: Config, signature: str) -> Representation | None:
    """Load the cache iff it exists and matches the current input signature."""
    cache_dir = Path(cfg.cache_dir)
    needed = ["matrices.npz", "lsa_pipeline.joblib", "frame.pkl", "meta.joblib"]
    if not all((cache_dir / f).exists() for f in needed):
        return None

    try:
        meta = joblib.load(cache_dir / "meta.joblib")
        if meta.get("signature") != signature:
            return None
        matrices = np.load(cache_dir / "matrices.npz")
        frame = pd.read_pickle(cache_dir / "frame.pkl")
        pipeline = joblib.load(cache_dir / "lsa_pipeline.joblib")
    except Exception as exc:  # noqa: BLE001 - a corrupt cache should just rebuild
        log.warning("Cache unreadable (%s); rebuilding.", exc)
        return None

    from .query.expansion.rules import lexicon_content_hash
    from .ranking.features import FEATURE_HASH
    cached_runtime = meta.get("runtime_signature", {})
    current_runtime = {
        "lexicon_hash": lexicon_content_hash(),
        "use_query_expansion": cfg.use_query_expansion,
        "ranking_feature_hash": FEATURE_HASH,
    }
    if cached_runtime and cached_runtime != current_runtime:
        log.warning(
            "Query-time lexicon/ranking settings changed since this representation "
            "was built; reusing matrices because runtime changes do not affect them."
        )

    return Representation(
        frame=frame,
        docs=meta["docs"],
        dense=matrices["dense"],
        lsa=matrices["lsa"],
        lsa_pipeline=pipeline,
        dense_model_name=meta["dense_model_name"],
        # Old caches omitted this field and used a one-column zero matrix. Treat
        # that shape as lexical/LSA fallback, never as a native encoder.
        dense_mode=meta.get("dense_mode", "lsa-fallback"),
        field_vocab=build_field_vocab(frame),
        raw_row_count=meta["raw_row_count"],
        encoder=None,  # loaded lazily on first query
        import_summary=meta.get("import_summary"),
    )


def load_or_build(cfg: Config, rebuild: bool = False, verbose: bool = True) -> Representation:
    """Main entry point: cached representation, or build and cache it."""
    signature = _cache_signature(cfg, Path(cfg.data_path))

    if not rebuild:
        cached = try_load(cfg, signature)
        if cached is not None:
            if verbose:
                print(f"Loaded cache ({len(cached)} report families).")
            return cached

    if verbose:
        print("Building representations (first run or --rebuild)...")
    rep = build(cfg, verbose=verbose)
    save(rep, cfg, signature)
    if verbose:
        print(f"Cached to {cfg.cache_dir}")
    return rep
