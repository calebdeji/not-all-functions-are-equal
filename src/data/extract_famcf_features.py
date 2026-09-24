"""
Extract FAMCF-style features (permissions, API calls, opcodes) from raw APKs.

Vocabularies are derived from the corpus itself (Stage 1 saves everything
raw so we can analyse the corpus and decide on filtering before encoding).

Pipeline:
  Stage 1 -- extract ALL permissions, ALL platform API calls, and opcode 3-grams
             per APK; saves incrementally to raw_features.pkl.
  Stage 2 -- build corpus-wide vocabulary, encode as binary vectors, save features.pkl.

Output (saved to dataset/higraph/famcf/ by default):
  raw_features.pkl   -- dict: sha256 -> {perms, apis, opcode_ngrams, family}
  vocab.pkl          -- dict: {perms: [...], apis: [...], opcodes: [...]}
  features.pkl       -- dict: sha256 -> {perm_vec, api_vec, opc_vec, family}
"""

import os
import sys
import pickle
import logging
import argparse
from itertools import islice
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Dalvik opcode grouping (simplified after Chen et al. 2018)
# ---------------------------------------------------------------------------
OPCODE_GROUPS = {
    "add": "arith", "sub": "arith", "mul": "arith", "div": "arith",
    "rem": "arith", "neg": "arith", "not": "arith",
    "and": "bit", "or": "bit", "xor": "bit",
    "shl": "bit", "shr": "bit", "ushr": "bit",
    "cmpl": "cmp", "cmpg": "cmp",
    "if-eq": "branch", "if-ne": "branch", "if-lt": "branch",
    "if-ge": "branch", "if-gt": "branch", "if-le": "branch",
    "if-eqz": "branch", "if-nez": "branch", "if-ltz": "branch",
    "if-gez": "branch", "if-gtz": "branch", "if-lez": "branch",
    "move": "move", "array-length": "move",
    "return": "ret",
    "goto": "goto", "packed-switch": "switch", "sparse-switch": "switch",
    "throw": "throw", "monitor": "monitor",
    "new-instance": "new", "new-array": "new",
    "filled-new-array": "new", "fill-array-data": "new",
    "check-cast": "cast", "instance-of": "cast",
    "iget": "fget", "iput": "fput", "sget": "sget", "sput": "sput",
    "invoke-virtual": "inv_virt", "invoke-super": "inv_super",
    "invoke-direct": "inv_dir", "invoke-static": "inv_stat",
    "invoke-interface": "inv_iface",
    "invoke-virtual/range": "inv_virt", "invoke-super/range": "inv_super",
    "invoke-direct/range": "inv_dir", "invoke-static/range": "inv_stat",
    "invoke-interface/range": "inv_iface",
    "const": "const",
    "aget": "aget", "aput": "aput",
    "int-to": "conv", "long-to": "conv", "float-to": "conv",
    "double-to": "conv", "byte-to": "conv", "char-to": "conv",
    "short-to": "conv",
    "nop": "nop",
}


def map_opcode(raw_opcode: str) -> str:
    raw = raw_opcode.lower()
    if raw in OPCODE_GROUPS:
        return OPCODE_GROUPS[raw]
    for prefix, sym in OPCODE_GROUPS.items():
        if raw.startswith(prefix):
            return sym
    return "other"


def ngrams(seq, n):
    it = iter(seq)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for item in it:
        window = window[1:] + (item,)
        yield window


# API prefixes to keep — platform APIs only, excludes app-internal classes
API_PREFIXES = (
    "Landroid/", "Ldalvik/", "Ljava/", "Ljavax/",
    "Lcom/google/android/", "Lorg/apache/",
)


def extract_features_from_apk(apk_path: str):
    """
    Returns (perms, apis, opcode_3grams):
      perms         -- all permissions declared in the manifest
      apis          -- all platform API calls invoked (android/java/etc.)
      opcode_3grams -- set of abstract Dalvik opcode 3-grams
    """
    from androguard.misc import AnalyzeAPK

    apk, dex_list, dx = AnalyzeAPK(apk_path)

    # --- Permissions: everything in the manifest ---
    perms = set(apk.get_permissions())

    # --- API calls: all invoked platform API methods ---
    apis = set()
    for method_analysis in dx.get_methods():
        for _, callee, _ in method_analysis.get_xref_to():
            if callee.class_name.startswith(API_PREFIXES):
                apis.add(f"{callee.class_name}->{callee.name}")

    # --- Opcodes: abstract + 3-gram ---
    opcode_seq = []
    for dex in dex_list:
        for cls in dex.get_classes():
            for method in cls.get_methods():
                for instr in method.get_instructions():
                    opcode_seq.append(map_opcode(instr.get_name()))

    opcode_3grams = set(ngrams(opcode_seq, 3))

    return perms, apis, opcode_3grams


def setup_logger(name: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(sh)
    return logger


def load_hash_family_map(csv_dir: str) -> dict:
    mapping = {}
    for csv_path in Path(csv_dir).glob("labelled_samples_with_family_*.csv"):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            mapping[row["hash"]] = row["family"]
    return mapping


def stage1_extract_raw(apk_dirs: list, hash_family: dict, out_path: str, logger):
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            raw = pickle.load(f)
        logger.info(f"Resuming from {len(raw)} already-processed APKs.")
    else:
        raw = {}

    for apk_dir in apk_dirs:
        apk_dir = Path(apk_dir)
        if not apk_dir.exists():
            continue
        apk_files = sorted(apk_dir.glob("*.apk"))
        logger.info(f"Processing {len(apk_files)} APKs in {apk_dir}")

        for apk_path in apk_files:
            sha = apk_path.stem
            if sha in raw:
                continue
            if sha not in hash_family:
                logger.warning(f"No label for {sha}, skipping.")
                continue

            try:
                perms, apis, opcode_3grams = extract_features_from_apk(str(apk_path))
                raw[sha] = {
                    "family": hash_family[sha],
                    "perms": perms,
                    "apis": apis,
                    "opcode_ngrams": opcode_3grams,
                }
                logger.info(f"OK  {sha[:16]}  family={hash_family[sha]}  "
                            f"perms={len(perms)}  apis={len(apis)}  "
                            f"ngrams={len(opcode_3grams)}")
            except Exception as e:
                logger.error(f"FAIL {sha[:16]}: {e}")

            if len(raw) % 50 == 0:
                with open(out_path, "wb") as f:
                    pickle.dump(raw, f)

    with open(out_path, "wb") as f:
        pickle.dump(raw, f)

    logger.info(f"Stage 1 complete. {len(raw)} APKs -> {out_path}")
    return raw


def stage2_encode_with_vocab(raw: dict, out_dir: str, vocab: dict, logger):
    """
    Encode raw features using an existing vocabulary instead of building a new one.
    Used for obfuscated APKs so that feature vector positions match the trained encoders.
    """
    perm_idx  = {p: i for i, p in enumerate(vocab["perms"])}
    api_idx   = {a: i for i, a in enumerate(vocab["apis"])}
    ngram_idx = {ng: i for i, ng in enumerate(vocab["opcodes"])}

    logger.info(f"Encoding with existing vocab: "
                f"perms={len(perm_idx)}, apis={len(api_idx)}, opcodes={len(ngram_idx)}")

    features = {}
    for sha, entry in raw.items():
        perm_vec = np.zeros(len(perm_idx), dtype=np.float32)
        for p in entry["perms"]:
            if p in perm_idx:
                perm_vec[perm_idx[p]] = 1.0

        api_vec = np.zeros(len(api_idx), dtype=np.float32)
        for a in entry["apis"]:
            if a in api_idx:
                api_vec[api_idx[a]] = 1.0

        opc_vec = np.zeros(len(ngram_idx), dtype=np.float32)
        for ng in entry["opcode_ngrams"]:
            if ng in ngram_idx:
                opc_vec[ngram_idx[ng]] = 1.0

        features[sha] = {
            "family":   entry["family"],
            "perm_vec": perm_vec,
            "api_vec":  api_vec,
            "opc_vec":  opc_vec,
        }

    feat_path = os.path.join(out_dir, "features.pkl")
    with open(feat_path, "wb") as f:
        pickle.dump(features, f)
    logger.info(f"Encoded features saved -> {feat_path} ({len(features)} samples)")
    return features


def stage2_build_and_encode(raw: dict, out_dir: str, logger,
                            top_perms: int = 57,
                            top_apis: int = 78,
                            top_ngrams: int = 343):
    """
    Build top-K frequency-based vocabularies and encode each APK as binary vectors.

    K values match the feature dimensionalities reported in the FAMCF paper
    (permissions=57, API calls=78, opcode 3-grams=343). The paper's original
    predefined lists were not publicly available and authors were unresponsive,
    so we select the top-K most frequent features from the training corpus instead.
    """
    from collections import Counter

    # Count how many samples each feature appears in
    perm_counts  = Counter(p  for entry in raw.values() for p  in entry["perms"])
    api_counts   = Counter(a  for entry in raw.values() for a  in entry["apis"])
    ngram_counts = Counter(ng for entry in raw.values() for ng in entry["opcode_ngrams"])

    logger.info(f"Full vocab sizes -- perms: {len(perm_counts)}, "
                f"apis: {len(api_counts)}, opcode 3-grams: {len(ngram_counts)}")

    # Top-K most frequent, sorted alphabetically within the top-K for stability
    all_perms  = sorted(f for f, _ in perm_counts.most_common(top_perms))
    all_apis   = sorted(f for f, _ in api_counts.most_common(top_apis))
    all_ngrams = sorted(f for f, _ in ngram_counts.most_common(top_ngrams))

    logger.info(f"Top-K vocab sizes -- perms: {len(all_perms)} (K={top_perms}), "
                f"apis: {len(all_apis)} (K={top_apis}), "
                f"opcode 3-grams: {len(all_ngrams)} (K={top_ngrams})")

    perm_idx  = {p: i for i, p in enumerate(all_perms)}
    api_idx   = {a: i for i, a in enumerate(all_apis)}
    ngram_idx = {ng: i for i, ng in enumerate(all_ngrams)}

    vocab = {"perms": all_perms, "apis": all_apis, "opcodes": all_ngrams}
    vocab_path = os.path.join(out_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"Vocab saved -> {vocab_path}")

    features = {}
    for sha, entry in raw.items():
        perm_vec = np.zeros(len(all_perms), dtype=np.float32)
        for p in entry["perms"]:
            if p in perm_idx:
                perm_vec[perm_idx[p]] = 1.0

        api_vec = np.zeros(len(all_apis), dtype=np.float32)
        for a in entry["apis"]:
            if a in api_idx:
                api_vec[api_idx[a]] = 1.0

        opc_vec = np.zeros(len(all_ngrams), dtype=np.float32)
        for ng in entry["opcode_ngrams"]:
            if ng in ngram_idx:
                opc_vec[ngram_idx[ng]] = 1.0

        features[sha] = {
            "family": entry["family"],
            "perm_vec": perm_vec,
            "api_vec": api_vec,
            "opc_vec": opc_vec,
        }

    feat_path = os.path.join(out_dir, "features.pkl")
    with open(feat_path, "wb") as f:
        pickle.dump(features, f)
    logger.info(f"Encoded features saved -> {feat_path} ({len(features)} samples)")

    return features, vocab


def main():
    parser = argparse.ArgumentParser(description="Extract FAMCF features from HiGraph APKs")
    parser.add_argument("--apk_root", default=None,
                        help="Root directory containing year-subdirs of APKs. "
                             "Defaults to raw_format/higraph.")
    parser.add_argument("--csv_dir", default=None,
                        help="Directory with labelled_samples_with_family_*.csv. "
                             "Defaults to dataset_refined_truth/higraph.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory. Defaults to dataset/higraph/famcf.")
    parser.add_argument("--years", nargs="+", default=None,
                        help="Year subdirs to process (default: all)")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="Run only stage 1 (extract) or stage 2 (encode). Default: both.")
    parser.add_argument("--top_perms",  type=int, default=57,
                        help="Top-K most frequent permissions to keep (default: 57, matches FAMCF paper).")
    parser.add_argument("--top_apis",   type=int, default=78,
                        help="Top-K most frequent API calls to keep (default: 78, matches FAMCF paper).")
    parser.add_argument("--top_ngrams", type=int, default=343,
                        help="Top-K most frequent opcode 3-grams to keep (default: 343, matches FAMCF paper).")
    parser.add_argument("--existing_vocab", default=None,
                        help="Path to an existing vocab.pkl. If provided, stage 2 encodes "
                             "using this vocabulary instead of building a new one from the "
                             "input corpus. Use this for obfuscated APKs.")
    args = parser.parse_args()

    if args.existing_vocab and args.stage == 1:
        parser.error("--existing_vocab only applies to stage 2 encoding")
    from src.core.utils import Utils
    project_root = Path(Utils.get_project_root())
    apk_root = Path(args.apk_root) if args.apk_root is not None \
               else Path(Utils.get_raw_apk_root('higraph'))
    csv_dir  = Path(args.csv_dir)  if args.csv_dir  is not None \
               else Path(Utils.get_refined_truth_root('higraph'))
    out_dir  = Path(args.out_dir)  if args.out_dir  is not None \
               else Path(Utils.get_famcf_root('higraph'))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "extraction.log"
    logger = setup_logger("famcf_extract", str(log_path))

    if args.years:
        year_dirs = [apk_root / y for y in args.years]
    else:
        year_dirs = sorted([d for d in apk_root.iterdir() if d.is_dir()])

    logger.info(f"APK dirs: {[str(d) for d in year_dirs]}")

    hash_family = load_hash_family_map(str(csv_dir))
    logger.info(f"Loaded {len(hash_family)} hash->family mappings")

    raw_path = out_dir / "raw_features.pkl"

    if args.stage in (1, None):
        raw = stage1_extract_raw(year_dirs, hash_family, str(raw_path), logger)

    if args.stage in (2, None):
        if args.stage == 2:
            with open(raw_path, "rb") as f:
                raw = pickle.load(f)
        if args.existing_vocab:
            with open(args.existing_vocab, "rb") as f:
                vocab = pickle.load(f)
            logger.info(f"Using existing vocab from: {args.existing_vocab}")
            stage2_encode_with_vocab(raw, str(out_dir), vocab, logger)
        else:
            stage2_build_and_encode(raw, str(out_dir), logger,
                                    top_perms=args.top_perms,
                                    top_apis=args.top_apis,
                                    top_ngrams=args.top_ngrams)


if __name__ == "__main__":
    main()
