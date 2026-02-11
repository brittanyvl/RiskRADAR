"""Text preprocessing for BM25 tokenization."""

import re

# Common English stopwords (no external dependency needed)
STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "let", "me", "might", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "shouldn't", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won't", "would",
    "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "also", "been", "being", "did", "does", "had", "has", "have", "its",
    "shall", "should", "was", "were", "will", "would",
})

# Aviation-specific terms to always preserve (never filter as stopwords)
AVIATION_TERMS = frozenset({
    "atc", "vfr", "ifr", "ils", "ndb", "vor", "dme", "rnav", "gps",
    "stall", "cfit", "ntsb", "faa", "easa", "icao", "pic", "sic", "cfi",
    "atp", "ppl", "metar", "taf", "sigmet", "notam", "atis", "tcas",
    "gpws", "egpws", "taws",
})

# Regex: remove punctuation but keep hyphens within words (e.g., LOC-I)
_PUNCT_RE = re.compile(r"(?<!\w)-|-(?!\w)|[^\w\s-]", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.

    - Lowercase
    - Remove punctuation (keep intra-word hyphens like LOC-I)
    - Split on whitespace
    - Filter stopwords (preserve aviation terms)

    Returns:
        List of tokens.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    return [
        t for t in tokens
        if t in AVIATION_TERMS or (t not in STOPWORDS and len(t) > 1)
    ]
