"""
extraction/processing/tokenizer.py
----------------------------------
Tokenizer wrapper for tiktoken cl100k_base encoding.

Provides token counting for chunk size management. Uses cl100k_base
encoding which is compatible with GPT-4 and text-embedding models,
and provides a good approximation for sentence-transformers.

Configuration is tracked for reproducibility.
"""

import tiktoken

# Configuration - tracked for reproducibility
# v2: Increased min to 400 (enforced), max to 800, overlap to 25%
TOKENIZER_CONFIG = {
    "encoding": "cl100k_base",
    "library": "tiktoken",
    "chunk_target_tokens": 600,
    "chunk_min_tokens": 400,  # v2: Hard minimum (enforced with forward borrowing)
    "chunk_max_tokens": 800,  # v2: Increased from 700 to accommodate minimum
    "overlap_ratio": 0.25,    # v2: Increased from 0.20 to 0.25
}

# Lazy initialization
_encoder = None


def get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder (lazy initialization)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(TOKENIZER_CONFIG["encoding"])
    return _encoder


def get_config() -> dict:
    """
    Get tokenizer configuration for tracking.

    Returns config dict with encoding, library version, and chunk settings.
    """
    config = TOKENIZER_CONFIG.copy()
    config["tiktoken_version"] = tiktoken.__version__
    config["overlap_tokens"] = int(
        config["chunk_min_tokens"] * config["overlap_ratio"]
    )
    return config


def count_tokens(text: str) -> int:
    """
    Count tokens in text using cl100k_base encoding.

    Args:
        text: Text to tokenize

    Returns:
        Number of tokens
    """
    if not text:
        return 0
    encoder = get_encoder()
    return len(encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text (may be original if already under limit)
    """
    if not text:
        return ""
    encoder = get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def get_token_chunks(text: str, chunk_size: int) -> list[tuple[str, int]]:
    """
    Split text into token-sized chunks (no overlap).

    Useful for understanding token boundaries.

    Args:
        text: Text to split
        chunk_size: Target tokens per chunk

    Returns:
        List of (chunk_text, token_count) tuples
    """
    if not text:
        return []

    encoder = get_encoder()
    tokens = encoder.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append((chunk_text, len(chunk_tokens)))

    return chunks


def estimate_tokens(char_count: int) -> int:
    """
    Rough estimate of token count from character count.

    For English text, roughly 4 chars per token on average.
    This is a fast approximation when exact count isn't needed.

    Args:
        char_count: Number of characters

    Returns:
        Estimated token count
    """
    return max(1, char_count // 4)


# Chunk size helpers

def is_under_target(token_count: int) -> bool:
    """Check if token count is under minimum target (500)."""
    return token_count < TOKENIZER_CONFIG["chunk_min_tokens"]


def is_in_target_range(token_count: int) -> bool:
    """Check if token count is in target range (500-700)."""
    return (TOKENIZER_CONFIG["chunk_min_tokens"] <=
            token_count <=
            TOKENIZER_CONFIG["chunk_max_tokens"])


def is_over_target(token_count: int) -> bool:
    """Check if token count exceeds maximum target (700)."""
    return token_count > TOKENIZER_CONFIG["chunk_max_tokens"]


def get_overlap_tokens() -> int:
    """Get the configured overlap token count (~100-120)."""
    return int(
        TOKENIZER_CONFIG["chunk_min_tokens"] *
        TOKENIZER_CONFIG["overlap_ratio"]
    )
