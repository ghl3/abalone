"""Run-id generation: `<adjective>-<noun>-<UTC timestamp>`.

Examples:
    bold-otter-20260504-2045
    crimson-falcon-20260504-2052

The wordlist is curated for: family-friendly, single-token (no hyphens),
positive-affect, no political/cultural baggage. 100 × 100 = 10,000 unique
adjective-noun pairs; with a minute-precision timestamp suffix the
collision probability is essentially zero.
"""

from __future__ import annotations

import datetime as _dt
import random
from pathlib import Path

# fmt: off
ADJECTIVES = [
    "amber", "azure", "blazing", "bold", "bouncy", "breezy", "bright",
    "brisk", "bubbly", "buoyant", "calm", "cheery", "cobalt", "cosmic",
    "cozy", "crimson", "crystal", "curious", "daring", "dazzling",
    "deft", "dreamy", "eager", "earnest", "ember", "fearless", "feisty",
    "fierce", "fleet", "fluffy", "gentle", "giddy", "glad", "gleaming",
    "glowing", "graceful", "grand", "happy", "hardy", "hazy", "heroic",
    "honest", "humble", "iridescent", "ivory", "jade", "jaunty", "joyful",
    "jovial", "keen", "kindly", "lively", "lucky", "lush", "luminous",
    "majestic", "mellow", "merry", "mighty", "misty", "noble", "nimble",
    "opal", "peaceful", "perky", "playful", "plucky", "polite", "proud",
    "quick", "quiet", "radiant", "regal", "ruby", "rustic", "sapphire",
    "scarlet", "serene", "shiny", "silken", "silver", "skillful", "snappy",
    "sparkling", "spirited", "splendid", "spry", "stalwart", "stellar",
    "sturdy", "sunny", "swift", "tender", "thoughtful", "tidy", "twilight",
    "valiant", "velvet", "verdant", "vibrant", "warm", "wistful", "zesty",
]

NOUNS = [
    "antler", "arrow", "ash", "badger", "beacon", "beetle", "bison",
    "bramble", "branch", "breeze", "brook", "canyon", "cascade", "cedar",
    "cinder", "clover", "comet", "compass", "coral", "cougar", "crane",
    "crest", "cypress", "dawn", "dell", "delta", "dragon", "dune", "echo",
    "ember", "fable", "falcon", "feather", "fern", "ferret", "finch",
    "firefly", "flame", "fjord", "forge", "forest", "fox", "frost",
    "galaxy", "glade", "glacier", "grove", "harbor", "harvest", "haven",
    "hawk", "heron", "hollow", "horizon", "ibex", "island", "jaguar",
    "kelpie", "kestrel", "lagoon", "lake", "lantern", "leaf", "lemur",
    "lichen", "lotus", "lynx", "marsh", "meadow", "mirage", "mist",
    "moose", "mountain", "nebula", "oak", "orchid", "otter", "panther",
    "petal", "pine", "prairie", "puma", "quail", "raven", "reef", "river",
    "robin", "sable", "sage", "sequoia", "signet", "skylark", "sonata",
    "sparrow", "spruce", "stag", "starling", "stream", "summit", "thicket",
    "thunder", "tide", "tiger", "torrent", "trail", "tundra", "valley",
    "vine", "vista", "voyage", "warbler", "wave", "willow", "wisp",
    "yarrow", "zephyr",
]
# fmt: on

# Trim to exactly 100 each — the lists above are slightly oversized to
# survive an occasional taste-edit without breaking the contract.
ADJECTIVES = ADJECTIVES[:100]
NOUNS = NOUNS[:100]
assert len(ADJECTIVES) == 100, f"adjectives = {len(ADJECTIVES)}"
assert len(NOUNS) == 100, f"nouns = {len(NOUNS)}"


def _utc_timestamp() -> str:
    """`YYYYMMDD-HHMM` in UTC. Minute precision."""
    now = _dt.datetime.now(tz=_dt.UTC)
    return now.strftime("%Y%m%d-%H%M")


def generate(rng: random.Random | None = None) -> str:
    """Generate a fresh run-id."""
    r = rng or random.Random()
    adj = r.choice(ADJECTIVES)
    noun = r.choice(NOUNS)
    return f"{adj}-{noun}-{_utc_timestamp()}"


def generate_unique(runs_root: Path, rng: random.Random | None = None) -> str:
    """Generate a run-id that doesn't collide with an existing
    `runs_root/<run-id>/` directory. Tries 32 times with random suffixes;
    after that, appends `-2`, `-3`, ... to the candidate."""
    r = rng or random.Random()
    for _ in range(32):
        rid = generate(r)
        if not (runs_root / rid).exists():
            return rid
    # Extremely unlikely path: fall back to numeric disambiguation.
    base = generate(r)
    n = 2
    while (runs_root / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


if __name__ == "__main__":
    # Quick visual sanity check.
    for _ in range(5):
        print(generate())
