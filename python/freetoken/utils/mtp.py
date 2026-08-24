"""Single switch for MTP speculative decoding (milestone: env-var opt-in).

Consulted by the model (build the draft head), the weight loader (keep mtp.*
tensors), the model config (register the MTP layer in the full-attention KV
group), and the engine/scheduler (draft + verify serve path). A proper
``--mtp-drafts`` engine flag replaces this once the feature stabilizes.
"""

from __future__ import annotations

import os


def mtp_enabled() -> bool:
    return os.environ.get("FREETOKEN_MTP", "0") == "1"


__all__ = ["mtp_enabled"]
