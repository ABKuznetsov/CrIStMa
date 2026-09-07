from __future__ import annotations

import cristma


def test_local_development_version_is_beta6() -> None:
    assert cristma.__version__ == "0.1.0b6"
