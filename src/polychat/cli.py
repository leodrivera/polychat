"""Console entrypoint exposed as the ``polychat`` command.

Declared in ``pyproject.toml`` under ``[project.scripts]``. Running ``polychat``
is equivalent to ``streamlit run src/polychat/app.py`` — any extra CLI arguments
are forwarded to Streamlit, so ``polychat --server.port=8080`` works.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
