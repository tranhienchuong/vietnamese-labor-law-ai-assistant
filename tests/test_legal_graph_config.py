from __future__ import annotations

import unittest
from unittest.mock import patch

from vn_labor_law_ai_assistant.rag.graph.config import LegalGraphConfig


class LegalGraphConfigTests(unittest.TestCase):
    def test_rejects_default_neo4j_password_in_production(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "APP_ENV": "production",
                    "LEGAL_GRAPH_ENABLED": "true",
                    "NEO4J_PASSWORD": "password",
                },
                clear=False,
            ),
            self.assertRaisesRegex(RuntimeError, "default.*NEO4J_PASSWORD"),
        ):
            LegalGraphConfig.from_env()


if __name__ == "__main__":
    unittest.main()
