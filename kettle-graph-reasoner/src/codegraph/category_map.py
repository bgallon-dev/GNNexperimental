r"""60-repo category map for stratified train/test splits.

Mirrors the user-curated corpus: 60 Python repos across 37 categories
selected for *structural* diversity (dropped overlapping siblings, kept
the architecturally distinctive member of each cluster). Used by
``make_repo_split.py`` to produce a category-stratified 80/20 split so
the test set spans the category space rather than concentrating in one
slice.

If a repo's export directory name differs from the key here (e.g., your
exporter wrote ``plotly_py/`` instead of ``plotly.py/``), edit this
mapping or override per-repo via the ``aliases`` arg to
``make_repo_split.build_split``.
"""

from __future__ import annotations

CATEGORY_MAP: dict[str, str] = {
    # Web frameworks (6)
    "django": "web-framework",
    "flask": "web-framework",
    "fastapi": "web-framework",
    "starlette": "web-framework",
    "tornado": "web-framework",
    "pyramid": "web-framework",
    # HTTP / network (3)
    "uvicorn": "http-network",
    "httpx": "http-network",
    "aiohttp": "http-network",
    # Templating / WSGI (2)
    "jinja": "templating-wsgi",
    "werkzeug": "templating-wsgi",
    # Property testing (1)
    "hypothesis": "property-testing",
    # GUI / event-driven (1)
    "kivy": "gui",
    # Data / numeric (5)
    "pandas": "data-numeric",
    "numpy": "data-numeric",
    "scipy": "data-numeric",
    "scikit-learn": "data-numeric",
    "sympy": "data-numeric",
    # Visualization (2)
    "matplotlib": "visualization",
    "plotly.py": "visualization",
    # Imaging (1)
    "pillow": "imaging",
    # Statistics (1)
    "statsmodels": "statistics",
    # Graph algorithms (1)
    "networkx": "graph-algorithms",
    # Distributed data (1)
    "dask": "distributed-data",
    # ML core (4)
    "pytorch": "ml-core",
    "transformers": "ml-core",
    "keras": "ml-core",
    "datasets": "ml-core",
    # ML framework (1)
    "ray": "ml-framework",
    # ML ops (1)
    "mlflow": "ml-ops",
    # NLP pipeline (1)
    "spacy": "nlp",
    # Container orchestration (1)
    "kubernetes-python": "container-orchestration",
    # Config management (2)
    "ansible": "config-management",
    "salt": "config-management",
    # Workflow orchestration (2)
    "airflow": "workflow-orchestration",
    "celery": "workflow-orchestration",
    # Reactive app (1)
    "streamlit": "reactive-app",
    # Caching client (1)
    "redis-py": "caching-client",
    # Observability (1)
    "opentelemetry-python": "observability",
    # Data transformation (1)
    "dbt-core": "data-transformation",
    # Home automation (1)
    "home-assistant-core": "home-automation",
    # Packaging (2)
    "pip": "packaging",
    "poetry": "packaging",
    # Test runners (2)
    "tox": "test-runner",
    "pytest": "test-runner",
    # Type checking (1)
    "mypy": "type-checking",
    # Formatter (1)
    "black": "formatter",
    # Coverage instrumentation (1)
    "coveragepy": "coverage",
    # Validation / serialization (2)
    "pydantic": "validation",
    "marshmallow": "validation",
    # Cloud SDK (1)
    "botocore": "cloud-sdk",
    # ORM (2)
    "sqlalchemy": "orm",
    "asyncpg": "orm",
    # Crypto / cert app (1)
    "certbot": "crypto-cert",
    # DB client (1)
    "pymongo": "db-client",
    # Async runtime (2)
    "trio": "async-runtime",
    "anyio": "async-runtime",
    # Web scraping (1)
    "scrapy": "web-scraping",
    # Documentation (1)
    "sphinx": "documentation",
    # SSH / transport (1)
    "paramiko": "ssh-transport",
    # --- Leftovers from the original 6-repo experiment that aren't in
    # the user's curated 60-list. Singleton categories ⇒ they go entirely
    # into the train split (the stratifier only holds out from categories
    # with ≥2 repos), so they contribute to Stage-A training without
    # contaminating the category-stratified test claim. ---
    "attrs": "attribute-management",
    "click": "cli-tooling",
    "requests": "http-legacy",
}

assert len(CATEGORY_MAP) == 63, (
    f"CATEGORY_MAP has {len(CATEGORY_MAP)} repos; expected 63 "
    "(60 curated + 3 original-experiment leftovers in singleton "
    "categories so they train without affecting test stratification)."
)
