"""Feature engineering modules.

Heavy notebook-derived builders live in `notebook_features.py` and are imported
directly by the training pipeline. Keeping this package init lightweight makes
small imports, docs checks and metadata access work before optional NLP
dependencies are installed.
"""

__all__: list[str] = []
