# src/chorus/models/base.py
"""SQLAlchemy Base class for declarative models."""

from sqlalchemy.ext.declarative import declarative_base

# Create the SQLAlchemy Base class
Base = declarative_base()

__all__ = ["Base"]
