"""Add the metadata columns introduced after the original Flask schema.

Revision ID: 20261102_0002
Revises: 20261102_0001
Create Date: 2026-11-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20261102_0002"
down_revision: Union[str, None] = "20261102_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("optimizations", sa.Column("category", sa.String(length=40), nullable=True))
    op.add_column("optimizations", sa.Column("month", sa.Integer(), nullable=True))
    op.add_column("optimizations", sa.Column("rating", sa.Float(), nullable=True))
    op.add_column("optimizations", sa.Column("ad_spend", sa.Float(), nullable=True))
    op.add_column("optimizations", sa.Column("method", sa.String(length=40), nullable=True))
    op.add_column("optimizations", sa.Column("product_name", sa.String(length=200), nullable=True))
    op.add_column("optimizations", sa.Column("quality", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("optimizations", "quality")
    op.drop_column("optimizations", "product_name")
    op.drop_column("optimizations", "method")
    op.drop_column("optimizations", "ad_spend")
    op.drop_column("optimizations", "rating")
    op.drop_column("optimizations", "month")
    op.drop_column("optimizations", "category")
