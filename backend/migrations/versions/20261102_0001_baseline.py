"""Baseline matching the Flask-SQLAlchemy optimizations table.

Revision ID: 20261102_0001
Revises:
Create Date: 2026-11-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20261102_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "optimizations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.String(length=100), nullable=True),
        sa.Column("cost_price", sa.Float(), nullable=False),
        sa.Column("fixed_costs", sa.Float(), nullable=True),
        sa.Column("available_units", sa.Integer(), nullable=True),
        sa.Column("recommended_price", sa.Float(), nullable=True),
        sa.Column("expected_profit", sa.Float(), nullable=True),
        sa.Column("expected_units_sold", sa.Integer(), nullable=True),
        sa.Column("competitor_prices", sa.JSON(), nullable=True),
        sa.Column("competitor_stats", sa.JSON(), nullable=True),
        sa.Column("price_analysis", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_optimizations_product_id", "optimizations", ["product_id"], unique=False)
    op.create_index("ix_optimizations_timestamp", "optimizations", ["timestamp"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_optimizations_timestamp", table_name="optimizations")
    op.drop_index("ix_optimizations_product_id", table_name="optimizations")
    op.drop_table("optimizations")
