# Database maintenance

## Archival

The archival job prunes ledger partitions older than 18 months. It runs nightly and alerts on failure; the alert was missing before INC-2026-02-17.

## Migrations

Migrations adding a non-null column to a table above 10 million rows must use a backfill rather than a default, to avoid an exclusive lock.
