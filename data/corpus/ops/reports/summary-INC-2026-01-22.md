# Summary: INC-2026-01-22

**Service:** `settlement-worker` · **Severity:** SEV2

## What happened

Settlement for 48,900 captured payments was delayed by 24 hours. Merchant payouts arrived one business day late.

## Why

The nightly settlement batch read the acquirer cut-off time from the pod's local timezone rather than UTC, so the batch ran one hour late and missed the acquirer window.

## Resolution

Resolved at 2026-01-22T13:50:00Z. Caused by release settle-2.1.4.
