# settlement-worker runbook

**Service:** `settlement-worker`

## Scope and ownership

Nightly batch job that settles captured payments with acquirers. Owned by the Payments Core team and operated as a tier-2 service. The paging rotation is `payments-core-oncall` and the source lives in `northwind/settlement-worker`.

## Service level objectives

Availability target 99.9% measured monthly. p99 latency budget 800ms. Error budget is 43 minutes per month.

## Dependencies

settlement-worker depends on the following, and each failure mode differs:

- `ledger-service` — if unavailable, captures fail while authorisations continue.
- `settlement-queue` — if unavailable, settlement is delayed until the queue drains.

## Incident history

Previous incidents affecting this service:

- INC-2026-01-22 (SEV2, 2026-01-22): The nightly settlement batch read the acquirer cut-off time from the pod's local timezone rather than UTC, so the batch ran one hour late and missed the acquirer window.
- INC-2026-08-07 (SEV3, 2026-08-07): The settlement queue accumulated a backlog after an acquirer returned HTTP 429 for sustained periods, and the worker had no adaptive rate limiting.

## Known failure modes

Check these first, in order, because each has occurred in production:

- The nightly settlement batch read the acquirer cut-off time from the pod's local timezone rather than UTC, so the batch ran one hour late and missed the acquirer window. See INC-2026-01-22.
- The settlement queue accumulated a backlog after an acquirer returned HTTP 429 for sustained periods, and the worker had no adaptive rate limiting. See INC-2026-08-07.

## Escalation

Page `payments-core-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Payments Core team lead and then to the incident commander rotation. For tier-2 services, declare an incident before beginning mitigation rather than afterwards.
