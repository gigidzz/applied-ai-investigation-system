# webhook-dispatcher runbook

**Service:** `webhook-dispatcher`

## Scope and ownership

Delivers event webhooks to merchant endpoints with retry and backoff. Owned by the Platform team and operated as a tier-2 service. The paging rotation is `platform-oncall` and the source lives in `northwind/webhook-dispatcher`.

## Service level objectives

Availability target 99.9% measured monthly. p99 latency budget 800ms. Error budget is 43 minutes per month.

## Dependencies

webhook-dispatcher depends on the following, and each failure mode differs:

- `ledger-service` — if unavailable, captures fail while authorisations continue.
- `settlement-queue` — if unavailable, settlement is delayed until the queue drains.

## Incident history

Previous incidents affecting this service:

- INC-2026-03-21 (SEV2, 2026-03-21): Retry backoff was changed from exponential to linear, so a single unreachable merchant endpoint consumed the whole dispatch worker pool.

## Known failure modes

Check these first, in order, because each has occurred in production:

- Retry backoff was changed from exponential to linear, so a single unreachable merchant endpoint consumed the whole dispatch worker pool. See INC-2026-03-21.

## Escalation

Page `platform-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Platform team lead and then to the incident commander rotation. For tier-2 services, declare an incident before beginning mitigation rather than afterwards.
