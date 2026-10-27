# feature-store runbook

**Service:** `feature-store`

## Scope and ownership

Serves precomputed risk features to fraud-scorer with a 5ms budget. Owned by the Risk team and operated as a tier-2 service. The paging rotation is `risk-oncall` and the source lives in `northwind/feature-store`.

## Service level objectives

Availability target 99.9% measured monthly. p99 latency budget 800ms. Error budget is 43 minutes per month.

## Dependencies

feature-store depends on the following, and each failure mode differs:

- `redis-cluster` — if unavailable, cache misses fall through to the database and latency rises.

## Blast radius

A failure of feature-store propagates to `fraud-scorer`. Notify the owning teams when declaring an incident, because their alerts will fire as a consequence rather than as a separate cause.

## Incident history

Previous incidents affecting this service:

- INC-2026-06-14 (SEV2, 2026-06-14): Redis memory pressure triggered eviction of risk features, and fraud-scorer treated missing features as zero rather than unknown.

## Known failure modes

Check these first, in order, because each has occurred in production:

- Redis memory pressure triggered eviction of risk features, and fraud-scorer treated missing features as zero rather than unknown. See INC-2026-06-14.

## Escalation

Page `risk-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Risk team lead and then to the incident commander rotation. For tier-2 services, declare an incident before beginning mitigation rather than afterwards.
