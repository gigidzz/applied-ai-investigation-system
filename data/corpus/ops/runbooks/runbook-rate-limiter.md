# rate-limiter runbook

**Service:** `rate-limiter`

## Scope and ownership

Token-bucket rate limiting shared by all edge services. Owned by the Platform team and operated as a tier-1 service. The paging rotation is `platform-oncall` and the source lives in `northwind/rate-limiter`.

## Service level objectives

Availability target 99.95% measured monthly. p99 latency budget 350ms. Error budget is 21 minutes per month; exhausting it freezes non-critical deploys for the remainder of the period.

## Dependencies

rate-limiter depends on the following, and each failure mode differs:

- `redis-cluster` — if unavailable, cache misses fall through to the database and latency rises.

## Blast radius

A failure of rate-limiter propagates to `payment-gateway`. Notify the owning teams when declaring an incident, because their alerts will fire as a consequence rather than as a separate cause.

## Incident history

Previous incidents affecting this service:

- INC-2026-02-03 (SEV1, 2026-02-03): A Redis cluster failover promoted a replica that was 40 seconds behind, resetting token buckets and briefly allowing unlimited request rates.

## Known failure modes

Check these first, in order, because each has occurred in production:

- A Redis cluster failover promoted a replica that was 40 seconds behind, resetting token buckets and briefly allowing unlimited request rates. See INC-2026-02-03.

## Escalation

Page `platform-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Platform team lead and then to the incident commander rotation. For tier-1 services, declare an incident before beginning mitigation rather than afterwards.
