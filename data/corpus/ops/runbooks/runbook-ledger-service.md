# ledger-service runbook

**Service:** `ledger-service`

## Scope and ownership

Double-entry ledger recording every authorisation, capture and refund. Owned by the Payments Core team and operated as a tier-1 service. The paging rotation is `payments-core-oncall` and the source lives in `northwind/ledger-service`.

## Service level objectives

Availability target 99.95% measured monthly. p99 latency budget 350ms. Error budget is 21 minutes per month; exhausting it freezes non-critical deploys for the remainder of the period.

## Dependencies

ledger-service depends on the following, and each failure mode differs:

- `postgres-primary` — if unavailable, all writes fail.

## Blast radius

A failure of ledger-service propagates to `payment-gateway`, `settlement-worker`, `merchant-portal`, `webhook-dispatcher`. Notify the owning teams when declaring an incident, because their alerts will fire as a consequence rather than as a separate cause.

## Incident history

Previous incidents affecting this service:

- INC-2026-02-17 (SEV1, 2026-02-17): The primary Postgres volume reached 94% capacity because an archival job had silently failed for eleven days. Write latency degraded until the ledger began rejecting transactions.
- INC-2026-07-19 (SEV2, 2026-07-19): A migration added a non-null column with a default to a 400 million row table, holding an exclusive lock for the duration.

## Known failure modes

Check these first, in order, because each has occurred in production:

- The primary Postgres volume reached 94% capacity because an archival job had silently failed for eleven days. See INC-2026-02-17.
- A migration added a non-null column with a default to a 400 million row table, holding an exclusive lock for the duration. See INC-2026-07-19.

## Escalation

Page `payments-core-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Payments Core team lead and then to the incident commander rotation. For tier-1 services, declare an incident before beginning mitigation rather than afterwards.
