# reporting-api runbook

**Service:** `reporting-api`

## Scope and ownership

Aggregated reporting queries over the ledger read replica. Owned by the Merchant Experience team and operated as a tier-3 service. The paging rotation is `merchant-exp-oncall` and the source lives in `northwind/reporting-api`.

## Service level objectives

Availability target 99.5% measured monthly. No hard latency budget. Degradation is acceptable during incident response on tier-1 services.

## Dependencies

reporting-api depends on the following, and each failure mode differs:

- `postgres-replica` — if unavailable, reporting queries fail; transactional paths are unaffected.

## Blast radius

A failure of reporting-api propagates to `merchant-portal`. Notify the owning teams when declaring an incident, because their alerts will fire as a consequence rather than as a separate cause.

## Known failure modes

No production incident has been recorded for reporting-api. Treat unfamiliar symptoms as a dependency problem first and check postgres-replica before investigating this service directly.

## Escalation

Page `merchant-exp-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Merchant Experience team lead and then to the incident commander rotation. For tier-3 services, declare an incident before beginning mitigation rather than afterwards.
