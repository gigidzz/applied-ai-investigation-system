# fraud-scorer runbook

**Service:** `fraud-scorer`

## Scope and ownership

Scores each authorisation attempt for fraud risk within 40ms. Owned by the Risk team and operated as a tier-1 service. The paging rotation is `risk-oncall` and the source lives in `northwind/fraud-scorer`.

## Service level objectives

Availability target 99.95% measured monthly. p99 latency budget 350ms. Error budget is 21 minutes per month; exhausting it freezes non-critical deploys for the remainder of the period.

## Dependencies

fraud-scorer depends on the following, and each failure mode differs:

- `feature-store` — if unavailable, risk features are missing and scores are understated.

## Blast radius

A failure of fraud-scorer propagates to `payment-gateway`. Notify the owning teams when declaring an incident, because their alerts will fire as a consequence rather than as a separate cause.

## Incident history

Previous incidents affecting this service:

- INC-2026-01-09 (SEV2, 2026-01-09): A model artefact was published without its feature schema, so every scoring request fell back to the default risk score of 0.5.

## Known failure modes

Check these first, in order, because each has occurred in production:

- A model artefact was published without its feature schema, so every scoring request fell back to the default risk score of 0. See INC-2026-01-09.

## Escalation

Page `risk-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Risk team lead and then to the incident commander rotation. For tier-1 services, declare an incident before beginning mitigation rather than afterwards.
