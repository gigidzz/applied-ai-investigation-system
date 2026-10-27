# merchant-portal runbook

**Service:** `merchant-portal`

## Scope and ownership

Web application where merchants view transactions and payouts. Owned by the Merchant Experience team and operated as a tier-2 service. The paging rotation is `merchant-exp-oncall` and the source lives in `northwind/merchant-portal`.

## Service level objectives

Availability target 99.9% measured monthly. p99 latency budget 800ms. Error budget is 43 minutes per month.

## Dependencies

merchant-portal depends on the following, and each failure mode differs:

- `payment-gateway` — if unavailable, no new payments can be authorised.
- `ledger-service` — if unavailable, captures fail while authorisations continue.
- `reporting-api` — if unavailable, merchant reporting pages degrade.

## Incident history

Previous incidents affecting this service:

- INC-2026-05-29 (SEV3, 2026-05-29): A reporting query lost its merchant_id predicate during a refactor, so the transaction list query scanned the whole ledger replica.

## Known failure modes

Check these first, in order, because each has occurred in production:

- A reporting query lost its merchant_id predicate during a refactor, so the transaction list query scanned the whole ledger replica. See INC-2026-05-29.

## Escalation

Page `merchant-exp-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Merchant Experience team lead and then to the incident commander rotation. For tier-2 services, declare an incident before beginning mitigation rather than afterwards.
