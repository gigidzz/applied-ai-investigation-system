# checkout-api runbook

**Service:** `checkout-api`

## Scope and ownership

Orchestrates the customer checkout flow and calls payment-gateway. Owned by the Checkout team and operated as a tier-1 service. The paging rotation is `checkout-oncall` and the source lives in `northwind/checkout-api`.

## Service level objectives

Availability target 99.95% measured monthly. p99 latency budget 350ms. Error budget is 21 minutes per month; exhausting it freezes non-critical deploys for the remainder of the period.

## Dependencies

checkout-api depends on the following, and each failure mode differs:

- `payment-gateway` — if unavailable, no new payments can be authorised.
- `kyc-service` — if unavailable, merchant onboarding stalls but existing merchants are unaffected.
- `notify-service` — if unavailable, customers stop receiving receipts; payments are unaffected.

## Incident history

Previous incidents affecting this service:

- INC-2026-05-11 (SEV2, 2026-05-11): A new idempotency key derived from the cart contents rather than the client-supplied key, so a customer editing their cart could create two authorisations.

## Known failure modes

Check these first, in order, because each has occurred in production:

- A new idempotency key derived from the cart contents rather than the client-supplied key, so a customer editing their cart could create two authorisations. See INC-2026-05-11.

## Escalation

Page `checkout-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Checkout team lead and then to the incident commander rotation. For tier-1 services, declare an incident before beginning mitigation rather than afterwards.
