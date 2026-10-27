# notify-service runbook

**Service:** `notify-service`

## Scope and ownership

Sends transactional email and SMS to cardholders and merchants. Owned by the Platform team and operated as a tier-3 service. The paging rotation is `platform-oncall` and the source lives in `northwind/notify-service`.

## Service level objectives

Availability target 99.5% measured monthly. No hard latency budget. Degradation is acceptable during incident response on tier-1 services.

## Dependencies

notify-service depends on the following, and each failure mode differs:

- `email-relay` — if unavailable, outbound mail queues rather than failing.

## Blast radius

A failure of notify-service propagates to `checkout-api`. Notify the owning teams when declaring an incident, because their alerts will fire as a consequence rather than as a separate cause.

## Incident history

Previous incidents affecting this service:

- INC-2026-02-26 (SEV3, 2026-02-26): A templating change escaped HTML entities twice, so merchant emails rendered with visible entity codes.

## Known failure modes

Check these first, in order, because each has occurred in production:

- A templating change escaped HTML entities twice, so merchant emails rendered with visible entity codes. See INC-2026-02-26.

## Escalation

Page `platform-oncall` first. If unacknowledged after 10 minutes the alert escalates to the Platform team lead and then to the incident commander rotation. For tier-3 services, declare an incident before beginning mitigation rather than afterwards.
