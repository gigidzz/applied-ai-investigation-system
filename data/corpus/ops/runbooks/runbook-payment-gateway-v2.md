# payment-gateway runbook (v2)

**Service:** `payment-gateway`

## Scope

Operational procedures for payment-gateway, the edge service that authorises card payments. Owned by Payments Core. Page payments-core-oncall for anything affecting authorisation success rate.

## Connection pool sizing

payment-gateway holds a pgbouncer connection pool per pod. Under normal afternoon load each pod requires approximately 28 concurrent connections. The minimum supported pool size is 25; never configure below this value. A pre-deploy check now rejects any configuration below the minimum, added after INC-2026-03-14. The database is Postgres 15 and the global connection ceiling is 400, which bounds the pool at 50 per pod across eight pods.

## Authorisation failure triage

Check the authorise error-rate dashboard first. If errors are HTTP 503 with rising pgbouncer cl_waiting, suspect pool exhaustion and compare against the most recent deploy in the change log. If errors are 502 from a single acquirer, check certificate expiry before anything else.

## Rollback procedure

Rollbacks are driven from the deploy pipeline. A full rollback across all eight pods takes between 8 and 12 minutes. Confirm the error rate has returned to baseline before declaring the incident resolved.

## Capacity planning

Headroom targets and growth projections are maintained in the capacity planning document reviewed each quarter by Payments Core.
