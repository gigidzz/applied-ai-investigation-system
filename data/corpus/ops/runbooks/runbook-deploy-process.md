# Deployment process

## Pipeline

All services deploy through the shared pipeline. A canary at 10% runs for 8 minutes before full rollout.

## Change log

Every deploy writes a row to the change log with release, service, timestamp, author and risk level. This is the authoritative record of what changed.

## Risk classification

Configuration changes affecting connection pools, timeouts or rate limits are at least medium risk and require a load test.
