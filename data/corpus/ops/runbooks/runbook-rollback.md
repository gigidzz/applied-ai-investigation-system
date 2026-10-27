# Rollback procedures

## When to roll back

Roll back rather than fix forward whenever a SEV1 correlates with a deploy in the last 60 minutes.

## Duration

Rollback takes 8 to 12 minutes for edge services and up to 25 minutes for services requiring a schema revert.
