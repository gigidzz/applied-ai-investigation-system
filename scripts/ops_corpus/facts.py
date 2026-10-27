"""The single source of truth for the synthetic operations corpus.

Every generated document is rendered from the structures below. Keeping the
facts in one place rather than writing 85 documents by hand buys three things:

  * Planted contradictions are deliberate and auditable. Each one is declared
    in PLANTED, naming the documents that disagree and what a correct system
    should do about it.
  * The corpus is reproducible. All variation comes from a seeded generator,
    so re-running produces byte-identical files, which evaluation requires.
  * Evaluation labels can be derived rather than guessed. We know which
    document holds which fact because we put it there.

The company, people and systems here are fictional.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

COMPANY = "Northwind Payments"
SEED = 20260314  # the date of the flagship incident; any fixed value would do


# --------------------------------------------------------------------------- #
# structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Service:
    name: str
    team: str
    tier: int  # 1 = customer-facing and revenue-critical, 3 = internal
    summary: str
    depends_on: tuple[str, ...]
    repo: str
    on_call: str


@dataclass(frozen=True)
class Deploy:
    release: str
    service: str
    deployed_at: str  # ISO-8601, UTC
    author: str
    summary: str
    risk: str  # low | medium | high
    rolled_back: bool = False


@dataclass(frozen=True)
class TimelineEntry:
    at: str
    note: str


@dataclass(frozen=True)
class Incident:
    incident_id: str
    title: str
    service: str
    severity: str  # SEV1 | SEV2 | SEV3
    started_at: str
    detected_at: str
    resolved_at: str
    impact: str
    root_cause: str
    caused_by_release: str | None
    affected_services: tuple[str, ...]
    timeline: tuple[TimelineEntry, ...]
    action_items: tuple[str, ...]
    author: str


@dataclass(frozen=True)
class Message:
    at: str
    who: str
    text: str


@dataclass(frozen=True)
class Thread:
    thread_id: str
    channel: str
    date: str
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Adr:
    adr_id: str
    title: str
    date: str
    status: str  # accepted | superseded | proposed
    context: str
    decision: str
    consequences: tuple[str, ...]
    author: str


@dataclass(frozen=True)
class Runbook:
    slug: str
    title: str
    service: str
    sections: tuple[tuple[str, str], ...]  # (heading, body)


@dataclass(frozen=True)
class OpsReport:
    slug: str
    period: str
    highlights: tuple[str, ...]
    incident_ids: tuple[str, ...]
    deploy_count: int


@dataclass(frozen=True)
class Planted:
    """A deliberately planted retrieval or reasoning challenge."""

    kind: str  # contradiction | gap | near_duplicate | multi_hop
    label: str
    explanation: str
    documents: tuple[str, ...]
    probe_question: str
    expected_behaviour: str
    answer: str | None = None
    category: str = ""
    gold_documents: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# services
# --------------------------------------------------------------------------- #

SERVICES: tuple[Service, ...] = (
    Service(
        "payment-gateway",
        "Payments Core",
        1,
        "Edge service that authorises card payments against upstream acquirers.",
        ("fraud-scorer", "ledger-service", "rate-limiter"),
        "northwind/payment-gateway",
        "payments-core-oncall",
    ),
    Service(
        "checkout-api",
        "Checkout",
        1,
        "Orchestrates the customer checkout flow and calls payment-gateway.",
        ("payment-gateway", "kyc-service", "notify-service"),
        "northwind/checkout-api",
        "checkout-oncall",
    ),
    Service(
        "ledger-service",
        "Payments Core",
        1,
        "Double-entry ledger recording every authorisation, capture and refund.",
        ("postgres-primary",),
        "northwind/ledger-service",
        "payments-core-oncall",
    ),
    Service(
        "settlement-worker",
        "Payments Core",
        2,
        "Nightly batch job that settles captured payments with acquirers.",
        ("ledger-service", "settlement-queue"),
        "northwind/settlement-worker",
        "payments-core-oncall",
    ),
    Service(
        "fraud-scorer",
        "Risk",
        1,
        "Scores each authorisation attempt for fraud risk within 40ms.",
        ("feature-store",),
        "northwind/fraud-scorer",
        "risk-oncall",
    ),
    Service(
        "notify-service",
        "Platform",
        3,
        "Sends transactional email and SMS to cardholders and merchants.",
        ("email-relay",),
        "northwind/notify-service",
        "platform-oncall",
    ),
    Service(
        "merchant-portal",
        "Merchant Experience",
        2,
        "Web application where merchants view transactions and payouts.",
        ("payment-gateway", "ledger-service", "reporting-api"),
        "northwind/merchant-portal",
        "merchant-exp-oncall",
    ),
    Service(
        "kyc-service",
        "Compliance",
        2,
        "Verifies merchant identity documents against third-party registries.",
        ("document-store",),
        "northwind/kyc-service",
        "compliance-oncall",
    ),
    Service(
        "webhook-dispatcher",
        "Platform",
        2,
        "Delivers event webhooks to merchant endpoints with retry and backoff.",
        ("ledger-service", "settlement-queue"),
        "northwind/webhook-dispatcher",
        "platform-oncall",
    ),
    Service(
        "rate-limiter",
        "Platform",
        1,
        "Token-bucket rate limiting shared by all edge services.",
        ("redis-cluster",),
        "northwind/rate-limiter",
        "platform-oncall",
    ),
    Service(
        "reporting-api",
        "Merchant Experience",
        3,
        "Aggregated reporting queries over the ledger read replica.",
        ("postgres-replica",),
        "northwind/reporting-api",
        "merchant-exp-oncall",
    ),
    Service(
        "feature-store",
        "Risk",
        2,
        "Serves precomputed risk features to fraud-scorer with a 5ms budget.",
        ("redis-cluster",),
        "northwind/feature-store",
        "risk-oncall",
    ),
)

SERVICE_NAMES = tuple(s.name for s in SERVICES)

ENGINEERS = (
    "a.kovacs",
    "m.oyelaran",
    "s.petrova",
    "d.nakamura",
    "r.achterberg",
    "l.mwangi",
    "t.lindqvist",
    "j.barros",
    "n.haddad",
    "e.sorensen",
)


# --------------------------------------------------------------------------- #
# the flagship scenario: the March payment disruption
#
# Answering "which change caused the March payment disruption, which services
# were affected, and which documents confirm it" requires four separate
# documents: the change log row, the incident report, the service catalogue
# entry (for downstream impact) and the chat thread (for the rollback).
# No single document carries the whole answer.
# --------------------------------------------------------------------------- #

FLAGSHIP_RELEASE = "pay-gw-4.2.1"
FLAGSHIP_INCIDENT = "INC-2026-03-14"

FLAGSHIP_DEPLOY = Deploy(
    release=FLAGSHIP_RELEASE,
    service="payment-gateway",
    deployed_at="2026-03-14T14:20:00Z",
    author="r.achterberg",
    summary=(
        "Upgrade pgbouncer client library to 3.4.0 and reduce the per-pod "
        "connection pool from 50 to 10 to cut idle connections."
    ),
    risk="medium",
    rolled_back=True,
)

FLAGSHIP_INCIDENT_RECORD = Incident(
    incident_id=FLAGSHIP_INCIDENT,
    title="Payment authorisation failures following connection pool change",
    service="payment-gateway",
    severity="SEV1",
    started_at="2026-03-14T14:32:00Z",
    detected_at="2026-03-14T14:38:00Z",
    resolved_at="2026-03-14T15:10:00Z",
    impact=(
        "Approximately 40% of checkout authorisation requests returned HTTP 503 "
        "for 38 minutes. An estimated 11,400 payment attempts failed. Merchants "
        "saw elevated decline rates and the merchant portal transaction view "
        "returned partial data."
    ),
    root_cause=(
        "Release pay-gw-4.2.1 reduced the per-pod database connection pool from "
        "50 to 10. Under normal afternoon load the gateway requires roughly 28 "
        "concurrent connections per pod, so the pool was exhausted within twelve "
        "minutes of the rollout completing. Exhausted checkouts waited on pool "
        "acquisition until the 2s upstream timeout fired and returned 503."
    ),
    caused_by_release=FLAGSHIP_RELEASE,
    affected_services=("payment-gateway", "checkout-api", "merchant-portal"),
    timeline=(
        TimelineEntry("14:20", f"{FLAGSHIP_RELEASE} rollout begins, canary at 10%."),
        TimelineEntry("14:28", "Rollout reaches 100% of payment-gateway pods."),
        TimelineEntry("14:32", "First 503 responses observed on /v2/authorise."),
        TimelineEntry("14:38", "Error-rate alert pages payments-core-oncall."),
        TimelineEntry("14:44", "checkout-api circuit breaker opens, shedding load."),
        TimelineEntry("14:51", "Pool exhaustion identified from pgbouncer metrics."),
        TimelineEntry("14:58", f"Decision taken to roll back {FLAGSHIP_RELEASE}."),
        TimelineEntry("15:10", "Rollback complete, error rate returns to baseline."),
    ),
    action_items=(
        "Add a pre-deploy check rejecting pool sizes below the documented minimum.",
        "Add a pgbouncer pool-saturation alert at 80% utilisation.",
        "Require load testing for any change to connection pool configuration.",
    ),
    author="s.petrova",
)

# The chat thread disagrees with the post-mortem about when the rollback
# finished: 15:40 here against 15:10 above. This is contradiction C1.
FLAGSHIP_THREAD = Thread(
    thread_id="THREAD-2026-03-14-incident",
    channel="#inc-payment-gateway",
    date="2026-03-14",
    messages=(
        Message("14:39", "a.kovacs", "Paged. Seeing 503s on authorise, about 40% of traffic."),
        Message("14:41", "s.petrova", "Confirmed. Started around 14:32. Anything deployed?"),
        Message("14:42", "r.achterberg", "4.2.1 went out at 14:20. Pool size change in there."),
        Message(
            "14:45", "s.petrova", "checkout-api breaker just opened. Merchant portal degraded too."
        ),
        Message(
            "14:52",
            "a.kovacs",
            "pgbouncer shows cl_waiting climbing, pool is at 10/10 on every pod.",
        ),
        Message("14:58", "s.petrova", "Rolling back 4.2.1. r.achterberg can you drive it?"),
        Message("15:02", "r.achterberg", "Rollback started."),
        Message("15:40", "r.achterberg", "Rollback finished on all pods, errors are gone."),
        Message("15:44", "s.petrova", "Agreed, calling it resolved. I will write the post-mortem."),
        Message(
            "16:05",
            "l.mwangi",
            "For the record this is the same class of failure as the April event.",
        ),
    ),
)


# --------------------------------------------------------------------------- #
# the remaining incidents
#
# Note the deliberate absence of any April 2026 incident: several documents
# refer to "the April event", but no post-mortem for it exists. That absence
# is gap G1, and a question about it must be answered with an abstention.
# --------------------------------------------------------------------------- #

_INCIDENT_SEEDS: tuple[tuple[str, str, str, str, str | None, str, str], ...] = (
    (
        "INC-2026-01-09",
        "2026-01-09",
        "fraud-scorer",
        "SEV2",
        "fraud-3.8.0",
        "A model artefact was published without its feature schema, so every "
        "scoring request fell back to the default risk score of 0.5.",
        "Fraud scoring was effectively disabled for 2 hours 14 minutes. No "
        "fraudulent transactions were confirmed in the window, but 340 payments "
        "were later manually reviewed.",
    ),
    (
        "INC-2026-01-22",
        "2026-01-22",
        "settlement-worker",
        "SEV2",
        "settle-2.1.4",
        "The nightly settlement batch read the acquirer cut-off time from the "
        "pod's local timezone rather than UTC, so the batch ran one hour late "
        "and missed the acquirer window.",
        "Settlement for 48,900 captured payments was delayed by 24 hours. "
        "Merchant payouts arrived one business day late.",
    ),
    (
        "INC-2026-02-03",
        "2026-02-03",
        "rate-limiter",
        "SEV1",
        "rl-1.9.2",
        "A Redis cluster failover promoted a replica that was 40 seconds behind, "
        "resetting token buckets and briefly allowing unlimited request rates.",
        "All edge services accepted unthrottled traffic for 90 seconds. "
        "payment-gateway latency rose to 4.1s at p99 under the surge.",
    ),
    (
        "INC-2026-02-17",
        "2026-02-17",
        "ledger-service",
        "SEV1",
        None,
        "The primary Postgres volume reached 94% capacity because an archival "
        "job had silently failed for eleven days. Write latency degraded until "
        "the ledger began rejecting transactions.",
        "Ledger writes failed for 22 minutes, which blocked all captures. "
        "Authorisations continued but could not be recorded.",
    ),
    (
        "INC-2026-02-26",
        "2026-02-26",
        "notify-service",
        "SEV3",
        "notify-5.0.1",
        "A templating change escaped HTML entities twice, so merchant emails "
        "rendered with visible entity codes.",
        "Roughly 6,200 notification emails were sent with malformed subject "
        "lines. No payment flow was affected.",
    ),
    (
        "INC-2026-03-02",
        "2026-03-02",
        "kyc-service",
        "SEV2",
        None,
        "A third-party identity registry changed its response schema without "
        "notice, and the parser rejected every response as invalid.",
        "Merchant onboarding was blocked for 3 hours 40 minutes. 71 merchant "
        "applications queued and were processed after the fix.",
    ),
    (
        "INC-2026-03-21",
        "2026-03-21",
        "webhook-dispatcher",
        "SEV2",
        "wh-2.6.0",
        "Retry backoff was changed from exponential to linear, so a single "
        "unreachable merchant endpoint consumed the whole dispatch worker pool.",
        "Webhook delivery to all merchants was delayed by up to 55 minutes. No events were lost.",
    ),
    (
        "INC-2026-05-11",
        "2026-05-11",
        "checkout-api",
        "SEV2",
        "checkout-7.3.0",
        "A new idempotency key derived from the cart contents rather than the "
        "client-supplied key, so a customer editing their cart could create two "
        "authorisations.",
        "19 duplicate authorisations were created over 4 hours. All were voided "
        "within one business day.",
    ),
    (
        "INC-2026-05-29",
        "2026-05-29",
        "merchant-portal",
        "SEV3",
        "portal-4.0.2",
        "A reporting query lost its merchant_id predicate during a refactor, so "
        "the transaction list query scanned the whole ledger replica.",
        "The merchant portal transaction page timed out for 35 minutes. No data "
        "was exposed across merchant boundaries; the query failed before "
        "returning rows.",
    ),
    (
        "INC-2026-06-14",
        "2026-06-14",
        "feature-store",
        "SEV2",
        None,
        "Redis memory pressure triggered eviction of risk features, and "
        "fraud-scorer treated missing features as zero rather than unknown.",
        "Fraud scores were systematically understated for 48 minutes. 1,100 "
        "transactions were rescored retrospectively.",
    ),
    (
        "INC-2026-06-30",
        "2026-06-30",
        "payment-gateway",
        "SEV2",
        "pay-gw-4.6.0",
        "A TLS certificate for one acquirer endpoint expired. The renewal "
        "automation covered only endpoints registered before January 2026.",
        "Authorisations routed to one acquirer failed for 26 minutes, roughly "
        "8% of total volume. Traffic was manually rerouted to a secondary "
        "acquirer.",
    ),
    (
        "INC-2026-07-19",
        "2026-07-19",
        "ledger-service",
        "SEV2",
        "ledger-9.2.0",
        "A migration added a non-null column with a default to a 400 million "
        "row table, holding an exclusive lock for the duration.",
        "Ledger writes were blocked for 9 minutes during the migration window. "
        "Captures retried successfully afterwards.",
    ),
    (
        "INC-2026-08-07",
        "2026-08-07",
        "settlement-worker",
        "SEV3",
        None,
        "The settlement queue accumulated a backlog after an acquirer returned "
        "HTTP 429 for sustained periods, and the worker had no adaptive rate "
        "limiting.",
        "Settlement completed 6 hours late for 12,300 payments. Payouts were "
        "unaffected as the acquirer window had not closed.",
    ),
)


def _timeline_for(seed: tuple, rng: random.Random) -> tuple[TimelineEntry, ...]:
    """Build a plausible timeline from the incident's own facts."""
    incident_id, date, service, severity, release, _cause, _impact = seed
    start_hour = rng.choice((2, 9, 11, 13, 16, 19, 22))
    detect_delay = rng.choice((3, 5, 7, 11, 16))
    mitigate_delay = detect_delay + rng.choice((8, 14, 21, 30))
    resolve_delay = mitigate_delay + rng.choice((9, 17, 26, 41))

    def at(minutes: int) -> str:
        return f"{start_hour:02d}:{minutes % 60:02d}"

    entries = [TimelineEntry(at(0), f"First symptoms observed on {service}.")]
    if release:
        entries.insert(
            0, TimelineEntry(at(0), f"Release {release} completed rollout earlier in the day.")
        )
    entries += [
        TimelineEntry(at(detect_delay), f"Alert pages the on-call engineer for {service}."),
        TimelineEntry(at(mitigate_delay), "Cause identified and mitigation begins."),
        TimelineEntry(at(resolve_delay), "Service returns to baseline; incident resolved."),
    ]
    return tuple(entries)


def build_incidents() -> tuple[Incident, ...]:
    """The flagship incident plus thirteen generated ones, deterministically."""
    rng = random.Random(SEED)
    incidents = [FLAGSHIP_INCIDENT_RECORD]

    for seed in _INCIDENT_SEEDS:
        incident_id, date, service, severity, release, cause, impact = seed
        svc = next(s for s in SERVICES if s.name == service)
        downstream = tuple(other.name for other in SERVICES if service in other.depends_on)
        timeline = _timeline_for(seed, rng)
        incidents.append(
            Incident(
                incident_id=incident_id,
                title=f"{svc.summary.split('.')[0][:60]} degradation",
                service=service,
                severity=severity,
                started_at=f"{date}T{timeline[0].at}:00Z",
                detected_at=f"{date}T{timeline[-3].at}:00Z",
                resolved_at=f"{date}T{timeline[-1].at}:00Z",
                impact=impact,
                root_cause=cause,
                caused_by_release=release,
                affected_services=(service,) + downstream,
                timeline=timeline,
                action_items=tuple(
                    rng.sample(
                        [
                            f"Add a pre-deploy validation check for {service}.",
                            f"Add saturation alerting to {service} at 80% of limits.",
                            "Require a load test for configuration changes of this class.",
                            "Document the mitigation steps in the service runbook.",
                            "Add an integration test covering the failure mode.",
                            "Review the third-party contract for change notification.",
                        ],
                        k=3,
                    )
                ),
                author=rng.choice(ENGINEERS),
            )
        )
    return tuple(incidents)


# --------------------------------------------------------------------------- #
# deploy history (the change log)
# --------------------------------------------------------------------------- #

_CHANGE_SUMMARIES = (
    "Bump base image to the January security patch level.",
    "Add structured request logging behind a feature flag.",
    "Increase HTTP client timeout from 2s to 5s for acquirer calls.",
    "Replace the retry helper with a shared library implementation.",
    "Add a database index on (merchant_id, created_at).",
    "Reduce container memory request from 1Gi to 768Mi.",
    "Enable gzip compression on API responses.",
    "Migrate the config loader to environment variables.",
    "Add a readiness probe that checks database connectivity.",
    "Upgrade the Postgres driver to 3.2.1.",
    "Split the authorisation handler into smaller functions.",
    "Add per-merchant request metrics.",
    "Raise the connection pool from 10 to 50 after incident review.",
    "Remove the deprecated v1 endpoints.",
    "Cache acquirer routing rules for 60 seconds.",
    "Add a circuit breaker around the fraud-scorer call.",
    "Correct the timezone handling in the settlement cut-off calculation.",
    "Add an adaptive rate limiter for acquirer 429 responses.",
)


def build_deploys() -> tuple[Deploy, ...]:
    """Roughly 300 deploy records across eight months, deterministically.

    The flagship deploy and every release named by an incident are included,
    so that a question about what caused an incident can actually be joined
    from the change log to the post-mortem.
    """
    rng = random.Random(SEED + 1)
    deploys: list[Deploy] = [FLAGSHIP_DEPLOY]

    # releases that incidents point at must exist in the change log
    for seed in _INCIDENT_SEEDS:
        incident_id, date, service, _sev, release, _cause, _impact = seed
        if release is None:
            continue
        deploys.append(
            Deploy(
                release=release,
                service=service,
                deployed_at=f"{date}T{rng.randint(8, 16):02d}:{rng.randint(0, 59):02d}:00Z",
                author=rng.choice(ENGINEERS),
                summary=rng.choice(_CHANGE_SUMMARIES),
                risk=rng.choice(("medium", "high")),
                rolled_back=rng.random() < 0.4,
            )
        )

    # routine deploys filling out the history
    counters = {s.name: 1 for s in SERVICES}
    for month, days in (
        (1, 31),
        (2, 28),
        (3, 31),
        (4, 30),
        (5, 31),
        (6, 30),
        (7, 31),
        (8, 20),
    ):
        for _ in range(36):
            svc = rng.choice(SERVICES)
            counters[svc.name] += 1
            prefix = svc.name.split("-")[0][:6]
            deploys.append(
                Deploy(
                    release=f"{prefix}-{rng.randint(1, 9)}.{counters[svc.name] % 12}."
                    f"{rng.randint(0, 9)}",
                    service=svc.name,
                    deployed_at=(
                        f"2026-{month:02d}-{rng.randint(1, days):02d}"
                        f"T{rng.randint(7, 19):02d}:{rng.randint(0, 59):02d}:00Z"
                    ),
                    author=rng.choice(ENGINEERS),
                    summary=rng.choice(_CHANGE_SUMMARIES),
                    risk=rng.choices(("low", "medium", "high"), weights=(6, 3, 1))[0],
                    rolled_back=rng.random() < 0.06,
                )
            )

    return tuple(sorted(deploys, key=lambda d: d.deployed_at))


# --------------------------------------------------------------------------- #
# runbooks
#
# kyc-service deliberately has no runbook: that absence is gap G2.
# payment-gateway has two near-identical versions: that is near-duplicate N1.
# The pool-sizing runbook states Postgres 15, which ADR-0009 contradicts (C4).
# --------------------------------------------------------------------------- #

_PAY_GW_RUNBOOK_BODY = (
    (
        "Scope",
        "Operational procedures for payment-gateway, the edge service that "
        "authorises card payments. Owned by Payments Core. Page "
        "payments-core-oncall for anything affecting authorisation success rate.",
    ),
    (
        "Connection pool sizing",
        "payment-gateway holds a pgbouncer connection pool per pod. Under "
        "normal afternoon load each pod requires approximately 28 concurrent "
        "connections. The minimum supported pool size is 25; never configure "
        "below this value. The database is Postgres 15 and the global "
        "connection ceiling is 400, which bounds the pool at 50 per pod across "
        "eight pods.",
    ),
    (
        "Authorisation failure triage",
        "Check the authorise error-rate dashboard first. If errors are HTTP 503 "
        "with rising pgbouncer cl_waiting, suspect pool exhaustion and compare "
        "against the most recent deploy in the change log. If errors are 502 "
        "from a single acquirer, check certificate expiry before anything else.",
    ),
    (
        "Rollback procedure",
        "Rollbacks are driven from the deploy pipeline. A full rollback across "
        "all eight pods takes between 8 and 12 minutes. Confirm the error rate "
        "has returned to baseline before declaring the incident resolved.",
    ),
    (
        "Capacity planning",
        "Headroom targets and growth projections are maintained in the capacity "
        "planning document reviewed each quarter by Payments Core.",
    ),
)

# Near-duplicate N1: identical except for the two sentences noted below.
_PAY_GW_RUNBOOK_V2_BODY = (
    _PAY_GW_RUNBOOK_BODY[0],
    (
        "Connection pool sizing",
        "payment-gateway holds a pgbouncer connection pool per pod. Under "
        "normal afternoon load each pod requires approximately 28 concurrent "
        "connections. The minimum supported pool size is 25; never configure "
        "below this value. A pre-deploy check now rejects any configuration "
        "below the minimum, added after INC-2026-03-14. The database is "
        "Postgres 15 and the global connection ceiling is 400, which bounds "
        "the pool at 50 per pod across eight pods.",
    ),
    _PAY_GW_RUNBOOK_BODY[2],
    _PAY_GW_RUNBOOK_BODY[3],
    _PAY_GW_RUNBOOK_BODY[4],
)

_TIER_OBJECTIVES = {
    1: (
        "Availability target 99.95% measured monthly. p99 latency budget 350ms. "
        "Error budget is 21 minutes per month; exhausting it freezes "
        "non-critical deploys for the remainder of the period."
    ),
    2: (
        "Availability target 99.9% measured monthly. p99 latency budget 800ms. "
        "Error budget is 43 minutes per month."
    ),
    3: (
        "Availability target 99.5% measured monthly. No hard latency budget. "
        "Degradation is acceptable during incident response on tier-1 services."
    ),
}

_DEPENDENCY_CONSEQUENCE = {
    "fraud-scorer": "authorisations fall back to the default risk score",
    "ledger-service": "captures fail while authorisations continue",
    "rate-limiter": "throttling is bypassed and upstream load becomes unbounded",
    "payment-gateway": "no new payments can be authorised",
    "kyc-service": "merchant onboarding stalls but existing merchants are unaffected",
    "notify-service": "customers stop receiving receipts; payments are unaffected",
    "settlement-queue": "settlement is delayed until the queue drains",
    "redis-cluster": "cache misses fall through to the database and latency rises",
    "postgres-primary": "all writes fail",
    "postgres-replica": "reporting queries fail; transactional paths are unaffected",
    "feature-store": "risk features are missing and scores are understated",
    "document-store": "identity documents cannot be retrieved for review",
    "email-relay": "outbound mail queues rather than failing",
    "reporting-api": "merchant reporting pages degrade",
}


def _runbook_sections_for(
    service: Service, incidents: tuple[Incident, ...]
) -> tuple[tuple[str, str], ...]:
    """Build a runbook from the service's own facts and incident history.

    Deriving each section from the service rather than filling one template
    keeps the runbooks genuinely different from each other. An earlier version
    used a shared template with the name substituted, which produced documents
    with Jaccard similarity above 0.6 to each other — accidental near-duplicates
    that would have polluted deduplication measurements.
    """
    own = tuple(i for i in incidents if i.service == service.name)
    downstream = tuple(o.name for o in SERVICES if service.name in o.depends_on)

    dependency_lines = "\n".join(
        f"- `{dep}` — if unavailable, {_DEPENDENCY_CONSEQUENCE.get(dep, 'behaviour is undefined')}."
        for dep in service.depends_on
    )

    sections: list[tuple[str, str]] = [
        (
            "Scope and ownership",
            f"{service.summary} Owned by the {service.team} team and operated as "
            f"a tier-{service.tier} service. The paging rotation is "
            f"`{service.on_call}` and the source lives in `{service.repo}`.",
        ),
        ("Service level objectives", _TIER_OBJECTIVES[service.tier]),
        (
            "Dependencies",
            f"{service.name} depends on the following, and each failure mode "
            f"differs:\n\n{dependency_lines}",
        ),
    ]

    if downstream:
        sections.append(
            (
                "Blast radius",
                f"A failure of {service.name} propagates to "
                f"{', '.join(f'`{d}`' for d in downstream)}. Notify the owning "
                f"teams when declaring an incident, because their alerts will "
                f"fire as a consequence rather than as a separate cause.",
            )
        )

    if own:
        history = "\n".join(
            f"- {i.incident_id} ({i.severity}, {i.started_at[:10]}): {i.root_cause}" for i in own
        )
        sections.append(
            (
                "Incident history",
                f"Previous incidents affecting this service:\n\n{history}",
            )
        )
        sections.append(
            (
                "Known failure modes",
                "Check these first, in order, because each has occurred in "
                "production:\n\n"
                + "\n".join(f"- {i.root_cause.split('.')[0]}. See {i.incident_id}." for i in own),
            )
        )
    else:
        sections.append(
            (
                "Known failure modes",
                f"No production incident has been recorded for {service.name}. "
                f"Treat unfamiliar symptoms as a dependency problem first and "
                f"check {service.depends_on[0]} before investigating this "
                f"service directly.",
            )
        )

    sections.append(
        (
            "Escalation",
            f"Page `{service.on_call}` first. If unacknowledged after 10 "
            f"minutes the alert escalates to the {service.team} team lead and "
            f"then to the incident commander rotation. For tier-"
            f"{service.tier} services, declare an incident before beginning "
            f"mitigation rather than afterwards.",
        )
    )
    return tuple(sections)


_TOPIC_RUNBOOKS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "incident-response",
        "Incident response process",
        (
            (
                "Severity definitions",
                "SEV1 means customer payments are failing or money is at risk. "
                "SEV2 means degraded service without payment loss. SEV3 means "
                "internal or cosmetic impact only.",
            ),
            (
                "Declaring an incident",
                "Any engineer may declare. Open a channel named #inc-<service>, "
                "post the symptoms, and assign an incident commander before "
                "beginning mitigation.",
            ),
            (
                "Post-mortem requirement",
                "Every SEV1 and SEV2 requires a written post-mortem within five "
                "business days, naming the causing release where one exists.",
            ),
        ),
    ),
    (
        "on-call",
        "On-call expectations",
        (
            (
                "Rotation",
                "Each team runs a weekly rotation with a primary and a "
                "secondary. Handover happens Monday at 10:00 UTC.",
            ),
            (
                "Response targets",
                "Acknowledge a SEV1 page within 5 minutes and a SEV2 within 15 minutes.",
            ),
        ),
    ),
    (
        "deploy-process",
        "Deployment process",
        (
            (
                "Pipeline",
                "All services deploy through the shared pipeline. A "
                "canary at 10% runs for 8 minutes before full rollout.",
            ),
            (
                "Change log",
                "Every deploy writes a row to the change log with "
                "release, service, timestamp, author and risk level. "
                "This is the authoritative record of what changed.",
            ),
            (
                "Risk classification",
                "Configuration changes affecting connection "
                "pools, timeouts or rate limits are at least "
                "medium risk and require a load test.",
            ),
        ),
    ),
    (
        "rollback",
        "Rollback procedures",
        (
            (
                "When to roll back",
                "Roll back rather than fix forward whenever a "
                "SEV1 correlates with a deploy in the last 60 "
                "minutes.",
            ),
            (
                "Duration",
                "Rollback takes 8 to 12 minutes for edge services and "
                "up to 25 minutes for services requiring a schema "
                "revert.",
            ),
        ),
    ),
    (
        "database-maintenance",
        "Database maintenance",
        (
            (
                "Archival",
                "The archival job prunes ledger partitions older than "
                "18 months. It runs nightly and alerts on failure; the "
                "alert was missing before INC-2026-02-17.",
            ),
            (
                "Migrations",
                "Migrations adding a non-null column to a table above "
                "10 million rows must use a backfill rather than a "
                "default, to avoid an exclusive lock.",
            ),
        ),
    ),
    (
        "alert-triage",
        "Alert triage",
        (
            (
                "Order of checks",
                "Confirm the alert is real, check the change log "
                "for recent deploys, then check dependencies "
                "before investigating the service itself.",
            ),
            (
                "Noisy alerts",
                "Alerts firing more than three times a week without "
                "action are reviewed at the weekly operations "
                "meeting.",
            ),
        ),
    ),
)


def build_runbooks() -> tuple[Runbook, ...]:
    incidents = build_incidents()
    books: list[Runbook] = [
        Runbook(
            "runbook-payment-gateway-v1",
            "payment-gateway runbook (v1)",
            "payment-gateway",
            _PAY_GW_RUNBOOK_BODY,
        ),
        Runbook(
            "runbook-payment-gateway-v2",
            "payment-gateway runbook (v2)",
            "payment-gateway",
            _PAY_GW_RUNBOOK_V2_BODY,
        ),
    ]

    for svc in SERVICES:
        # gap G2: kyc-service has no runbook, so its escalation path is
        # genuinely undocumented and a question about it must abstain.
        if svc.name in {"payment-gateway", "kyc-service"}:
            continue
        books.append(
            Runbook(
                f"runbook-{svc.name}",
                f"{svc.name} runbook",
                svc.name,
                _runbook_sections_for(svc, incidents),
            )
        )

    for slug, title, sections in _TOPIC_RUNBOOKS:
        books.append(Runbook(f"runbook-{slug}", title, "", sections))

    return tuple(books)


# --------------------------------------------------------------------------- #
# architecture decision records
#
# ADR-0007 contradicts the service catalogue on settlement-queue ownership (C2).
# ADR-0009 contradicts the payment-gateway runbook on Postgres version (C4).
# --------------------------------------------------------------------------- #

ADRS: tuple[Adr, ...] = (
    Adr(
        "ADR-0001",
        "Use double-entry bookkeeping in the ledger",
        "2025-09-12",
        "accepted",
        "Single-entry records made reconciliation with acquirers ambiguous.",
        "ledger-service records every movement as a balanced pair of entries.",
        ("Reconciliation becomes mechanical.", "Storage roughly doubles."),
        "s.petrova",
    ),
    Adr(
        "ADR-0002",
        "Shared rate limiting at the edge",
        "2025-10-02",
        "accepted",
        "Each edge service implemented its own throttling with different semantics.",
        "A single rate-limiter service backed by Redis serves all edge services.",
        (
            "One place to reason about limits.",
            "rate-limiter becomes a tier-1 dependency and a shared failure domain.",
        ),
        "t.lindqvist",
    ),
    Adr(
        "ADR-0003",
        "Synchronous fraud scoring with a 40ms budget",
        "2025-10-28",
        "accepted",
        "Asynchronous scoring allowed authorisations to complete before scoring.",
        "fraud-scorer is called synchronously with a hard 40ms timeout and "
        "fails open to a configured default score.",
        (
            "Fraud decisions are made before authorisation.",
            "A scorer outage silently degrades to the default score.",
        ),
        "l.mwangi",
    ),
    Adr(
        "ADR-0004",
        "Idempotency keys supplied by the client",
        "2025-11-19",
        "accepted",
        "Server-derived keys could not distinguish a retry from a new intent.",
        "checkout-api requires a client-supplied Idempotency-Key header.",
        ("Retries are unambiguous.", "Clients that omit the header are rejected."),
        "j.barros",
    ),
    Adr(
        "ADR-0005",
        "Webhook delivery with exponential backoff",
        "2025-12-04",
        "accepted",
        "Linear retries let one slow merchant endpoint starve the worker pool.",
        "webhook-dispatcher retries with exponential backoff and per-endpoint concurrency caps.",
        (
            "One bad endpoint can no longer exhaust the pool.",
            "Worst-case delivery latency increases.",
        ),
        "n.haddad",
    ),
    Adr(
        "ADR-0006",
        "Keep settlement as a nightly batch",
        "2026-01-15",
        "accepted",
        "Real-time settlement was considered but acquirer windows are daily.",
        "settlement-worker remains a nightly batch aligned to acquirer cut-off.",
        ("Simpler failure model.", "A missed window costs a full day."),
        "d.nakamura",
    ),
    Adr(
        "ADR-0007",
        "Transfer settlement-queue ownership to Platform",
        "2026-02-11",
        "accepted",
        "settlement-queue is shared infrastructure used by settlement-worker "
        "and webhook-dispatcher, but was operated by Payments Core, who were "
        "not paged for webhook-side failures.",
        "Ownership of settlement-queue transfers from Payments Core to the "
        "Platform team, effective 2026-02-20. Platform becomes the paging "
        "destination for all settlement-queue alerts.",
        (
            "One owner for shared infrastructure.",
            "The service catalogue must be updated to reflect the new owner.",
        ),
        "t.lindqvist",
    ),
    Adr(
        "ADR-0008",
        "Fail closed on KYC registry errors",
        "2026-03-05",
        "accepted",
        "INC-2026-03-02 showed the parser rejecting all responses silently.",
        "kyc-service fails closed and queues applications when the registry "
        "returns an unparseable response.",
        ("No unverified merchant is onboarded.", "Onboarding stalls during outages."),
        "e.sorensen",
    ),
    Adr(
        "ADR-0009",
        "Migrate the ledger to Postgres 16",
        "2026-01-28",
        "accepted",
        "Postgres 15 lacked the logical replication features needed for the read replica strategy.",
        "The ledger primary and replica were migrated to Postgres 16. The "
        "migration completed on 2026-01-31 and Postgres 15 is no longer in use "
        "anywhere in the payments estate.",
        (
            "Logical replication becomes available.",
            "All runbooks referencing Postgres 15 need revising.",
        ),
        "a.kovacs",
    ),
    Adr(
        "ADR-0010",
        "Adaptive rate limiting for acquirer 429s",
        "2026-08-12",
        "proposed",
        "INC-2026-08-07 showed settlement-worker had no backpressure response.",
        "settlement-worker will reduce its request rate in response to 429 "
        "responses, with a floor of 10% of nominal throughput.",
        (
            "Backlogs recover without manual intervention.",
            "Settlement may take longer under sustained throttling.",
        ),
        "d.nakamura",
    ),
)


# --------------------------------------------------------------------------- #
# chat threads
# --------------------------------------------------------------------------- #

_THREAD_TEMPLATES = (
    (
        "Anyone else seeing elevated latency on {service}?",
        "Checking now.",
        "Dashboard looks normal to me, p99 is 180ms.",
        "False alarm, it was my local network.",
    ),
    (
        "{service} alert fired and self-resolved in 90 seconds.",
        "Same thing happened on Tuesday.",
        "Raising it at the operations meeting.",
        "Added to the noisy-alert list.",
    ),
    (
        "Deploying {release} to {service} in 10 minutes.",
        "Any schema changes?",
        "No, config only.",
        "Go ahead.",
    ),
    (
        "Who owns {dep} these days?",
        "Good question, the catalogue says one thing and I thought it changed.",
        "Check the decision records, I think there was an ADR.",
        "Will follow up.",
    ),
    (
        "Merchant reports a missing payout for yesterday.",
        "Settlement ran late, see the operations report.",
        "Confirmed, it landed this morning.",
        "Replied to the merchant.",
    ),
    (
        "Can we raise the pool size on {service}?",
        "The runbook has a documented minimum, check it before changing.",
        "Right, 25. Current config is fine then.",
        "Closing this.",
    ),
)


def _fake_release(service: str, rng: random.Random) -> str:
    prefix = service.split("-")[0][:6]
    return f"{prefix}-{rng.randint(1, 9)}.{rng.randint(0, 9)}.0"


def build_threads() -> tuple[Thread, ...]:
    rng = random.Random(SEED + 2)
    threads = [FLAGSHIP_THREAD]

    for index in range(19):
        template = _THREAD_TEMPLATES[index % len(_THREAD_TEMPLATES)]
        svc = rng.choice(SERVICES)
        month = rng.randint(1, 8)
        day = rng.randint(1, 28)
        hour = rng.randint(8, 18)
        people = rng.sample(ENGINEERS, k=min(4, len(template)))
        messages = tuple(
            Message(
                f"{hour:02d}:{(index * 7 + position * 11) % 60:02d}",
                people[position % len(people)],
                text.format(
                    service=svc.name,
                    release=_fake_release(svc.name, rng),
                    dep=svc.depends_on[0] if svc.depends_on else "settlement-queue",
                ),
            )
            for position, text in enumerate(template)
        )
        threads.append(
            Thread(
                thread_id=f"THREAD-2026-{month:02d}-{day:02d}-{svc.name}",
                channel=f"#{svc.team.lower().replace(' ', '-')}",
                date=f"2026-{month:02d}-{day:02d}",
                messages=messages,
            )
        )
    return tuple(threads)


# --------------------------------------------------------------------------- #
# monthly operations reports
#
# The March report states a 12% impact figure, contradicting the post-mortem's
# 40% (C3). The May report refers to "the April event", for which no incident
# report exists (G1).
# --------------------------------------------------------------------------- #

OPS_REPORTS: tuple[OpsReport, ...] = (
    OpsReport(
        "ops-2026-01",
        "January 2026",
        (
            "Two SEV2 incidents, both resolved within target.",
            "Fraud scoring defaulted open for over two hours on the 9th.",
            "Settlement missed the acquirer window on the 22nd.",
        ),
        ("INC-2026-01-09", "INC-2026-01-22"),
        38,
    ),
    OpsReport(
        "ops-2026-02",
        "February 2026",
        (
            "One SEV1 caused by a Redis failover resetting rate limits.",
            "Ledger write failure traced to a silently failing archival job.",
            "Ownership of settlement-queue moved to the Platform team.",
        ),
        ("INC-2026-02-03", "INC-2026-02-17", "INC-2026-02-26"),
        41,
    ),
    OpsReport(
        "ops-2026-03",
        "March 2026",
        (
            "One SEV1 on the 14th following a connection pool change to "
            "payment-gateway. Approximately 12% of transactions were "
            "affected during the 38-minute window.",
            "Merchant onboarding blocked for close to four hours on the 2nd.",
            "Webhook delivery delayed on the 21st after a backoff change.",
        ),
        ("INC-2026-03-02", "INC-2026-03-14", "INC-2026-03-21"),
        44,
    ),
    OpsReport(
        "ops-2026-04",
        "April 2026",
        (
            "A quiet month operationally.",
            "Deploy volume was the lowest since September.",
            "No post-mortems were filed.",
        ),
        (),
        22,
    ),
    OpsReport(
        "ops-2026-05",
        "May 2026",
        (
            "Duplicate authorisations created by an idempotency key change.",
            "Reporting query regression degraded the merchant portal.",
            "Follow-up actions from the April event remain open.",
        ),
        ("INC-2026-05-11", "INC-2026-05-29"),
        39,
    ),
    OpsReport(
        "ops-2026-06",
        "June 2026",
        (
            "Risk features evicted under Redis memory pressure.",
            "An expired acquirer certificate blocked 8% of authorisations.",
            "Certificate renewal automation extended to all endpoints.",
        ),
        ("INC-2026-06-14", "INC-2026-06-30"),
        36,
    ),
    OpsReport(
        "ops-2026-07",
        "July 2026",
        (
            "A migration lock blocked ledger writes for nine minutes.",
            "Migration guidance added to the database runbook.",
        ),
        ("INC-2026-07-19",),
        35,
    ),
    OpsReport(
        "ops-2026-08",
        "August 2026",
        (
            "Settlement backlog after sustained acquirer throttling.",
            "Adaptive rate limiting proposed in ADR-0010.",
        ),
        ("INC-2026-08-07",),
        25,
    ),
    OpsReport(
        "ops-2026-q1",
        "Q1 2026 summary",
        (
            "Six incidents across the quarter, two of them SEV1.",
            "Both SEV1s involved shared infrastructure rather than application logic.",
            "Change-related incidents outnumbered capacity-related ones two to one.",
        ),
        ("INC-2026-01-09", "INC-2026-02-03", "INC-2026-03-14"),
        123,
    ),
    OpsReport(
        "ops-2026-q2",
        "Q2 2026 summary",
        (
            "Four incidents, none SEV1.",
            "Third-party dependencies caused half of the quarter's incidents.",
            "Mean time to resolution improved from 54 to 38 minutes.",
        ),
        ("INC-2026-05-11", "INC-2026-06-14", "INC-2026-06-30"),
        110,
    ),
)


# --------------------------------------------------------------------------- #
# planted challenges
#
# Every contradiction, gap and near-duplicate in the corpus is declared here.
# build_ops_corpus.py renders this list into DESIGN.md and derives evaluation
# questions from it, so the labels are authored rather than reverse-engineered.
# --------------------------------------------------------------------------- #

PLANTED: tuple[Planted, ...] = (
    Planted(
        kind="multi_hop",
        label="M1 — the March payment disruption",
        explanation=(
            "The complete answer requires four documents: the change log row "
            "for pay-gw-4.2.1, the INC-2026-03-14 post-mortem, the "
            "payment-gateway catalogue entry for downstream services, and the "
            "incident chat thread. No single document names the change, the "
            "affected services and the confirmation together."
        ),
        documents=(
            "changes/change-log.csv",
            "incidents/INC-2026-03-14.pdf",
            "services/payment-gateway.json",
            "threads/THREAD-2026-03-14-incident.txt",
        ),
        gold_documents=(
            "changes/change-log.csv",
            "incidents/INC-2026-03-14.pdf",
            "services/payment-gateway.json",
        ),
        probe_question=(
            "Which system change caused the March payment processing "
            "disruption, which services were affected, and which documents "
            "confirm this?"
        ),
        expected_behaviour=(
            "Name release pay-gw-4.2.1 deployed at 14:20 UTC on 2026-03-14, "
            "identify payment-gateway, checkout-api and merchant-portal as "
            "affected, and cite the change log, the post-mortem and the "
            "service catalogue."
        ),
        answer=(
            "Release pay-gw-4.2.1, which reduced the connection pool from 50 "
            "to 10, caused INC-2026-03-14. payment-gateway, checkout-api and "
            "merchant-portal were affected."
        ),
        category="cross_document_synthesis",
    ),
    Planted(
        kind="contradiction",
        label="C1 — rollback completion time",
        explanation=(
            "The post-mortem timeline records the rollback completing at 15:10 "
            "and resolves the incident at that time. The chat thread has the "
            "engineer who performed it reporting completion at 15:40."
        ),
        documents=(
            "incidents/INC-2026-03-14.pdf",
            "threads/THREAD-2026-03-14-incident.txt",
        ),
        probe_question="When did the rollback of pay-gw-4.2.1 finish?",
        expected_behaviour=(
            "Report both times, attribute each to its source, and state that "
            "the sources disagree rather than silently choosing one."
        ),
        category="contradiction",
    ),
    Planted(
        kind="contradiction",
        label="C2 — settlement-queue ownership",
        explanation=(
            "The service catalogue lists settlement-queue under Payments Core. "
            "ADR-0007 transfers ownership to the Platform team effective "
            "2026-02-20 and was accepted."
        ),
        documents=("services/settlement-worker.json", "adr/ADR-0007.docx"),
        probe_question="Which team owns the settlement queue?",
        expected_behaviour=(
            "Surface both claims, note that ADR-0007 is dated later and marked "
            "accepted, and flag that the catalogue appears stale rather than "
            "asserting one owner as fact."
        ),
        category="contradiction",
    ),
    Planted(
        kind="contradiction",
        label="C3 — customer impact of the March incident",
        explanation=(
            "The post-mortem states approximately 40% of checkout "
            "authorisation requests failed. The March operations report states "
            "approximately 12% of transactions were affected."
        ),
        documents=("incidents/INC-2026-03-14.pdf", "reports/ops-2026-03.md"),
        probe_question=(
            "What proportion of transactions failed during the March payment incident?"
        ),
        expected_behaviour=("Report both figures with their sources and note the discrepancy."),
        category="contradiction",
    ),
    Planted(
        kind="contradiction",
        label="C4 — Postgres version for the ledger",
        explanation=(
            "The payment-gateway runbook states the database is Postgres 15. "
            "ADR-0009 records a completed migration to Postgres 16 on "
            "2026-01-31 and states Postgres 15 is no longer in use."
        ),
        documents=("runbooks/runbook-payment-gateway-v2.md", "adr/ADR-0009.docx"),
        probe_question="Which version of Postgres does the ledger run on?",
        expected_behaviour=(
            "Note that the runbook and ADR-0009 disagree, and that the ADR is "
            "the later document and reports the migration as complete."
        ),
        category="contradiction",
    ),
    Planted(
        kind="gap",
        label="G1 — the April incident does not exist",
        explanation=(
            "The March incident chat thread and the May operations report both "
            "refer to 'the April event', and the April operations report "
            "explicitly states no post-mortems were filed. No April incident "
            "report exists anywhere in the corpus."
        ),
        documents=("reports/ops-2026-04.md", "reports/ops-2026-05.md"),
        probe_question="What caused the April 2026 payment outage?",
        expected_behaviour=(
            "Abstain. State that documents refer to an April event but that no "
            "incident report for April exists in the corpus, so the cause "
            "cannot be determined."
        ),
        category="unanswerable",
    ),
    Planted(
        kind="gap",
        label="G2 — kyc-service has no runbook",
        explanation=(
            "kyc-service appears in the service catalogue and in "
            "INC-2026-03-02, but no runbook was ever written for it, so its "
            "escalation procedure is undocumented."
        ),
        documents=("services/kyc-service.json",),
        probe_question="What is the alert escalation procedure for kyc-service?",
        expected_behaviour=(
            "Abstain on the specifics. The catalogue names compliance-oncall as "
            "the rotation, but no runbook documents an escalation procedure, "
            "and the system should say so rather than inventing one or "
            "borrowing another service's."
        ),
        category="unanswerable",
    ),
    Planted(
        kind="gap",
        label="G3 — the capacity planning document is absent",
        explanation=(
            "The payment-gateway runbook refers to a quarterly capacity "
            "planning document. No such document exists in the corpus."
        ),
        documents=("runbooks/runbook-payment-gateway-v1.md",),
        probe_question=(
            "What headroom target does the capacity planning document set for payment-gateway?"
        ),
        expected_behaviour=(
            "Abstain, noting that the runbook references the document but that it is not present."
        ),
        category="unanswerable",
    ),
    Planted(
        kind="near_duplicate",
        label="N1 — two versions of the payment-gateway runbook",
        explanation=(
            "v1 and v2 are identical except that v2's pool-sizing section adds "
            "two sentences about the pre-deploy check introduced after "
            "INC-2026-03-14. Retrieval should not return both as separate "
            "evidence for the same claim."
        ),
        documents=(
            "runbooks/runbook-payment-gateway-v1.md",
            "runbooks/runbook-payment-gateway-v2.md",
        ),
        probe_question="What is the minimum supported connection pool size?",
        expected_behaviour=(
            "Answer 25, citing one version rather than presenting both as independent confirmation."
        ),
        answer="25 connections per pod.",
        category="simple",
    ),
    Planted(
        kind="near_duplicate",
        label="N2 — settlement incident exists as PDF and Markdown",
        explanation=(
            "INC-2026-01-22 is published both as a post-mortem PDF and as a "
            "condensed Markdown summary sharing most of its wording, to give "
            "deduplication a cross-format case."
        ),
        documents=(
            "incidents/INC-2026-01-22.pdf",
            "reports/summary-INC-2026-01-22.md",
        ),
        probe_question="Why were merchant payouts delayed in January?",
        expected_behaviour=(
            "Explain the timezone error in the settlement cut-off calculation, "
            "without double-counting the two near-identical sources."
        ),
        answer=(
            "The settlement batch read the acquirer cut-off in local time "
            "rather than UTC, ran an hour late and missed the window."
        ),
        category="simple",
    ),
)
