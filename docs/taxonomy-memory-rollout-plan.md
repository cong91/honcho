# Honcho taxonomy memory rollout plan

## Goal
Ship taxonomy-native memory on the existing Honcho codebase as the canonical production memory platform, without hybrid external governance. End-state memory must support domain classification, horizon-aware retrieval, expiry/lifecycle handling, thesis extraction, and durable API/query surfaces while keeping Hermes as a client.

## End-state architecture
- Canonical carrier remains `Document/Conclusion` in the short term, enriched via `internal_metadata.memory`.
- Memory taxonomy fields are first-class in write, query, retrieval, and API layers:
  - `domain`
  - `horizon` (`short|medium|long`)
  - `thesis_kind` (`preference|fact|decision|plan|state|rule`)
  - `expiry` (`none|review|date|event`)
- Retrieval honors taxonomy filters and excludes expired memories by default.
- Prompt/deriver paths produce reusable thesis-bearing observations with taxonomy attached.
- API surfaces expose taxonomy fields directly instead of requiring raw filter injection.
- Lifecycle semantics become native over time: review due, event invalidation, supersession, promotion/demotion.

## Current state after phases 1-6
- Taxonomy schema and storage path implemented.
- Nested filters on `metadata.memory.*` implemented.
- Expiry-aware retrieval groundwork implemented.
- Taxonomy-aware prompting and lightweight prioritization implemented.
- Peer/conclusion API taxonomy fields implemented.
- Lifecycle projection groundwork implemented in response paths.
- Dev test environment dependencies now install via `uv sync --group dev`.
- Host-shell pytest now targets localhost DB automatically when runtime config uses Docker hostname `database`.

## Phase breakdown

### Phase A — Foundation hardening
Status: mostly done
- Schema + storage path in `internal_metadata.memory`
- Nested JSON filters
- Exclude expired in retrieval
- Prompt/tool guidance for taxonomy
- API request surfaces for peer/conclusion retrieval

### Phase B — Test and infra stabilization
Status: in progress
- Make host-shell tests resolve DB correctly (`database` -> `localhost` fallback in tests)
- Standardize dev bring-up:
  - `docker compose up -d database redis`
  - `uv sync --group dev`
  - `.venv/bin/pytest ...`
- Add focused test subsets for taxonomy paths before broad suite runs
- Keep full suite green across both Docker-networked and host-shell execution

### Phase C — Retrieval quality upgrade
Status: next
- Replace lightweight heuristic prioritization with explicit scoring helpers
- Blend semantic score + taxonomy priority + recency + derivation score
- Add domain-aware boosts and query-intent routing
- Add observability for why a memory was surfaced

### Phase D — Lifecycle native semantics
Status: next
- Add canonical lifecycle helper module
- Introduce native fields/derived projections for:
  - `review_due_at`
  - `pending_event_key`
  - `superseded_by`
  - `supersedes`
  - `demote_after`
- Add review sweeps / lifecycle workers
- Add event invalidation hooks
- Add soft supersession without destructive delete

### Phase E — Deriver/thesis productionization
Status: next
- Tighten prompts so stable user/workspace/project knowledge consistently becomes thesis-bearing memory
- Add explicit confidence calibration
- Separate temporary state vs durable rule/decision/preference more reliably
- Add evaluation fixtures for false-positive durable memory writes

### Phase F — Migration and rollout
Status: pending
- Backfill taxonomy for existing high-value documents using offline classifier pass
- Default unknown legacy memories to safe fallback buckets
- Reindex vector metadata after taxonomy enrichment
- Add feature flag for stricter taxonomy gating if needed during rollout
- Ship dashboard metrics:
  - memory writes by thesis kind/domain/horizon
  - expired filtered count
  - review-due count
  - superseded count
  - retrieval mix by source/rank reason

## Recommended execution order
1. Stabilize test infra and get focused taxonomy suite green.
2. Promote retrieval scoring from heuristic to explicit scoring helpers.
3. Implement lifecycle helpers + supersession model.
4. Improve deriver extraction quality with evals.
5. Run migration/backfill + reindex.
6. Turn on stricter retrieval/lifecycle defaults in production.

## Operational runbook
### Local dev / host shell
1. `docker compose up -d database redis`
2. `uv sync --group dev`
3. `TEST_DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@localhost:5432/postgres .venv/bin/pytest -q tests/routes/test_conclusions.py tests/routes/test_peers.py tests/crud/test_document.py tests/crud/test_representation_manager.py tests/test_schema_validations.py`

### Docker-networked runtime
- Keep `.env` using `database` hostname for in-compose services.
- Do not reuse Docker hostname assumptions for host-shell pytest.

## Risks
- Lifecycle semantics can drift if kept only as response projection for too long.
- Legacy memories without taxonomy may distort ranking until backfill lands.
- Over-aggressive derivation can promote transient state into long-term memory.
- Test environment split (Docker hostname vs localhost) can regress if not codified.

## Definition of done for production memory v1
- Taxonomy fields are present on write/query/retrieval/API paths.
- Expired memories are excluded by default everywhere relevant.
- Review/event lifecycle is queryable and operationally actionable.
- Retrieval ranking is taxonomy-aware and measurable.
- Existing memory corpus is backfilled/reindexed.
- Focused and full test suites pass in documented environments.
