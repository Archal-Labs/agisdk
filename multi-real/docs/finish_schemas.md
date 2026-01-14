# Multi-REAL Finish JSON Schema Documentation

*Generated: 2026-01-14 16:18 UTC*

This document describes the finish JSON schema for each web application.
Re-run `tools/discover_schemas.py` as you add more ground truth files.

## Coverage Summary

| Metric | Value |
|--------|-------|
| Total tasks | 72 |
| Tasks with ground truth | 6 (8%) |
| Tasks without ground truth | 66 |
| Apps used in tasks | 12 |
| Apps with schema knowledge | 10 |

**Apps missing schema (need GT):** `networkin, omnizon`

### Tasks Needing Ground Truth

Tasks without GT but with all app schemas known can still have queries validated for syntax/schema.

| Status | Count |
|--------|-------|
| ✅ Has GT | 6 |
| ⚠️ No GT, schema known | 57 |
| ❌ No GT, missing app schema | 10 |

## App Schema Overview

| App | Has `differences` | Has `initialfinaldiff` | App-Specific Diffs | Sources |
|-----|-------------------|------------------------|-------------------|---------|
| dashdish | ✅ | ✅ | None | 1 |
| flyunified | ✅ | ✅ | None | 1 |
| gocalendar | ✅ | ✅ | None | 1 |
| gomail | ✅ | ✅ | None | 4 |
| marrisuite | ❌ | ✅ | bookingDetailsDiff, cancellationDetailsDiff, recentSearchesDiff, roomRequestsDiff | 1 |
| opendining | ✅ | ✅ | None | 1 |
| staynb | ❌ | ✅ | None | 1 |
| topwork | ❌ | ✅ | None | 1 |
| udriver | ❌ | ✅ | None | 1 |
| zilloft | ✅ | ✅ | None | 1 |

## Per-App Schema Details

### dashdish

**Sources:** dashdish-gomail-1.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `dashdish.differences.foodOrders.added` | dict | ❌ | - |
| `dashdish.differences.foodOrders.deleted` | dict | ❌ | - |
| `dashdish.differences.foodOrders.updated` | dict | ❌ | - |
| `dashdish.initialfinaldiff.added` | dict | ❌ | cart, config |
| `dashdish.initialfinaldiff.deleted` | dict | ❌ | cart, config |
| `dashdish.initialfinaldiff.updated` | dict | ❌ | - |

---

### flyunified

**Sources:** flyunified-gocalendar-staynb-1.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `flyunified.differences.bookedFlights.added` | dict | ✅ | 0, 1 |
| `flyunified.differences.bookedFlights.deleted` | dict | ❌ | - |
| `flyunified.differences.bookedFlights.updated` | dict | ❌ | - |
| `flyunified.differences.selectedFlightCartIds.added` | dict | ✅ | 0 |
| `flyunified.differences.selectedFlightCartIds.deleted` | dict | ❌ | - |
| `flyunified.differences.selectedFlightCartIds.updated` | dict | ❌ | - |
| `flyunified.differences.purchaseDetails.added` | dict | ❌ | - |
| `flyunified.differences.purchaseDetails.deleted` | dict | ❌ | - |
| `flyunified.differences.purchaseDetails.updated` | dict | ❌ | - |
| `flyunified.initialfinaldiff.deleted` | dict | ❌ | booking, ui, miles |
| `flyunified.initialfinaldiff.updated` | dict | ❌ | flightSearch, booking, ui, miles |

---

### gocalendar

**Sources:** flyunified-gocalendar-staynb-1.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `gocalendar.differences.events.added` | dict | ✅ | e789f6b8-298c-4c4b-b530-ba1501e4e708 |
| `gocalendar.differences.events.deleted` | dict | ❌ | - |
| `gocalendar.differences.events.updated` | dict | ❌ | - |
| `gocalendar.differences.calendars.added` | dict | ❌ | - |
| `gocalendar.differences.calendars.deleted` | dict | ❌ | - |
| `gocalendar.differences.calendars.updated` | dict | ❌ | - |
| `gocalendar.differences.joinedEvents.added` | dict | ❌ | - |
| `gocalendar.differences.joinedEvents.deleted` | dict | ❌ | - |
| `gocalendar.differences.joinedEvents.updated` | dict | ❌ | - |
| `gocalendar.initialfinaldiff.calendar` | dict | ❌ | myEvents, filteredEvents, other_updated |

---

### gomail

**Sources:** dashdish-gomail-1.json, gomail-marrsuite-1.json, gomail-topwork-3.json, gomail-zilloft-1.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `gomail.differences.emails.added` | list | ✅ | id, from, to, cc, bcc |
| `gomail.differences.emails.deleted` | list | ✅ | id, from, to, cc, subject |
| `gomail.differences.emails.updated` | list | ✅ | id |
| `gomail.initialfinaldiff.added` | dict | ❌ | email |
| `gomail.initialfinaldiff.deleted` | dict | ❌ | ui |
| `gomail.initialfinaldiff.updated` | dict | ❌ | ui |

---

### marrisuite

**Sources:** gomail-marrsuite-1.json

**Top-level keys:** `actionhistory, bookingDetailsDiff, cancellationDetailsDiff, finalstate, initialfinaldiff, initialstate, recentSearchesDiff, roomRequestsDiff, state, submitMessage, submitStatus`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `marrisuite.initialfinaldiff.added` | dict | ❌ | booking |
| `marrisuite.initialfinaldiff.updated` | dict | ❌ | search, guest, booking, points |
| `marrisuite.bookingDetailsDiff.added` | list | ❌ | - |
| `marrisuite.bookingDetailsDiff.deleted` | list | ❌ | - |
| `marrisuite.bookingDetailsDiff.updated` | list | ❌ | - |
| `marrisuite.cancellationDetailsDiff.added` | list | ❌ | - |
| `marrisuite.cancellationDetailsDiff.deleted` | list | ❌ | - |
| `marrisuite.cancellationDetailsDiff.updated` | list | ❌ | - |
| `marrisuite.recentSearchesDiff.added` | list | ❌ | - |
| `marrisuite.recentSearchesDiff.deleted` | list | ❌ | - |
| `marrisuite.recentSearchesDiff.updated` | list | ❌ | - |
| `marrisuite.roomRequestsDiff.added` | list | ❌ | - |
| `marrisuite.roomRequestsDiff.deleted` | list | ❌ | - |
| `marrisuite.roomRequestsDiff.updated` | list | ❌ | - |

---

### opendining

**Sources:** opendining-udriver-1.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `opendining.differences.bookings.added` | list | ✅ | restaurant, date, time, guests, tel |
| `opendining.differences.bookings.deleted` | list | ❌ | - |
| `opendining.differences.bookings.updated` | list | ❌ | - |
| `opendining.differences.reviews.added` | list | ❌ | - |
| `opendining.differences.reviews.deleted` | list | ❌ | - |
| `opendining.differences.reviews.updated` | list | ❌ | - |
| `opendining.differences.savedRestaurants.added` | list | ❌ | - |
| `opendining.differences.savedRestaurants.deleted` | list | ❌ | - |
| `opendining.differences.savedRestaurants.updated` | list | ❌ | - |
| `opendining.initialfinaldiff.added` | dict | ❌ | booking |
| `opendining.initialfinaldiff.updated` | dict | ❌ | booking, search, config |

---

### staynb

**Sources:** flyunified-gocalendar-staynb-1.json

**Top-level keys:** `actionhistory, finalstate, initialfinaldiff, initialstate, query, state, url`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `staynb.initialfinaldiff.added` | dict | ❌ | - |
| `staynb.initialfinaldiff.deleted` | dict | ❌ | - |
| `staynb.initialfinaldiff.updated` | dict | ❌ | - |

---

### topwork

**Sources:** gomail-topwork-3.json

**Top-level keys:** `actionhistory, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `topwork.initialfinaldiff.added` | dict | ❌ | jobs |
| `topwork.initialfinaldiff.updated` | dict | ❌ | config, jobs |

---

### udriver

**Sources:** opendining-udriver-1.json

**Top-level keys:** `actionhistory, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `udriver.initialfinaldiff.added` | dict | ❌ | ride, ui, user |
| `udriver.initialfinaldiff.updated` | dict | ❌ | ride, ui, router, user |

---

### zilloft

**Sources:** gomail-zilloft-1.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `zilloft.differences.requestTours.added` | dict | ❌ | - |
| `zilloft.differences.requestTours.deleted` | dict | ❌ | - |
| `zilloft.differences.requestTours.updated` | dict | ❌ | - |
| `zilloft.differences.contactAgents.added` | dict | ❌ | - |
| `zilloft.differences.contactAgents.deleted` | dict | ❌ | - |
| `zilloft.differences.contactAgents.updated` | dict | ❌ | - |
| `zilloft.differences.savedHomes.added` | dict | ✅ | 0, 1, 2 |
| `zilloft.differences.savedHomes.deleted` | dict | ❌ | - |
| `zilloft.differences.savedHomes.updated` | dict | ❌ | - |
| `zilloft.initialfinaldiff.added` | dict | ❌ | savedHomes, config |
| `zilloft.initialfinaldiff.deleted` | dict | ❌ | - |
| `zilloft.initialfinaldiff.updated` | dict | ❌ | - |

---

## Query Pattern Guidelines

### Standard Pattern (when `differences` exists):
```
app_id.differences.collection.added[?filter_condition]
app_id.differences.collection.added[?field == 'value'] | length(@) >= `1`
```

### InitialFinalDiff Pattern:
```
app_id.initialfinaldiff.added.collection != null
values(app_id.initialfinaldiff.added.collection)[?condition]
```

### App-Specific Diff Pattern:
```
app_id.bookingDetailsDiff.added[?condition]
app_id.foodOrders.added (note: may be dict, not array)
```