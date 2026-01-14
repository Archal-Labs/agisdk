# Multi-REAL Finish JSON Schema Documentation

*Generated: 2026-01-14 21:48 UTC*

This document describes the finish JSON schema for each web application.
Re-run `tools/discover_schemas.py` as you add more ground truth files.

## Coverage Summary

| Metric | Value |
|--------|-------|
| Total tasks | 69 |
| Tasks with ground truth | 9 (13%) |
| Tasks without ground truth | 60 |
| Apps used in tasks | 12 |
| Apps with schema knowledge | 12 |

### Tasks Needing Ground Truth

Tasks without GT but with all app schemas known can still have queries validated for syntax/schema.

| Status | Count |
|--------|-------|
| ✅ Has GT | 9 |
| ⚠️ No GT, schema known | 61 |
| ❌ No GT, missing app schema | 0 |

## App Schema Overview

| App | Has `differences` | Has `initialfinaldiff` | App-Specific Diffs | Sources |
|-----|-------------------|------------------------|-------------------|---------|
| dashdish | ✅ | ✅ | None | 1 |
| flyunified | ✅ | ✅ | None | 2 |
| gocalendar | ✅ | ✅ | None | 2 |
| gomail | ✅ | ✅ | None | 6 |
| marrisuite | ❌ | ✅ | bookingDetailsDiff, cancellationDetailsDiff, recentSearchesDiff, roomRequestsDiff | 1 |
| networkin | ❌ | ✅ | None | 1 |
| omnizon | ❌ | ✅ | cancelledOrderDetailsDiff, cartDetailsDiff, orderDetailsDiff | 1 |
| opendining | ✅ | ✅ | None | 1 |
| staynb | ❌ | ✅ | None | 2 |
| topwork | ❌ | ✅ | None | 2 |
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

**Sources:** flyunified-gocalendar-staynb-1.json, flyunified-gocalendar-staynb-2.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `flyunified.differences.bookedFlights.added` | dict | ✅ | 0, 1, 2, 3 |
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
| `flyunified.initialfinaldiff.added` | dict | ❌ | booking, ui, miles |

---

### gocalendar

**Sources:** flyunified-gocalendar-staynb-1.json, flyunified-gocalendar-staynb-2.json

**Top-level keys:** `actionhistory, differences, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `gocalendar.differences.events.added` | dict | ✅ | e789f6b8-298c-4c4b-b530-ba1501e4e708, 2515c670-0620-490b-8791-b6075a011740 |
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

**Sources:** dashdish-gomail-1.json, gomail-marrsuite-1.json, gomail-networkin-topwork-1.json, gomail-omnizon-1.json, gomail-topwork-3.json, gomail-zilloft-1.json

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

### networkin

**Sources:** gomail-networkin-topwork-1.json

**Top-level keys:** `actionhistory, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `networkin.initialfinaldiff.added` | dict | ❌ | jobs, notifications |
| `networkin.initialfinaldiff.deleted` | dict | ❌ | notifications |
| `networkin.initialfinaldiff.updated` | dict | ❌ | config, jobs, notifications |

---

### omnizon

**Sources:** gomail-omnizon-1.json

**Top-level keys:** `actionhistory, cancelledOrderDetailsDiff, cartDetailsDiff, finalstate, initialfinaldiff, initialstate, orderDetailsDiff, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `omnizon.initialfinaldiff.added` | dict | ❌ | order, similarProducts |
| `omnizon.initialfinaldiff.deleted` | dict | ❌ | cart |
| `omnizon.initialfinaldiff.updated` | dict | ❌ | cart, ui, filter, router, similarProducts |
| `omnizon.orderDetailsDiff.added` | dict | ✅ | 0, 1, 2, 3, 4 |
| `omnizon.orderDetailsDiff.deleted` | dict | ❌ | - |
| `omnizon.orderDetailsDiff.updated` | dict | ❌ | - |
| `omnizon.cancelledOrderDetailsDiff.added` | dict | ✅ | 0, 1 |
| `omnizon.cancelledOrderDetailsDiff.deleted` | dict | ❌ | - |
| `omnizon.cancelledOrderDetailsDiff.updated` | dict | ❌ | - |
| `omnizon.cartDetailsDiff.added` | dict | ✅ | 0 |
| `omnizon.cartDetailsDiff.deleted` | dict | ❌ | - |
| `omnizon.cartDetailsDiff.updated` | dict | ❌ | - |

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

**Sources:** flyunified-gocalendar-staynb-1.json, flyunified-gocalendar-staynb-2.json

**Top-level keys:** `actionhistory, finalstate, initialfinaldiff, initialstate, query, state, url`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `staynb.initialfinaldiff.added` | dict | ❌ | - |
| `staynb.initialfinaldiff.deleted` | dict | ❌ | - |
| `staynb.initialfinaldiff.updated` | dict | ❌ | - |

---

### topwork

**Sources:** gomail-networkin-topwork-1.json, gomail-topwork-3.json

**Top-level keys:** `actionhistory, finalstate, initialfinaldiff, initialstate, state`

**Queryable Diff Paths:**

| Path | Type | Has Data | Sample Keys |
|------|------|----------|-------------|
| `topwork.initialfinaldiff.added` | dict | ❌ | jobs, notifications |
| `topwork.initialfinaldiff.deleted` | dict | ❌ | notifications |
| `topwork.initialfinaldiff.updated` | dict | ❌ | config, jobs, notifications |

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