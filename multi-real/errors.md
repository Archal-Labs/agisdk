MESPath query failed: udriver.finalstate.ride.calculatedPrice != null || udriver.finalstate.ride.bookedTrip != null. Error: 'WebCloneEvaluator' object has no attribute 'jmespath_search'
2026-01-14 19:36:12,829 - WARNING - JMESPath query failed: length(gocalendar.differences.events.added) >= `1`. Error: 'WebCloneEvaluator' object has no attribute 'jmespath_search'
2026-01-14 19:36:12,829 - WARNING - JMESPath query failed: gocalendar.differences.events.added.values(@)[?contains(title, 'Meeting')] | [?attendees[?email=='ben.carter@example.com']] | length(@) >= `1`. Error: 'WebCloneEvaluator' object has no attribute 'jmespath_search'
2026-01-14 19:36:12,829 - WARNING - JMESPath query failed: gocalendar.differences.events.added.values(@)[0].start != null && gocalendar.differences.events.added.values(@)[0].end != null. Error: 'WebCloneEvaluator' object has no attribute 'jmespath_search'

2026-01-14 19:49:04,017 - ERROR - Error during action execution attempt: Error: Locator.press: Unknown key: "F6 F6"
Call log:
  - waiting for get_by_test_id("47")
    - locator resolved to <div bid="47" role="alert" aria-live="assertive" browsergym_set_of_marks="0" id="__next-route-announcer__" browsergym_visibility_ratio="0"></div>
  - elementHandle.press("F6 F6")
Traceback (most recent call last):
  File "/home/noahsong/work/archal/agisdk/src/agisdk/REAL/browsergym/core/env.py", line 540, in step
    execute_python_code(
  File "/home/noahsong/work/archal/agisdk/src/agisdk/REAL/browsergym/core/action/base.py", line 60, in execute_python_code
    exec(code, globals)
  File "<string>", line 551, in <module>
  File "<string>", line 467, in press
  File "/home/noahsong/work/archal/agisdk/.venv/lib/python3.12/site-packages/playwright/sync_api/_generated.py", line 17365, in press
    self._sync(
  File "/home/noahsong/work/archal/agisdk/.venv/lib/python3.12/site-packages/playwright/_impl/_sync_base.py", line 115, in _sync
    return task.result()
           ^^^^^^^^^^^^^
  File "/home/noahsong/work/archal/agisdk/.venv/lib/python3.12/site-packages/playwright/_impl/_locator.py", line 544, in press
    return await self._frame.press(self._selector, strict=True, **params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noahsong/work/archal/agisdk/.venv/lib/python3.12/site-packages/playwright/_impl/_frame.py", line 835, in press
    await self._channel.send("press", self._timeout, locals_to_params(locals()))
  File "/home/noahsong/work/archal/agisdk/.venv/lib/python3.12/site-packages/playwright/_impl/_connection.py", line 69, in send
    return await self._connection.wrap_api_call(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/noahsong/work/archal/agisdk/.venv/lib/python3.12/site-packages/playwright/_impl/_connection.py", line 559, in wrap_api_call
    raise rewrite_error(error, f"{parsed_st['apiName']}: {error}") from None
playwright._impl._errors.Error: Locator.press: Unknown key: "F6 F6"
Call log:
  - waiting for get_by_test_id("47")
    - locator resolved to <div bid="47" role="alert" aria-live="assertive" browsergym_set_of_marks="0" id="__next-route-announcer__" browsergym_visibility_ratio="0"></div>
  - elementHandle.press("F6 F6")




# every single task:

agisdk.REAL.browsergym.core.env - INFO - The active page and / or page history has changed during task.validate(). A recovery fix will be applied.


# many tasks

  "error": "Exception uncaught by agent or environment in task browsergym/multi.gomail-topwork-1.\nInternalServerError:\nError code: 500 - {'type': 'error', 'error': {'type': 'api_error', 'message': 'Internal server error'}, 'request_id': 'req_011CX8QpyLgs1TDFz8G4RPmY'}",

'WebCloneEvaluator' object has no attribute 'jmespath_search'"
