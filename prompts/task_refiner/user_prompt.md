Raw user request:
{{user_prompt}}

Known start URL:
{{start_url}}

Rewrite the request for the browser agent so it can execute autonomously in one run.
Do not include any user-specified return format (e.g. "as JSON", "in a table", schema) in `refined_user_prompt`—put it only in `return_format`.
Return JSON only using the required schema.
