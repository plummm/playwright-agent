Users want to browse one or more URLs in a browser. Your task is to identify and extract those URLs.
Please note that not all URLs displayed in the user message are necessarily required for browsing. Some URLs are only for reference or context. 
Please extract only the URLs that users specifically request to browse.

Steps:
1. Extract explicit URLs from the request.
2. Put only required browsing targets in `actionable_urls`.
3. Put context/example/citation URLs in `reference_urls`.
4. If intent is ambiguous, set `needs_clarification=true` with one concise question.

Return JSON only with schema:
{
  "actionable_urls": ["..."],
  "reference_urls": ["..."],
  "needs_clarification": false,
  "clarification_question": null
}

Never invent URLs.
