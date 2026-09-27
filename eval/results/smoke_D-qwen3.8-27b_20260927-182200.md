# smoke run 20260927-182200

Model `qwen/qwen3.8-27b` (reasoning_effort None), commit `b167461`, prompt fingerprint `eca9cd476398`.

| metric | count |
|---|---|
| outcome correct | 3 of 6 |
| tools correct | 5 of 6 |
| constraints correct | 6 of 6 |
| relaxed correct | 1 of 4 |
| false fits | 0 of 6 |
| relaxed answers breaking another constraint | 0 of 6 |
| gave up (step cap) | 1 of 6 |
| errors | 0 of 6 |
| LLM calls | 25 |
| input tokens | 55711 |
| output tokens | 2087 |
| wall-clock seconds (incl. pacing waits) | 547.25 |
| HTTP 429 retries | 1 |
| malformed tool call retries (Groq tool_use_failed) | 0 |

| task | expected | got | tools | constraints | relaxed | false fit | calls | tokens in/out | s |
|---|---|---|---|---|---|---|---|---|---|
| s02 | relaxed_match | no_match | yes | yes | no | no | 5 | 11072/469 | 61.992 |
| s04 | relaxed_match | relaxed_match | yes | yes | yes | no | 4 | 8577/413 | 121.009 |
| s05 | relaxed_match | gave_up | no | yes | no | no | 6 | 16057/406 | 124.642 |
| s06 | relaxed_match | no_match | yes | yes | no | no | 4 | 8265/361 | 117.834 |
| s07 | no_match | no_match | yes | yes | - | no | 3 | 5912/213 | 60.858 |
| s08 | no_match | no_match | yes | yes | - | no | 3 | 5828/225 | 60.917 |
