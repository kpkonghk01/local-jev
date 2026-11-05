# Local-Jev

在本機用小型模型（≤ 3B）提供 Jev-compatible 的決策服務：說 TypeSafe 的 `/v1/systemone` wire format，讓 jevbench 內建的 `typesafe` adapter 不需修改就能評測它。每個 server process 只跑一個 Decider，啟動時選定。

目前可用的 Decider：

- `lfm3`：LFM3 Option-ID Decider。`LiquidAI/LFM3-1.7B`（Chat model）透過 Option-ID scoring 回答：關掉 thinking，一次 forward pass 讀出每個 Option ID 的 next-token 機率，再在 Label set 上正規化。固定在 revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`。
- `d1`：`LiquidAI/d1-2B`（Decision model），透過它自己的 `system_one` head 一次 forward pass 直接回答，固定在 revision `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`。

## 在 Mac 上原生執行（`uv run`）

需求：Apple Silicon Mac、[uv](https://docs.astral.sh/uv/)。Python 3.12 與所有相依套件都由 uv 依 `uv.lock` 安裝，不需另外準備。

```sh
uv sync                                  # 第一次：建立 .venv 並安裝鎖定的版本
uv run local-jev serve --decider lfm3    # 或 --decider d1；預設聽 127.0.0.1:8000
```

> **第一次啟動會下載模型。** `LiquidAI/LFM3-1.7B` 約 3.2 GB、`LiquidAI/d1-2B` 約 3.6 GB，會下載到 Hugging Face 快取（預設 `~/.cache/huggingface/hub`，可用 `HF_HOME` 改位置），之後的啟動直接讀快取。下載期間 server 已經在聽，但 health 會回報尚未就緒。下載完成後可以設 `HF_HUB_OFFLINE=1`，省掉啟動時對 Hub 的查詢。

選項：

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--decider` | （必填） | `lfm3` 或 `d1`，見下方「選擇 Decider」 |
| `--host` | `127.0.0.1` | 要讓其他機器連線時改成 `0.0.0.0` |
| `--port` | `8000` | |
| `--device` | `auto` | 自動依序選 CUDA → MPS → CPU；可指定 `cuda`、`mps`、`cpu` |
| `--dtype` | `auto` | CUDA/MPS 用 `bfloat16`、CPU 用 `float32`；可指定 `bfloat16`、`float16`、`float32` |
| `--model` | Decider 自己的 model | Hugging Face model id；也可用環境變數 `LOCAL_JEV_MODEL` |
| `--revision` | Decider 自己固定的 revision | 給了 `--model` 就必須一起給；也可用 `LOCAL_JEV_REVISION` |
| `--chat-template-options` | `{"enable_thinking": false}` | 只適用 `lfm3`：傳給 chat template 的選項（JSON 物件），會合併在預設值之上；也可用 `LOCAL_JEV_CHAT_TEMPLATE_OPTIONS` |

在 Mac 上 `auto` 會選 MPS（Metal）。容器裡沒有 MPS，只能用 CPU。

### 選擇 Decider

每個 server process 只載入一個 Decider，啟動時用 `--decider` 選定；要比較兩者就各開一個 server（不同 port）分別跑 benchmark。兩個 Decider 對外的 wire format 完全一樣。

| | `lfm3` | `d1` |
| --- | --- | --- |
| Model | `LiquidAI/LFM3-1.7B`，Chat model | `LiquidAI/d1-2B`，Decision model |
| 怎麼得到 Distribution | Option-ID scoring：`choice` 的選項依 Criteria 送來的順序標成 A–Z，`noul` 讀 yes/no（`Yes`/`yes`、`No`/`no` 合併計算），`score` 讀等級數字 0–9；在 Label set 上正規化 | d1 自己的 decision head |
| 每個 Decision 的成本 | 一次 forward pass，不生成任何 token | 一次 forward pass |
| MPS 上的延遲（M3 Pro） | 每個 Decision 約 115–240 ms（prompt 80–530 tokens） | 約 65–120 ms（prompt 約 65–75 tokens） |
| 記憶體（MPS bf16，driver） | 約 4.1 GB | 約 5.4 GB |

`lfm3` 在啟動時會先確認每個 Option ID（A–Z、yes/Yes/no/No、0–9）在被讀取的位置剛好是一個 token，不是的話 health 會回報 `failed`，不會等到第一個請求才出錯。

**換成另一個 ≤ 3B 的 Chat model 不需要改程式**，例如：

```sh
uv run local-jev serve --decider lfm3 \
  --model Qwen/Qwen3-1.7B --revision <commit sha> \
  --chat-template-options '{"enable_thinking": false}'
```

- 一定要給 `--revision`：model 用 `trust_remote_code` 載入，固定 revision 才能固定實際執行的程式碼。
- chat template 選項依 model 而定；Option ID 直接接在 chat template 的 generation prompt 之後讀取，所以 template 必須在那裡結束（例如 LFM3 的 `<think></think>`）。
- `--model`/`--revision` 也適用 `d1`（換另一個 d1 checkpoint）。

**已知風險：`lfm3` 有很強的 recency bias。** 研究時用同一個三選一的問題試過全部 6 種選項順序，LFM3-1.7B 每次都選兩個合理選項中「排在後面」的那個，機率 0.85–1.00。這會拉高 Brier 分數，temperature calibration 也修不了 argmax。v1 不處理它：把多種選項順序平均起來可以抵銷順序的影響，但每個 Decision 就不再只要一次 forward pass。

### 確認服務就緒

```sh
curl -s http://127.0.0.1:8000/health
# 載入中：HTTP 503 {"status":"loading"}
# 就緒：  HTTP 200 {"status":"ready"}
# 載入失敗：HTTP 503 {"status":"failed","error":"…"}
```

模型載入並暖機完成之前，`/v1/systemone` 一律回 503。請等 health 回報 ready 再送流量（jevbench 連續三次失敗就會停止整個 run）。

### 送一個 `choice` Question

```sh
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state": "Where is my package? I ordered it last week and it still hasn'"'"'t arrived.",
       "model": "d1-2B",
       "questions": {"decision": {"type": "choice",
         "instructions": "Which intent does the user'"'"'s message express?",
         "criteria": {"track_order": "Wants to know where an order is or when it arrives",
                      "cancel_order": "Wants to cancel an order",
                      "billing_question": "Asks about a charge, invoice or payment"}}}}'
```

回應裡 `answers` 底下用同樣的名稱（`decision`）回答：`choice` 是最可能的 Label，`probabilities` 是這個 Question 的 Label set 上加總為 1 的 Distribution；`model` 原樣回傳請求裡的值。State 可以是純文字，也可以是 JSON 物件。

一個請求可以帶好幾個具名的 Question（都針對同一個 State），回應的 `answers` 底下用同樣的名稱各回一個答案。每個 Question 各自呼叫一次 Decider，所以答案不會因為同一個請求裡還有哪些其他 Question 而改變（d1 若把多個 question 塞進同一次呼叫，答案會偏移約 0.03）。

回應裡還有：

```json
{"usage":   {"input_tokens": 73, "output_tokens": 0},
 "timing":  {"total_ms": 98.512, "decider_ms": 97.904},
 "runtime": {"decider": "d1", "model": "LiquidAI/d1-2B",
             "revision": "652f920e544ea37f81e4b0ce89b81d4ed0da90b1", "device": "mps",
             "total_ms": 98.512, "decider_ms": 97.904}}
```

- `usage`：所有 Question 加總的 token 數。兩個 Decider 都只讀 prompt 不生成，所以 `output_tokens` 一律是 0。
- `timing`：server 自己量的時間（毫秒）。`total_ms` 從解析完請求到產生回應為止（包含等前一個請求的時間），`decider_ms` 是其中花在 Decider 上的部分。拿來跟 client 端的 wall-clock 對照，就能分出網路與 client 的額外開銷。
- `runtime`：同樣的時間，再加上 Decider 名稱、model id、revision 與 device（`cuda`、`mps` 或 `cpu`）。jevbench 會把回應頂層的 `runtime` 原樣抄進 `results.jsonl` 的每一筆紀錄，所以每筆結果都帶著它是在什麼硬體上跑的。

請求一次處理一個：同時送來的請求會排隊，不會交錯使用模型。

### 三種 Question type

| type | `criteria` | 答案 |
| --- | --- | --- |
| `choice` | `{label: 說明}`，1 到 26 個選項（兩個 Decider 都一樣） | `{"type": "choice", "choice": <最可能的 Label>, "probabilities": {label: …}}` |
| `noul` | 選用，`{"true": "yes 代表什麼", "false": "no 代表什麼"}` | `{"type": "noul", "noul": <P(yes)>}` |
| `score` | 2 到 10 個等級說明的 list，由低到高 | `{"type": "score", "score": <最可能的等級>, "probabilities": {"0": …, "1": …}}` |

`score` 的 `probabilities` 以等級 index 字串（`"0"` 到 `"N-1"`）為 key。d1 自己的 `score` 欄位是中位數等級，不一定是最可能的等級，所以不採用。

### 錯誤的請求回 422

不支援的 Question type（`text`、`number`、`date`、`box`、`boxes`、多選 `choice`，也就是 `"multiple": true`）、超過 26 個選項的 `choice`（`lfm3` 用 A–Z 標選項；為了兩個 Decider 行為一致，`d1` 也一樣拒絕）與格式錯誤的請求，一律回 HTTP 422，`detail` 是一句話，指出哪裡錯、該怎麼改，例如：

```json
{"detail": "questions.decision: a score's \"criteria\" lists 2 to 10 levels, lowest first, as [\"Can wait\", \"Today\", \"Now\"]; this one lists 11"}
```

整個請求只要有一個 Question 有問題就整個拒絕，不會先回答其他 Question。用 422 而不是 400，是因為 jevbench 把 422 算成答錯，但不計入「連續三次失敗就停止整個 run」的規則。

## 用 compose 在容器裡執行（CPU 與 GPU）

需求：Docker 與 Compose v2（Mac 上用 [OrbStack](https://orbstack.dev/)）。`compose.yaml` 每個 Decider 一個 service，各自放在同名的 profile 底下，所以只有指定的那個會啟動；沒給 `--profile` 就什麼都不啟動。

### 在 Mac 上（OrbStack，CPU）

```sh
docker compose --profile d1 up -d --wait   # 建 image、啟動 d1，等到 health 回報 ready
uv run local-jev bench --limit 10          # 從 Mac 對容器跑 smoke run（d1 在 port 8000）
docker compose --profile d1 down           # 停掉並移除容器；模型留在 volume 裡
```

- `--profile lfm3` 啟動 LFM3，聽 port 8001：`uv run local-jev bench --endpoint http://127.0.0.1:8001`。
- **一次只跑一個 profile。** 兩個 Decider 在 CPU 上用 fp32，加起來約 15 GB，剛好是 OrbStack 給容器的上限（15.7 GiB）。
- 容器裡沒有 MPS（macOS 上沒有任何 container runtime 能讓 Linux 容器用 Metal），只能用 CPU，每個 Decision 要一秒多。拿來開發與驗證可以，延遲數字請用原生的 `uv run`（MPS）或 GPU host 量。
- **Model 不在 image 裡。** 第一次啟動時 server 會把 model 下載到 named volume `hf-cache`（容器裡的 `HF_HOME=/data/hf`），固定在 Decider 的 pinned revision，之後的啟動直接讀它；兩個 service 共用這個 volume。整個 Hugging Face cache 一起掛載，因為 hub 把各 model 的檔案連到一個共用的 blob store，只掛一個 model 的目錄會留下斷掉的 symlink。
- 第一次下載 3–4 GB 需要一段時間，health check 給它 30 分鐘的 `start_period`；`docker compose logs -f d1` 可以看進度。`docker compose down -v` 會連 volume 一起刪掉，下次啟動就要重新下載。

用環境變數設定（可以寫在 repo 根目錄的 `.env`，已 git-ignored；compose 會自動讀）：

| 環境變數 | 預設 | 說明 |
| --- | --- | --- |
| `LOCAL_JEV_TOKEN` | 空 | 有設就要求 bearer token（見「用 bearer token 保護」）；`bench` 在同一個 shell 讀同一個變數 |
| `LOCAL_JEV_D1_MODEL`、`LOCAL_JEV_D1_REVISION` | 空：d1 自己的 model 與 pinned revision | 傳給 `d1` service 的 `LOCAL_JEV_MODEL`、`LOCAL_JEV_REVISION`；給了 model 就必須給 revision |
| `LOCAL_JEV_LFM3_MODEL`、`LOCAL_JEV_LFM3_REVISION`、`LOCAL_JEV_LFM3_CHAT_TEMPLATE_OPTIONS` | 空：LFM3 與它的 pinned revision | 傳給 `lfm3` service 的 `LOCAL_JEV_MODEL`、`LOCAL_JEV_REVISION`、`LOCAL_JEV_CHAT_TEMPLATE_OPTIONS` |
| `LOCAL_JEV_D1_PORT`、`LOCAL_JEV_LFM3_PORT` | `8000`、`8001` | 主機上的 port |
| `LOCAL_JEV_BIND` | `127.0.0.1` | 主機上 publish 的位址；要讓其他機器連線就設 `0.0.0.0`，並一定要設 `LOCAL_JEV_TOKEN` |

不用 compose 也可以直接跑 image（`--decider` 之後的參數都交給 `local-jev serve`）：

```sh
docker run --rm -p 127.0.0.1:8000:8000 -v local-jev_hf-cache:/data/hf local-jev:runtime --decider d1
```

### Image

`Dockerfile` 是 multi-stage，`compose.yaml` 用的是 `runtime` target：

- `ubuntu:24.04`，uv binary 從固定版本（含 digest）的 `ghcr.io/astral-sh/uv` image 複製，Python 3.12 由 uv 安裝；
- `uv sync --frozen --no-dev`：照 `uv.lock` 裝 runtime 相依套件，不含 dev 套件；
- 以 non-root user `app`（UID 10001）執行；
- 保留 gcc：LFM3 的 conv kernel 在 CUDA 上用 Triton，Triton 在 run time 編譯 kernel。

torch 依平台來自不同的 index（都鎖在 `uv.lock` 裡，版本都是 2.11.0）：linux/amd64 用 PyTorch 的 CUDA 12.8 index，linux/arm64 用 PyTorch 的 CPU index（PyPI 的 Linux wheel 在 arm64 上也會帶進 CUDA 13 的 library），Mac 用 PyPI（支援 MPS 的版本）。linux/arm64 的 runtime image 約 1.4 GB。

另一個 target `dev` 是 dev container 用的：多了 dev 相依套件與 git，source 在執行時掛載到 `/app`。

### 在 Linux NVIDIA GPU host 上（未驗證）

```sh
docker compose -f compose.yaml -f compose.gpu.yaml --profile d1 up -d --wait
```

- **`compose.gpu.yaml` 還沒在真的 GPU host 上跑過。** 它只替每個 service 加上 NVIDIA GPU 的 device reservation（`driver: nvidia`、`count: all`、`capabilities: [gpu]`），其他設定都跟 CPU 一樣；`--device auto` 會自動選 CUDA 與 bfloat16。
- Host 需要 NVIDIA driver（570.26 以上，CUDA 12.8 的要求）與 NVIDIA Container Toolkit。Image 不用 `nvidia/cuda` 當 base：torch 的 wheel 自帶 CUDA library。
- 在 GPU host 上 compose 會直接建 linux/amd64 的 image，裡面是 CUDA 12.8 版的 torch。
- GPU 記憶體夠的話，兩個 Decider 可以同時跑（`--profile d1 --profile lfm3`，port 各自是 8000 與 8001）。

### 用 bearer token 保護（選用）

```sh
LOCAL_JEV_TOKEN=換成你的-token uv run local-jev serve --decider d1 --host 0.0.0.0
```

- 有設 `LOCAL_JEV_TOKEN`（非空字串）時，`/v1/systemone` 必須帶 `Authorization: Bearer <token>`，沒帶或錯了都回 401。`/health` 不需要 token，方便 compose 做 health check。
- 沒設時，帶任何 token 或不帶都接受。jevbench 的 `typesafe` adapter 預設一定會送一個 token（環境變數空著它就不肯跑），所以本機跑 benchmark 時隨便設一個假值即可，例如 `TYPESAFE_API_KEY=local`。`local-jev bench` 會自動處理（見下節）。
- Server 有設 token 時，把同一個值放進 jevbench `--key-env` 指定的環境變數。用 `local-jev bench` 時，只要在同一個 shell 設好 `LOCAL_JEV_TOKEN` 即可。

## 跑 benchmark（jevbench）

`local-jev bench` 用 jevbench 自己的 `typesafe` adapter 評測一個**已經在跑**的 server，一個指令完成 run 與 summarize。請在 repo 根目錄執行（預設路徑都相對於目前目錄）。

```sh
uv run local-jev serve --decider d1      # 終端機 1：先啟動 server（任何 Decider 都可以）
uv run local-jev bench                   # 終端機 2：跑全部 231 個 public Tasks
uv run local-jev bench --task-set hard   # 只跑 hard
uv run local-jev bench --limit 10        # smoke run：10 個 Tasks，不到一分鐘
```

它會依序：

1. 把 jevbench 抓到固定的 commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`，放在 `.cache/jevbench/<commit>/`（git-ignored）。快取的 checkout 若已在這個 commit 就直接重用，不再連網；不對（或抓到一半中斷）就重新抓。要改從 mirror 抓，設 `LOCAL_JEV_JEVBENCH_URL=<git URL>`。
2. 等 server 的 `/health` 回報 ready（Decider 載入中最多等 10 分鐘）。連不上 server、或 Decider 載入失敗，就直接結束、不建立 run 目錄。jevbench 連續三次失敗就會停止整個 run，所以絕不能讓它先碰到還沒就緒的 server。
3. 在 `runs/` 底下開一個新的目錄，用 jevbench 跑選定的 Task set，再 summarize。

| 選項 | 預設 | 說明 |
| --- | --- | --- |
| `--endpoint` | `http://127.0.0.1:8000` | Server 的 base URL（不要加 `/v1`） |
| `--task-set` | `public` | `easy`（48）、`original`（72）、`hard`（111）、`public`（三者合計 231） |
| `--limit N` | 全部 | 只跑 N 個 Tasks：輪流從每個（Task set 的部分 × Question type）群組依檔案順序各取一個，所以 `--limit 10` 也會涵蓋 `choice`、`noul`、`score` 三種。 |
| `--runs-dir` | `runs` | Run 目錄放在這裡（git-ignored） |
| `--cache-dir` | `.cache` | jevbench 的快取放在這裡（git-ignored） |

Server 有設 `LOCAL_JEV_TOKEN` 時，在跑 `bench` 的 shell 設同一個值；沒設時 `bench` 會給 jevbench 一個假的 key。

跑完會印出 run 目錄和一行重點數字，例如：

```text
[local-jev] run: runs/20261105T090000Z-public
[local-jev] d1 LiquidAI/d1-2B@652f920 on mps, Task set public, 231 Tasks: Valid 100.0%, Accuracy …, Brier …, ECE …, client p50 … s, p95 … s
```

### Run 目錄裡有什麼

每次 run 都是一個新目錄 `runs/<UTC 時間>-<Task set>[-limit<N>]`（同一秒內再跑一次會加 `-2`），絕不覆寫之前的 run。

| 檔案 | 內容 |
| --- | --- |
| `tasks.jsonl` | 這次實際跑的 Tasks，逐行照抄 jevbench 的 dataset。`run` 和 `summarize` 都讀這個檔，所以 `--limit` 的 run 一樣是 `complete: true`。 |
| `results.jsonl` | jevbench 的逐 Task 結果（`probs`、`valid`、`correct`、`latency_s`，以及 server 回應頂層的 `runtime`） |
| `raw/` | jevbench 的 raw evidence：每個 Task 一個檔，存完整的 request 與 response |
| `ledger.jsonl`、`manifest.json` | jevbench 的成本 ledger 與 run 設定（含 `dataset_hash`） |
| `summary.json` | jevbench `summarize` 的輸出：`accuracy`、`brier_mean`、`ece`、`schema_validity`（Valid answers）、`latency.p50_s`/`p95_s`（client 端）等 |
| `decisions.jsonl` | 每個 Task 一行：server 給的 Distribution 和預期的 Label 放在一起，給 calibration 用（格式見下） |
| `run.json` | 這次 run 的紀錄：jevbench 的 URL 與 commit、endpoint、Task set、`limit`、Task 數、起訖時間，以及 `server`：server 自己回報的 Decider、model id、revision 與硬體（`cuda`、`mps` 或 `cpu`） |

`run.json` 只在整個 run 成功跑完才會寫入。jevbench 中途停止（遇到 401、403、429，或連續三次失敗）時，`local-jev bench` 以非零 exit code 結束並說明跑到第幾個 Task；目錄裡留著 jevbench 已寫的結果與 raw evidence 供除錯，但沒有 `summary.json`、`decisions.jsonl`、`run.json`。

`decisions.jsonl` 每行的欄位（jevbench 的結果裡沒有預期的 Label，也沒有 Question type，所以這裡用 `task_id` 跟 Tasks 對起來）：

| 欄位 | 內容 |
| --- | --- |
| `task_id` | jevbench 的 Task id |
| `task_set` | 這個 Task 來自哪個部分：`easy`、`original` 或 `hard` |
| `question_type` | `choice`、`noul` 或 `score` |
| `labels` | 這個 Task 的 Label set（jevbench 作者的順序；`noul` 是 `["no", "yes"]`，`score` 是 `["0", …]`） |
| `expected` | 預期的 Label，一律是字串（`score` 也是，例如 `"2"`） |
| `distribution` | Server 給的 Distribution，以 Label 為 key（jevbench 的 `probs_as_returned`；`noul` 是 `{"yes": p, "no": 1 - p}`）。沒有可用的答案時是 `null` |
| `valid` | jevbench 是否接受這個 Distribution（無效的答案算答錯，calibration 應略過） |
| `predicted`、`correct` | jevbench 的 argmax Label，以及它是否等於 `expected` |
| `latency_s` | jevbench 量的 client 端 wall-clock（秒） |
| `runtime` | Server 這次回應的 `runtime`：Decider、model、revision、device，以及 server 端的 `total_ms`、`decider_ms` |

### jevbench 的眉角（`bench` 已處理）

- `typesafe` adapter 的 key 環境變數空著就一個請求都不送：沒設 token 時給它一個假的 `TYPESAFE_API_KEY`。
- 未知 endpoint 沒有價格時，jevbench 每個 Task 預留 $0.02、累計到 $15 上限就停，而且 cost 記成 null：一律傳 `--price-in-per-m 0 --price-out-per-m 0 --reserve-usd 0`。
- jevbench 的結果與 raw evidence 只能寫到新檔，而且必須在它的 checkout 之外：每次 run 都用新目錄、絕對路徑。
- jevbench 是用目前的 Python（`python -m jevbench.cli`）在它的 checkout 裡執行，不是 `uv run`（checkout 裡有它自己的 `pyproject.toml`）。它只用標準函式庫，不是 local-jev 的相依套件。
- jevbench 不會送 `explain`，所以 benchmark 一律不產生 Reason。

## 測試

```sh
uv run pytest                 # 快速測試：in-process 打 HTTP contract，用假的 d1 model 與假的 LM backend，不載入權重
uv run pytest -m real_model   # 真模型測試：載入真正的 LFM3-1.7B 與 d1-2B（第一次會下載），按需執行
uv run pytest -m real_model tests/test_real_lfm3.py   # 只跑其中一個 model
uv run ruff check
```

快速測試裡的 `tests/test_bench.py` 會跑真正的 jevbench：第一次執行時把它抓進 `.cache/`（需要網路），之後離線也能跑。`tests/test_lockfile.py` 確認 `uv.lock` 在每個平台鎖的 torch 來自正確的 index；`tests/test_compose.py` 用 `docker compose config` 確認 profile、共用的 volume、環境變數與 GPU override，沒有 docker CLI 時自動略過。容器實際啟動（health 與 10 個 Tasks 的 smoke run）要照上面的 compose 步驟手動驗證。`real_model` 裡的 `tests/test_real_bench.py` 用 jevbench 的 `typesafe` adapter 對真正的 d1 server 跑 10 個 Tasks，10 個答案都必須有效。
