# 프롬프트 온도(temperature) 조절 실험

> 
> 
> 
> **모델:** Ollama + gemma3:4b (로컬, 무료)
> 
> **핵심 질문:** temperature=0.0 이면 정말 항상 같은 결과가 나올까?
> 

---

## 1. 왜 이 실험을 했는가

LLM API를 처음 쓸 때 가장 많이 듣는 파라미터가 `temperature`다.

문서에는 **"높을수록 창의적, 낮을수록 보수적"** 이라고 쓰여 있다.

근데 실제로 얼마나 달라지는지 수치로 본 사람이 의외로 적다.

특히 두 가지 의문이 출발점이었다.

- `temperature=0.0` 이면 정말 항상 같은 결과가 나오는가?
- 코드 생성과 창의 글쓰기에서 최적 온도가 다른가?

---

## 2. 실험 설계

### 태스크 유형

출력 성격이 다른 세 가지 프롬프트를 골랐다.

| 태스크 | 유형 | 프롬프트 |
| --- | --- | --- |
| `creative` | 창의 글쓰기 | 한 문장으로 파란 하늘을 시적으로 묘사해줘. |
| `code` | 코드 생성 | 파이썬에서 리스트 중복 제거하는 가장 간단한 방법 한 줄로만 알려줘. |
| `factual` | 사실 질문 | 대한민국의 수도는 어디야? 한 단어로만 답해. |

### 측정 지표

- **엔트로피 (Entropy):** 출력에 사용된 단어 분포의 Shannon entropy. 높을수록 다양한 단어 사용
    - Shannon entropy 참고 : https://velog.io/@zlddp723/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B8%B0%EC%B4%88%EC%88%98%ED%95%99-%EC%A0%95%EB%B3%B4%EC%99%80-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC-Shannon-%ED%81%AC%EB%A1%9C%EC%8A%A4-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC-Cross
- **Unique Ratio:** n회 중 완전히 다른 응답의 비율 — `1.0` = 전부 다름, `0.0` = 전부 동일
- **평균 길이:** 응답 글자 수 평균. 온도가 길이에도 영향을 주는지 확인

### 실험 조건

- 온도 범위: `0.0 ~ 1.5`, 0.1 단위 (총 16단계)
- 온도별 반복: 20회
- 모델: `gemma3:4b` (Ollama 로컬)
- `seed=-1` 고정 — 매번 다른 시드로 temperature 효과만 순수하게 관찰

---

## 3. 핵심 코드

외부 라이브러리 없이 표준 라이브러리(`urllib`, `json`, `math`)만 사용한다.

### Ollama 호출

```python
import json
import urllib.request

def call_ollama(prompt: str, temperature: float, model: str) -> str | None:
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 120,
            "seed": -1,   # 매번 다른 시드 → temperature 효과만 격리
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
        return data.get("response", "").strip()
```

### 다양성 측정 — Shannon Entropy

```python
from collections import Counter
import math

def token_entropy(texts: list[str]) -> float:
    """
    출력 다양성을 Shannon entropy 로 측정.
    entropy ~= 0  : 모든 출력이 거의 동일
    entropy 1~2   : 약간 다양
    entropy 3+    : 매우 다양
    """
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())

    counts = Counter(all_words)
    total  = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def unique_ratio(texts: list[str]) -> float:
    """완전히 동일한 응답 제외한 고유 응답 비율."""
    return len(set(texts)) / len(texts)
```

### 실험 루프

```python
TEMPERATURES = [round(i * 0.1, 1) for i in range(0, 16)]  # 0.0 ~ 1.5

for temp in TEMPERATURES:
    outputs = [call_ollama(prompt, temp, model) for _ in range(N)]

    print(
        f"temp={temp:.1f}  "
        f"entropy={token_entropy(outputs):.3f}  "
        f"unique={unique_ratio(outputs):.2f}  "
        f"avg_len={sum(len(t) for t in outputs) / len(outputs):.0f}"
    )
```

---

## 4. 테스트 기록

### #1 — `temperature=0.0` 은 결정론적이다 — 근데 그게 문제였다

처음엔 `temperature=0.0` 이면 다양성이 떨어지더라도 어느 정도 변화는 있을 거라 생각했다.

결과: `creative` 태스크에서 `entropy=3.17`, `unique_ratio=0.05` — **20회 중 19회가 완전히 동일한 문장이었다.**

```
"파란 하늘은 마치 거대한 캔버스 위에 펼쳐진 꿈결 같았다."
"파란 하늘은 마치 거대한 캔버스 위에 펼쳐진 꿈결 같았다."
"파란 하늘은 마치 거대한 캔버스 위에 펼쳐진 꿈결 같았다."
... (19회 반복)
```

`code` 태스크는 더 심각했다. `temperature=1.5`까지 올려도 출력 코드는 단 한 번도 바뀌지 않았다.

```python
# 0.0에서도, 1.5에서도 동일한 코드
my_list = list(dict.fromkeys(my_list))
```

**원인:** `temperature=0.0`은 매 스텝에서 확률이 가장 높은 토큰 하나만 선택(greedy decoding)한다.
모델이 특정 패턴에 강하게 수렴해 있으면 온도를 올려도 그 패턴이 쉽게 깨지지 않는다.

> `temperature=0.0` 은 "고착 구간"이다.
> 
> 
> 창의 태스크에 `0.0~0.1`을 쓰면 같은 문장만 반복해서 나온다.
> 
> 다양성이 필요하다면 **최소 `0.2` 이상**을 써야 분기가 시작된다.
> 

```python
# 고착 구간 — 창의 태스크에 쓰면 안 됨
"options": {"temperature": 0.0}  # unique_ratio=0.05

# 다양성 분기점 — 여기서부터 출력이 달라지기 시작
"options": {"temperature": 0.2}  # unique_ratio=0.45

# 창의 태스크 스위트 스팟
"options": {"temperature": 0.8}  # unique_ratio=1.0, 문법 유지
```

---

### #2 — `code` 태스크에서 `0.9~1.1` 구간 응답 시간이 폭발했다

`temperature=0.9`에서 elapsed가 갑자기 **169초**, `1.0`에서는 **229초**까지 튀었다.
다른 구간은 평균 60~70초인데 이 구간만 3~4배 느렸다.

| temperature | elapsed (sec) |
| --- | --- |
| 0.8 | 61.6 |
| **0.9** | **169.2** |
| **1.0** | **229.6** |
| **1.1** | **181.3** |
| 1.2 | 64.0 |

처음엔 네트워크 문제인 줄 알았다. 재현해보니 동일한 패턴이 반복됐다.

**원인:** 고온 구간에서 모델이 설명을 길게 늘이려는 경향이 생기는데,
`num_predict` 한도에 도달하기 전에 스스로 멈추지 못하고 한도까지 생성하다 보니 지연이 발생했다.
`1.2` 이상에서는 오히려 짧게 끊기는 패턴으로 바뀌어서 속도가 회복됐다.

**fix:** `timeout=60`으로 상한을 걸었다.
프로덕션에서 `0.9~1.1` 구간을 쓴다면 반드시 타임아웃을 설정해야 한다.

```python
# timeout 없이 날리면 229초짜리 응답을 하염없이 기다리게 된다
with urllib.request.urlopen(req, timeout=60) as resp:  # 반드시 timeout 명시
    data = json.loads(resp.read())
    return data.get("response", "").strip()
```

---

### #3 — Ollama 서버 미실행 상태에서 960번 호출 날림

`ollama serve`를 안 띄운 상태로 실험을 돌려서
`ConnectionRefusedError`가 **3개 태스크 × 16온도 × 20회 = 960번** 찍혔다.
에러 로그가 960줄 터미널에 쏟아지고 결과 JSON은 텅 비어있었다.

이후 실행 전 서버 상태를 먼저 체크하는 로직을 추가했다.

```python
def check_ollama(model: str) -> None:
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(req.read())
    except Exception:
        raise SystemExit("[오류] ollama serve 를 먼저 실행하세요.")

    available = [m["name"] for m in data.get("models", [])]
    if not any(m.startswith(model.split(":")[0]) for m in available):
        raise SystemExit(f"[오류] 모델 없음. 'ollama pull {model}' 실행 필요.")
```

실험 루프 진입 전에 이걸 먼저 호출하면 서버·모델 문제를 초기에 잡아낼 수 있다.

---

## 5. 결과 분석

> **실험 조건:** `gemma3:4b` (Ollama 로컬), `n=20` / 온도별, 총 960회 호출
> 

---

### creative 태스크 (시적 묘사)

| temperature | entropy | unique ratio | avg length | 비고 |
| --- | --- | --- | --- | --- |
| 0.0 | 3.17 | 0.05 | 32 | **거의 동일한 문장** — "캔버스 위에 펼쳐진 꿈결" 고착 |
| 0.1 | 3.42 | 0.10 | 32.7 | 단어 하나씩 미세하게 바뀌기 시작 |
| 0.2 | 4.35 | 0.45 | 38.5 | 문장 길이가 갑자기 늘어남 — 분기점 |
| 0.3 | 4.41 | 0.55 | 39.5 | 은하수·캔버스·꿈결 메타포 혼합 |
| 0.5 | 5.01 | 0.80 | 42.6 | 다양성 급증 구간 |
| **0.6** | **5.11** | **0.85** | **46.2** | **다양성·문법 균형 최적** |
| **0.8** | **5.17** | **1.00** | **43.5** | **unique ratio 첫 1.0 달성** |
| 1.0 | 5.63 | 1.00 | 43.8 | 완전 고유 유지, 문학성 정점 |
| 1.3 | 5.98 | 1.00 | 44.4 | "투명한 자수 속에 은하수" — 약간 추상적 |
| 1.5 | 6.55 | 1.00 | 47.5 | 여전히 읽을 수 있는 수준 유지 |

**실제 샘플 비교**

- `temp=0.0` → `"파란 하늘은 마치 거대한 캔버스 위에 펼쳐진 꿈결 같았다."` (20회 중 19회 동일)
- `temp=0.8` → `"파란 하늘은 마치 거대한 캔버스 위에 흩뿌려진 잔잔한 푸른 꿈과 같다."` / `"푸른빛으로 펼쳐진 하늘은 별들의 꿈처럼 깊고 아름답다."` / `"무한한 평화와 자유를 꿈꾸는 듯 황홀하다."` (20회 전부 다름)
- `temp=1.3` → `"투명한 자수 속에 무한히 펼쳐진 파란색 꿈"` (독창적이지만 약간 어색)

> **💡 인사이트:** 예상보다 낮은 `0.2` 구간에서 **다양성 분기점**이 나타났다.
> 
> 
> `0.0~0.1` 은 사실상 같은 문장만 반복하는 "고착 구간"이었다.
> 
> 창의 태스크 스위트 스팟은 `0.6~1.0` — unique ratio 1.0을 유지하면서 문법도 멀쩡하다.
> 
> `1.3` 이상은 퀄리티 저하 없이 엔트로피만 계속 올라간다. gemma3:4b는 고온에서도 꽤 안정적이었다.
> 

---

### code 태스크 (코드 생성)

| temperature | entropy | unique ratio | avg length | 비고 |
| --- | --- | --- | --- | --- |
| **0.0** | **5.13** | **0.05** | **239** | **코드 자체는 100% 동일 — 설명문만 미세하게 다름** |
| 0.1 | 5.15 | 0.20 | 237.8 | 거의 차이 없음 |
| 0.2 | 5.37 | 0.55 | 231.4 | 설명 구조가 바뀌기 시작 |
| 0.4 | 5.73 | 0.75 | 216.8 | 응답 길이 감소 — 설명 간결화 |
| 0.6 | 6.11 | 0.95 | 215.4 | 거의 모든 응답이 고유 |
| **0.7** | **6.02** | **1.00** | **217.4** | **unique ratio 1.0 최초 달성** |
| 0.9 | 6.12 | 1.00 | 221.7 | elapsed 169초 — 응답 지연 첫 발생 |
| 1.0 | 6.28 | 1.00 | 205.8 | elapsed 229초 — 심각한 지연 |
| 1.3 | 6.49 | 1.00 | 229.6 | 지연 안정화 (74초), 코드 오류 없음 |
| 1.5 | 6.65 | 1.00 | 214.2 | 가장 높은 entropy, 코드 자체는 여전히 정상 |

**실제 샘플 비교**

- `temp=0.0` → 전 구간 `list(dict.fromkeys(my_list))` 로 **완전히 동일한 코드** 출력. 설명 문구만 조금씩 다름.
- `temp=0.7` → 번호 목록 방식 설명, 불릿 방식 설명, 한 줄 요약 방식 — **설명 스타일만 다양**, 코드 자체는 동일
- `temp=1.5` → `"then list()를 사용하여"` 처럼 영어가 섞이는 현상 간헐적 발생

> **💡 인사이트:** 코드 자체(`list(dict.fromkeys(...))`)는 **전 온도 구간에서 변하지 않았다.**
> 
> 
> 변한 것은 설명문의 구조와 길이뿐. 즉, **코드 생성 태스크에서 temperature는 로직이 아니라 "말투"를 바꾼다.**
> 
> `0.9~1.1` 구간에서 응답 시간이 최대 229초까지 튀었다 — 고온에서 긴 설명을 생성하려다 지연 발생.
> 
> 코드 생성 목적이면 `0.0~0.3`이 가장 효율적이다 (속도·일관성 모두).
> 

---

### factual 태스크 (사실 질문)

| temperature | entropy | unique ratio | avg length | 비고 |
| --- | --- | --- | --- | --- |
| 0.0 | 0.00 | 0.05 | 2 | "서울" 단독 출력, 20회 중 19회 완전 동일 |
| 0.3 | 0.00 | 0.05 | 2 | 변화 없음 |
| 0.7 | 0.00 | 0.05 | 2 | 변화 없음 |
| 1.0 | 0.00 | 0.05 | 2 | 변화 없음 |
| 1.3 | 0.00 | 0.05 | 2 | 변화 없음 |
| **1.5** | **0.00** | **0.05** | **2** | **여전히 "서울" — 전 구간 완전 고착** |

> **💡 인사이트 (그리고 가장 큰 반전):**
> 
> 
> `temperature=1.5`까지 올려도 단답형 사실 질문에서는 출력이 전혀 바뀌지 않았다.
> 
> entropy=0.00, unique_ratio=0.05(≈ 20회 중 1회만 다름)가 전 구간 유지됐다.
> 
> "온도가 높으면 사실 답변도 흔들린다"는 예측은 완전히 틀렸다.
> 
> **gemma3:4b는 단답형 프롬프트에서는 temperature 영향을 거의 받지 않는다.**
> 
> 이는 모델이 "가장 확률 높은 단어"가 압도적으로 높을 때 temperature가 사실상 무의미해지는 현상이다.
> 

---

### 세 태스크 비교 요약

| 태스크 | 온도 영향 | unique=1.0 달성 온도 | 특이점 |
| --- | --- | --- | --- |
| creative | **크다** | 0.8 | 0.2에서 다양성 분기점 발생 |
| code | **중간** | 0.7 | 코드 로직은 불변, 설명문만 변함 |
| factual | **거의 없음** | 달성 불가 | 1.5에서도 entropy=0.00 |

---

## 6. 결론 및 실전 가이드

| 태스크 유형 | 권장 온도 | 이유 |
| --- | --- | --- |
| 창의 글쓰기 / 마케팅 문구 | `0.7~0.9` | 다양성과 문법 유지의 균형점 |
| 코드 생성 (단일 정답) | `0.1~0.3` | 일관성 우선. 단, `0.0`보다 `0.2`가 나은 경우 있음 |
| RAG 답변 생성 | `0.0~0.2` | 사실 기반이므로 변동 최소화 |
| 브레인스토밍 / 아이디어 | `1.0~1.2` | 독창성이 목표, 어느 정도 일탈 허용 |
| 구조화 출력 (JSON 파싱) | `0.0` | 형식 오류 최소화 최우선 |

---

## 7. 직접 실행하기

### 사전 준비

```bash
# Ollama 설치: https://ollama.com
ollama pull gemma3:4b
ollama serve
```

### 실행

```bash
# 빠른 테스트 (n=3, 온도 5단계, 약 2분)
python experiment_01_temperature.py --quick

# 기본 실험 (n=20, 16단계, 약 15분)
python experiment_01_temperature.py

# 풀 실험 (n=50)
python experiment_01_temperature.py --n 50

# 다른 모델 비교
python experiment_01_temperature.py --model llama3.2:3b
```

### 결과 확인

```bash
# 콘솔 요약은 실험 종료 후 자동 출력
# JSON 전체 결과 확인
cat results_experiment_01.json | python -m json.tool | head -80
```
