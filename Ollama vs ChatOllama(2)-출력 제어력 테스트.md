# 실험 목표

1. Ollama 기본 프롬프팅: 프롬프트만으로 JSON을 요구했을 때의 한계 확인
2. JSON Mode (ChatOllama): JSON 형태 자체를 강제했을 때의 안정성 비교
3. Structured Output (Pydantic + LangChain): [최신 실무 표준] 형태뿐만 아니라 데이터 스키마(key-value 구조)까지 완벽하게 통제하는 방법 테스트

### 실험 환경

- Model: gemma3:4b
- Python: 3.10+
- Libraries: requests, langchain-ollama, pydantic

# 2-1, 순수 프롬프트 의존(Ollama 기본 API) - 실패 케이스

- LLM에게 말로만 “JSON으로 줘”라고 부탁하는 가장 순진한 접근 방식

```python
import requests
import json

URL = "http://localhost:11434/api/generate"

def ollama_request(prompt):
    response = requests.post(URL, json={
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

prompt = """
다음 형식으로만 출력해 (설명 절대 금지, 마크다운 금지)
{"name": "", "age": 0}

입력: 홍길동 20살
"""

res = ollama_request(prompt)

print("=== Raw Output ===")
print(res)

print("\n=== JSON Parsing 시도 ===")
try:
    parsed = json.loads(res)
    print("성공:", parsed)
except Exception as e:
    print("실패:", e)
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/486ffc81-3da8-4838-8ac7-7b4c9faeb77a/image.png)


### 핵심 관찰

- LLM 특유의 친절함(오지랖) 때문에 JSON 앞뒤로 설명이나 마크다운(` ```json `)이 붙는다.
- 문자열 조작(’replace’, ‘regex’)으로 파싱을 시도하다가 결국 예외 처리 지옥에 빠지게 됨

# 2-2. JSON Mode 강제 (ChatOllama)

- LangChain의 ‘ChatOllama’를 사용하여 응답 포맷을 JSON으로 강제 함.

```python
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(
    model="gemma3:4b",
    format="json",   # 🔥 핵심: JSON 형식 강제
    temperature=0
)

res = llm.invoke("홍길동 20살을 JSON으로 만들어줘. 키 값은 name과 age로 해.")

print("=== Raw Output ===")
print(res.content)

print("\n=== JSON Parsing ===")
try:
    parsed = json.loads(res.content)
    print("성공:", parsed)
except Exception as e:
    print("실패:", e)
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/a83bbb49-0458-4e3e-a382-13434a326766/image.png)


### 관찰 및 한계

- 군더더기 텍스트가 사라지고 깔끔하게 파싱 됨.
- 하지만 치명적인 단점이 존재
- 포맷이 JSON일 뿐, 내부의 ‘Key’값을 내 마음대로 100% 통제할 수 없음.
- 프롬프트가 조금만 꼬여도 {”이름”: “홍길동”, “나이”: 20} 처럼 스키마가 마음대로 변할 위험이 있어 실무 도입 시 불안 요소가 남음

# 2-3. 최신 실무 표준: Structured Output (Pydantic)

- 현재 백엔드 및 AI 서비스 개발에서 가장 권장되는 패턴
- Pydantic을 이용해 “무엇을(Schema)” “어떻게(JSON)” 반환할지 완벽하게 구격화함.

### 코드

```python
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# 1. 원하는 데이터 구조를 Pydantic으로 정의
class PersonInfo(BaseModel):
    name: str = Field(description="사람의 이름")
    age: int = Field(description="사람의 나이, 숫자만")

llm = ChatOllama(model="gemma3:4b", temperature=0)

# 2. LLM에 구조 강제 (with_structured_output)
structured_llm = llm.with_structured_output(PersonInfo)

# 3. 실행 (JSON 파싱이 필요 없음!)
res = structured_llm.invoke("안녕? 나는 홍길동이고, 올해 스무 살이야.")

print("=== Pydantic Object Output ===")
print(type(res))
print(res)
print(f"이름: {res.name}, 나이: {res.age}")
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/d55af6f6-7d69-42ef-b47d-7eed17eaf69f/image.png)


### 핵심 차이

- json.loads() 조차 필요 없음 → 결과물이 완벽한 Python 객체(Pydantic Model)로 반환됨.
- 키 값이 틀리거나 데이터 타입이 맞지 않는 문제(예: 나이에 “스무살”이라는 문자열이 들어가는 문제)를 원천 차단.

# 2-4. 극단 테스트

- 이 구조화된 출력이 실무에서 얼마나 강력한지 보여주는 테스트
- 여려 명의 데이터를 추출해 배열 형태로 받아옴.

### 코드

```python
from typing import List
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="사람의 이름")
    age: int = Field(description="사람의 나이, 숫자만")

llm = ChatOllama(model="gemma3:4b", temperature=0)

class TeamData(BaseModel):
    team_name: str = Field(description="추론된 팀 이름이나 주제")
    members: List[PersonInfo] = Field(description="팀원들의 정보 목록")

structured_team_llm = llm.with_structured_output(TeamData)

text = """
우리 백엔드 팀을 소개할게.
김철수는 1999년생 개발자야. 그리고 새로 들어온 박영희는 25살이고 데이터베이스를 담당하지.
"""

res = structured_team_llm.invoke(text)
print(res.model_dump_json(indent=2))
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/17e9c41f-ebd3-49bb-af8a-352ff20876a3/image.png)


- 참고: 철수의 나이는 1999년생을 기준으로 LLM이 자체 계산하여 25||26 등으로 출력

# 결론

- Ollama 그대로 쓰면 (Prompt Engineering)
    - JSON 파싱 에러 발생 → 서버 터짐
    - 어떻게든 고쳐보려고 replace, Regex 등 예외 처리 코드가 기형적으로 늘어남
- Structured Output (Pydantic + ChatOllama) 쓰면
    - 응답이 보장된 객체(Object)로 즉시 튀어나옴.
    - 타입 검증(Validation)이 자동으로 처리
    - API 서버 (특히 FastAPI 등)와 결합할 때 개발 속도가 압도적으로 빨라짐
