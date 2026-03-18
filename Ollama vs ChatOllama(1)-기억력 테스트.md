# 실험 목적

- Ollama를 쓰다 보면 한 번쯤 드는 생각
- 그냥 API로 쓰면 되는데 왜 굳이 ChatOllama 써야 하지?

# 실험 환경

```bash
OS: macOS
Ollama: latest
Model: gemma3:4b
LangChain: 최신
```

# 실험 1 - “기억력 테스트”

### 가설

- Ollama 기본: 기억 못할 것
- ChatOllama: 기억할 것

### 실험 목표

1. Ollama는 왜 기억 못하는지 구조적으로 확인
2. ChatOllama도 사실 “자동 기억 아님” 증명
3. 진짜 메모리 붙이면 어떻게 달라지는지 확인

### 실험 환경

```python
model: gemma3:4b
OS: mac
```

# 1-1. Ollama 기본 통신 (완전 Stateless 확인)

```python
import requests

URL = "http://localhost:11434/api/generate"

def ask(prompt):
    res = requests.post(URL, json={
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"]

print("=== 1차 질문 ===")
print(ask("내 이름은 승희야"))

print("\n=== 2차 질문 ===")
print(ask("내 이름 기억해? 기억한다면 내 이름이 뭐야?"))
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/5218ee44-cfab-4775-b565-0dd837ec4d60/image.png)


### 관찰

- 요청마다 완전히 독립적
- 컨텍스트 유지 안됌
- 그냥 stateless API

<aside>
🔍

[요청1] → 끝
[요청2] → 완전 새 시작

</aside>

# 1-2. ChatOllama (잘못된 사용법 실험)

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma3:4b")

messages1 = [
    HumanMessage(content="내 이름은 승희야"),
]

messages2 = [
    HumanMessage(content="내 이름 기억해? 기억한다면 내 이름은 뭐야?")
]

print("=== 1차 ===")
print(llm.invoke(messages1).content)

print("\n=== 2차 ===")
print(llm.invoke(messages2).content)
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/4a5c121d-a399-411f-852e-63599513a806/image.png)


### 관찰

- ChatOllama도 결국

<aside>
🔍

messages = context

</aside>

- 즉, messages 안 넘기면 = Ollama랑 똑같음

# 1-3. ChatOllama (정상 사용)

### 코드

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma3:4b")

messages = [
    HumanMessage(content="내 이름은 승희야"),
    HumanMessage(content="내 이름 기억해? 기억한다면 내 이름은 뭐야?")
]

res = llm.invoke(messages)
print(res.content)
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/7de2dcb6-246a-4d22-81b6-34346f35bdbc/image.png)


# 1-4. “대화형” 시물레이션 (진짜 챗처럼)

### 코드

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma3:4b")

messages = []

# 1차
messages.append(HumanMessage(content="내 이름은 승희야"))
res = llm.invoke(messages)
print("AI:", res.content)

# 2차
messages.append(HumanMessage(content="내 이름 기억해? 기억한다면 내 이름은 뭐야?"))
res = llm.invoke(messages)
print("AI:", res.content)
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/3aa16111-999c-42d6-92ef-f1fbde40b537/image.png)


# 1-5. 한계 테스트 (토큰 길이)

### 코드

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma3:4b")

messages = []

messages.append(HumanMessage(content="내 이름은 승희야"))

# 의미 없는 대화 20번 반복
for i in range(20):
    messages.append(HumanMessage(content=f"쓸데없는 대화 {i}"))

messages.append(HumanMessage(content="내 이름 기억해? 기억한다면 내 이름은 뭐야?"))

res = llm.invoke(messages)
print(res.content)
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/838a1ef7-57c7-4717-b2a6-deb9bd4d2ca0/image.png)


# 1-6. 진짜 Memory 적용 (자동 기억)

### 핵심 변화

- 기존
    - ConversationBufferMemory - 레거시
- 현재
    - RunnableWithMessageHistory - 공식 권장

### 코드

```python
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 모델
llm = ChatOllama(model="gemma3:4b")

# 세션별 메모리 저장소
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable 생성
with_history = RunnableWithMessageHistory(
    llm,
    get_session_history
)

# 실행
def chat(session_id, message):
    return with_history.invoke(
        message,
        config={"configurable": {"session_id": session_id}}
    ).content

# 🧪 테스트
print(chat("user1", "내 이름은 승희야"))
print(chat("user1", "내 이름 기억해? 기억한다면 내 이름은 뭐야?"))

print(chat("user2", "내 이름은 철수야"))
print(chat("user2", "내 이름 기억해? 기억한다면 내 이름은 뭐야?"))
```

### 결과
![](https://velog.velcdn.com/images/fpalzntm/post/8069c6d5-7630-4943-aa5e-2a4007ecd35b/image.png)


### 관찰

1. 자동 기억 동작
    - 별도로 messages를 관리하지 않아도
    - 내부에서 자동으로 history가 붙는다.
2. 세션 기반 기억 분리
    
    ```python
    chat("user1", ...)
    chat("user2", ...)
    ```
    
3. 사용자별 상태 관리 기능, 확장 가능 (DB, Redis 등)

# 실험 중 발견한 중요한 사실

- LLM은 “기억하는 존재”가 아니다.
- 이전 대화(history) + 현재 입력 → 다시 넣어서 생성

### 그래서 생기는 문제..

- 대화 길어지면 성능 저하
- 토큰 비용 증가
- 오래된 정보는 잘림

# 기존 방식 vs 최신 방식

| 방식 | 특징 | 상태 |
| --- | --- | --- |
| messages 직접 관리 | 수동 | ❌ 비효율 |
| ConversationBufferMemory | 간편 | ⚠️ 레거시 |
| RunnableWithMessageHistory | 자동 + 세션 기반 | ✅ 현재 표준 |

# 핵심 구조 이해

<aside>
🔍

invoke()
↓
session_id 확인
↓
history 가져옴
↓
history + input → LLM
↓
결과 저장

</aside>

- 이 과정이 자동으로 돌아감
