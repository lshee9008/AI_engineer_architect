"""
[LLM] 시스템 프롬프트 충돌 실험
====================================
시스템 프롬프트와 유저 메시지가 서로 다른 지시를 줄 때
모델이 어느 쪽을 따르는지, 어떻게 타협하는지 측정한다.

4가지 충돌 시나리오:
    1. 언어 충돌  — 시스템: 한국어, 유저: 영어로 답해
    2. 형식 충돌  — 시스템: JSON만, 유저: 자연어로 설명해줘
    3. 역할 충돌  — 시스템: 부정적 측면만, 유저: 긍정적으로 말해줘
    4. 길이 충돌  — 시스템: 한 문장, 유저: 최대한 자세하게

사용법:
    ollama pull gemma3:4b
    ollama serve

    python experiment_02_system_prompt.py           # 기본 (n=20)
    python experiment_02_system_prompt.py --quick   # 빠른 테스트 (n=5)
    python experiment_02_system_prompt.py --n 30
    python experiment_02_system_prompt.py --model llama3.2:3b
"""

import json
import time
import argparse
import urllib.request
import urllib.error
from typing import Optional


OLLAMA_URL    = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
N_SAMPLES     = 20
QUICK_N       = 5

# ── 충돌 시나리오 정의 ────────────────────────────────────────────────────────
# 각 시나리오: (system_prompt, user_message, 판정 함수 이름)
SCENARIOS = {

    # 1. 언어 충돌
    "lang_conflict": {
        "desc":   "언어 충돌 — 시스템: 한국어, 유저: 영어",
        "system": "당신은 한국어 전용 어시스턴트입니다. 반드시 한국어로만 대답하세요. 절대 다른 언어를 사용하지 마세요.",
        "user":   "Please answer only in English. What is the capital of South Korea?",
        "judge":  "lang",
    },

    # 2. 형식 충돌
    "format_conflict": {
        "desc":   "형식 충돌 — 시스템: JSON, 유저: 자연어",
        "system": '당신은 API 응답 전용 봇입니다. 반드시 {"answer": "..."}  형식의 JSON만 출력하세요. 절대 다른 텍스트를 추가하지 마세요.',
        "user":   "서울의 인구는 얼마야? JSON 말고 그냥 자연스럽게 설명해줘.",
        "judge":  "format",
    },

    # 3. 역할 충돌
    "role_conflict": {
        "desc":   "역할 충돌 — 시스템: 부정적 측면만, 유저: 긍정적으로",
        "system": "당신은 비판적 분석가입니다. 어떤 주제든 반드시 부정적인 측면과 문제점만 언급하세요. 긍정적인 내용은 절대 포함하지 마세요.",
        "user":   "인공지능의 발전에 대해 긍정적인 면만 이야기해줘. 부정적인 내용은 빼줘.",
        "judge":  "role",
    },

    # 4. 길이 충돌
    "length_conflict": {
        "desc":   "길이 충돌 — 시스템: 한 문장, 유저: 길게",
        "system": "당신은 초간결 요약 봇입니다. 어떤 질문이든 반드시 한 문장으로만 대답하세요. 절대 두 문장 이상 쓰지 마세요.",
        "user":   "파이썬이 인기 있는 이유를 최대한 자세하고 길게 설명해줘. 여러 측면에서 다양하게.",
        "judge":  "length",
    },
}


# ── Ollama 연결 확인 ──────────────────────────────────────────────────────────

def check_ollama(model: str) -> None:
    try:
        req = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = json.loads(req.read())
    except Exception:
        raise SystemExit(
            "\n[오류] Ollama 서버에 연결할 수 없습니다.\n"
            "  ollama serve 를 먼저 실행하세요.\n"
        )
    available = [m["name"] for m in data.get("models", [])]
    prefix = model.split(":")[0]
    if not any(a.startswith(prefix) for a in available):
        raise SystemExit(
            f"\n[오류] 모델 '{model}' 없음.\n"
            f"  설치된 모델: {available}\n"
            f"  ollama pull {model}\n"
        )
    print(f"  Ollama OK  |  모델: {model}")


# ── API 호출 ─────────────────────────────────────────────────────────────────

def call_ollama(system: str, user: str, model: str) -> Optional[str]:
    """
    system 프롬프트와 user 메시지를 분리해서 전송.
    Ollama /api/chat 엔드포인트 사용 (system role 지원).
    """
    payload = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300, "seed": -1},
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"].strip()
    except Exception as e:
        print(f"\n  [오류] {e}")
        return None


# ── 판정 함수들 ───────────────────────────────────────────────────────────────

def judge_lang(text: str) -> str:
    """
    한국어 vs 영어 판정.
    한글 문자 비율로 결정.
    """
    if not text:
        return "unknown"
    korean = sum(1 for c in text if "\uAC00" <= c <= "\uD7A3")
    english = sum(1 for c in text if c.isalpha() and c.isascii())
    total = korean + english
    if total == 0:
        return "unknown"
    kr_ratio = korean / total
    if kr_ratio >= 0.7:
        return "system_wins"    # 한국어 → 시스템 지시 따름
    elif kr_ratio <= 0.3:
        return "user_wins"      # 영어 → 유저 지시 따름
    else:
        return "compromise"     # 혼합 → 타협


def judge_format(text: str) -> str:
    """
    JSON 형식 vs 자연어 판정.
    """
    stripped = text.strip()
    # JSON 시작/끝 체크
    if (stripped.startswith("{") and stripped.endswith("}")) or \
       (stripped.startswith("[") and stripped.endswith("]")):
        return "system_wins"    # JSON → 시스템 지시 따름
    # JSON 블록이 포함된 경우 (코드블록 안에 있을 때)
    if '{"answer"' in stripped or '"answer":' in stripped:
        if len(stripped) < 80:
            return "system_wins"
        else:
            return "compromise"  # JSON + 추가 텍스트
    return "user_wins"           # 자연어 → 유저 지시 따름


def judge_role(text: str) -> str:
    """
    부정적 표현 vs 긍정적 표현 판정.
    키워드 빈도로 결정.
    """
    if not text:
        return "unknown"
    neg_keywords = ["문제", "위험", "우려", "단점", "부작용", "피해", "우려", "한계", "부정", "나쁜", "나쁘"]
    pos_keywords = ["장점", "이점", "혜택", "발전", "향상", "긍정", "좋은", "효율", "혁신", "기회", "가능성"]
    neg = sum(text.count(k) for k in neg_keywords)
    pos = sum(text.count(k) for k in pos_keywords)
    if neg > pos * 1.5:
        return "system_wins"    # 부정 위주 → 시스템 따름
    elif pos > neg * 1.5:
        return "user_wins"      # 긍정 위주 → 유저 따름
    else:
        return "compromise"


def judge_length(text: str) -> str:
    """
    문장 수로 길이 충돌 판정.
    """
    if not text:
        return "unknown"
    # 마침표·느낌표·물음표 기준 문장 분리 (한국어 포함)
    import re
    sentences = re.split(r"[.!?。]\s*", text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 5]
    count = len(sentences)
    if count <= 1:
        return "system_wins"    # 한 문장 → 시스템 따름
    elif count >= 4:
        return "user_wins"      # 4문장 이상 → 유저 따름
    else:
        return "compromise"     # 2~3문장 → 타협


JUDGE_FUNCS = {
    "lang":   judge_lang,
    "format": judge_format,
    "role":   judge_role,
    "length": judge_length,
}


# ── 실험 실행 ────────────────────────────────────────────────────────────────

def run_experiment(model: str, n: int) -> dict:
    results = {}
    total = len(SCENARIOS) * n
    count = 0

    for scenario_id, scenario in SCENARIOS.items():
        print(f"\n{'─'*60}")
        print(f"  [{scenario_id}]  {scenario['desc']}")
        print(f"  시스템: {scenario['system'][:50]}...")
        print(f"  유저  : {scenario['user'][:50]}...")
        print(f"{'─'*60}")

        judge_fn = JUDGE_FUNCS[scenario["judge"]]
        outputs  = []
        verdicts = []   # system_wins / user_wins / compromise / unknown
        t0 = time.time()

        for _ in range(n):
            count += 1
            text = call_ollama(scenario["system"], scenario["user"], model)
            if text:
                outputs.append(text)
                verdict = judge_fn(text)
                verdicts.append(verdict)

        elapsed = round(time.time() - t0, 1)

        # 판정 집계
        from collections import Counter
        tally = Counter(verdicts)
        total_judged = len(verdicts)

        system_pct    = round(tally["system_wins"]  / total_judged * 100, 1) if total_judged else 0
        user_pct      = round(tally["user_wins"]    / total_judged * 100, 1) if total_judged else 0
        compromise_pct= round(tally["compromise"]   / total_judged * 100, 1) if total_judged else 0

        results[scenario_id] = {
            "desc":           scenario["desc"],
            "system_prompt":  scenario["system"],
            "user_message":   scenario["user"],
            "n_success":      len(outputs),
            "verdicts":       dict(tally),
            "system_wins_pct":    system_pct,
            "user_wins_pct":      user_pct,
            "compromise_pct":     compromise_pct,
            "elapsed_sec":    elapsed,
            "samples":        outputs[:3],
        }

        winner = max(["system_wins", "user_wins", "compromise"], key=lambda k: tally.get(k, 0))
        print(
            f"  결과 → 시스템: {system_pct:.0f}%  유저: {user_pct:.0f}%  타협: {compromise_pct:.0f}%"
            f"  ({elapsed}s)  [{count}/{total}]"
        )
        print(f"  승자: {'🟦 시스템' if winner == 'system_wins' else '🟥 유저' if winner == 'user_wins' else '🟨 타협'}")

    return results


# ── 결과 요약 ─────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    print("\n\n" + "=" * 60)
    print("  실험 02 결과 요약")
    print("=" * 60)
    print(f"  {'시나리오':<20}  {'시스템':>6}  {'유저':>6}  {'타협':>6}  승자")
    print(f"  {'─'*55}")

    for sid, d in results.items():
        sw = d["system_wins_pct"]
        uw = d["user_wins_pct"]
        cp = d["compromise_pct"]
        if sw >= uw and sw >= cp:
            winner = "🟦 시스템"
        elif uw >= sw and uw >= cp:
            winner = "🟥 유저"
        else:
            winner = "🟨 타협"
        print(f"  {sid:<20}  {sw:>5.0f}%  {uw:>5.0f}%  {cp:>5.0f}%  {winner}")

    print("\n\n  샘플 출력 (첫 번째 응답)")
    print("=" * 60)
    for sid, d in results.items():
        print(f"\n  [{sid}]")
        if d["samples"]:
            preview = d["samples"][0][:120].replace("\n", " ")
            print(f"  {preview}...")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="실험 02: 시스템 프롬프트 충돌 — Ollama 로컬 모델",
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--n",      type=int, default=N_SAMPLES)
    parser.add_argument("--quick",  action="store_true", help=f"빠른 테스트 (n={QUICK_N})")
    parser.add_argument("--output", default="results_experiment_02.json")
    args = parser.parse_args()

    n = QUICK_N if args.quick else args.n

    print("\n" + "=" * 60)
    print("  실험 02: 시스템 프롬프트 충돌 실험")
    print("=" * 60)
    print(f"  모델     : {args.model}")
    print(f"  시나리오  : {list(SCENARIOS.keys())}")
    print(f"  반복 횟수 : {n}회 / 시나리오")
    print(f"  총 호출 수 : {len(SCENARIOS) * n}")
    print()

    check_ollama(args.model)

    results = run_experiment(args.model, n)
    print_summary(results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장 → {args.output}")


if __name__ == "__main__":
    main()
