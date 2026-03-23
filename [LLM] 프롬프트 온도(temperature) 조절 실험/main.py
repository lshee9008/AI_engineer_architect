"""
실험 01: 프롬프트 온도(temperature) 조절 실험
========================================
같은 질문을 temperature=0.0 ~ 1.5 구간에서 반복 실행하여
출력 다양성(entropy)과 품질 변화를 측정한다.

요구사항:
    Ollama 설치 후 gemma3:4b 모델 pull 필요
    https://ollama.com 에서 설치

    ollama pull gemma3:4b
    ollama serve          # 이미 실행 중이면 생략

사용법:
    # 기본 실행 (n=20, 약 10~20분)
    python experiment_01_temperature.py

    # 빠른 테스트 (n=3, 온도 5단계, 약 2분)
    python experiment_01_temperature.py --quick

    # 반복 횟수 지정
    python experiment_01_temperature.py --n 10

    # 다른 Ollama 모델로 실행
    python experiment_01_temperature.py --model llama3.2:3b

    # 결과 파일 경로 지정
    python experiment_01_temperature.py --output my_results.json
"""

import json
import math
import time
import argparse
import statistics
import urllib.request
import urllib.error
from collections import Counter
from typing import Optional


# ── 실험 설정 ────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"

# 0.0 ~ 1.5, 0.1 단위 (총 16단계)
TEMPERATURES = [round(i * 0.1, 1) for i in range(0, 16)]

N_SAMPLES = 20   # 온도별 반복 횟수 (실제 실험은 50~100 권장)
QUICK_N   = 3    # --quick 모드 반복 횟수

# 세 가지 태스크 유형: 창의 / 코드 / 사실
PROMPTS = {
    "creative": "한 문장으로 파란 하늘을 시적으로 묘사해줘.",
    "code":     "파이썬에서 리스트 중복 제거하는 가장 간단한 방법 한 줄로만 알려줘.",
    "factual":  "대한민국의 수도는 어디야? 한 단어로만 답해.",
}


# ── Ollama 연결 확인 ──────────────────────────────────────────────────────────

def check_ollama(model: str) -> None:
    """Ollama 서버와 모델이 사용 가능한지 사전 확인."""
    try:
        req = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = json.loads(req.read())
    except Exception:
        raise SystemExit(
            "\n[오류] Ollama 서버에 연결할 수 없습니다.\n"
            "  1. https://ollama.com 에서 Ollama 를 설치하세요.\n"
            "  2. 터미널에서 'ollama serve' 를 실행하세요.\n"
        )

    available = [m["name"] for m in data.get("models", [])]
    prefix = model.split(":")[0]
    if not any(a.startswith(prefix) for a in available):
        raise SystemExit(
            f"\n[오류] 모델 '{model}' 을 찾을 수 없습니다.\n"
            f"  설치된 모델: {available or '없음'}\n"
            f"  다음 명령으로 설치하세요: ollama pull {model}\n"
        )

    print(f"  Ollama 서버 연결 OK  |  모델: {model}")


# ── API 호출 ─────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, temperature: float, model: str) -> Optional[str]:
    """
    Ollama /api/generate 엔드포인트 호출.
    stream=False 로 전체 응답을 한 번에 받는다.

    삽질 포인트:
        seed=-1 이면 매번 다른 시드 → temperature=0 에서도 비결정론적 출력 가능.
        재현 가능한 실험을 원하면 seed=42 같이 고정하면 되지만,
        그러면 temperature 효과가 희석되므로 여기서는 -1 로 둔다.
    """
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 120,
            "seed": -1,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"\n  [호출 실패] temp={temperature:.1f} -> {e}")
        return None
    except Exception as e:
        print(f"\n  [오류] temp={temperature:.1f} -> {e}")
        return None


# ── 다양성 측정 ───────────────────────────────────────────────────────────────

def token_entropy(texts: list) -> float:
    """
    출력 텍스트 목록의 단어 단위 Shannon entropy.
    높을수록 다양한 단어 사용.
        entropy ~= 0   : 모든 출력이 거의 동일
        entropy 1~2    : 약간 다양
        entropy 3+     : 매우 다양
    """
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())
    if not all_words:
        return 0.0
    counts = Counter(all_words)
    total  = sum(counts.values())
    return round(
        -sum((c / total) * math.log2(c / total) for c in counts.values()),
        4,
    )


def unique_ratio(texts: list) -> float:
    """완전히 동일한 응답 제외한 고유 응답 비율 (1.0=전부 다름, 0.0=전부 같음)."""
    if not texts:
        return 0.0
    return round(len(set(texts)) / len(texts), 4)


def avg_char_length(texts: list) -> float:
    """평균 응답 글자 수."""
    if not texts:
        return 0.0
    return round(statistics.mean(len(t) for t in texts), 1)


# ── 실험 실행 ────────────────────────────────────────────────────────────────

def run_experiment(model: str, n: int, temperatures: list) -> dict:
    results     = {}
    total_calls = len(temperatures) * len(PROMPTS) * n
    call_count  = 0

    for task_name, prompt in PROMPTS.items():
        print(f"\n{'─'*60}")
        print(f"  태스크 [{task_name}]  |  {prompt}")
        print(f"{'─'*60}")
        results[task_name] = {}

        for temp in temperatures:
            outputs = []
            t0 = time.time()

            for _ in range(n):
                call_count += 1
                out = call_ollama(prompt, temp, model)
                if out:
                    outputs.append(out)

            elapsed = round(time.time() - t0, 1)
            entropy = token_entropy(outputs)
            uniq    = unique_ratio(outputs)
            avg_len = avg_char_length(outputs)

            results[task_name][str(temp)] = {
                "temperature":  temp,
                "n_success":    len(outputs),
                "entropy":      entropy,
                "unique_ratio": uniq,
                "avg_length":   avg_len,
                "elapsed_sec":  elapsed,
                "samples":      outputs[:3],
            }

            progress = f"{call_count}/{total_calls}"
            print(
                f"  temp={temp:.1f}  "
                f"entropy={entropy:.3f}  "
                f"unique={uniq:.2f}  "
                f"len={avg_len:.0f}  "
                f"({elapsed}s) [{progress}]"
            )

    return results


# ── 결과 요약 ─────────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """실험 결과 요약 + 자동 인사이트 출력."""

    print("\n\n" + "=" * 60)
    print("  결과 요약")
    print("=" * 60)

    for task_name, task_data in results.items():
        print(f"\n  [{task_name.upper()}]")
        print(f"  {'temp':>5}  {'entropy':>8}  {'unique':>7}  {'len':>6}  샘플")
        print(f"  {'─'*65}")

        for temp_str, d in task_data.items():
            sample = ""
            if d["samples"]:
                sample = d["samples"][0][:38].replace("\n", " ")
            print(
                f"  {d['temperature']:>5.1f}  "
                f"{d['entropy']:>8.3f}  "
                f"{d['unique_ratio']:>7.2f}  "
                f"{d['avg_length']:>6.0f}  "
                f"{sample}..."
            )

    print("\n\n" + "=" * 60)
    print("  자동 인사이트")
    print("=" * 60)

    for task_name, task_data in results.items():
        entropies = {float(t): d["entropy"] for t, d in task_data.items()}
        best_temp  = max(entropies, key=entropies.get)
        worst_temp = min(entropies, key=entropies.get)
        zero_ent   = entropies.get(0.0, 0.0)
        zero_uniq  = task_data.get("0.0", {}).get("unique_ratio", 0.0)

        print(f"\n  [{task_name}]")
        print(f"    가장 다양한 출력 : temp={best_temp:.1f}  (entropy={entropies[best_temp]:.3f})")
        print(f"    가장 일관된 출력 : temp={worst_temp:.1f}  (entropy={entropies[worst_temp]:.3f})")

        if zero_ent > 0.3:
            print(f"    ⚠  temp=0.0 entropy={zero_ent:.3f} — 비결정론적 출력 확인!")
        if zero_uniq > 0.5:
            print(f"    ⚠  temp=0.0 unique_ratio={zero_uniq:.2f} — seed 미고정 효과")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="실험 01: Temperature 조절 — Ollama 로컬 모델",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Ollama 모델 이름 (기본: {DEFAULT_MODEL})")
    parser.add_argument("--n",      type=int, default=N_SAMPLES,
                        help=f"온도별 반복 횟수 (기본: {N_SAMPLES})")
    parser.add_argument("--quick",  action="store_true",
                        help="빠른 테스트 (n=3, 5단계 온도만)")
    parser.add_argument("--output", default="results_experiment_01.json",
                        help="결과 JSON 저장 경로")
    args = parser.parse_args()

    n    = QUICK_N if args.quick else args.n
    temps = [0.0, 0.3, 0.7, 1.0, 1.3] if args.quick else TEMPERATURES

    print("\n" + "=" * 60)
    print("  실험 01: 프롬프트 온도 조절 실험  (Ollama 로컬)")
    print("=" * 60)
    print(f"  모델     : {args.model}")
    print(f"  온도 구간 : {temps[0]} ~ {temps[-1]}  ({len(temps)}단계)")
    print(f"  반복 횟수 : {n}회 / 온도")
    print(f"  총 호출 수 : {len(temps) * len(PROMPTS) * n}")
    print()

    check_ollama(args.model)

    results = run_experiment(args.model, n, temps)
    print_summary(results)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장 -> {args.output}")
    print("\n  다음 단계:")
    print(f"    python {__file__} --n 50          # 더 많은 샘플")
    print(f"    python {__file__} --model llama3.2:3b  # 다른 모델 비교")


if __name__ == "__main__":
    main()
