"""Переводит словарь жестов РЖЯ на казахский один раз, заранее.

Живой режим получает казахское слово из готового файла мгновенно:
ни запроса к LLM на каждый жест, ни нагрузки на сервер.

    ../venv/bin/python build_kk.py        # докачивает недостающее

Результат: model/RSL_class_list_kk.json  {"привет": "сәлем", ...}
"""
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "model" / "RSL_class_list.txt"
OUT = BASE / "model" / "RSL_class_list_kk.json"
BATCH = 60

for line in (BASE.parent / ".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
URL = os.environ.get("ALEM_LLM_URL", "https://llm.alem.ai/v1").rstrip("/") + "/chat/completions"
KEY = os.environ["ALEM_LLM_KEY"]
MODEL = os.environ.get("ALEM_LLM_MODEL", "gemma4")

PROMPT = """Ты — переводчик словаря жестового языка с русского на казахский.
Каждая строка — название одного жеста: слово, фраза, число или буква.

Правила:
- Переводи естественно, как сказал бы носитель казахского языка.
- Глаголы давай в начальной форме на -у (учиться → оқу).
- Числа и время переводи словами только если в оригинале слова (2 часа → 2 сағат).
- Отдельные буквы русского алфавита оставь как есть.
- Никаких пояснений.

Ответ — строго JSON-объект {"русское": "казахское"} для ВСЕХ строк:
"""


def ask(words: list[str]) -> dict:
    return ask_raw(PROMPT + json.dumps(words, ensure_ascii=False))


def ask_raw(content: str) -> dict:
    """Один запрос к LLM; из ответа достаёт JSON-объект."""
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4000,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def main() -> None:
    labels = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip() and parts[1].strip() != "no":
            labels.append(parts[1].strip())
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    # буквы и числа не переводятся
    for w in labels:
        if w not in done and (len(w) == 1 or w.isdigit()):
            done[w] = w
    todo = [w for w in labels if w not in done]
    print(f"всего {len(labels)}, готово {len(done)}, осталось {len(todo)}")

    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]

    def one(batch: list[str]) -> None:
        try:
            got = ask(batch)
        except Exception as e:
            print("  ошибка пачки:", e, file=sys.stderr)
            return
        for w in batch:
            v = got.get(w)
            if isinstance(v, str) and v.strip():
                done[w] = v.strip()
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
        print(f"  +{len(batch)} → {len(done)}/{len(labels)}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, batches))
    OUT.write_text(json.dumps(done, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    left = [w for w in labels if w not in done]
    print(f"итог: {len(done)}/{len(labels)}; без перевода: {len(left)}", left[:20])


if __name__ == "__main__":
    main()
