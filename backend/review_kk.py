"""Вычитка казахского словаря: второй проход LLM ищет ошибки перевода.

Первый проход (build_kk.py) иногда берёт не то значение многозначного
слова: «ехать» → «жүргізу» (водить) вместо «бару». Здесь модель видит
готовые пары и возвращает только исправления.

    ../venv/bin/python review_kk.py
"""
import json
from concurrent.futures import ThreadPoolExecutor

from build_kk import OUT, ask_raw

BATCH = 80

PROMPT = """Ты — редактор казахско-русского словаря жестового языка.
Ниже пары «русское название жеста → казахский перевод».
Найди ошибки: неверное значение многозначного слова, не та часть речи,
калька, орфография. Глаголы должны быть в начальной форме на -у.
Правильные пары НЕ включай.

Ответ — строго JSON-объект только с исправлениями {"русское": "верный казахский"},
или {} если ошибок нет:
"""


def main() -> None:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    pairs = [(k, v) for k, v in d.items() if len(k) > 1 and not k.isdigit()]
    batches = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]
    fixes: dict[str, str] = {}

    def one(batch):
        text = "\n".join(f"{k} → {v}" for k, v in batch)
        try:
            got = ask_raw(PROMPT + text)
        except Exception as e:
            print("  ошибка пачки:", e)
            return
        for k, v in got.items():
            if k in d and isinstance(v, str) and v.strip() and v.strip() != d[k]:
                fixes[k] = v.strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, batches))
    for k, v in sorted(fixes.items()):
        print(f"  {k}: {d[k]} → {v}")
    d.update(fixes)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"исправлено {len(fixes)} из {len(pairs)}")


if __name__ == "__main__":
    main()
