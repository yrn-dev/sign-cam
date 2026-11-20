"""
SignCam backend — real-time RSL sign-language recognition over WebSocket.

Flow:
  Browser (camera) --base64 JPEG 224x224 frames-->  /ws
  This server buffers 32 frames per connection, runs S3D inference,
  and pushes back {label, conf, ms} for each completed window.

Run:
  uvicorn app:app --host 0.0.0.0 --port 8600
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent


def _load_env() -> None:
    """Читает .env в корне проекта, не затирая уже заданные переменные."""
    env = BASE.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
MODEL_DIR = BASE / "model"
FRONTEND_DIR = BASE.parent / "frontend"

WINDOW = 32          # frames per S3D clip
# Сдвиг окна: новая попытка каждые ~0.5 с вместо 1.3 с.
# На слабом CPU шаг стоит увеличить, чтобы инференс успевал за потоком
# кадров: при 25 fps шаг N кадров = N*40 мс на одно решение.
STRIDE = int(os.environ.get("SIGNCAM_STRIDE", "12"))
INTRA_THREADS = int(os.environ.get("SIGNCAM_THREADS", "8"))
SMOOTH = 3           # сколько последних окон усредняем перед решением
THRESHOLD = 0.5      # confidence gate
TOPK = 5             # candidates returned for diagnostics
LABEL_NO = "no"      # class index 0 means "no gesture"
RU_LETTERS = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

# --- смысловой слой: мультимодальная LLM собирает фразу из кадров ---
LLM_URL = os.environ.get("ALEM_LLM_URL", "https://llm.alem.ai/v1")
LLM_KEY = os.environ.get("ALEM_LLM_KEY", "")
LLM_MODEL = os.environ.get("ALEM_LLM_MODEL", "gemma4")

# Зрение можно увести на другого провайдера, не трогая остальное:
#   VISION_URL=https://openrouter.ai/api/v1 VISION_KEY=sk-or-... VISION_MODEL=...
VISION_URL = os.environ.get("VISION_URL") or LLM_URL
VISION_KEY = os.environ.get("VISION_KEY") or os.environ.get("OPENROUTER_API_KEY") or LLM_KEY
VISION_MODEL = os.environ.get("VISION_MODEL") or LLM_MODEL

PROMPT = """Ты — сурдопереводчик. На картинке кадры одного видео, \
разложенные по сетке слева направо и сверху вниз: это подряд идущие \
моменты, пока человек показывал жест или фразу на жестовом языке \
(русский РЖЯ, возможен казахский КЖЯ — они близки).

Смотри на кисти рук, их форму, положение относительно лица и груди, \
направление движения между кадрами, а также на мимику и артикуляцию губ.

{draft}

Ответь строго в формате JSON, без markdown:
{{"phrase": "фраза на русском языке обычными словами",
  "kk": "та же фраза на казахском",
  "confidence": 0.0-1.0,
  "notes": "что именно ты увидел в руках, кратко"}}

Если жест разобрать невозможно — верни phrase: "" и объясни в notes."""


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Predictor:
    """Thin wrapper around the S3D ONNX model."""

    def __init__(self) -> None:
        available = rt.get_available_providers()
        self.provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in available
            else "CPUExecutionProvider"
        )
        opts = rt.SessionOptions()
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 8 потоков вдвое быстрее, чем все 16 логических ядер:
        # на гипертрединге 3D-свёртки только дерутся за кэш (223 мс против 392)
        opts.intra_op_num_threads = INTRA_THREADS
        self.sess = rt.InferenceSession(
            str(MODEL_DIR / "S3D.onnx"), sess_options=opts, providers=[self.provider]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.labels: dict[int, str] = {}
        for line in (MODEL_DIR / "RSL_class_list.txt").read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 2:
                self.labels[int(parts[0])] = parts[1]
        # перевод названий жестов на казахский, собран заранее build_kk.py
        kk_path = MODEL_DIR / "RSL_class_list_kk.json"
        self.kk: dict[str, str] = (
            json.loads(kk_path.read_text(encoding="utf-8")) if kk_path.exists() else {})

    def to_kk(self, label: str) -> str:
        return self.kk.get(label, label)

    def predict(self, frames: deque[np.ndarray]) -> dict:
        """frames: 32 x [224,224,3] uint8 (RGB). Returns {label, conf, ms, top}."""
        clip = np.asarray(frames, dtype=np.float32) / 255.0  # [32,224,224,3]
        clip = clip.transpose(3, 0, 1, 2)[None]              # [1,3,32,224,224]
        t0 = time.perf_counter()
        logits = self.sess.run([self.out_name], {self.in_name: clip})[0][0]
        ms = (time.perf_counter() - t0) * 1000.0
        return {"probs": _softmax(logits), "ms": round(ms, 1)}

    def decode(self, p: np.ndarray) -> dict:
        """Вероятности -> метка + топ-5 кандидатов."""
        order = np.argsort(p)[::-1][:TOPK]
        idx = int(order[0])
        label = self.labels.get(idx, "?")
        return {
            "label": label,
            "kk": self.to_kk(label),
            "conf": round(float(p[idx]), 4),
            "top": [
                {"label": (lb := self.labels.get(int(i), "?")), "kk": self.to_kk(lb),
                 "conf": round(float(p[i]), 4)}
                for i in order
            ],
        }


STATE: dict[str, "Session"] = {}


class Session:
    """Per-connection frame buffer + inference state."""

    __slots__ = ("buf", "busy", "windows", "last_ms", "hist")

    def __init__(self) -> None:
        self.buf = deque(maxlen=WINDOW)
        self.busy = False
        self.windows = 0
        self.last_ms = 0.0
        # последние SMOOTH распределений — усреднение гасит случайные всплески
        self.hist: deque[np.ndarray] = deque(maxlen=SMOOTH)


_PREDICTOR: Predictor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _PREDICTOR
    _PREDICTOR = Predictor()
    print(f"[signcam] model loaded on {_PREDICTOR.provider} "
          f"({len(_PREDICTOR.labels)} classes)", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="SignCam", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "provider": _PREDICTOR.provider if _PREDICTOR else "loading",
        "classes": len(_PREDICTOR.labels) if _PREDICTOR else 0,
        "window": WINDOW,
        "stride": STRIDE,
        "smooth": SMOOTH,
        "threshold": THRESHOLD,
        "vision_model": VISION_MODEL,
    }


def _decode_frame(msg: str | bytes) -> np.ndarray | None:
    """Decode an incoming frame into a 224x224 RGB array.

    Принимаем и сырые байты JPEG (быстрый путь), и base64/data-URL —
    браузер шлёт бинарь, но текстовый вариант оставлен для совместимости.
    """
    try:
        if isinstance(msg, str):
            if msg.startswith("data:"):
                msg = msg.split(",", 1)[1]
            raw = base64.b64decode(msg)
        else:
            raw = msg
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sid = str(ws.client)
    sess = Session()
    STATE[sid] = sess
    try:
        while True:
            packet = await ws.receive()
            if packet.get("type") == "websocket.disconnect":
                break
            msg = packet.get("bytes")
            if msg is None:
                msg = packet.get("text")
                # control messages
                if msg in ("reset", "ping"):
                    if msg == "reset":
                        sess.buf.clear()
                        sess.hist.clear()
                    await ws.send_text(json.dumps({"type": "ack"}))
                    continue
            if msg is None:
                continue

            img = await asyncio.to_thread(_decode_frame, msg)
            if img is None:
                continue
            sess.buf.append(img)

            if len(sess.buf) == WINDOW and not sess.busy:
                sess.busy = True
                sess.windows += 1
                n_window = sess.windows
                try:
                    raw = await asyncio.to_thread(_PREDICTOR.predict, sess.buf)
                    sess.last_ms = raw["ms"]
                    sess.hist.append(raw["probs"])
                    # решение принимаем по среднему за последние окна
                    avg = np.mean(sess.hist, axis=0)
                    dec = _PREDICTOR.decode(avg)
                    hit = dec["label"] != LABEL_NO and dec["conf"] >= THRESHOLD
                    if hit:
                        # слово зафиксировано — историю и буфер начинаем заново
                        sess.buf.clear()
                        sess.hist.clear()
                    else:
                        # иначе двигаем окно на STRIDE: жест не потеряется
                        # на стыке и получит несколько попыток
                        for _ in range(STRIDE):
                            if sess.buf:
                                sess.buf.popleft()
                    result = {
                        "type": "result",
                        "label": dec["label"],
                        "kk": dec["kk"],
                        "conf": dec["conf"],
                        "ms": raw["ms"],
                        "top": dec["top"],
                        "window": n_window,
                        "gesture": hit,
                    }
                    await ws.send_text(json.dumps(result, ensure_ascii=False))
                finally:
                    sess.busy = False
    except WebSocketDisconnect:
        pass
    finally:
        STATE.pop(sid, None)


# ---------------------------------------------------------------- разбор файлов

UPLOAD_MAX_MB = 300
UPLOAD_MAX_SECONDS = 180      # больше трёх минут не берём: это уже пакетная задача
TARGET_FPS = 25.0             # к этой частоте приводим любое видео
JOBS: dict[str, dict] = {}


def _frames_from_video(path: str, fps_hint: float) -> tuple[list[np.ndarray], float]:
    """Читает видео и возвращает кадры 224x224 RGB, прореженные до ~25 fps.

    Приведение частоты обязательно: модель обучалась на 25-30 fps, и если
    подать 60 fps как есть, жест окажется вдвое «медленнее» и не распознается.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, "файлды ашу мүмкін болмады — формат қолдау көрсетілмейтін шығар")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or TARGET_FPS
    if not (1 <= src_fps <= 240):
        src_fps = TARGET_FPS
    step = max(1.0, src_fps / TARGET_FPS)

    frames: list[np.ndarray] = []
    idx = 0.0
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n >= idx:
            h, w = img.shape[:2]
            side = min(h, w)
            img = img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
            img = cv2.resize(img, (224, 224))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            idx += step
        n += 1
        if len(frames) > UPLOAD_MAX_SECONDS * TARGET_FPS:
            break
    cap.release()
    return frames, src_fps


def _frames_from_image(path: str) -> list[np.ndarray]:
    """Одно фото -> 32 одинаковых кадра: модель принимает только клип."""
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(400, "суретті оқу мүмкін болмады")
    h, w = img.shape[:2]
    side = min(h, w)
    img = img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
    img = cv2.resize(img, (224, 224))
    return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)] * WINDOW


def _run_job(job_id: str, path: str, is_image: bool) -> None:
    """Прогоняет файл скользящим окном и копит распознанные слова."""
    job = JOBS[job_id]
    try:
        if is_image:
            # у фото свой путь: зрячая LLM вместо видеомодели
            job["status"] = "смотрим фото"
            job["vision"] = _vision_gesture(path)
            frames = _frames_from_image(path)
            src_fps = 0.0
        else:
            frames, src_fps = _frames_from_video(path, TARGET_FPS)
        if len(frames) < WINDOW:
            raise HTTPException(
                400, f"үзінді тым қысқа: кемінде {WINDOW} кадр "
                     f"(~1,3 с) керек, табылғаны {len(frames)}")

        # на длинном видео шаг увеличиваем, иначе разбор затянется
        stride = STRIDE if len(frames) <= TARGET_FPS * 40 else STRIDE * 2
        starts = list(range(0, len(frames) - WINDOW + 1, stride))
        job.update(total=len(starts), fps=round(src_fps, 1),
                   seconds=round(len(frames) / TARGET_FPS, 1))

        hist: deque[np.ndarray] = deque(maxlen=SMOOTH)
        last = None
        gap = 0
        for i, st in enumerate(starts):
            clip = deque(frames[st:st + WINDOW])
            hist.append(_PREDICTOR.predict(clip)["probs"])
            dec = _PREDICTOR.decode(np.mean(hist, axis=0))
            t = round(st / TARGET_FPS, 2)
            if dec["label"] != LABEL_NO and dec["conf"] >= THRESHOLD:
                if dec["label"] != last or gap >= 2:
                    job["words"].append(
                        {"word": dec["label"], "kk": dec["kk"], "conf": dec["conf"], "t": t})
                    last = dec["label"]
                hist.clear()
                gap = 0
            else:
                gap += 1
                if gap >= 2:
                    last = None
            job["done"] = i + 1
            job["preview"] = dec["top"]
        # Видео: отдаём отрезки зрячей модели сетками кадров.
        # S3D даёт черновик слов, LLM собирает из них смысл.
        if not is_image and VISION_KEY:
            job["status"] = "читаем кадры"
            seg_len = int(SEGMENT_SECONDS * TARGET_FPS)
            bounds = [b for b in range(0, len(frames), seg_len)
                      if len(frames[b:b + seg_len]) >= WINDOW // 2][:MAX_SEGMENTS]
            job["vsegs_total"] = len(bounds)
            job["segments"] = [None] * len(bounds)

            def one(i_st: tuple[int, int]) -> None:
                i, st = i_st
                chunk = frames[st:st + seg_len]
                t0s, t1s = st / TARGET_FPS, (st + len(chunk)) / TARGET_FPS
                draft = [w["word"] for w in job["words"] if t0s <= w["t"] < t1s]
                res = _vision_clip(chunk, draft)
                if not res.get("error") and "confidence" not in res:
                    res = _vision_clip(chunk, draft)
                phrase = (res.get("phrase") or "").strip()
                if phrase and phrase.lower() in {v.lower() for v in _PREDICTOR.labels.values()}:
                    res["kk"] = _PREDICTOR.to_kk(phrase.lower())
                elif phrase and float(res.get("confidence") or 0) < VISION_MIN_CONF:
                    res["rejected"] = phrase
                    res["phrase"] = ""
                    res["kk"] = ""
                res["from"] = round(t0s, 1)
                res["to"] = round(t1s, 1)
                job["segments"][i] = res
                job["vsegs_done"] = sum(x is not None for x in job["segments"])

            # отрезки независимы, поэтому читаем их одновременно:
            # последовательно минута видео заняла бы несколько минут
            with ThreadPoolExecutor(max_workers=VISION_WORKERS) as pool:
                list(pool.map(one, enumerate(bounds)))
            job["segments"] = [x for x in job["segments"] if x]

        vis = job.get("vision") or {}
        draft_ru = " ".join(w["word"] for w in job["words"])
        draft_kk = " ".join(w["kk"] for w in job["words"])
        if is_image and vis.get("gesture"):
            job["text"] = vis["gesture"]
            job["text_kk"] = vis.get("kk") or _PREDICTOR.to_kk(vis["gesture"])
        elif job.get("segments"):
            segs = [x for x in job["segments"] if x.get("phrase")]
            job["text"] = " ".join(x["phrase"].strip() for x in segs) or draft_ru
            job["text_kk"] = " ".join(
                (x.get("kk") or _PREDICTOR.to_kk(x["phrase"])).strip() for x in segs) or draft_kk
        else:
            job["text"] = draft_ru
            job["text_kk"] = draft_kk
        job["status"] = "ready"
    except HTTPException as e:
        job.update(status="error", error=e.detail)
    except Exception as e:
        job.update(status="error", error=str(e))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Принимает фото или видео с жестами и запускает разбор в фоне."""
    name = (file.filename or "").lower()
    ext = os.path.splitext(name)[1]
    is_image = ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_video = ext in {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif", ".m4v"}
    if not (is_image or is_video):
        raise HTTPException(400, f"{ext or '?'} форматы қолдау көрсетілмейді")

    fd, path = tempfile.mkstemp(suffix=ext)
    size = 0
    with os.fdopen(fd, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > UPLOAD_MAX_MB << 20:
                out.close()
                os.unlink(path)
                raise HTTPException(413, f"файл {UPLOAD_MAX_MB} МБ-тан үлкен")
            out.write(chunk)

    # задачи живут в памяти, поэтому старые выбрасываем
    now = time.time()
    for old in [k for k, v in JOBS.items() if now - v.get("at", now) > 3600]:
        JOBS.pop(old, None)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "работаем", "done": 0, "total": 0, "words": [],
                    "text": "", "kind": "image" if is_image else "video",
                    "name": file.filename, "size": size, "at": now}
    asyncio.create_task(asyncio.to_thread(_run_job, job_id, path, is_image))
    return {"job": job_id, "kind": JOBS[job_id]["kind"]}


@app.get("/job/{job_id}")
async def job_status(job_id: str) -> dict:
    """Прогресс и результат разбора."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "тапсырма табылмады")
    return job


VISION_PROMPT = """На фотографии неслышащий человек говорит жестом. \
Назови слово, которое он показывает.

Правила, без которых легко ошибиться:
1. Дактильная азбука РЖЯ показывается ОДНОЙ рукой. Если работают обе \
руки — это слово или фраза, а НЕ буква алфавита. Никогда не называй \
двуручный жест буквой.
2. Не объясняй форму кисти через латинские буквы и не опирайся на \
американский язык ASL: там другая азбука, и её формы к РЖЯ не подходят.
3. По одному кадру не видно движения. Многие жесты РЖЯ различаются \
именно траекторией, и тогда честный ответ — что нужен видеоряд.
4. Не завышай уверенность. 0.9 ставь только если жест однозначен. \
Если сомневаешься — 0.3-0.5 и несколько вариантов.

ВАЖНО: значение жеста выбирай ТОЛЬКО из словаря, который дан ниже. \
Это полный список жестов, которые система умеет распознавать. Если \
увиденное не подходит ни под один жест из словаря — оставь gesture \
пустым. Не придумывай значений вне списка.

Ответь строго в формате JSON, без markdown. Поля заполняй по порядку: \
сначала опиши увиденное, и только потом делай вывод.
{"hands": "одна" или "обе",
 "description": "что видно: форма кисти, какие пальцы выпрямлены или \
согнуты, положение рук относительно лица и груди, направление ладоней, \
выражение лица. 1-3 предложения, без выводов",
 "static": true если жест неподвижный и по кадру его можно определить, \
false если смысл в движении и нужен видеоряд,
 "gesture": "наиболее вероятное значение по-русски, одно-два слова; \
пустая строка, если определить нельзя",
 "alternatives": ["другие возможные значения, 0-3 варианта"],
 "kk": "gesture по-казахски",
 "description_kk": "description на казахском языке",
 "confidence": число от 0 до 1}

Если человека или жеста на фото нет — gesture пустой, объясни в description.

СЛОВАРЬ ДОСТУПНЫХ ЖЕСТОВ:
{vocab}"""


SYSTEM_ROLE = """Ты профессиональный сурдопереводчик русского жестового \
языка (РЖЯ) и казахского (КЖЯ).

Ты работаешь внутри системы, которая помогает слышащим людям понимать \
глухих и слабослышащих. На фотографиях и кадрах видео, которые тебе \
присылают, неслышащий человек ГОВОРИТ на жестовом языке: его руки — \
это его речь. Каждый жест — это слово или фраза, которую он произносит.

Твоя задача одна: назвать слово, которое человек говорит жестом.

Как ты смотришь на кадр:
— форма кисти: какие пальцы выпрямлены, какие согнуты, куда смотрит ладонь;
— место: где рука относительно лица, лба, подбородка, груди, плеча;
— движение: как положение рук меняется от кадра к кадру;
— мимика и губы: в жестовом языке лицо несёт грамматику, а губы часто \
беззвучно проговаривают само слово.

От твоей точности зависит, поймут ли человека. Поэтому ты никогда не \
выдумываешь: если жест непонятен, ты честно говоришь об этом, а не \
подставляешь похожее слово."""

CLIP_PROMPT = """На картинке кадры ОДНОГО отрезка видео, разложенные по сетке слева \
направо и сверху вниз. Это последовательные моменты, пока человек \
показывал жест или короткую фразу. Читай их как покадровую съёмку \
движения: сравнивай, как меняется положение рук от кадра к кадру.

Правила:
1. Дактильная азбука РЖЯ показывается ОДНОЙ рукой. Двуручный жест \
буквой не называй.
2. Не опирайся на американский ASL — там другая азбука.
3. Значение выбирай ТОЛЬКО из словаря в конце. Не подходит ничего — \
оставь phrase пустым.
4. Не завышай уверенность.
5. Если на кадрах нет человека, показывающего жест руками, — phrase \
пустой, а в motion напиши, что именно на кадрах. Не подгоняй под \
словарь абстрактные фигуры и посторонние объекты.

{hint}

Ответь строго в формате JSON, без markdown, поля по порядку:
{"motion": "как меняется положение рук между кадрами, 1-2 предложения",
 "phrase": "значение жеста по-русски, слово или короткая фраза из \
словаря; пустая строка, если определить нельзя",
 "kk": "phrase по-казахски",
 "motion_kk": "motion на казахском языке",
 "confidence": число от 0 до 1}

СЛОВАРЬ ДОСТУПНЫХ ЖЕСТОВ:
{vocab}"""

# Как режется видео:
#   отрезок 3 с  ->  16 кадров в сетке 4x4  ->  кадр каждые ~190 мс.
# Жест длится 1-2 с, то есть попадает в 5-10 кадров — движение видно.
VISION_MIN_CONF = 0.4         # ниже этого фото считаем нераспознанным
PHOTO_DESC_MIN_CONF = 0.75    # ниже этого слова из PHOTO_WORDS считаем описанием кадра, а не жестом
# слова, которыми модель описывает сам снимок; как перевод жеста не годятся
PHOTO_WORDS = {"человек", "лицо", "рука", "руки", "подбородок", "мужчина",
               "женщина", "улыбка", "голова", "палец", "пальцы", "щека", "губы"}
GRID_COLS = 4                # кадров в ряду сетки
GRID_FRAMES = 16             # кадров на отрезок (сетка 4x4)
SEGMENT_SECONDS = 3.0        # длительность отрезка видео
MAX_SEGMENTS = 60            # 60 x 3 с = 180 с, ровно предел загрузки
VISION_WORKERS = 4           # столько отрезков читаем одновременно


def _grid_b64(frames: list[np.ndarray], cell: int = 224) -> str:
    """Склеивает кадры в одну картинку-сетку: так модель видит движение.

    Модель принимает изображения, но не видео, поэтому отрезок
    превращаем в контактный лист — кадры идут слева направо, сверху вниз.
    """
    picked = frames
    if len(picked) > GRID_FRAMES:
        step = len(picked) / GRID_FRAMES
        picked = [picked[int(i * step)] for i in range(GRID_FRAMES)]

    cols = min(GRID_COLS, len(picked))
    rows = (len(picked) + cols - 1) // cols
    sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i, f in enumerate(picked):
        r, c = divmod(i, cols)
        img = f if f.shape[0] == cell else cv2.resize(f, (cell, cell))
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
        # номер кадра: без него модель путает порядок
        cv2.putText(sheet, str(i + 1), (c * cell + 8, r * cell + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    sheet = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)   # кадры хранятся в RGB
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "не удалось собрать сетку кадров")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _ask_vision(content: list[dict], max_tokens: int = 500) -> dict:
    """Один запрос к модели зрения. Возвращает разобранный JSON."""
    if not VISION_KEY:
        return {"error": "көру моделінің кілті берілмеген"}
    body = {"model": VISION_MODEL, "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_ROLE},
                {"role": "user", "content": content},
            ]}
    req = urllib.request.Request(
        VISION_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {VISION_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            raw = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"көру моделі {e.code} қатесін қайтарды"}
    except Exception as e:
        return {"error": f"көру моделі қолжетімсіз: {e}"}

    text = raw["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = {"phrase": "", "motion": text, "confidence": 0.0}
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def _vocab_text() -> str:
    return ", ".join(sorted(set(_PREDICTOR.labels.values()) - {LABEL_NO})) \
        if _PREDICTOR else ""


def _vision_clip(frames: list[np.ndarray], draft: list[str]) -> dict:
    """Смотрит отрезок видео как сетку кадров и называет жест."""
    hint = ("Быстрая видеомодель предположила в этом отрезке: "
            + ", ".join(draft)
            + ". Она часто ошибается — считай это подсказкой, а не истиной."
            ) if draft else "Подсказок от быстрой модели нет."
    prompt = CLIP_PROMPT.replace("{hint}", hint).replace("{vocab}", _vocab_text())
    return _ask_vision([
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _grid_b64(frames)}},
    ])


def _image_b64(path: str, side: int = 768) -> str:
    """Готовит фото для LLM: ужимаем большую сторону, отдаём data-URL."""
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(400, "суретті оқу мүмкін болмады")
    h, w = img.shape[:2]
    if max(h, w) > side:
        k = side / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(400, "не удалось закодировать изображение")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _vision_gesture(path: str) -> dict:
    """Распознаёт жест на фото.

    S3D для одиночного снимка не годится: она обучена на движении, а в
    фото движения нет. Зрячая модель смотрит на форму кисти и потому
    справляется со статичными жестами заметно лучше.
    """
    content = [
        {"type": "text", "text": VISION_PROMPT.replace("{vocab}", _vocab_text())},
        {"type": "image_url", "image_url": {"url": _image_b64(path)}},
    ]
    out = _ask_vision(content)
    # модель изредка отвечает не JSON — тогда поля gesture нет; второй попытки хватает
    if not out.get("error") and "gesture" not in out:
        out = _ask_vision(content)
    if out.get("error"):
        return out

    # двуручный жест не может быть буквой — модель об этом всё равно забывает
    word = (out.get("gesture") or "").strip()
    if word and out.get("hands") == "обе" and len(word.replace("буква", "").strip()) <= 1:
        out["warning"] = ("модель екі қолмен көрсетілген ымды әріп деп атады — орыс ым тілінде "
                          "дактиль бір қолмен көрсетіледі, жауап сенімсіз")
        out["confidence"] = min(float(out.get("confidence") or 0), 0.3)

    # есть ли такое слово среди 1598 жестов, которым обучена S3D
    if word and _PREDICTOR:
        known = {v.lower() for v in _PREDICTOR.labels.values()}
        norm = word.lower().replace("буква", "").strip()
        out["in_vocab"] = norm in known
        # Ответ не из словаря жестов — это не жест, а описание кадра
        # («подбородок» у руки возле лица). Показывать его как перевод нельзя.
        # Слова из PHOTO_WORDS («человек», «лицо»...) — настоящие жесты из
        # словаря, но при слабой уверенности модель ими скорее описывает
        # кадр. Принимать их как жест можно только при высокой уверенности.
        conf = float(out.get("confidence") or 0)
        descriptive = norm in PHOTO_WORDS and conf < PHOTO_DESC_MIN_CONF
        if not out["in_vocab"] or conf < VISION_MIN_CONF or descriptive:
            out["rejected"] = word
            out["gesture"] = ""
            out["kk"] = ""
        else:
            out["gesture"] = norm
            out["kk"] = _PREDICTOR.to_kk(norm)
            if norm in PHOTO_WORDS:
                out["warning"] = ("модель назвала жест словом, которым часто описывают кадр — "
                                   "убедитесь, что это именно тот жест, который вы показали")
        alts = [a.strip().lower() for a in out.get("alternatives") or [] if isinstance(a, str)]
        out["alternatives"] = [_PREDICTOR.to_kk(a) for a in alts if a in known][:3]

    out["model"] = VISION_MODEL
    return out


class TranslateReq(BaseModel):
    """Сетка кадров (data:image/jpeg;base64,...) + черновик слов от S3D."""

    grid: str
    draft: list[str] = []


def _call_llm(grid_b64: str, draft: list[str]) -> dict:
    """Синхронный запрос к мультимодальной LLM. Вызывается в потоке."""
    if not LLM_KEY:
        raise HTTPException(503, "ALEM_LLM_KEY не задан")

    hint = (
        "Быстрая модель распознала в этом фрагменте слова-кандидаты: "
        + ", ".join(draft)
        + ". Она часто ошибается — используй их как подсказку, но верь глазам."
        if draft
        else "Подсказок от быстрой модели нет — определи жест сам."
    )
    body = {
        "model": LLM_MODEL,
        "max_tokens": 400,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(draft=hint)},
            {"type": "image_url", "image_url": {"url": grid_b64}},
        ]}],
    }
    req = urllib.request.Request(
        LLM_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {LLM_KEY}",
                 "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = json.load(r)
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"LLM {e.code}: {e.read()[:200].decode()}")
    except Exception as e:
        raise HTTPException(502, f"LLM недоступна: {e}")

    text = raw["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = {"phrase": text, "kk": "", "confidence": 0.0, "notes": "не JSON"}
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


@app.get("/vocab")
async def vocab() -> dict:
    """Словарь для режима дактиля: по нему сверяются набранные по буквам слова."""
    words = sorted({
        w for w in _PREDICTOR.labels.values()
        if len(w) > 2 and w.isalpha()
    }) if _PREDICTOR else []
    kk = {w: _PREDICTOR.to_kk(w) for w in words} if _PREDICTOR else {}
    return {"words": words, "kk": kk, "letters": sorted(RU_LETTERS)}


@app.post("/translate")
async def translate(req: TranslateReq) -> dict:
    """Смысловой перевод фрагмента: кадры + черновик -> человеческая фраза."""
    return await asyncio.to_thread(_call_llm, req.grid, req.draft)


class NoCacheStatic(StaticFiles):
    """Отдаёт фронтенд без кэширования — правки видны после обычного F5."""

    def is_not_modified(self, *_args, **_kwargs) -> bool:
        return False

    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store"
        return resp


# serve the frontend (index.html + assets)
app.mount("/", NoCacheStatic(directory=str(FRONTEND_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("SIGNCAM_HOST", "0.0.0.0"), port=8600)
