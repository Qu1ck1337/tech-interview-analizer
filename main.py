import asr
import asyncio
from analizers.mistake_analyzer import MistakeAnalyzer
from analizers.qa_evaluator import QAEvaluator
import json
from pathlib import Path
from typing import Optional
from config import load_config

def _find_cached_transcript_json(file_path: str, output_dir: Optional[str]) -> Optional[str]:
    """Находит ранее сохранённый транскрипт.

    Ищем в приоритетном порядке:
    1) <output_dir>/<stem>/<stem>_transcript.json
    2) results/<stem>/<stem>_transcript.json
    3) Рядом с исходным файлом: *_transcript.json (legacy)
    """
    p = Path(file_path)
    stem = p.stem
    candidates = []
    if output_dir:
        candidates.append(str(Path(output_dir) / stem / f"{stem}_transcript.json"))
    else:
        base = Path.cwd() / "results" / stem
        candidates.append(str(base / f"{stem}_transcript.json"))
    # legacy рядом с файлом
    candidates.append(str(p.with_suffix('')) + "_transcript.json")
    for c in candidates:
        if Path(c).is_file():
            return c
    return None

def _prepare_output_dir(file_path: str, output_dir: Optional[str]) -> Path:
    p = Path(file_path)
    stem = p.stem
    base_dir = Path(output_dir) if output_dir else (Path.cwd() / "results")
    out_dir = base_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _load_segments_from_json(json_path: str):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Наш формат: transcript, возможны варианты: segments
            segments = data.get("transcript") or data.get("segments")
            # Иногда могут быть вложенные структуры
            if segments is None and "result" in data and isinstance(data["result"], dict):
                segments = data["result"].get("segments")
        elif isinstance(data, list):
            segments = data
        else:
            segments = None
        return segments
    except Exception as e:
        print(f"⚠️ Не удалось загрузить транскрипт из {json_path}: {e}")
        return None


async def analyze_interview_file(file_path: str, segments=None, output_dir: Optional[str] = None):
    """Анализирует собеседование: использует готовый транскрипт или транскрибирует.

    - Если переданы `segments`, пропускает транскрибацию.
    - Если `file_path` указывает на .json или рядом/в results есть *_transcript.json с полем transcript, использует его.
    - Иначе запускает WhisperX транскрибацию.
    """
    print(f"🔍 Начинаю анализ входа: {file_path}")

    # 0. Пытаемся использовать переданный транскрипт
    if segments is not None:
        print("📝 Использую уже готовый транскрипт (передан в функцию)")
    else:
        p = Path(file_path)
        used_cached = False
        if p.suffix.lower() == ".json":
            # Пользователь дал json с транскриптом
            print("📝 Обнаружен JSON. Пытаюсь загрузить транскрипт...")
            segments = _load_segments_from_json(str(p))
            used_cached = segments is not None
        else:
            # Ищем ранее сохранённый транскрипт (legacy и новая схема папок)
            candidate = _find_cached_transcript_json(str(p), output_dir)
            if candidate and Path(candidate).is_file():
                print(f"📝 Найден сохранённый транскрипт: {candidate}")
                segments = _load_segments_from_json(candidate)
                used_cached = segments is not None

        if used_cached:
            print(f"✅ Загружено {len(segments)} сегментов из готового транскрипта")
        else:
            # 1. Транскрибируем аудио
            print("📝 Готового транскрипта нет. Транскрибирую аудио...")
            result = await asr.transcribe(file_path)
            if not result or "segments" not in result:
                print("❌ Ошибка: не удалось получить транскрипцию")
                return
            segments = result["segments"]
            print(f"✅ Транскрипция завершена. Получено {len(segments)} сегментов")
            # Сохраняем транскрипт немедленно, чтобы не потерять при сбое LLM
            out_dir_early = _prepare_output_dir(file_path, output_dir)
            stem = Path(file_path).stem
            transcript_file = out_dir_early / f"{stem}_transcript.json"
            try:
                with open(transcript_file, 'w', encoding='utf-8') as tf:
                    json.dump({"segments": segments}, tf, ensure_ascii=False, indent=2)
                print(f"💾 Транскрипт сохранён: {transcript_file}")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить транскрипт: {e}")
    
    # 2. Анализируем только ошибки кандидата
    print("🧠 Анализирую ошибки кандидата...")
    # Загрузим конфиг, чтобы прокинуть модели в аналайзеры
    cfg = load_config(None)
    models_cfg = cfg.get("models", {}) if isinstance(cfg, dict) else {}
    mistakes_model = models_cfg.get("mistakes") or "gemini-2.0-flash"
    qa_model = models_cfg.get("qa") or "gemini-2.0-flash"

    # Чанк-настройки из конфига (по умолчанию выключены)
    chunk_cfg = cfg.get("chunking", {}) if isinstance(cfg, dict) else {}
    m_chunk = (chunk_cfg.get("mistakes") or {}) if isinstance(chunk_cfg, dict) else {}
    q_chunk = (chunk_cfg.get("qa") or {}) if isinstance(chunk_cfg, dict) else {}

    m_enabled = bool(m_chunk.get("enabled", False))
    m_chunk_sec = int(m_chunk.get("chunk_duration_sec", 0) or 0) if m_enabled else None
    m_overlap = int(m_chunk.get("overlap_sec", 15) or 15)
    m_max_chunks = m_chunk.get("max_chunks")
    m_max_hint = m_chunk.get("max_issues_hint")

    mistakes_analyzer = MistakeAnalyzer(
        model_name=mistakes_model,
        chunk_duration_sec=m_chunk_sec,
        overlap_sec=m_overlap,
        max_chunks=m_max_chunks,
        max_issues_hint=m_max_hint,
    )
    mistakes_analysis = mistakes_analyzer.analyze_mistakes(segments)

    # 3. Генерируем только отчет по ошибкам
    print("📊 Генерирую отчет по ошибкам...")
    mistakes_report = mistakes_analyzer.generate_report(mistakes_analysis)

    # 4. Оценка ответов по вопросам интервьюера (QA)
    print("🧪 Оцениваю ответы по вопросам интервьюера...")
    q_enabled = bool(q_chunk.get("enabled", False))
    q_chunk_sec = int(q_chunk.get("chunk_duration_sec", 0) or 0) if q_enabled else None
    q_overlap = int(q_chunk.get("overlap_sec", 15) or 15)
    q_max_chunks = q_chunk.get("max_chunks")
    q_max_hint = q_chunk.get("max_items_hint")

    qa = QAEvaluator(
        model_name=qa_model,
        chunk_duration_sec=q_chunk_sec,
        overlap_sec=q_overlap,
        max_chunks=q_max_chunks,
        max_items_hint=q_max_hint,
    )
    candidate_speaker = (mistakes_analysis.assumptions or {}).get("candidate_speaker")
    qa_eval = qa.analyze_qa(segments, candidate_speaker=candidate_speaker)
    qa_report = qa.generate_report(qa_eval)

    # 5. Сохраняем результаты (ошибки + QA) в папку для файла
    out_dir = _prepare_output_dir(file_path, output_dir)
    stem = Path(file_path).stem

    mistakes_output_file = out_dir / f"{stem}_mistakes.md"
    with open(mistakes_output_file, 'w', encoding='utf-8') as f:
        f.write(mistakes_report)

    qa_output_file = out_dir / f"{stem}_qa_review.md"
    with open(qa_output_file, 'w', encoding='utf-8') as f:
        f.write(qa_report)

    # 5.1 Study guide (Hotspots + План прокачки) и CSV с Q&A
    def _build_study_guide(low_threshold: float = 6.0, max_qa_hotspots: int = 20) -> str:
        guide = ["# План прокачки по результатам собеседования\n"]

        # Hotspots: худшие ответы и критичные ошибки
        guide.append("\n## Hotspots\n")
        # QA hotspots
        qa_hot = [it for it in (qa_eval.items or []) if (it.correctness_score or 0) < low_threshold or (it.verdict or '').lower() in {"incorrect", "unclear"}]
        qa_hot = sorted(qa_hot, key=lambda x: (x.correctness_score, x.timestamp))[:max_qa_hotspots]
        if qa_hot:
            guide.append("### Слабые ответы (Q&A)\n")
            for it in qa_hot:
                guide.append(f"- [{it.timestamp}] {it.topic} ({it.difficulty}) — {it.correctness_score}/10 ({it.verdict})\n  Вопрос: {it.question}\n  Что ответил: {it.answer_summary}\n  Правильный ответ: {it.ideal_answer}\n  Подтянуть: {', '.join(it.what_to_improve or [])}\n  Чего не хватило: {', '.join(it.missing_points or [])}\n")
        # Mistakes hotspots
        high_mistakes = [m for m in (mistakes_analysis.issues or []) if (m.severity or '').lower() == 'high']
        if high_mistakes:
            guide.append("\n### Критичные ошибки\n")
            for m in high_mistakes:
                guide.append(f"- [{m.timestamp}] {m.type} — {m.explanation}\n  Фраза: \"{m.quote}\"\n  Правильно: {m.correction}\n  Теги: {', '.join(m.tags or [])}\n")

        # Темы для подтяжки по средним баллам
        guide.append("\n## Темы для подтяжки\n")
        topics = list(getattr(qa_eval.summary, 'by_topic', []) or [])
        try:
            topics_sorted = sorted(topics, key=lambda t: float(t.get('avg_score', 0)))
        except Exception:
            topics_sorted = topics
        worst_topics = topics_sorted[:3] if topics_sorted else []
        if worst_topics:
            for t in worst_topics:
                topic = t.get('topic', 'other')
                avg = t.get('avg_score', 0)
                guide.append(f"### {topic}: средний балл {avg}/10\n")
                # Соберём чеклист по теме из Q&A
                checklist = []
                for it in (qa_eval.items or []):
                    if it.topic == topic:
                        checklist.extend(it.what_to_improve or [])
                        checklist.extend(it.missing_points or [])
                # Дедуп и обрезка
                seen = set()
                uniq = []
                for x in checklist:
                    x = x.strip()
                    if not x or x.lower() in seen:
                        continue
                    seen.add(x.lower())
                    uniq.append(x)
                if uniq:
                    guide.append("Чеклист:\n")
                    for i, item in enumerate(uniq[:10], 1):
                        guide.append(f"- [ ] {item}\n")
                guide.append("\n")

        # Сильные стороны и ключевые пробелы из сводки
        if getattr(qa_eval.summary, 'top_strengths', None):
            guide.append("## Сильные стороны\n")
            for x in qa_eval.summary.top_strengths:
                guide.append(f"- {x}\n")
            guide.append("\n")
        if getattr(qa_eval.summary, 'key_gaps', None):
            guide.append("## Ключевые пробелы\n")
            for x in qa_eval.summary.key_gaps:
                guide.append(f"- {x}\n")
            guide.append("\n")

        return "".join(guide)

    # Порог можно переопределить через config.json/yaml (CLI его прокидывает в выходной JSON, а здесь оставляем дефолт)
    cfg = load_config(None)
    low_thr = float(cfg.get("qa_low_threshold", 6.0))
    max_hot = int((cfg.get("study_guide") or {}).get("max_qa_hotspots", 20))
    study_md = _build_study_guide(low_threshold=low_thr, max_qa_hotspots=max_hot)
    study_file = out_dir / f"{stem}_study_guide.md"
    with open(study_file, 'w', encoding='utf-8') as f:
        f.write(study_md)


    print("✅ Анализ завершен!")
    print(f"🚩 Ошибки и неточности сохранены в: {mistakes_output_file}")
    print(f"🧾 QA-оценка ответов сохранена в: {qa_output_file}")
    print(f"🧠 План прокачки сохранён в: {study_file}")

    # Краткий отчет в консоль по количеству ошибок
    by_sev = {"high": 0, "medium": 0, "low": 0}
    for it in (mistakes_analysis.issues or []):
        sev = (it.severity or '').lower()
        if sev not in by_sev:
            by_sev[sev] = 0
        by_sev[sev] += 1

    print("\n" + "=" * 50)
    print("КРАТКИЙ ОТЧЕТ ПО ОШИБКАМ")
    print("=" * 50)
    print(f"High: {by_sev.get('high', 0)} | Medium: {by_sev.get('medium', 0)} | Low: {by_sev.get('low', 0)}")
    print(f"Всего ошибок: {len(mistakes_analysis.issues or [])}")

    # Краткая сводка по QA
    print("\n" + "=" * 50)
    print("КРАТКАЯ QA-СВОДКА")
    print("=" * 50)
    print(f"Вопросов: {qa_eval.summary.total_questions} | Средний балл корректности: {qa_eval.summary.average_score:.1f}/10")
    if qa_eval.summary.by_topic:
        try:
            sorted_topics = sorted(qa_eval.summary.by_topic, key=lambda t: float(t.get('avg_score', 0)))
            worst = sorted_topics[:3]
            if worst:
                print("Темы для подтяжки:")
                for t in worst:
                    print(f"- {t.get('topic')}: {t.get('avg_score')}/10")
        except Exception:
            pass

    # Вернём объект с ошибками и добавим QA-оценку для дальнейшей сводки в CLI
    try:
        setattr(mistakes_analysis, 'qa_eval', qa_eval)
    except Exception:
        pass
    return mistakes_analysis
