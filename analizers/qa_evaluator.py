import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()


@dataclass
class QAItem:
    timestamp: str
    topic: str
    difficulty: str
    question: str
    answer_summary: str
    correctness_score: float
    verdict: str
    ideal_answer: str = ""
    what_to_improve: List[str] = field(default_factory=list)
    missing_points: List[str] = field(default_factory=list)
    followup_recommendations: List[str] = field(default_factory=list)


@dataclass
class QASummary:
    average_score: float
    total_questions: int
    by_topic: List[Dict[str, Any]] = field(default_factory=list)
    top_strengths: List[str] = field(default_factory=list)
    key_gaps: List[str] = field(default_factory=list)


@dataclass
class QAEvaluation:
    items: List[QAItem]
    summary: QASummary
    assumptions: Dict[str, Any] = field(default_factory=dict)


class QAEvaluator:
    """Выделяет вопросы интервьюера и оценивает ответы кандидата по темам."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        *,
        chunk_duration_sec: Optional[int] = None,
        overlap_sec: int = 15,
        max_chunks: Optional[int] = None,
        max_items_hint: Optional[int] = None,
    ):
        print(model_name)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        self.chunk_duration_sec = chunk_duration_sec if (chunk_duration_sec or 0) > 0 else None
        self.overlap_sec = max(0, overlap_sec)
        self.max_chunks = max_chunks
        self.max_items_hint = max_items_hint

        # Системная инструкция: извлечь Q&A пары, оценить ответы, сгруппировать по темам
        base = (
            "Ты — ведущий технический интервьюер. Выдели вопросы интервьюера, сверь их с ответами кандидата и оцени ответы.\n"
            "Определи тему вопроса, сложность, кратко резюмируй ответ кандидата, поставь оценку корректности (0-10),\n"
            "дай вердикт (correct | partially_correct | incorrect | unclear), укажи 1–3 направления для улучшения,\n"
            "и добавь эталонный ответ (ideal_answer) — как следовало бы ответить.\n\n"
        )
        constraints = ""
        if isinstance(self.max_items_hint, int) and self.max_items_hint > 0:
            constraints = (
                f"Ограничения: выделяй не более {self.max_items_hint} Q&A элементов. "
                "Тексты держи краткими (до 200 символов).\n"
            )
        schema = (
            "Всегда возвращай СТРОГО ВАЛИДНЫЙ JSON без markdown и без комментариев.\n\n"
            "Ответ строго в JSON без markdown. Схема:\n"
            "{\n"
            "  \"assumptions\": { \"candidate_speaker\": \"SPEAKER_01\", \"interviewer_speakers\": [\"SPEAKER_00\"], \"confidence\": 0.9 },\n"
            "  \"summary\": {\n"
            "     \"average_score\": 7.2,\n"
            "     \"total_questions\": 12,\n"
            "     \"by_topic\": [ {\"topic\": \"system_design\", \"count\": 4, \"avg_score\": 6.5, \"priority\": \"high|medium|low\"} ],\n"
            "     \"top_strengths\": [""],\n"
            "     \"key_gaps\": ["" ]\n"
            "  },\n"
            "  \"items\": [\n"
            "    {\n"
            "      \"timestamp\": \"MM:SS\",\n"
            "      \"topic\": \"algorithms|system_design|python|db|infra|security|ml|devops|testing|communication|other\",\n"
            "      \"difficulty\": \"easy|medium|hard\",\n"
            "      \"question\": \"вопрос интервьюера\",\n"
            "      \"answer_summary\": \"краткое содержание ответа кандидата\",\n"
            "      \"ideal_answer\": \"краткая эталонная формулировка правильного ответа\",\n"
            "      \"correctness_score\": 0-10,\n"
            "      \"verdict\": \"correct|partially_correct|incorrect|unclear\",\n"
            "      \"what_to_improve\": [""],\n"
            "      \"missing_points\": [""],\n"
            "      \"followup_recommendations\": ["" ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )
        self.system_prompt = base + constraints + schema

    def _format_transcript(self, segments: List[Dict]) -> str:
        lines = []
        for s in segments:
            sp = s.get("speaker", "Unknown")
            txt = (s.get("text", "") or "").strip()
            start = s.get("start", 0)
            if not txt:
                continue
            m = int(start // 60)
            sec = int(start % 60)
            lines.append(f"[{m:02d}:{sec:02d}] {sp}: {txt}")
        return "\n".join(lines)

    def _chunk_segments_by_time(
        self, segments: List[Dict], chunk_sec: int, overlap_sec: int
    ) -> List[List[Dict]]:
        if not segments or chunk_sec <= 0:
            return [segments]
        chunks: List[List[Dict]] = []
        current: List[Dict] = []
        if not segments:
            return chunks
        window_start = float(segments[0].get("start", 0))
        for seg in segments:
            st = float(seg.get("start", 0) or 0)
            if st - window_start < chunk_sec:
                current.append(seg)
            else:
                if current:
                    chunks.append(current)
                window_start = st - max(0, overlap_sec)
                current = [seg]
        if current:
            chunks.append(current)
        return chunks

    def _dedup_items(self, items: List[QAItem]) -> List[QAItem]:
        seen: set[Tuple[str, str]] = set()
        out: List[QAItem] = []
        for it in items:
            key = (it.timestamp.strip(), (it.question or "").strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    def _extract_json_from_response(self, response_content: str) -> str:
        content = (response_content or "").strip()
        # Снимем обрамление кодовыми блоками, если есть
        start = content.find("```json")
        if start != -1:
            content = content[start + 7 :]
        else:
            start = content.find("```")
            if start != -1:
                content = content[start + 3 :]
        end = content.rfind("```")
        if end != -1:
            content = content[:end]
        content = content.strip()

        # Попробуем выделить сбалансированный JSON-блок
        def _find_balanced_json_block(s: str) -> str:
            for opener, closer in (("{", "}"), ("[", "]")):
                st = s.find(opener)
                if st == -1:
                    continue
                depth = 0
                in_str = False
                esc = False
                for i in range(st, len(s)):
                    ch = s[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == opener:
                            depth += 1
                        elif ch == closer:
                            depth -= 1
                            if depth == 0:
                                return s[st : i + 1].strip()
            return s.strip()

        return _find_balanced_json_block(content)

    def analyze_qa(self, segments: List[Dict], candidate_speaker: Optional[str] = None) -> QAEvaluation:
        def _analyze_chunked(segments_local: List[Dict]) -> QAEvaluation:
            # Используем конфиг, а если он не задан — безопасные дефолты
            eff_chunk_sec = self.chunk_duration_sec or 480
            eff_overlap = self.overlap_sec if hasattr(self, 'overlap_sec') else 20
            eff_max_chunks = self.max_chunks if hasattr(self, 'max_chunks') else None

            all_items: List[QAItem] = []
            best_assumptions: Dict[str, Any] = {}
            best_conf = -1.0
            chunks = self._chunk_segments_by_time(segments_local, eff_chunk_sec, eff_overlap)
            if eff_max_chunks:
                chunks = chunks[: eff_max_chunks]

            for idx, seg_chunk in enumerate(chunks, 1):
                transcript_chunk = self._format_transcript(seg_chunk)
                extra_hint_chunk = (
                    f"Кандидат — {candidate_speaker}. Оценивай только ответы кандидата относительно вопросов интервьюера."
                    if candidate_speaker
                    else "Определи самостоятельно, кто кандидат и кто интервьюер."
                )
                messages_chunk = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(
                        content=(
                            f"Это часть {idx} из {len(chunks)}. Выдели Q&A и оцени ответы по этой части. Верни JSON.\n"
                            + extra_hint_chunk
                            + "\n\n"
                            + transcript_chunk
                        )
                    ),
                ]
                try:
                    response_chunk = self.llm.invoke(messages_chunk)
                    data_chunk = _parse_or_retry(response_chunk.content, transcript_chunk)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Ошибка парсинга JSON в QAEvaluator (chunk {idx}): {e}")
                    continue

                items_raw_chunk = (data_chunk.get("items") if isinstance(data_chunk, dict) else []) or []
                for it in items_raw_chunk:
                    if not isinstance(it, dict):
                        continue
                    all_items.append(
                        QAItem(
                            timestamp=str(it.get("timestamp", "00:00")),
                            topic=str(it.get("topic", "other")),
                            difficulty=str(it.get("difficulty", "medium")),
                            question=str(it.get("question", "")),
                            answer_summary=str(it.get("answer_summary", "")),
                            correctness_score=float(it.get("correctness_score", 0)),
                            verdict=str(it.get("verdict", "unclear")),
                            ideal_answer=str(it.get("ideal_answer", "")),
                            what_to_improve=[str(x) for x in (it.get("what_to_improve") or [])],
                            missing_points=[str(x) for x in (it.get("missing_points") or [])],
                            followup_recommendations=[str(x) for x in (it.get("followup_recommendations") or [])],
                        )
                    )

                ass_chunk = data_chunk.get("assumptions") if isinstance(data_chunk, dict) else {}
                if isinstance(ass_chunk, dict):
                    conf = float(ass_chunk.get("confidence", 0) or 0)
                    if conf > best_conf and ass_chunk.get("candidate_speaker"):
                        best_conf = conf
                        best_assumptions = ass_chunk

            # Дедуп и сводка
            def _dedup_items(items: List[QAItem]) -> List[QAItem]:
                seen: set[Tuple[str, str]] = set()
                out: List[QAItem] = []
                for it in items:
                    key = (it.timestamp.strip(), (it.question or "").strip())
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(it)
                return out

            all_items = _dedup_items(all_items)
            # Построим summary локально
            if all_items:
                total = len(all_items)
                avg = sum(float(it.correctness_score or 0.0) for it in all_items) / max(1, total)
                summary = QASummary(average_score=round(avg, 2), total_questions=total)
            else:
                summary = QASummary(average_score=0.0, total_questions=0)
            return QAEvaluation(items=all_items, summary=summary, assumptions=best_assumptions)
        def _build_summary(items: List[QAItem]) -> QASummary:
            if not items:
                return QASummary(average_score=0.0, total_questions=0)
            total = len(items)
            avg = sum(float(it.correctness_score or 0.0) for it in items) / max(1, total)
            by: Dict[str, Dict[str, float]] = {}
            for it in items:
                t = by.setdefault(it.topic or "other", {"sum": 0.0, "count": 0.0})
                t["sum"] += float(it.correctness_score or 0.0)
                t["count"] += 1
            by_topic = []
            for topic, v in by.items():
                avg_t = v["sum"] / max(1.0, v["count"])
                pr = "high" if avg_t < 6.0 else ("medium" if avg_t < 7.5 else "low")
                by_topic.append({"topic": topic, "count": int(v["count"]), "avg_score": round(avg_t, 2), "priority": pr})
            return QASummary(average_score=round(avg, 2), total_questions=total, by_topic=by_topic)

        def _parse_or_retry(raw_text: str, transcript_chunk: str):
            content = self._extract_json_from_response(raw_text)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
            # Вторая попытка с жёсткими ограничениями
            retry_sys = (
                "СТРОГО верни ВАЛИДНЫЙ JSON (без markdown). Не более 20 элементов items."
                " Все строки короткие (<200 символов). Если не умещается — сократи."
            )
            retry_messages = [
                SystemMessage(content=self.system_prompt + "\n\n" + retry_sys),
                HumanMessage(content=(
                    "Ниже тот же транскрипт. Верни корректный JSON согласно схеме.\n\n"
                    + transcript_chunk
                )),
            ]
            retry_resp = self.llm.invoke(retry_messages)
            retry_content = self._extract_json_from_response(retry_resp.content)
            return json.loads(retry_content)

        # Без чанков — прежний путь
        if not self.chunk_duration_sec:
            transcript = self._format_transcript(segments)
            extra_hint = (
                f"Кандидат — {candidate_speaker}. Оценивай только ответы кандидата относительно вопросов интервьюера."
                if candidate_speaker
                else "Определи самостоятельно, кто кандидат и кто интервьюер."
            )
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=(
                        "Ниже транскрипт собеседования. Выдели вопросы интервьюера, оцени ответы кандидата и верни JSON.\n"
                        + extra_hint
                        + "\n\n"
                        + transcript
                    )
                ),
            ]
            response = self.llm.invoke(messages)
            try:
                data = _parse_or_retry(response.content, transcript)
                # Если модель вернула не объект, а список элементов — поддержим это
                if isinstance(data, list):
                    items_raw = data
                    summary_raw = {}
                    assumptions = {}
                else:
                    items_raw = data.get("items") or []
                    summary_raw = data.get("summary") or {}
                    assumptions = data.get("assumptions") if isinstance(data, dict) else {}

                items: List[QAItem] = []
                for it in items_raw:
                    if not isinstance(it, dict):
                        continue
                    items.append(
                        QAItem(
                            timestamp=str(it.get("timestamp", "00:00")),
                            topic=str(it.get("topic", "other")),
                            difficulty=str(it.get("difficulty", "medium")),
                            question=str(it.get("question", "")),
                            answer_summary=str(it.get("answer_summary", "")),
                            correctness_score=float(it.get("correctness_score", 0)),
                            verdict=str(it.get("verdict", "unclear")),
                            ideal_answer=str(it.get("ideal_answer", "")),
                            what_to_improve=[str(x) for x in (it.get("what_to_improve") or [])],
                            missing_points=[str(x) for x in (it.get("missing_points") or [])],
                            followup_recommendations=[str(x) for x in (it.get("followup_recommendations") or [])],
                        )
                    )

                if summary_raw:
                    summary = QASummary(
                        average_score=float(summary_raw.get("average_score", 0.0)),
                        total_questions=int(summary_raw.get("total_questions", len(items))),
                        by_topic=list(summary_raw.get("by_topic", [])),
                        top_strengths=[str(x) for x in (summary_raw.get("top_strengths") or [])],
                        key_gaps=[str(x) for x in (summary_raw.get("key_gaps") or [])],
                    )
                else:
                    # Если summary нет — посчитаем краткую сводку
                    total = len(items)
                    avg = sum(float(it.correctness_score or 0.0) for it in items) / max(1, total)
                    summary = QASummary(average_score=round(avg, 2), total_questions=total)

                if not isinstance(assumptions, dict):
                    assumptions = {}

                # Если получили пустой результат — падаем в чанк‑режим по дефолтным настройкам
                if not items:
                    return _analyze_chunked(segments)
                return QAEvaluation(items=items, summary=summary, assumptions=assumptions)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON в QAEvaluator: {e}")
                print(f"Ответ LLM: {response.content[:400]}...")
                # Попробуем чанк‑режим как fallback
                return _analyze_chunked(segments)

        # Чанковый режим: собираем items по окнам, summary считаем локально
        all_items: List[QAItem] = []
        best_assumptions: Dict[str, Any] = {}
        best_conf = -1.0
        chunks = self._chunk_segments_by_time(segments, self.chunk_duration_sec or 0, self.overlap_sec)
        if self.max_chunks:
            chunks = chunks[: self.max_chunks]

        for idx, seg_chunk in enumerate(chunks, 1):
            transcript = self._format_transcript(seg_chunk)
            extra_hint = (
                f"Кандидат — {candidate_speaker}. Оценивай только ответы кандидата относительно вопросов интервьюера."
                if candidate_speaker
                else "Определи самостоятельно, кто кандидат и кто интервьюер."
            )
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=(
                        f"Это часть {idx} из {len(chunks)}. Выдели Q&A и оцени ответы по этой части. Верни JSON.\n"
                        + extra_hint
                        + "\n\n"
                        + transcript
                    )
                ),
            ]
            try:
                response = self.llm.invoke(messages)
                data = _parse_or_retry(response.content, transcript)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON в QAEvaluator (chunk {idx}): {e}")
                continue

            items_raw = (data.get("items") if isinstance(data, dict) else []) or []
            for it in items_raw:
                if not isinstance(it, dict):
                    continue
                all_items.append(
                    QAItem(
                        timestamp=str(it.get("timestamp", "00:00")),
                        topic=str(it.get("topic", "other")),
                        difficulty=str(it.get("difficulty", "medium")),
                        question=str(it.get("question", "")),
                        answer_summary=str(it.get("answer_summary", "")),
                        correctness_score=float(it.get("correctness_score", 0)),
                        verdict=str(it.get("verdict", "unclear")),
                        ideal_answer=str(it.get("ideal_answer", "")),
                        what_to_improve=[str(x) for x in (it.get("what_to_improve") or [])],
                        missing_points=[str(x) for x in (it.get("missing_points") or [])],
                        followup_recommendations=[str(x) for x in (it.get("followup_recommendations") or [])],
                    )
                )

            ass = data.get("assumptions") if isinstance(data, dict) else {}
            if isinstance(ass, dict):
                conf = float(ass.get("confidence", 0) or 0)
                if conf > best_conf and ass.get("candidate_speaker"):
                    best_conf = conf
                    best_assumptions = ass

        all_items = self._dedup_items(all_items)
        summary = _build_summary(all_items)
        return QAEvaluation(items=all_items, summary=summary, assumptions=best_assumptions)

    def generate_report(self, evaluation: QAEvaluation) -> str:
        report = "# Оценка ответов по вопросам интервьюера\n\n"
        s = evaluation.summary
        report += "## Краткая сводка\n"
        report += f"- Всего вопросов: {s.total_questions}\n"
        report += f"- Средняя корректность: {s.average_score:.1f}/10\n\n"
        if s.by_topic:
            report += "### По темам (средние баллы)\n"
            for t in s.by_topic:
                topic = t.get("topic", "other")
                cnt = t.get("count", 0)
                avg = t.get("avg_score", 0)
                pr = t.get("priority", "")
                pr_str = f", приоритет: {pr}" if pr else ""
                report += f"- {topic}: {avg}/10 (вопросов: {cnt}{pr_str})\n"
            report += "\n"

        if s.top_strengths:
            report += "### Сильные стороны\n"
            for i, x in enumerate(s.top_strengths, 1):
                report += f"{i}. {x}\n"
            report += "\n"

        if s.key_gaps:
            report += "### Зоны роста\n"
            for i, x in enumerate(s.key_gaps, 1):
                report += f"{i}. {x}\n"
            report += "\n"

        # Детализация по темам
        # Группируем
        by_topic: Dict[str, List[QAItem]] = {}
        for it in evaluation.items:
            by_topic.setdefault(it.topic, []).append(it)

        for topic, items in by_topic.items():
            report += f"## Тема: {topic}\n"
            for it in items:
                report += (
                    f"- [{it.timestamp}] {it.difficulty} — Оценка: {it.correctness_score}/10 ({it.verdict})\n"
                )
                if it.question:
                    report += f"  Вопрос: {it.question}\n"
                if it.answer_summary:
                    report += f"  Ответ (кратко): {it.answer_summary}\n"
                if it.ideal_answer:
                    report += f"  Правильный ответ: {it.ideal_answer}\n"
                if it.missing_points:
                    report += f"  Чего не хватило: {', '.join(it.missing_points)}\n"
                if it.what_to_improve:
                    report += f"  Подтянуть: {', '.join(it.what_to_improve)}\n"
                if it.followup_recommendations:
                    report += f"  Рекомендуемые темы/практика: {', '.join(it.followup_recommendations)}\n"
            report += "\n"

        if not evaluation.items:
            report += "Нет распознанных вопросов/ответов для оценки.\n"

        return report
