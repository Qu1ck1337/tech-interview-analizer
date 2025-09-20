import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()


@dataclass
class Issue:
    type: str
    timestamp: str
    quote: str
    explanation: str
    correction: str
    severity: str
    tags: List[str] = field(default_factory=list)


@dataclass
class MistakesAnalysis:
    issues: List[Issue]
    assumptions: Dict[str, Any] = field(default_factory=dict)


class MistakeAnalyzer:
    """Анализатор ошибок и неточностей в ответах кандидата."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        *,
        chunk_duration_sec: Optional[int] = None,
        overlap_sec: int = 15,
        max_chunks: Optional[int] = None,
        max_issues_hint: Optional[int] = None,
    ):
        print(model_name)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.chunk_duration_sec = chunk_duration_sec if (chunk_duration_sec or 0) > 0 else None
        self.overlap_sec = max(0, overlap_sec)
        self.max_chunks = max_chunks
        self.max_issues_hint = max_issues_hint

        base_rules = (
            "Ты — опытный технический интервьюер и тимлид. Твоя задача — выявить все ошибки, "
            "неточности, пробелы и спорные моменты в ответах кандидата. Работай ТОЛЬКО по высказываниям "
            "кандидата (не интервьюера). Если нужно, сначала определи, какой SPEAKER — кандидат.\n\n"
            "Правила: \n"
            "- Приводи точный таймкод и цитату кандидата для каждой проблемы.\n"
            "- Кратко объясняй, что не так, и давай корректную формулировку/ответ.\n"
            "- Классифицируй по типам (пример): factual_error, wrong_complexity, incorrect_definition, "
            "missed_requirement, architecture_flaw, security_issue, incomplete_answer, contradiction, "
            "terminology_misuse, unclear_explanation, performance_misconception, db_mistake, concurrency_mistake, testing_gap.\n"
            "- Укажи серьёзность: high | medium | low.\n"
            "- Добавь 1-3 тематических тега (например: algorithms, system_design, python, db, security).\n\n"
        )

        constraints = ""
        if isinstance(self.max_issues_hint, int) and self.max_issues_hint > 0:
            constraints = (
                f"Ограничения: приведи не более {self.max_issues_hint} пунктов (issues). "
                "Делай поля explanation/correction лаконичными (до 200 символов).\n"
            )

        schema = (
            "Всегда возвращай СТРОГО ВАЛИДНЫЙ JSON: без markdown, без комментариев, без хвостового текста.\n\n"
            "Ответ строго в JSON без пояснений и без markdown. Схема:\n"
            "{\n"
            "  \"assumptions\": { \"candidate_speaker\": \"SPEAKER_01\", \"confidence\": 0.9 },\n"
            "  \"issues\": [\n"
            "    {\n"
            "      \"type\": \"factual_error\",\n"
            "      \"timestamp\": \"MM:SS\",\n"
            "      \"quote\": \"точная цитата кандидата\",\n"
            "      \"explanation\": \"почему это неверно\",\n"
            "      \"correction\": \"как правильно\",\n"
            "      \"severity\": \"high|medium|low\",\n"
            "      \"tags\": [\"algorithms\", \"python\"]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        self.system_prompt = base_rules + constraints + schema

    def _format_transcript(self, segments: List[Dict]) -> str:
        """Форматирует сегменты транскрипции в читаемый текст."""
        formatted = []
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = (segment.get("text", "") or "").strip()
            start_time = segment.get("start", 0)
            if not text:
                continue
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            formatted.append(f"[{time_str}] {speaker}: {text}")
        return "\n".join(formatted)

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
        last_time = window_start
        for seg in segments:
            st = float(seg.get("start", 0) or 0)
            last_time = st
            if st - window_start < chunk_sec:
                current.append(seg)
            else:
                if current:
                    chunks.append(current)
                # новая оконная рамка с перекрытием
                window_start = st - max(0, overlap_sec)
                current = [seg]
        if current:
            chunks.append(current)
        return chunks

    def _dedup_issues(self, items: List[Issue]) -> List[Issue]:
        seen: set[Tuple[str, str]] = set()
        out: List[Issue] = []
        for it in items:
            key = (it.timestamp.strip(), (it.quote or "").strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    def _extract_json_from_response(self, response_content: str) -> str:
        """Пытается извлечь JSON-тело из ответа LLM, убирая markdown и лишний текст."""
        content = (response_content or "").strip()
        # Убираем возможные markdown-кодовые блоки
        json_start = content.find('```json')
        if json_start != -1:
            content = content[json_start + 7:]
        else:
            json_start = content.find('```')
            if json_start != -1:
                content = content[json_start + 3:]
        json_end = content.rfind('```')
        if json_end != -1:
            content = content[:json_end]
        content = content.strip()

        # Если в тексте есть префиксы/суффиксы, попробуем выделить сбалансированный JSON-объект/массив
        def _find_balanced_json_block(s: str) -> str:
            # Ищем либо объект {...}, либо массив [...]
            for opener, closer in (("{", "}"), ("[", "]")):
                start = s.find(opener)
                if start == -1:
                    continue
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(s)):
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
                                return s[start : i + 1].strip()
                # если не нашли закрывающую скобку — пробуем следующий тип
            return s.strip()

        candidate = _find_balanced_json_block(content)
        return candidate

    def analyze_mistakes(self, segments: List[Dict]) -> MistakesAnalysis:
        """Возвращает список ошибок и неточностей кандидата.

        Если задан chunk_duration_sec, разбивает транскрипт на окна и объединяет результаты без потерь.
        """

        def _parse_or_retry(raw_text: str, transcript_chunk: str):
            # 1) Пытаемся распарсить как есть (после извлечения)
            content = self._extract_json_from_response(raw_text)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            # 2) Пытаемся найти сбалансированный блок в полном ответе
            try:
                balanced = self._extract_json_from_response(raw_text)
                return json.loads(balanced)
            except json.JSONDecodeError:
                pass

            # 3) Повторный запрос с более строгими ограничениями
            retry_system = (
                "СТРОГО верни ВАЛИДНЫЙ JSON (без markdown). Не более 20 issues."
                " Все строки короткие (<200 символов). Если не умещается — сократи."
            )
            retry_messages = [
                SystemMessage(content=self.system_prompt + "\n\n" + retry_system),
                HumanMessage(content=(
                    "Ниже тот же транскрипт. Верни корректный JSON согласно схеме.\n\n"
                    f"{transcript_chunk}"
                )),
            ]
            retry_resp = self.llm.invoke(retry_messages)
            retry_content = self._extract_json_from_response(retry_resp.content)
            return json.loads(retry_content)

        # Обычный режим (без чанков)
        if not self.chunk_duration_sec:
            transcript = self._format_transcript(segments)
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=(
                    "Ниже транскрипт собеседования. Найди и перечисли проблемы только в ответах кандидата.\n\n"
                    f"{transcript}"
                )),
            ]
            response = self.llm.invoke(messages)
            try:
                data = _parse_or_retry(response.content, transcript)
                raw_issues = data.get("issues")
                if raw_issues is None and isinstance(data, list):
                    raw_issues = data
                if raw_issues is None:
                    raw_issues = []
                issues: List[Issue] = []
                for it in raw_issues:
                    if not isinstance(it, dict):
                        continue
                    issues.append(
                        Issue(
                            type=str(it.get("type", "unspecified")),
                            timestamp=str(it.get("timestamp", "00:00")),
                            quote=str(it.get("quote", "")),
                            explanation=str(it.get("explanation", "")),
                            correction=str(it.get("correction", "")),
                            severity=str(it.get("severity", "medium")),
                            tags=[str(t) for t in (it.get("tags") or [])],
                        )
                    )
                assumptions = data.get("assumptions") if isinstance(data, dict) else {}
                if not isinstance(assumptions, dict):
                    assumptions = {}
                return MistakesAnalysis(issues=issues, assumptions=assumptions)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON в MistakeAnalyzer: {e}")
                print(f"Ответ LLM: {response.content[:400]}...")
                return MistakesAnalysis(issues=[], assumptions={})

        # Чанковый режим
        all_issues: List[Issue] = []
        best_assumptions: Dict[str, Any] = {}
        best_conf = -1.0
        chunks = self._chunk_segments_by_time(segments, self.chunk_duration_sec or 0, self.overlap_sec)
        if self.max_chunks:
            chunks = chunks[: self.max_chunks]
        for idx, seg_chunk in enumerate(chunks, 1):
            transcript = self._format_transcript(seg_chunk)
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=(
                    f"Это часть {idx} из {len(chunks)} транскрипта. Анализируй ТОЛЬКО эту часть. "
                    "Найди и перечисли проблемы только в ответах кандидата. Верни JSON.\n\n"
                    f"{transcript}"
                )),
            ]
            try:
                response = self.llm.invoke(messages)
                data = _parse_or_retry(response.content, transcript)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ошибка парсинга JSON в MistakeAnalyzer (chunk {idx}): {e}")
                continue

            raw_issues = data.get("issues")
            if raw_issues is None and isinstance(data, list):
                raw_issues = data
            if raw_issues is None:
                raw_issues = []
            for it in raw_issues:
                if not isinstance(it, dict):
                    continue
                all_issues.append(
                    Issue(
                        type=str(it.get("type", "unspecified")),
                        timestamp=str(it.get("timestamp", "00:00")),
                        quote=str(it.get("quote", "")),
                        explanation=str(it.get("explanation", "")),
                        correction=str(it.get("correction", "")),
                        severity=str(it.get("severity", "medium")),
                        tags=[str(t) for t in (it.get("tags") or [])],
                    )
                )

            if isinstance(data, dict):
                ass = data.get("assumptions") or {}
                if isinstance(ass, dict):
                    conf = float(ass.get("confidence", 0) or 0)
                    if conf > best_conf and ass.get("candidate_speaker"):
                        best_conf = conf
                        best_assumptions = ass

        all_issues = self._dedup_issues(all_issues)
        return MistakesAnalysis(issues=all_issues, assumptions=best_assumptions)

    def generate_report(self, analysis: MistakesAnalysis) -> str:
        """Генерирует Markdown-отчет по ошибкам и неточностям."""
        issues = analysis.issues or []
        # Группируем по серьёзности
        sev_order = {"high": 0, "medium": 1, "low": 2}
        issues_sorted = sorted(issues, key=lambda x: (sev_order.get(x.severity.lower(), 1), x.timestamp))

        # Подсчёты
        by_sev: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        by_type: Dict[str, int] = {}
        for i in issues_sorted:
            by_sev[i.severity.lower()] = by_sev.get(i.severity.lower(), 0) + 1
            by_type[i.type] = by_type.get(i.type, 0) + 1

        report = "# Ошибки и неточности в ответах кандидата\n\n"
        if analysis.assumptions.get("candidate_speaker"):
            cs = analysis.assumptions.get("candidate_speaker")
            conf = analysis.assumptions.get("confidence")
            if conf is not None:
                report += f"Предположительный кандидат: {cs} (доверие {conf})\n\n"
            else:
                report += f"Предположительный кандидат: {cs}\n\n"

        report += "## Итоговая сводка\n"
        report += f"- High: {by_sev['high']}\n"
        report += f"- Medium: {by_sev['medium']}\n"
        report += f"- Low: {by_sev['low']}\n\n"

        if by_type:
            report += "### По типам\n"
            for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
                report += f"- {t}: {c}\n"
            report += "\n"

        def fmt_issue(it: Issue) -> str:
            base = f"- [{it.timestamp}] {it.type} ({it.severity}) — {it.explanation}\n"
            if it.quote:
                base += f"  Фраза: \"{it.quote}\"\n"
            if it.correction:
                base += f"  Правильно: {it.correction}\n"
            if it.tags:
                base += f"  Теги: {', '.join(it.tags)}\n"
            return base

        sections = [
            ("High", [i for i in issues_sorted if i.severity.lower() == "high"]),
            ("Medium", [i for i in issues_sorted if i.severity.lower() == "medium"]),
            ("Low", [i for i in issues_sorted if i.severity.lower() == "low"]),
        ]

        for title, items in sections:
            if not items:
                continue
            report += f"## {title} severity\n"
            for it in items:
                report += fmt_issue(it)
            report += "\n"

        if not issues_sorted:
            report += "Нет явных ошибок или неточностей, отличная работа!\n"

        return report
