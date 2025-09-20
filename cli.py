#!/usr/bin/env python3
"""
CLI интерфейс для анализатора собеседований
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from main import analyze_interview_file
from config import load_config

def main():
    parser = argparse.ArgumentParser(
        description="Анализатор собеседований - AI-агент для анализа качества собеседований",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python cli.py interview.mov                    # Анализ одного файла
  python cli.py *.mov                           # Анализ всех .mov файлов
  python cli.py --output-dir reports/ *.mp4     # Сохранение в папку reports/
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="Файлы для анализа (поддерживаются .mov, .mp4, .avi, .mkv, .wav, .mp3)"
    )
    
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Папка для сохранения результатов (по умолчанию - та же папка, что и файлы)"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        help="Путь к конфигу (yaml/json). По умолчанию: config.yaml|yml|json в корне"
    )
    
    parser.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-flash",
        help="Модель Google Gemini для анализа (по умолчанию: gemini-2.5-flash)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    # Проверяем наличие API ключей
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Ошибка: не найден GOOGLE_API_KEY в переменных окружения")
        print("Добавьте ваш API ключ в файл .env или установите переменную окружения")
        print("Получите ключ здесь: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    if not os.getenv("HF_TOKEN"):
        print("⚠️  Предупреждение: не найден HF_TOKEN для диаризации")
        print("Диаризация может работать некорректно")
    
    # Поддерживаемые форматы
    supported_formats = {'.mov', '.mp4', '.avi', '.mkv', '.wav', '.mp3', '.m4a', '.flac', '.json'}
    
    # Фильтруем файлы
    valid_files = []
    for file_pattern in args.files:
        path = Path(file_pattern)
        
        if path.is_file():
            if path.suffix.lower() in supported_formats:
                valid_files.append(str(path))
            else:
                print(f"⚠️  Пропускаю {path}: неподдерживаемый формат")
        elif path.is_dir():
            # Если указана папка, ищем файлы в ней
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                    valid_files.append(str(file_path))
        else:
            # Возможно, это glob pattern
            import glob
            for file_path in glob.glob(file_pattern):
                path_obj = Path(file_path)
                if path_obj.is_file() and path_obj.suffix.lower() in supported_formats:
                    valid_files.append(file_path)
    
    if not valid_files:
        print("❌ Не найдено файлов для анализа")
        sys.exit(1)
    
    print(f"🎯 Найдено {len(valid_files)} файлов для анализа:")
    for file_path in valid_files:
        print(f"  - {file_path}")
    
    # Конфиг и базовая папка результатов
    cfg = load_config(args.config)
    out_base = Path(args.output_dir or cfg.get("output_dir", "results"))
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"📁 Базовая папка результатов: {out_base}")
    
    # Анализируем файлы
    results = []
    # Агрегаторы по всем сессиям
    total_issues = total_high = total_medium = total_low = 0
    qa_total_questions = 0
    qa_sum_scores = 0.0
    qa_topics = {}  # topic -> {sum: float, count: float}
    qa_hotspots = 0
    qa_low_threshold = float(cfg.get("qa_low_threshold", 6.0))
    for i, file_path in enumerate(valid_files, 1):
        print(f"\n{'='*60}")
        print(f"📊 Анализ файла {i}/{len(valid_files)}: {Path(file_path).name}")
        print(f"{'='*60}")
        
        try:
            result = asyncio.run(analyze_interview_file(file_path, output_dir=str(out_base)))
            results.append((file_path, result))

            # Агрегируем ошибки по файлу
            issues = getattr(result, 'issues', []) or []
            high = sum(1 for it in issues if (it.severity or '').lower() == 'high')
            medium = sum(1 for it in issues if (it.severity or '').lower() == 'medium')
            low = sum(1 for it in issues if (it.severity or '').lower() == 'low')
            total_issues += len(issues)
            total_high += high
            total_medium += medium
            total_low += low

            # Агрегируем QA по результату (без JSON файлов)
            qa_eval = getattr(result, 'qa_eval', None)
            if qa_eval is not None:
                items = getattr(qa_eval, 'items', []) or []
                summary = getattr(qa_eval, 'summary', None)
                if items:
                    qa_total_questions += len(items)
                    for it in items:
                        sc = float(getattr(it, 'correctness_score', 0.0))
                        vd = str(getattr(it, 'verdict', '')).lower()
                        topic = str(getattr(it, 'topic', 'other'))
                        qa_sum_scores += sc
                        t = qa_topics.setdefault(topic, {"sum": 0.0, "count": 0.0})
                        t["sum"] += sc
                        t["count"] += 1
                        if sc < qa_low_threshold or vd in {"incorrect", "unclear"}:
                            qa_hotspots += 1
                elif summary is not None:
                    # Если по каким-то причинам нет items, используем summary
                    qn = int(getattr(summary, 'total_questions', 0) or 0)
                    qa_total_questions += qn
                    qa_sum_scores += float(getattr(summary, 'average_score', 0.0) or 0.0) * qn
                    for tinfo in (getattr(summary, 'by_topic', []) or []):
                        try:
                            topic = str(tinfo.get('topic', 'other'))
                            cnt = float(tinfo.get('count', 0) or 0)
                            avg = float(tinfo.get('avg_score', 0.0) or 0.0)
                            t = qa_topics.setdefault(topic, {"sum": 0.0, "count": 0.0})
                            t["sum"] += avg * cnt
                            t["count"] += cnt
                        except Exception:
                            continue

            if args.verbose and result:
                print(f"\n📈 Краткие результаты для {Path(file_path).name}:")
                print(f"   Ошибок: {len(issues)} (High: {high}, Medium: {medium}, Low: {low})")
                
        except Exception as e:
            print(f"❌ Ошибка при анализе {file_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Итоговая сводка
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СВОДКА ПО НЕСКОЛЬКИМ СЕССИЯМ")
    print(f"{'='*60}")
    
    successful = len([r for r in results if r[1] is not None])
    print(f"✅ Успешно проанализировано: {successful}/{len(valid_files)} файлов")

    if successful > 0:
        print(f"🚩 Ошибки суммарно: {total_issues} (High: {total_high}, Medium: {total_medium}, Low: {total_low})")
        if qa_total_questions > 0:
            qa_avg = qa_sum_scores / qa_total_questions
            print(f"❓ QA: вопросов: {qa_total_questions}, средний балл: {qa_avg:.1f}/10, hotspots: {qa_hotspots} (порог {qa_low_threshold})")
            # Топ-3 слабых темы
            topic_avgs = [
                (t, (v["sum"] / v["count"])) for t, v in qa_topics.items() if v["count"] > 0
            ]
            topic_avgs.sort(key=lambda x: x[1])
            worst = topic_avgs[:3]
            if worst:
                print("   Слабые темы:")
                for t, avg in worst:
                    print(f"   - {t}: {avg:.1f}/10")
    
    print(f"\n🎉 Анализ завершен! Проверьте папки результатов: {out_base}")

if __name__ == "__main__":
    main()
