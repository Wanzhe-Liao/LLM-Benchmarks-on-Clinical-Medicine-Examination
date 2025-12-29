import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from exam.question import Question, TestResult
from .analyzer import ResultAnalyzer
from .visualizer import ResultVisualizer, SUBJECT_ORDER, SUBJECT_RANGES, get_subject


class ReportGenerator:
    def __init__(self, questions: List[Question], output_dir: str):
        self.questions = questions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = ResultAnalyzer(questions)

    def generate_report(
        self,
        baseline_results: Dict[str, List[TestResult]],
        agentic_results: Optional[Dict[str, List[TestResult]]] = None,
    ) -> str:
        lines: List[str] = []
        lines.append("# LLM医学考试测试报告\n")
        lines.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**题目总数**: {len(self.questions)}题\n")
        model_count = len(baseline_results) if baseline_results else (len(agentic_results) if agentic_results else 0)
        lines.append(f"**测试模型**: {model_count}个\n")
        lines.append("---\n")

        if baseline_results:
            lines.append("## 一、单体（Baseline）正确率\n")
            lines.append(self._generate_accuracy_table(baseline_results))

        if agentic_results is not None:
            header_idx = "二" if baseline_results else "一"
            lines.append(f"\n## {header_idx}、多智能体（Agentic）正确率\n")
            lines.append(self._generate_accuracy_table(agentic_results))

            if baseline_results:
                lines.append("\n## 三、模式对比（Agentic vs Baseline）\n")
                lines.append(self._generate_comparison_table(baseline_results, agentic_results))

        primary = agentic_results if agentic_results is not None else baseline_results
        lines.append("\n## 四、答案一致性分析\n")
        lines.append(self._generate_consistency_section(primary))

        consistency = self.analyzer.analyze_consistency(primary)
        if consistency.all_wrong:
            lines.append("\n## 五、共同错题列表\n")
            lines.append(self._generate_wrong_questions_section(primary, consistency.all_wrong))

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return str(report_path)

    def generate_multi_run_report(
        self,
        runs: List[Dict[str, Dict[str, List[TestResult]]]],
    ) -> str:
        """
        runs: [{ "baseline": {...}, "agentic": {...} }, ...]
        """
        lines: List[str] = []
        lines.append("# LLM医学考试多轮测试报告\n")
        lines.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**轮数**: {len(runs)}\n")
        lines.append(f"**题目总数**: {len(self.questions)}题\n")
        lines.append("---\n")

        if not runs:
            report_path = self.output_dir / "report.md"
            report_path.write_text("\n".join(lines), encoding="utf-8")
            return str(report_path)

        first_baseline = runs[0].get("baseline") or {}
        first_agentic = runs[0].get("agentic") or {}
        has_baseline = bool(first_baseline)
        has_agentic = runs[0].get("agentic") is not None and bool(first_agentic)
        model_ids = list(first_baseline.keys()) if has_baseline else list(first_agentic.keys())

        def accs(mode_key: str, model_id: str) -> List[float]:
            values: List[float] = []
            for run in runs:
                mode_results = run.get(mode_key) or {}
                if model_id not in mode_results:
                    continue
                values.append(self.analyzer.analyze_model(mode_results[model_id]).accuracy)
            return values

        def fmt_mean_sd(values: List[float]) -> str:
            if not values:
                return "0.0%"
            mean = statistics.mean(values) * 100
            sd = statistics.stdev(values) * 100 if len(values) > 1 else 0.0
            return f"{mean:.1f}%±{sd:.1f}%" if len(values) > 1 else f"{mean:.1f}%"

        def _get_question_id(r) -> int:
            if hasattr(r, "question_id"):
                return int(getattr(r, "question_id") or 0)
            if isinstance(r, dict):
                return int(r.get("question_id", 0) or 0)
            return 0

        def _is_correct(r) -> bool:
            if hasattr(r, "is_correct"):
                return bool(getattr(r, "is_correct"))
            if isinstance(r, dict):
                return bool(r.get("is_correct", False))
            return False

        def _is_strict_complete_agentic(r) -> bool:
            if r is None:
                return False
            if hasattr(r, "error"):
                if getattr(r, "error") is not None:
                    return False
                predicted = str(getattr(r, "predicted_answer", "") or "")
                meta = getattr(r, "meta", None) or {}
            elif isinstance(r, dict):
                if r.get("error") is not None:
                    return False
                predicted = str(r.get("predicted_answer", "") or "")
                meta = r.get("meta") or {}
            else:
                return False

            if not predicted.strip():
                return False
            if not isinstance(meta, dict):
                return False

            for role in ("planner", "reasoner", "critic"):
                blob = meta.get(role) or {}
                if not isinstance(blob, dict):
                    return False
                if blob.get("error") is not None:
                    return False
                if blob.get("repair_error") is not None:
                    return False
                parsed = blob.get("parsed")
                if not isinstance(parsed, dict) or not parsed:
                    return False
            return True

        lines.append("## 一、多轮平均正确率（Mean±SD）\n")
        header = ["模型"]
        if has_baseline:
            header.append("Baseline")
        if has_agentic:
            header.append("Agentic")
        if has_baseline and has_agentic:
            header.append("Improvement(Mean)")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["------"] * len(header)) + "|")

        for model_id in model_ids:
            row = [model_id]
            base = accs("baseline", model_id) if has_baseline else []
            ag = accs("agentic", model_id) if has_agentic else []

            if has_baseline:
                row.append(fmt_mean_sd(base))
            if has_agentic:
                row.append(fmt_mean_sd(ag))
            if has_baseline and has_agentic:
                imp = (statistics.mean(ag) - statistics.mean(base)) * 100 if ag and base else 0.0
                sign = "+" if imp >= 0 else ""
                row.append(f"{sign}{imp:.1f}%")

            lines.append("| " + " | ".join(row) + " |")

        if has_agentic:
            lines.append("\n## 二、严格口径运行完整率（Agentic）\n")
            lines.append("| 模型 | 完整条目/总条目 | 完整率 |")
            lines.append("|------|----------------|--------|")
            for model_id in model_ids:
                complete = 0
                total = 0
                for run in runs:
                    mode_results = run.get("agentic") or {}
                    rs = mode_results.get(model_id) or []
                    total += len(rs)
                    complete += sum(1 for r in rs if _is_strict_complete_agentic(r))
                rate = (complete / total * 100) if total else 0.0
                lines.append(f"| {model_id} | {complete}/{total} | {rate:.1f}% |")

            lines.append("\n## 三、JSON修复使用率（Agentic）\n")
            lines.append("| 模型 | 修复条目/总条目 | 修复率 |")
            lines.append("|------|----------------|--------|")
            for model_id in model_ids:
                repaired = 0
                total = 0
                for run in runs:
                    mode_results = run.get("agentic") or {}
                    rs = mode_results.get(model_id) or []
                    total += len(rs)
                    for r in rs:
                        meta = None
                        if hasattr(r, "meta"):
                            meta = getattr(r, "meta", None) or {}
                        elif isinstance(r, dict):
                            meta = r.get("meta") or {}
                        if not isinstance(meta, dict):
                            continue
                        any_repaired = any(bool((meta.get(role) or {}).get("repaired")) for role in ("planner", "reasoner", "critic"))
                        if any_repaired:
                            repaired += 1
                rate = (repaired / total * 100) if total else 0.0
                lines.append(f"| {model_id} | {repaired}/{total} | {rate:.1f}% |")

        def _type_mean_sd(mode_key: str, model_id: str, q_type: str) -> str:
            values: List[float] = []
            for run in runs:
                mode_results = run.get(mode_key) or {}
                rs = mode_results.get(model_id) or []
                if not rs:
                    continue
                stats = self.analyzer.analyze_model(rs)
                values.append(float(stats.by_type.get(q_type, {}).get("total", 0.0)))
            return fmt_mean_sd(values)

        if has_baseline and not has_agentic:
            lines.append("\n## 二、按题型平均正确率（Baseline, Mean±SD）\n")
            lines.append("| 模型 | A型题 | B型题 | X型题 | 总计 |")
            lines.append("|------|-------|-------|-------|------|")
            for model_id in model_ids:
                lines.append(
                    f"| {model_id} | {_type_mean_sd('baseline', model_id, 'A')} | {_type_mean_sd('baseline', model_id, 'B')} | {_type_mean_sd('baseline', model_id, 'X')} | {fmt_mean_sd(accs('baseline', model_id))} |"
                )

        if has_agentic:
            header_idx = "四" if has_agentic else "三"
            lines.append("\n## 四、按题型平均正确率（Agentic, Mean±SD）\n")
            lines.append("| 模型 | A型题 | B型题 | X型题 | 总计 |")
            lines.append("|------|-------|-------|-------|------|")
            for model_id in model_ids:
                lines.append(
                    f"| {model_id} | {_type_mean_sd('agentic', model_id, 'A')} | {_type_mean_sd('agentic', model_id, 'B')} | {_type_mean_sd('agentic', model_id, 'X')} | {fmt_mean_sd(accs('agentic', model_id))} |"
                )

        if has_baseline:
            lines.append("\n## 五、按题型平均正确率（Baseline, Mean±SD）\n")
            lines.append("| 模型 | A型题 | B型题 | X型题 | 总计 |")
            lines.append("|------|-------|-------|-------|------|")
            for model_id in model_ids:
                lines.append(
                    f"| {model_id} | {_type_mean_sd('baseline', model_id, 'A')} | {_type_mean_sd('baseline', model_id, 'B')} | {_type_mean_sd('baseline', model_id, 'X')} | {fmt_mean_sd(accs('baseline', model_id))} |"
                )

        if has_baseline and has_agentic:
            import statistics as _stats

            def _type_mean(mode_key: str, model_id: str, q_type: str) -> float:
                values: List[float] = []
                for run in runs:
                    mode_results = run.get(mode_key) or {}
                    rs = mode_results.get(model_id) or []
                    if not rs:
                        continue
                    stats = self.analyzer.analyze_model(rs)
                    values.append(float(stats.by_type.get(q_type, {}).get("total", 0.0)))
                return float(_stats.mean(values)) if values else 0.0

            lines.append("\n## 六、按题型提升（Agentic - Baseline, Mean）\n")
            lines.append("| 模型 | A型题Δ | B型题Δ | X型题Δ | 总计Δ |")
            lines.append("|------|--------|--------|--------|-------|")
            for model_id in model_ids:
                dA = (_type_mean("agentic", model_id, "A") - _type_mean("baseline", model_id, "A")) * 100
                dB = (_type_mean("agentic", model_id, "B") - _type_mean("baseline", model_id, "B")) * 100
                dX = (_type_mean("agentic", model_id, "X") - _type_mean("baseline", model_id, "X")) * 100
                dT = (_stats.mean(accs("agentic", model_id)) - _stats.mean(accs("baseline", model_id))) * 100
                fmt = lambda x: f"{x:+.1f}%"
                lines.append(f"| {model_id} | {fmt(dA)} | {fmt(dB)} | {fmt(dX)} | {fmt(dT)} |")

        def _subject_acc(rs: List, subject: str) -> Optional[float]:
            ranges = SUBJECT_RANGES[subject]
            hits = []
            for r in rs:
                qid = _get_question_id(r)
                for start, end in ranges:
                    if start <= qid <= end:
                        hits.append(1 if _is_correct(r) else 0)
                        break
            if not hits:
                return None
            return sum(hits) / len(hits)

        def _subject_mean_sd(mode_key: str, model_id: str, subject: str) -> str:
            values: List[float] = []
            for run in runs:
                mode_results = run.get(mode_key) or {}
                rs = mode_results.get(model_id) or []
                if not rs:
                    continue
                v = _subject_acc(rs, subject)
                if v is not None:
                    values.append(float(v))
            return fmt_mean_sd(values)

        if has_baseline and not has_agentic:
            lines.append("\n## 三、按学科平均正确率（Baseline, Mean±SD）\n")
            subj_cols = SUBJECT_ORDER
            lines.append("| 模型 | " + " | ".join(subj_cols) + " | 总计 |")
            lines.append("|" + "|".join(["------"] * (len(subj_cols) + 2)) + "|")
            for model_id in model_ids:
                parts = [model_id] + [_subject_mean_sd("baseline", model_id, s) for s in subj_cols] + [fmt_mean_sd(accs("baseline", model_id))]
                lines.append("| " + " | ".join(parts) + " |")

            total_by_q: Dict[int, int] = {}
            correct_by_q: Dict[int, int] = {}
            for run in runs:
                mode_results = run.get("baseline") or {}
                for _, rs in mode_results.items():
                    for r in rs:
                        qid = _get_question_id(r)
                        if qid <= 0:
                            continue
                        total_by_q[qid] = total_by_q.get(qid, 0) + 1
                        if _is_correct(r):
                            correct_by_q[qid] = correct_by_q.get(qid, 0) + 1

            scored = []
            for qid, total in total_by_q.items():
                acc = (correct_by_q.get(qid, 0) / total) if total else 0.0
                q = self.questions[qid - 1] if 1 <= qid <= len(self.questions) else None
                q_type = q.type if q else "?"
                subject = get_subject(qid)
                scored.append((acc, qid, subject, q_type))
            scored.sort(key=lambda x: x[0])

            lines.append("\n## 四、冷点题目（Baseline，平均正确率最低）\n")
            lines.append("完整列表见 `cold_spots_baseline.md`。\n")
            lines.append("| 排名 | 题号 | 学科 | 题型 | 平均正确率 |")
            lines.append("|------|------|------|------|------------|")
            for i, (acc, qid, subject, q_type) in enumerate(scored[:15], 1):
                lines.append(f"| {i} | {qid} | {subject} | {q_type} | {acc*100:.1f}% |")

        if has_agentic:
            lines.append("\n## 七、按学科平均正确率（Agentic, Mean±SD）\n")
            subj_cols = SUBJECT_ORDER
            lines.append("| 模型 | " + " | ".join(subj_cols) + " | 总计 |")
            lines.append("|" + "|".join(["------"] * (len(subj_cols) + 2)) + "|")
            for model_id in model_ids:
                parts = [model_id] + [_subject_mean_sd("agentic", model_id, s) for s in subj_cols] + [fmt_mean_sd(accs("agentic", model_id))]
                lines.append("| " + " | ".join(parts) + " |")

        if has_baseline:
            lines.append("\n## 八、按学科平均正确率（Baseline, Mean±SD）\n")
            subj_cols = SUBJECT_ORDER
            lines.append("| 模型 | " + " | ".join(subj_cols) + " | 总计 |")
            lines.append("|" + "|".join(["------"] * (len(subj_cols) + 2)) + "|")
            for model_id in model_ids:
                parts = [model_id] + [_subject_mean_sd("baseline", model_id, s) for s in subj_cols] + [fmt_mean_sd(accs("baseline", model_id))]
                lines.append("| " + " | ".join(parts) + " |")

        if has_baseline and has_agentic:
            import statistics as _stats

            def _subj_mean(mode_key: str, model_id: str, subject: str) -> float:
                values: List[float] = []
                for run in runs:
                    mode_results = run.get(mode_key) or {}
                    rs = mode_results.get(model_id) or []
                    if not rs:
                        continue
                    v = _subject_acc(rs, subject)
                    if v is not None:
                        values.append(float(v))
                return float(_stats.mean(values)) if values else 0.0

            lines.append("\n## 九、按学科提升（Agentic - Baseline, Mean）\n")
            subj_cols = SUBJECT_ORDER
            lines.append("| 模型 | " + " | ".join([f"{s}Δ" for s in subj_cols]) + " | 总计Δ |")
            lines.append("|" + "|".join(["------"] * (len(subj_cols) + 2)) + "|")
            for model_id in model_ids:
                deltas = [(_subj_mean("agentic", model_id, s) - _subj_mean("baseline", model_id, s)) * 100 for s in subj_cols]
                total_delta = (_stats.mean(accs("agentic", model_id)) - _stats.mean(accs("baseline", model_id))) * 100
                fmt = lambda x: f"{x:+.1f}%"
                parts = [model_id] + [fmt(x) for x in deltas] + [fmt(total_delta)]
                lines.append("| " + " | ".join(parts) + " |")

            # Cold spots: hardest questions overall across all models+runs
            total_by_q: Dict[int, int] = {}
            correct_by_q: Dict[int, int] = {}
            for run in runs:
                mode_results = run.get("agentic") or {}
                for _, rs in mode_results.items():
                    for r in rs:
                        qid = _get_question_id(r)
                        if qid <= 0:
                            continue
                        total_by_q[qid] = total_by_q.get(qid, 0) + 1
                        if _is_correct(r):
                            correct_by_q[qid] = correct_by_q.get(qid, 0) + 1

            scored = []
            for qid, total in total_by_q.items():
                acc = (correct_by_q.get(qid, 0) / total) if total else 0.0
                q = self.questions[qid - 1] if 1 <= qid <= len(self.questions) else None
                q_type = q.type if q else "?"
                subject = get_subject(qid)
                scored.append((acc, qid, subject, q_type))
            scored.sort(key=lambda x: x[0])

            lines.append("\n## 五、冷点题目（Agentic，平均正确率最低）\n")
            lines.append("完整列表见 `cold_spots_agentic.md`。\n")
            lines.append("| 排名 | 题号 | 学科 | 题型 | 平均正确率 |")
            lines.append("|------|------|------|------|------------|")
            for i, (acc, qid, subject, q_type) in enumerate(scored[:15], 1):
                lines.append(f"| {i} | {qid} | {subject} | {q_type} | {acc*100:.1f}% |")

        report_path = self.output_dir / "report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        visualizer = ResultVisualizer(self.questions, self.output_dir)
        visualizer.generate_all(runs)

        return str(report_path)

    def save_results(
        self,
        baseline_results: Dict[str, List[TestResult]],
        agentic_results: Optional[Dict[str, List[TestResult]]] = None,
    ):
        def serialize(results: Dict[str, List[TestResult]]) -> Dict[str, List[Dict]]:
            return {
                model_id: [
                    {
                        "question_id": r.question_id,
                        "model_id": r.model_id,
                        "mode": r.mode,
                        "predicted_answer": r.predicted_answer,
                        "correct_answer": r.correct_answer,
                        "is_correct": r.is_correct,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                        "meta": r.meta,
                        "raw_response": r.raw_response if not r.predicted_answer else None,
                    }
                    for r in model_results
                ]
                for model_id, model_results in results.items()
            }

        payload: Dict[str, Any] = {"baseline": serialize(baseline_results)}
        if agentic_results is not None:
            payload["agentic"] = serialize(agentic_results)

        path = self.output_dir / "results.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.save_per_model_results(baseline_results, agentic_results)
        self._save_metadata(baseline_results, agentic_results)

    def save_multi_run_results(self, runs: List[Dict[str, Dict[str, List[TestResult]]]]):
        def serialize(results: Dict[str, List[TestResult]]) -> Dict[str, List[Dict]]:
            return {
                model_id: [
                    {
                        "question_id": r.question_id,
                        "model_id": r.model_id,
                        "mode": r.mode,
                        "predicted_answer": r.predicted_answer,
                        "correct_answer": r.correct_answer,
                        "is_correct": r.is_correct,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                        "meta": r.meta,
                        "raw_response": r.raw_response if not r.predicted_answer else None,
                    }
                    for r in model_results
                ]
                for model_id, model_results in results.items()
            }

        out = []
        for idx, run in enumerate(runs, 1):
            entry = {"run": idx}
            if "baseline" in run and run["baseline"] is not None:
                entry["baseline"] = serialize(run["baseline"])
            if "agentic" in run and run["agentic"] is not None:
                entry["agentic"] = serialize(run["agentic"])
            out.append(entry)

        path = self.output_dir / "runs.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def _generate_accuracy_table(self, results: Dict[str, List[TestResult]]) -> str:
        lines: List[str] = []
        lines.append("| 模型 | A型题 | B型题 | X型题 | 总正确率 |")
        lines.append("|------|-------|-------|-------|----------|")

        for model_id, model_results in results.items():
            stats = self.analyzer.analyze_model(model_results)
            a_acc = stats.by_type.get("A", {}).get("total", 0) * 100
            b_acc = stats.by_type.get("B", {}).get("total", 0) * 100
            x_acc = stats.by_type.get("X", {}).get("total", 0) * 100
            total_acc = stats.accuracy * 100
            lines.append(f"| {model_id} | {a_acc:.1f}% | {b_acc:.1f}% | {x_acc:.1f}% | {total_acc:.1f}% |")

        return "\n".join(lines)

    def _generate_comparison_table(
        self,
        baseline: Dict[str, List[TestResult]],
        agentic: Dict[str, List[TestResult]],
    ) -> str:
        lines: List[str] = []
        lines.append("| 模型 | Baseline | Agentic | 提升 | 答案变化 | 变好 | 变差 |")
        lines.append("|------|----------|--------|------|----------|------|------|")

        for model_id in baseline:
            if model_id not in agentic:
                continue
            comp = self.analyzer.compare_modes(baseline[model_id], agentic[model_id])
            sign = "+" if comp.improvement >= 0 else ""
            lines.append(
                f"| {model_id} | {comp.baseline_accuracy*100:.1f}% | {comp.agentic_accuracy*100:.1f}% | "
                f"{sign}{comp.improvement*100:.1f}% | {comp.changed_answers} | {comp.improved_answers} | {comp.degraded_answers} |"
            )

        return "\n".join(lines)

    def _generate_consistency_section(self, results: Dict[str, List[TestResult]]) -> str:
        consistency = self.analyzer.analyze_consistency(results)

        lines: List[str] = []
        lines.append(f"- 所有模型一致正确: **{len(consistency.all_correct)}**题")
        lines.append(f"- 所有模型一致错误: **{len(consistency.all_wrong)}**题")

        if consistency.most_divisive:
            lines.append("\n### 答案分歧最大的题目\n")
            lines.append("| 题号 | 正确模型数 |")
            lines.append("|------|------------|")
            for q_id, correct_count in consistency.most_divisive[:5]:
                lines.append(f"| {q_id} | {correct_count}/{len(results)} |")

        return "\n".join(lines)

    def _generate_wrong_questions_section(self, results: Dict[str, List[TestResult]], wrong_ids: List[int]) -> str:
        details = self.analyzer.get_wrong_questions_detail(results, wrong_ids)

        lines: List[str] = []
        model_ids = list(results.keys())

        header = "| 题号 | 题型 | 正确答案 | " + " | ".join(model_ids) + " |"
        separator = "|------|------|----------|" + "|".join(["------"] * len(model_ids)) + "|"
        lines.append(header)
        lines.append(separator)

        for detail in details:
            answers = [detail["model_answers"].get(m, "-") for m in model_ids]
            row = f"| {detail['id']} | {detail['type']} | {detail['correct_answer']} | " + " | ".join(answers) + " |"
            lines.append(row)

        return "\n".join(lines)

    def save_per_model_results(self, baseline_results: Dict[str, List[TestResult]],
                               agentic_results: Optional[Dict[str, List[TestResult]]] = None):
        models_dir = self.output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        for model_id in baseline_results.keys():
            model_dir = models_dir / model_id
            model_dir.mkdir(exist_ok=True)

            self._save_model_file(model_dir / "baseline.json", baseline_results[model_id])

            comp = None
            if agentic_results and model_id in agentic_results:
                self._save_model_file(model_dir / "agentic.json", agentic_results[model_id])
                comp = self.analyzer.compare_modes(baseline_results[model_id], agentic_results[model_id])
                self._save_comparison_file(model_dir, comp)

            stats = self.analyzer.analyze_model(baseline_results[model_id])
            self._save_stats_file(model_dir, model_id, stats)
            self._generate_model_report(model_dir, model_id, stats, comp)

    def _save_model_file(self, file_path: Path, results: List[TestResult]):
        data = [
            {
                "question_id": r.question_id,
                "model_id": r.model_id,
                "mode": r.mode,
                "predicted_answer": r.predicted_answer,
                "correct_answer": r.correct_answer,
                "is_correct": r.is_correct,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "meta": r.meta,
                "raw_response": r.raw_response if not r.predicted_answer else None,
            }
            for r in results
        ]
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_comparison_file(self, model_dir: Path, comp):
        data = {
            "model_id": comp.model_id,
            "baseline_accuracy": comp.baseline_accuracy,
            "agentic_accuracy": comp.agentic_accuracy,
            "improvement": comp.improvement,
            "changed_answers": comp.changed_answers,
            "improved_answers": comp.improved_answers,
            "degraded_answers": comp.degraded_answers,
        }
        (model_dir / "comparison.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _save_stats_file(self, model_dir: Path, model_id: str, stats):
        data = {
            "model_id": stats.model_id,
            "total_correct": stats.total_correct,
            "total_count": stats.total_count,
            "accuracy": stats.accuracy,
            "by_type": {
                type_key: {
                    "correct": type_data.get("correct", 0),
                    "total": type_data.get("total", 0),
                }
                for type_key, type_data in stats.by_type.items()
            }
        }
        (model_dir / "stats.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _generate_model_report(self, model_dir: Path, model_id: str, stats, comp=None):
        lines: List[str] = []
        lines.append(f"# {model_id} 模型测试报告\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**题目总数**: {stats.total_count}题\n")
        lines.append(f"**正确题数**: {stats.total_correct}题\n")
        lines.append(f"**总正确率**: {stats.accuracy * 100:.1f}%\n")
        lines.append("---\n")

        lines.append("## 按题型正确率\n")
        lines.append("| 题型 | 正确数 | 总数 | 正确率 |")
        lines.append("|------|-------|------|--------|")
        for type_key, type_data in stats.by_type.items():
            correct = type_data.get("correct", 0)
            total = type_data.get("total", 0)
            acc = type_data.get("total", 0) * 100
            lines.append(f"| {type_key}型题 | {correct} | {total} | {acc:.1f}% |")

        if comp:
            lines.append("\n## 模式对比（Agentic vs Baseline）\n")
            lines.append(f"- Baseline正确率: **{comp.baseline_accuracy * 100:.1f}%**")
            lines.append(f"- Agentic正确率: **{comp.agentic_accuracy * 100:.1f}%**")
            sign = "+" if comp.improvement >= 0 else ""
            lines.append(f"- 提升: **{sign}{comp.improvement * 100:.1f}%**")
            lines.append(f"- 答案变化数: {comp.changed_answers}")
            lines.append(f"- 变好: {comp.improved_answers}")
            lines.append(f"- 变差: {comp.degraded_answers}")

        (model_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def _save_metadata(self, baseline_results: Dict[str, List[TestResult]],
                      agentic_results: Optional[Dict[str, List[TestResult]]] = None):
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(self.questions),
            "models": list(baseline_results.keys()),
            "modes": ["baseline"] + (["agentic"] if agentic_results else []),
            "file_structure": {
                "results.json": "全量数据（向后兼容）",
                "models/[model_id]/baseline.json": "该模型的baseline结果",
                "models/[model_id]/agentic.json": "该模型的agentic结果（如果有）",
                "models/[model_id]/comparison.json": "baseline vs agentic对比",
                "models/[model_id]/stats.json": "该模型的统计信息",
                "models/[model_id]/report.md": "该模型的独立报告",
                "charts/": "可视化图表目录（PNG/PDF/EPS格式）",
            },
            "summary": {}
        }

        for model_id in baseline_results.keys():
            stats_base = self.analyzer.analyze_model(baseline_results[model_id])
            model_summary = {
                "baseline": {
                    "accuracy": stats_base.accuracy,
                    "total_correct": stats_base.total_correct,
                    "total_count": stats_base.total_count,
                }
            }

            if agentic_results and model_id in agentic_results:
                stats_ag = self.analyzer.analyze_model(agentic_results[model_id])
                model_summary["agentic"] = {
                    "accuracy": stats_ag.accuracy,
                    "total_correct": stats_ag.total_correct,
                    "total_count": stats_ag.total_count,
                }
                comp = self.analyzer.compare_modes(baseline_results[model_id], agentic_results[model_id])
                model_summary["improvement"] = comp.improvement

            metadata["summary"][model_id] = model_summary

        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
