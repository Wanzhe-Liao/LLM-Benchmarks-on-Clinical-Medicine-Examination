import statistics
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from exam.question import Question


PUBLICATION_STYLE = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'axes.unicode_minus': False,
}

OKABE_ITO_COLORS = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442',
    '#0072B2', '#D55E00', '#CC79A7', '#999999',
    '#000000', '#DDCC77', '#117733', '#882255'
]

SUBJECT_RANGES = {
    "生理/生化": [(1, 28), (116, 123), (136, 145)],
    "病理": [(29, 40), (124, 129), (146, 153)],
    "内科": [(41, 58), (68, 92), (130, 131), (154, 159)],
    "外科": [(59, 67), (93, 107), (132, 135), (160, 165)],
    "伦理/法规": [(108, 115)],
}

SUBJECT_ORDER = ["生理/生化", "病理", "内科", "外科", "伦理/法规"]


def get_subject(question_id: int) -> str:
    for subject, ranges in SUBJECT_RANGES.items():
        for start, end in ranges:
            if start <= question_id <= end:
                return subject
    return "其他"


class ResultVisualizer:
    def __init__(self, questions: List[Question], output_dir: str):
        self.questions = questions
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.q_map = {q.id: q for q in questions}

        for key, value in PUBLICATION_STYLE.items():
            matplotlib.rcParams[key] = value

    def generate_all(self, runs: List[Dict[str, Any]]):
        if not runs:
            return

        has_baseline = any(bool((r.get("baseline") or {})) for r in runs)
        has_agentic = any(bool((r.get("agentic") or {})) for r in runs)

        if has_baseline:
            self.plot_accuracy_bar(runs, mode_key="baseline", mode_label="Baseline")
            self.plot_type_heatmap(runs, mode_key="baseline", mode_label="Baseline")
            self.plot_subject_heatmap(runs, mode_key="baseline", mode_label="Baseline")
            self.plot_stability_boxplot(runs, mode_key="baseline", mode_label="Baseline")
            self.plot_question_model_heatmap(runs, mode_key="baseline", mode_label="Baseline")

        if has_agentic:
            self.plot_accuracy_bar(runs, mode_key="agentic", mode_label="Agentic")
            self.plot_type_heatmap(runs, mode_key="agentic", mode_label="Agentic")
            self.plot_subject_heatmap(runs, mode_key="agentic", mode_label="Agentic")
            self.plot_stability_boxplot(runs, mode_key="agentic", mode_label="Agentic")
            self.plot_question_model_heatmap(runs, mode_key="agentic", mode_label="Agentic")

    def _get_model_accuracies(self, runs: List[Dict], mode_key: str) -> Dict[str, List[float]]:
        model_accs: Dict[str, List[float]] = {}
        for run in runs:
            mode_results = run.get(mode_key) or {}
            for model_id, results in mode_results.items():
                if model_id not in model_accs:
                    model_accs[model_id] = []
                correct = sum(1 for r in results if self._is_correct(r))
                total = len(results)
                acc = correct / total if total > 0 else 0
                model_accs[model_id].append(acc)
        return model_accs

    def _is_correct(self, r) -> bool:
        if hasattr(r, 'is_correct'):
            return r.is_correct
        if isinstance(r, dict):
            return r.get("is_correct", False)
        return False

    def _get_question_id(self, r) -> int:
        if hasattr(r, 'question_id'):
            return r.question_id
        if isinstance(r, dict):
            return r.get("question_id", 0)
        return 0

    def _calculate_figure_width(self, n_models: int) -> float:
        if n_models <= 5:
            return 10
        elif n_models <= 10:
            return 14
        else:
            return max(18, n_models * 1.2)

    def _save_figure(self, filename_base: str):
        for fmt, dpi in [('png', 300), ('pdf', 300), ('eps', 600)]:
            path = self.charts_dir / f'{filename_base}.{fmt}'
            plt.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')

    def plot_accuracy_bar(self, runs: List[Dict], mode_key: str, mode_label: str):
        model_accs = self._get_model_accuracies(runs, mode_key=mode_key)
        if not model_accs:
            return

        models = list(model_accs.keys())
        n_models = len(models)
        means = [statistics.mean(model_accs[m]) * 100 for m in models]
        stds = [statistics.stdev(model_accs[m]) * 100 if len(model_accs[m]) > 1 else 0 for m in models]

        fig_width = self._calculate_figure_width(n_models)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        x = range(n_models)

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                     color=OKABE_ITO_COLORS[:n_models],
                     edgecolor='black', linewidth=0.8, alpha=0.9)

        ax.set_xlabel('模型')
        ax.set_ylabel('正确率 (%)')
        ax.set_title(f'{mode_label} 模型正确率对比（含标准差）')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45 if n_models > 5 else 0, ha='right')
        ax.set_ylim(0, 100)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1,
                    f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        self._save_figure(f'{mode_key}_accuracy_bar')
        plt.close()

    def plot_type_heatmap(self, runs: List[Dict], mode_key: str, mode_label: str):
        model_accs = self._get_model_accuracies(runs, mode_key=mode_key)
        if not model_accs:
            return

        models = list(model_accs.keys())
        n_models = len(models)
        types = ['A', 'B', 'X']
        type_accs = {m: {t: [] for t in types} for m in models}

        for run in runs:
            mode_results = run.get(mode_key) or {}
            for model_id, results in mode_results.items():
                for t in types:
                    type_results = []
                    for r in results:
                        q_id = self._get_question_id(r)
                        q = self.q_map.get(q_id)
                        if q and q.type == t:
                            type_results.append(1 if self._is_correct(r) else 0)
                    if type_results:
                        type_accs[model_id][t].append(sum(type_results) / len(type_results))

        data = []
        for m in models:
            row = []
            for t in types:
                if type_accs[m][t]:
                    row.append(statistics.mean(type_accs[m][t]) * 100)
                else:
                    row.append(0)
            data.append(row)

        fig_width = max(8, n_models * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, max(6, n_models * 0.5)))
        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(range(len(types)))
        ax.set_xticklabels([f'{t}型题' for t in types])
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=8 if n_models > 8 else 10)

        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

        for i in range(n_models):
            for j in range(len(types)):
                text = ax.text(j, i, f'{data[i][j]:.1f}%', ha='center', va='center',
                               color='white' if data[i][j] < 60 else 'black', fontsize=8)

        ax.set_title(f'{mode_label} 模型×题型正确率热力图')
        cbar = fig.colorbar(im, ax=ax, label='正确率 (%)')
        cbar.set_ticks([0, 25, 50, 75, 100])

        plt.tight_layout()
        self._save_figure(f'{mode_key}_type_heatmap')
        plt.close()

    def plot_subject_heatmap(self, runs: List[Dict], mode_key: str, mode_label: str):
        model_accs = self._get_model_accuracies(runs, mode_key=mode_key)
        if not model_accs:
            return

        models = list(model_accs.keys())
        n_models = len(models)
        subjects = SUBJECT_ORDER
        subj_accs = {m: {s: [] for s in subjects} for m in models}

        for run in runs:
            mode_results = run.get(mode_key) or {}
            for model_id, results in mode_results.items():
                for subject in subjects:
                    ranges = SUBJECT_RANGES[subject]
                    hits = []
                    for r in results:
                        q_id = self._get_question_id(r)
                        for start, end in ranges:
                            if start <= q_id <= end:
                                hits.append(1 if self._is_correct(r) else 0)
                                break
                    if hits:
                        subj_accs[model_id][subject].append(sum(hits) / len(hits))

        data = []
        for m in models:
            row = []
            for s in subjects:
                row.append(statistics.mean(subj_accs[m][s]) * 100 if subj_accs[m][s] else 0)
            data.append(row)

        fig_width = max(8, n_models * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(fig_width, max(6, n_models * 0.5)))
        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=8 if n_models > 8 else 10)

        for i in range(n_models):
            for j in range(len(subjects)):
                ax.text(j, i, f'{data[i][j]:.1f}%', ha='center', va='center',
                        color='white' if data[i][j] < 60 else 'black', fontsize=8)

        ax.set_title(f'{mode_label} 模型×学科正确率热力图')
        cbar = fig.colorbar(im, ax=ax, label='正确率 (%)')
        cbar.set_ticks([0, 25, 50, 75, 100])

        plt.tight_layout()
        self._save_figure(f'{mode_key}_subject_heatmap')
        plt.close()

    def plot_stability_boxplot(self, runs: List[Dict], mode_key: str, mode_label: str):
        model_accs = self._get_model_accuracies(runs, mode_key=mode_key)
        if not model_accs:
            return

        models = list(model_accs.keys())
        n_models = len(models)
        data = [[acc * 100 for acc in model_accs[m]] for m in models]

        fig_width = max(10, n_models * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        bp = ax.boxplot(data, patch_artist=True, labels=models,
                       medianprops=dict(color='darkred', linewidth=2),
                       boxprops=dict(alpha=0.7),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.2))

        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)])

        ax.set_xlabel('模型')
        ax.set_ylabel('正确率 (%)')
        ax.set_title(f'{mode_label} 多轮测试稳定性（{len(runs)}轮）')
        ax.set_ylim(0, 100)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        if n_models <= 10:
            legend_elements = [plt.Rectangle((0,0),1,1, fc=OKABE_ITO_COLORS[i % len(OKABE_ITO_COLORS)],
                                            edgecolor='black', alpha=0.7, label=models[i])
                              for i in range(n_models)]
            ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

        plt.xticks(rotation=45 if n_models > 5 else 0, ha='right')
        plt.tight_layout()
        self._save_figure(f'{mode_key}_stability_boxplot')
        plt.close()

    def plot_question_model_heatmap(self, runs: List[Dict], mode_key: str, mode_label: str):
        if not runs:
            return

        first_mode = runs[0].get(mode_key) or {}
        if not first_mode:
            return

        models = list(first_mode.keys())
        n_models = len(models)
        question_ids = sorted(self.q_map.keys())

        sorted_questions = []
        for subject in SUBJECT_ORDER:
            ranges = SUBJECT_RANGES[subject]
            for q_id in question_ids:
                for start, end in ranges:
                    if start <= q_id <= end:
                        sorted_questions.append(q_id)
                        break

        q_model_acc: Dict[int, Dict[str, float]] = {q: {} for q in sorted_questions}

        for q_id in sorted_questions:
            for model_id in models:
                correct_count = 0
                total_count = 0
                for run in runs:
                    mode_results = run.get(mode_key) or {}
                    results = mode_results.get(model_id, [])
                    for r in results:
                        r_qid = self._get_question_id(r)
                        if r_qid == q_id:
                            total_count += 1
                            if self._is_correct(r):
                                correct_count += 1
                            break
                q_model_acc[q_id][model_id] = correct_count / total_count if total_count > 0 else 0

        data = []
        for q_id in sorted_questions:
            row = [q_model_acc[q_id].get(m, 0) for m in models]
            data.append(row)

        fig_width = max(12, n_models * 1.2)
        fig_height = max(20, len(sorted_questions) * 0.12)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8 if n_models > 8 else 10)

        y_labels = []
        for q_id in sorted_questions:
            subject = get_subject(q_id)
            y_labels.append(f'{q_id}')

        tick_fontsize = 6 if len(sorted_questions) > 100 else 8
        ax.set_yticks(range(len(sorted_questions)))
        ax.set_yticklabels(y_labels, fontsize=tick_fontsize)

        subject_boundaries = []
        current_idx = 0
        for subject in SUBJECT_ORDER:
            ranges = SUBJECT_RANGES[subject]
            count = sum(1 for q in sorted_questions if any(start <= q <= end for start, end in ranges))
            if count > 0:
                mid_idx = current_idx + count // 2
                subject_boundaries.append((current_idx, current_idx + count - 1, mid_idx, subject))
                current_idx += count

        for start_idx, end_idx, mid_idx, subject in subject_boundaries:
            if start_idx > 0:
                ax.axhline(y=start_idx - 0.5, color='black', linewidth=2)
            ax.text(-0.5, mid_idx, subject, ha='right', va='center',
                   fontsize=10, fontweight='bold', rotation=0)

        ax.set_title(f'{mode_label} 题目×模型正确率热力图（按学科分组）')
        ax.set_xlabel('模型')
        ax.set_ylabel('题目ID')

        cbar = fig.colorbar(im, ax=ax, label='正确率', shrink=0.5)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0%', '50%', '100%'])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        self._save_figure(f'{mode_key}_question_model_heatmap')
        plt.close()

        # Cold spots (hard questions): lowest mean accuracy across models
        q_mean = []
        for q_id in sorted_questions:
            row = [q_model_acc[q_id].get(m, 0) for m in models]
            q_mean.append((q_id, sum(row) / len(row) if row else 0))
        q_mean.sort(key=lambda x: x[1])

        cold_path = self.output_dir / f"cold_spots_{mode_key}.md"
        lines = [f"# 冷点题目（{mode_label}）", "按题目在所有模型上的平均正确率从低到高排序。", ""]
        lines.append("| 题号 | 学科 | 题型 | 平均正确率 |")
        lines.append("|------|------|------|------------|")
        for q_id, acc in q_mean[:30]:
            q = self.q_map.get(q_id)
            q_type = q.type if q else "?"
            subject = get_subject(q_id)
            lines.append(f"| {q_id} | {subject} | {q_type} | {acc*100:.1f}% |")

        cold_path.write_text("\n".join(lines), encoding="utf-8")
