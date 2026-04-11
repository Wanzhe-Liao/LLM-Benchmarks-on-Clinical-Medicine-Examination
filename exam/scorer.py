import re
from typing import Tuple


def extract_answer(response: str) -> str:
    text = response.strip()

    # Normalize common fullwidth letters used by some models (ＡＢＣＤ / ａｂｃｄ)
    text = text.translate(str.maketrans({
        "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D",
        "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d",
    }))

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\*+([^*]+)\*+', r'\1', text)
    text = text.strip()

    patterns = [
        r'答案[：:]\s*([A-Da-d]+)',
        r'【答案】\s*([A-Da-d]+)',
        r'选择?\s*([A-Da-d]+)',
        r'在给定的选项中[，,：:\s]*([A-Da-d]+)',
        r'在上述选项中[，,：:\s]*([A-Da-d]+)',
        r'([A-Da-d])[\(（][^\)）]{0,80}[\)）]?\s*(?:是|为)?\s*(?:最合适|最佳|最恰当|最可能)',
        r'^([A-Da-d]+)$',
        r'([A-Da-d]+)\s*$',
        r'正确[答选][案项][是为]?\s*([A-Da-d]+)',
        r'[Aa]nswer[:\s]+([A-Da-d]+)',
        r'[Tt]he\s+answer\s+is\s+([A-Da-d]+)',
        r'[Oo]ption\s+([A-Da-d]+)',
        r'"(?:final_)?answer"\s*:\s*"([A-Da-d]+)"',
        r'\(([A-Da-d]+)\)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            return ''.join(sorted(set(answer)))

    letters = re.findall(r'[A-Da-d]', text)
    if letters:
        unique_letters = []
        for letter in letters:
            upper_letter = letter.upper()
            if upper_letter not in unique_letters:
                unique_letters.append(upper_letter)
        if len(unique_letters) <= 4:
            return ''.join(sorted(unique_letters))

    return ''


def score_answer(predicted: str, correct: str) -> Tuple[bool, float]:
    predicted = ''.join(sorted(set(predicted.upper())))
    correct = ''.join(sorted(set(correct.upper())))

    is_correct = predicted == correct
    score = 1.0 if is_correct else 0.0

    return is_correct, score


def calculate_accuracy(results: list) -> dict:
    if not results:
        return {'total': 0.0, 'correct': 0, 'total_count': 0}

    def _is_correct(r) -> bool:
        if hasattr(r, "is_correct"):
            return bool(getattr(r, "is_correct"))
        if isinstance(r, dict):
            return bool(r.get("is_correct", False))
        return False

    correct_count = sum(1 for r in results if _is_correct(r))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return {
        'total': accuracy,
        'correct': correct_count,
        'total_count': total_count,
        'accuracy_percent': f"{accuracy * 100:.1f}%"
    }


def calculate_accuracy_by_type(results: list, questions: list) -> dict:
    q_type_map = {q.id: q.type for q in questions}

    type_results = {'A': [], 'B': [], 'X': []}
    for r in results:
        if hasattr(r, "question_id"):
            qid = getattr(r, "question_id")
        elif isinstance(r, dict):
            qid = r.get("question_id", 0)
        else:
            qid = 0
        q_type = q_type_map.get(int(qid or 0), 'A')
        type_results[q_type].append(r)

    return {
        'A': calculate_accuracy(type_results['A']),
        'B': calculate_accuracy(type_results['B']),
        'X': calculate_accuracy(type_results['X']),
        'total': calculate_accuracy(results)
    }
