import re
from typing import List, Dict, Optional, Tuple
from .question import Question


class ExamParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.questions: List[Question] = []

    def parse(self) -> List[Question]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.questions = []
        self.answers = self._parse_answer_table(content)
        self._parse_a_type(content)
        self._parse_b_type(content)
        self._parse_x_type(content)

        self.questions.sort(key=lambda q: q.id)
        return self.questions

    def _parse_answer_table(self, content: str) -> Dict[int, str]:
        answers = {}
        table_match = re.search(r'## 参考答案\s*\n(.*?)$', content, re.DOTALL)
        if table_match:
            table_content = table_match.group(1)
            row_pattern = r'\|\s*(\d+)\s*\|\s*([A-D]+)\s*'
            for match in re.finditer(row_pattern, table_content):
                q_id = int(match.group(1))
                answer = match.group(2).strip()
                answers[q_id] = answer
        if not answers:
            inline_pattern = r'\*\*(\d+)\.\*\*.*?\*\*【答案】([A-D]+)\*\*'
            for match in re.finditer(inline_pattern, content, re.DOTALL):
                q_id = int(match.group(1))
                answer = match.group(2).strip()
                answers[q_id] = answer
        return answers

    def _parse_a_type(self, content: str):
        a_section_match = re.search(
            r'## 一、A型题.*?\n(.*?)(?=## 二、B型题|## 参考答案)',
            content, re.DOTALL
        )
        if not a_section_match:
            return

        a_content = a_section_match.group(1)
        self._parse_single_questions(a_content, 'A')

    def _parse_single_questions(self, content: str, q_type: str):
        pattern = r'\*\*(\d+)\.\*\*\s*(.*?)(?=\*\*\d+\.\*\*|$)'
        matches = list(re.finditer(pattern, content, re.DOTALL))

        shared_stem_groups = {}
        for match in matches:
            q_id = int(match.group(1))
            q_text = match.group(2).strip()
            group_match = re.match(r'（(\d+)[～~-]+(\d+)题共用题干）\s*(.*)', q_text, re.DOTALL)
            if group_match:
                start_id = int(group_match.group(1))
                end_id = int(group_match.group(2))
                case_desc = group_match.group(3).strip()
                stem_part, _ = self._extract_stem_and_options(case_desc)
                for linked_id in range(start_id + 1, end_id + 1):
                    shared_stem_groups[linked_id] = stem_part

        for match in matches:
            q_id = int(match.group(1))
            q_text = match.group(2).strip()

            group_match = re.match(r'（\d+[～~-]+\d+题共用题干）\s*(.*)', q_text, re.DOTALL)
            if group_match:
                q_text = group_match.group(1).strip()

            stem, options = self._extract_stem_and_options(q_text)
            answer = self.answers.get(q_id, '')
            shared_stem = shared_stem_groups.get(q_id)

            if stem and answer:
                question = Question(
                    id=q_id,
                    type=q_type,
                    stem=stem,
                    options=options,
                    answer=answer,
                    shared_stem=shared_stem
                )
                self.questions.append(question)

    def _parse_b_type(self, content: str):
        b_section_match = re.search(
            r'## 二、B型题.*?\n(.*?)(?=## 三、X型题|## 参考答案)',
            content, re.DOTALL
        )
        if not b_section_match:
            return

        b_content = b_section_match.group(1)
        group_pattern = r'###\s*（(\d+)[～~-]+(\d+)题共用备选答案）\s*\n(.*?)(?=###\s*（|## 三|## 参考答案|$)'
        groups = re.finditer(group_pattern, b_content, re.DOTALL)

        for group in groups:
            start_id = int(group.group(1))
            end_id = int(group.group(2))
            group_content = group.group(3)

            shared_options = self._extract_shared_options(group_content)
            self._parse_b_group_questions(group_content, shared_options, start_id, end_id)

    def _extract_shared_options(self, content: str) -> Dict[str, str]:
        options = {}
        option_pattern = r'-\s*([A-D])\.\s*(.+?)(?=\n-\s*[A-D]\.|(?:\n\n|\*\*\d))'
        matches = re.finditer(option_pattern, content, re.DOTALL)
        for m in matches:
            opt_letter = m.group(1)
            opt_text = m.group(2).strip()
            options[opt_letter] = opt_text
        return options

    def _parse_b_group_questions(self, content: str, shared_options: Dict[str, str],
                                  start_id: int, end_id: int):
        for q_id in range(start_id, end_id + 1):
            pattern = rf'\*\*{q_id}\.\*\*\s*(.*?)(?=\*\*\d+\.\*\*|$)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                stem = match.group(1).strip()
                answer = self.answers.get(q_id, '')

                if stem and answer:
                    question = Question(
                        id=q_id,
                        type='B',
                        stem=stem,
                        options={},
                        answer=answer,
                        shared_options=shared_options
                    )
                    self.questions.append(question)

    def _parse_x_type(self, content: str):
        x_section_match = re.search(
            r'## 三、X型题.*?\n(.*?)(?=## 参考答案|$)',
            content, re.DOTALL
        )
        if not x_section_match:
            return

        x_content = x_section_match.group(1)
        self._parse_x_questions(x_content)

    def _parse_x_questions(self, content: str):
        pattern = r'\*\*(\d+)\.\*\*\s*(.*?)(?=\*\*\d+\.\*\*|$)'
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            q_id = int(match.group(1))
            q_text = match.group(2).strip()
            answer = self.answers.get(q_id, '')

            stem, options = self._extract_stem_and_options(q_text)

            if stem and answer:
                question = Question(
                    id=q_id,
                    type='X',
                    stem=stem,
                    options=options,
                    answer=answer
                )
                self.questions.append(question)

    def _extract_question_parts(self, q_text: str, full_content: str,
                                 q_id: int) -> Tuple[str, Dict[str, str], str]:
        answer_pattern = rf'\*\*{q_id}\.\*\*.*?\*\*【答案】\*\*\s*([A-D]+)'
        answer_match = re.search(answer_pattern, full_content, re.DOTALL)
        answer = answer_match.group(1) if answer_match else ''

        stem, options = self._extract_stem_and_options(q_text)
        return stem, options, answer

    def _extract_stem_and_options(self, text: str) -> Tuple[str, Dict[str, str]]:
        parts = re.split(r'\n-\s*[A-D]\.', text)
        stem = parts[0].strip() if parts else text.strip()

        options = {}
        option_pattern = r'-\s*([A-D])\.\s*(.+?)(?=\n-\s*[A-D]\.|$)'
        matches = re.finditer(option_pattern, text, re.DOTALL)
        for m in matches:
            opt_letter = m.group(1)
            opt_text = m.group(2).strip()
            options[opt_letter] = opt_text

        return stem, options

    def get_questions_by_type(self, q_type: str) -> List[Question]:
        return [q for q in self.questions if q.type == q_type]

    def get_question_by_id(self, q_id: int) -> Optional[Question]:
        for q in self.questions:
            if q.id == q_id:
                return q
        return None
