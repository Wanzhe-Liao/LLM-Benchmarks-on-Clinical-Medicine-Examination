from .question import Question


class PromptBuilder:
    SINGLE_CHOICE_TEMPLATE = """请回答以下医学选择题，只需直接回答选项字母（如A、B、C或D），不需要任何解释。

{stem}

{options}

答案："""

    MULTIPLE_CHOICE_TEMPLATE = """请回答以下医学多选题，可能有2-4个正确答案。只需直接回答所有正确选项的字母（如AB、BC或ABCD），不需要任何解释。

{stem}

{options}

答案："""

    B_TYPE_TEMPLATE = """请回答以下配伍选择题，根据题干从备选答案中选择一个最佳答案，只需直接回答选项字母（如A、B、C或D），不需要任何解释。

备选答案：
{shared_options}

题目：{stem}

答案："""

    @classmethod
    def build_prompt(cls, question: Question) -> str:
        if question.type == 'A':
            prompt = cls._build_a_type(question)
        elif question.type == 'B':
            prompt = cls._build_b_type(question)
        else:
            prompt = cls._build_x_type(question)

        return prompt

    @classmethod
    def _build_a_type(cls, question: Question) -> str:
        return cls.SINGLE_CHOICE_TEMPLATE.format(
            stem=question.get_full_stem(),
            options=question.format_options()
        )

    @classmethod
    def _build_b_type(cls, question: Question) -> str:
        shared_opts = question.get_full_options()
        shared_options_text = '\n'.join([f"{k}. {v}" for k, v in sorted(shared_opts.items())])

        return cls.B_TYPE_TEMPLATE.format(
            shared_options=shared_options_text,
            stem=question.stem
        )

    @classmethod
    def _build_x_type(cls, question: Question) -> str:
        return cls.MULTIPLE_CHOICE_TEMPLATE.format(
            stem=question.get_full_stem(),
            options=question.format_options()
        )
