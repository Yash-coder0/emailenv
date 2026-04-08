EPSILON = 1e-4


class DraftGrader:
    def grade(self, action, email: dict) -> tuple[float, dict, str]:
        """
        Grade draft action.
        Returns: (score, breakdown_dict, reason_str)
        Score is ALWAYS strictly between 0 and 1 (exclusive).
        """
        breakdown = {}
        reason = ""

        response_text = action.response_text or ""

        if not response_text or len(response_text.strip()) == 0:
            score = EPSILON
            breakdown = {
                "questions_covered": 0.0,
                "tone_match": 0.0,
                "length_ok": 0.0,
                "format": 0.0,
                "total": score,
            }
            return score, breakdown, "Empty response"

        response_lower = response_text.lower()

        questions = email.get("key_questions_to_address", [])
        questions_matched = 0
        if questions:
            for question in questions:
                keywords = question.lower().split()
                if any(kw in response_lower for kw in keywords):
                    questions_matched += 1
            questions_score = (questions_matched / len(questions)) * 0.3
        else:
            questions_score = 0.0

        tone = email.get("expected_response_tone", "professional")
        tone_score = 0.0
        if tone == "urgent":
            urgent_keywords = ["immediately", "asap", "priority", "urgent", "right away"]
            if any(kw in response_lower for kw in urgent_keywords):
                tone_score = 0.2
            else:
                tone_score = 0.05
        elif tone == "formal":
            formal_endings = ["sincerely", "best regards", "kind regards", "respectfully"]
            if any(ending in response_lower for ending in formal_endings):
                tone_score = 0.2
            else:
                tone_score = 0.05
        else:
            tone_score = 0.1

        word_count = len(response_text.split())
        if 50 <= word_count <= 500:
            length_score = 0.3
        elif word_count < 50:
            length_score = 0.1
        else:
            length_score = 0.2

        has_greeting = any(greeting in response_lower for greeting in ["hi ", "hello ", "dear "])
        has_sign_off = any(sign in response_lower for sign in ["sincerely", "best regards", "thanks", "thank you"])

        if has_greeting and has_sign_off:
            format_score = 0.2
        elif has_greeting or has_sign_off:
            format_score = 0.1
        else:
            format_score = 0.0

        raw_score = questions_score + tone_score + length_score + format_score

        # CLAMP: score must never be exactly 0.0 or 1.0
        score = max(EPSILON, min(1.0 - EPSILON, raw_score))
        score = round(score, 4)

        breakdown = {
            "questions_covered": questions_score,
            "tone_match": tone_score,
            "length_ok": length_score,
            "format": format_score,
            "total": score,
        }

        reason = (
            f"Word count: {word_count}. "
            f"Questions matched: {questions_matched}/{len(questions) if questions else 0}. "
            f"Tone: {tone}."
        )

        return score, breakdown, reason


if __name__ == "__main__":
    from types import SimpleNamespace

    grader = DraftGrader()
    email = {
        "key_questions_to_address": ["refund policy", "timeline"],
        "expected_response_tone": "formal",
    }

    # Worst case: empty response → EPSILON
    worst = SimpleNamespace(response_text=None)
    score = grader.grade(worst, email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    # Best case: full, formal, keyword-rich response
    best_text = (
        "Dear Customer, sincerely thank you for reaching out. "
        "Our refund policy allows returns within 30 days. "
        "The timeline for processing is approximately 5 business days. "
        "We are committed to resolving this with utmost priority. "
        "Please feel free to contact us with any further questions. "
        "Best regards, Support Team"
    )
    best = SimpleNamespace(response_text=best_text)
    score = grader.grade(best, email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    print("All grader bounds OK")
