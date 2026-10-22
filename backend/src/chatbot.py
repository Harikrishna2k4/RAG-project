import time
from typing import Dict, Optional, Tuple
from openai import OpenAI, BadRequestError
from typing import Callable

class Chatbot:
    def __init__(
        self,
        model_name: str,
        client: OpenAI,
        retrieve_context: Callable[[str], str]
    ):
        self.model_name = model_name
        self.client = client
        self.retrieve_context = retrieve_context
        self.last_context: Optional[str] = None

    def chat(self, user_message: str) -> Tuple[str, Dict[str, float]]:
        timings: Dict[str, float] = {}
        start = time.time()
        context = self.retrieve_context(user_message)
        timings["retrieve"] = time.time() - start
        self.last_context = context

        system_message = {
            "role": "system",
            "content": (
                "Answer using the supplied excerpts from the user's PDF. If the excerpts do not contain "
                "the answer, say so instead of guessing. Explain concepts in clear plain language. Be concise."
            ),
        }
        grounded_question = f"Document excerpts:\n\n{context or '(no excerpts found)'}\n\nUser request: {user_message}"
        kwargs = {
            "model": self.model_name,
            "messages": [
                system_message,
                {"role": "user", "content": grounded_question}
            ]
        }
        start = time.time()
        try:
            response = self.client.chat.completions.create(temperature=0.2, **kwargs)
        except BadRequestError as error:
            if "temperature" not in str(error).lower():
                raise
            response = self.client.chat.completions.create(**kwargs)
        timings["llm"] = time.time() - start
        answer = response.choices[0].message.content or "I couldn't generate an answer."
        return answer, timings
