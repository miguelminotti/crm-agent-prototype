from langchain_anthropic import ChatAnthropic


def build_chat_anthropic_model() -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )