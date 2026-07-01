"""
My Agent
--------

Seu agente de IA pessoal. Customize a persona em `agents/prompts.py`
e os guardrails em `agents/guardrails/security.py`.

Suporta OpenAI e OpenRouter (via OPENAI_BASE_URL).
"""

from os import getenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from agents.guardrails import ContentSafetyGuardrail, enforce_safe_whatsapp_output
from agents.hooks import prepare_multimodal_input
from agents.prompts import instructions
from db import get_postgres_db

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
agent_db = get_postgres_db()
model_id = getenv("OPENAI_MODEL", "gpt-4o")
api_key = getenv("OPENAI_API_KEY", "")
base_url = getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
my_agent = Agent(
    id="my-agent",
    name="My Agent",
    model=OpenAIChat(
        id=model_id,
        api_key=api_key,
        base_url=base_url,
    ),
    db=agent_db,
    instructions=instructions,
    enable_agentic_memory=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    read_chat_history=True,
    num_history_runs=5,
    tool_call_limit=5,
    markdown=True,
    pre_hooks=[prepare_multimodal_input, ContentSafetyGuardrail()],
    post_hooks=[enforce_safe_whatsapp_output],
)

# Compat: Agno 2.6.x nao aceita `max_iterations` no construtor, mas o atributo
# e mantido para alinhamento com configuracoes antigas do projeto.
setattr(my_agent, "max_iterations", 5)


if __name__ == "__main__":
    my_agent.print_response("Oi! Me conta um pouco sobre você.", stream=True)