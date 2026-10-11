"""Read one supplier email into an ExtractedQuote: a search_catalog tool call plus a schema-checked output.

At most two attempts. Anything the model cannot read stays null and becomes a follow-up, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from rfq_agent.llm.base import LlmProvider, LlmRequest, LlmResponse
from rfq_agent.models import ExtractedQuote, QuoteEmail, Supplier
from rfq_agent.tracing import Span, Tracer

MAX_ATTEMPTS = 2

EXTRACT_SYSTEM = """You read supplier quote emails for a pooled medicine RFQ and copy what they state into the output schema.

Rules:
- For every product in the email, call search_catalog with the product text and use the product_id of the match whose name and strength agree with the email. If none agrees, product_id is null.
- Copy values as written. Never compute, convert currencies or estimate. A field the email does not state is null.
- price_amount is the number as written, and price_per_quantity is how many single units (tablets, ampoules, sachets) that price covers: "per box of 100" is 100, "per 1,000 tablets" is 1000, "each" or "per ampoule" is 1.
- moq_units and max_units are in single units. Multiply boxes by the units per box only when the email states both.
- shelf_life_months is the remaining shelf life at dispatch. lead_time_days converts weeks to days (3 weeks is 21).
- coa_provided is true only if the email says a certificate of analysis is attached or included; "available on request" or "available" is false; not mentioned is null.
- If the quote itself is not in the email (for example it points to an attachment that is not included), return an empty items list and say why in unreadable_reason.
- Return only the structured output."""


@dataclass
class ExtractResult:
    quote: ExtractedQuote
    attempts: int
    responses: list[LlmResponse]


def extract_request(email: QuoteEmail, supplier: Supplier | None, feedback: str | None = None) -> LlmRequest:
    sender = supplier.name if supplier else "unknown sender"
    prompt = (
        f"Supplier on file: {sender}\n"
        f"Subject: {email.subject}\n\n"
        f"Email body:\n<<<\n{email.body}\n>>>"
    )
    if feedback:
        prompt += f"\n\nYour previous output was rejected by the schema check: {feedback}\nReturn a corrected output."
    return LlmRequest(
        task="extract_quote",
        system=EXTRACT_SYSTEM,
        prompt=prompt,
        output_schema=ExtractedQuote.model_json_schema(),
        tools=["search_catalog"],
        max_turns=6,
    )


def _schema_errors(output: dict[str, object] | None) -> tuple[ExtractedQuote | None, str | None]:
    if output is None:
        return None, "no structured output"
    try:
        return ExtractedQuote.model_validate(output), None
    except ValidationError as exc:
        return None, "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]
        )


async def extract_quote(
    email: QuoteEmail,
    supplier: Supplier | None,
    provider: LlmProvider,
    tracer: Tracer | None = None,
    parent: Span | None = None,
) -> ExtractResult:
    feedback: str | None = None
    responses: list[LlmResponse] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = extract_request(email, supplier, feedback)
        if tracer:
            async with tracer.span("llm", f"read {email.id}", parent, task=request.task) as span:
                response = await provider.complete(request)
                tracer.llm_done(span, response, attempt)
                quote, feedback = _schema_errors(response.output)
                if feedback:
                    span.status = "retry"
                    span.attrs["schema_error"] = feedback
        else:
            response = await provider.complete(request)
            quote, feedback = _schema_errors(response.output)
        responses.append(response)
        if quote is not None:
            return ExtractResult(quote=quote, attempts=attempt, responses=responses)
    unreadable = ExtractedQuote(
        supplier_name=None, items=[], unreadable_reason=f"output failed the schema check twice ({feedback})"
    )
    return ExtractResult(quote=unreadable, attempts=MAX_ATTEMPTS, responses=responses)
