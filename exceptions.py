# Genie — Shared exception hierarchy.
# All layers import from here. Never import exceptions from element_resolver.py.


class ElementResolverError(Exception):
    pass


class TransientError(ElementResolverError):
    pass


class EnvironmentalError(ElementResolverError):
    pass


class ResourceError(ElementResolverError):
    pass


class UnrecoverableError(ElementResolverError):
    pass


class SchemaValidationError(EnvironmentalError):
    pass


class ResponseTruncatedError(Exception):
    """Raised when the LLM response was truncated by max_tokens (finish_reason='length').

    Attributes:
        partial_content: The incomplete response text returned by the API.
        cost_usd: The cost incurred for this (wasted) call.
    """

    def __init__(self, partial_content: str, cost_usd: float) -> None:
        self.partial_content = partial_content
        self.cost_usd = cost_usd
        super().__init__(
            f"Response truncated by max_tokens ({len(partial_content)} chars)"
        )


class FetchError(Exception):
    """Raised when fetch_url exhausts all cascade tiers without valid content.

    Attributes:
        url: The URL that could not be fetched.
        tiers_tried: Number of cascade tiers attempted before giving up.
    """
