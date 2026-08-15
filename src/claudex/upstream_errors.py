class UpstreamError(Exception):
    """Base for provider upstream HTTP failures; carries status_code and body."""

    provider_label: str = ""

    def __init__(
        self, status_code: int, body: str, provider_label: str | None = None
    ) -> None:
        label = self.provider_label if provider_label is None else provider_label
        super().__init__(f"{label} upstream returned {status_code}: {body[:2000]}")
        self.status_code = status_code
        self.body = body


class UpstreamAuthError(Exception):
    """Base for provider credential failures."""
