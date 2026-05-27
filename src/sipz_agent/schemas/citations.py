from pydantic import BaseModel, Field, HttpUrl


class CandidateCitation(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl | None = None
    doi: str | None = None
    pmid: str | None = None
    year: int | None = None
    source: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    selection_reason: str | None = None
    abstract: str | None = None
    body_text: str | None = None
