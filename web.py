import importlib.util
from typing import List, Dict, Optional
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from pydantic import BaseModel, Field

class DuckDuckGoResult(BaseModel):
    """Pydantic model to represent a DuckDuckGo search result."""
    title: Optional[str] = Field(None, description="Title of the search result")
    href: Optional[str] = Field(None, description="URL of the search result")
    body: Optional[str] = Field(None, description="Snippet/body of the search result")

class DuckDuckGoSearchToolSpec(BaseToolSpec):
    """DuckDuckGoSearch tool spec."""

    spec_functions = ["duckduckgo_instant_search", "duckduckgo_full_search"]

    def __init__(self) -> None:
        if not importlib.util.find_spec("duckduckgo_search"):
            raise ImportError(
                "DuckDuckGoSearchToolSpec requires the duckduckgo_search package to be installed."
            )
        super().__init__()

    def duckduckgo_instant_search(self, query: str) -> DuckDuckGoResult:
        """
        Make a query to DuckDuckGo API to receive an instant answer.

        Args:
            query (str): The query to be passed to DuckDuckGo.
        """
        from duckduckgo_search import DDGS

        with DDGS() as ddg:
            results = list(ddg.answers(query))
            if results:
                return DuckDuckGoResult(body=results[0]['answer'])
            else:
                return DuckDuckGoResult(body="No instant answer found.")

    def duckduckgo_full_search(
        self,
        query: str,
        region: Optional[str] = "wt-wt",
        max_results: Optional[int] = 10,
    ) -> List[DuckDuckGoResult]:
        """
        Make a query to DuckDuckGo search to receive full search results.

        Args:
            query (str): The query to be passed to DuckDuckGo.
            region (Optional[str]): The region to be used for the search in [country-language] convention, ex us-en, uk-en, ru-ru, etc...
            max_results (Optional[int]): The maximum number of results to be returned.
        """
        from duckduckgo_search import DDGS

        params = {
            "keywords": query,
            "region": region,
            "max_results": max_results,
        }

        with DDGS() as ddg:
            results = list(ddg.text(**params))
            return [DuckDuckGoResult(**result) for result in results]