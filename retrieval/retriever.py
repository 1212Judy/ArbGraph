# retrieval/retriever.py

import requests
from typing import List
from dataclasses import dataclass


# ============================
# Document Abstraction
# ============================

@dataclass
class Document:
    """
    Minimal document abstraction used by the ArbGraph pipeline.
    """
    id: str
    text: str
    title: str
    url: str


# ============================
# Wikipedia Retriever
# ============================

class WikipediaRetriever:
    """
    Wikipedia-based document retriever.

    Given a query, retrieves top-k Wikipedia articles and
    returns them as Document objects.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "ArbGraph-Retriever/1.0"
        }

    def retrieve(self, query: str, max_docs: int = 3) -> List[Document]:
        search_results = self._search(query, max_docs)

        documents = []
        for idx, item in enumerate(search_results):
            documents.append(
                Document(
                    id=f"wiki_{idx}",
                    title=item["title"],
                    text=item["content"],
                    url=item["url"],
                )
            )

        return documents

    # ============================
    # Internal Helpers
    # ============================

    def _search(self, query: str, limit: int) -> List[dict]:
        """
        Query Wikipedia search API.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
        }

        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        search_items = data.get("query", {}).get("search", [])

        results = []
        for item in search_items:
            title = item["title"]
            content = self._fetch_page_content(title)

            results.append({
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "content": content,
            })

        return results

    def _fetch_page_content(self, title: str, max_chars: int = 3000) -> str:
        """
        Fetch plain text content of a Wikipedia page.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        }

        response = requests.get(url, params=params, headers=self.headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})

        content = page.get("extract", "") or ""
        return content[:max_chars]
