import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: Optional[str]
    arxiv_id: Optional[str]
    references: List['Paper']


class SemanticScholar:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Semantic Scholar API client.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}
    
    def search_paper(self, title: str) -> Optional[Paper]:
        """Search for a paper by title.
        
        Args:
            title: The title of the paper to search for
            
        Returns:
            Paper object if found, None otherwise
        """
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": title,
            "fields": "paperId,title,abstract,externalIds"
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("data"):
            return None
            
        paper_data = data["data"][0]
        return Paper(
            paper_id=paper_data["paperId"],
            title=paper_data["title"],
            abstract=paper_data.get("abstract"),
            arxiv_id=paper_data.get("externalIds", {}).get("ArXiv"),
            references=[]
        )
    
    def get_references(self, paper_id: str) -> List[Paper]:
        """Get all references for a paper.
        
        Args:
            paper_id: The Semantic Scholar paper ID
            
        Returns:
            List of Paper objects representing references
        """
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {
            "fields": "paperId,title,abstract,externalIds",
            "limit": 100  # Maximum allowed by API
        }
        
        references = []
        offset = 0
        
        def has_all_fields(paper: Dict) -> bool:
            return paper.get("paperId") and paper.get("title") and paper.get("externalIds", {}).get("ArXiv")
        
        while True:
            params["offset"] = offset
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if not data.get("data"):
                break
                
            for reference in data["data"]:
                referenced_paper = reference.get("citedPaper", {})
                if not referenced_paper or not has_all_fields(referenced_paper):
                    continue
                references.append(Paper(
                    paper_id=referenced_paper["paperId"],
                    title=referenced_paper["title"],
                    abstract=referenced_paper.get("abstract"),
                    arxiv_id=referenced_paper.get("externalIds", {}).get("ArXiv"),
                    references=[]
                ))
            
            if not data.get("next"):
                break
                
            offset += len(data["data"])
            
        return references
    
    def get_paper_with_references(self, title: str) -> Optional[Paper]:
        """Get a paper and all its references in one call.
        
        Args:
            title: The title of the paper to search for
            
        Returns:
            Paper object with references if found, None otherwise
        """
        paper = self.search_paper(title)
        if paper:
            paper.references = self.get_references(paper.paper_id)
        return paper
