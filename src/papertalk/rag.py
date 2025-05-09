from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
from pathlib import Path
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain.chains import LLMChain

from papertalk.paper_parser import ComprehensiveReference, InTextReferenceParser, ReferenceContext
from papertalk.semantic_scholar import Paper


@dataclass
class PaperSummary:
    """A class representing a paper summary."""
    title: str
    abstract: str
    main_points: List[str]
    methodology: str
    results: str
    conclusions: str


@dataclass
class CitationAnalysis:
    """A class representing the analysis of how a reference is cited in the original paper."""
    reference_key: str
    reference_title: str
    citation_contexts: List[ReferenceContext]
    relevance_summary: str
    key_contributions: List[str]
    methodology_mentions: List[str]
    results_mentions: List[str]
    limitations_mentioned: List[str]
    future_work_mentions: List[str]


class PaperVectorStore:
    """A class to manage the FAISS vector store of paper abstracts."""
    
    def __init__(self, index_path: str = "paper_index"):
        """
        Initialize the PaperVectorStore.
        
        Args:
            index_path: Path to store/load the FAISS index
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.index_path = index_path
        self.vectorstore = None
        
    def _create_documents(self, papers: List[Paper]) -> List[Document]:
        """Convert Paper objects to LangChain documents."""
        documents = []
        for paper in papers:
            if paper.abstract:
                doc = Document(
                    page_content=paper.abstract,
                    metadata={
                        'title': paper.title,
                        'paper_id': paper.paper_id,
                        'arxiv_id': paper.arxiv_id
                    }
                )
                documents.append(doc)
        return documents
    
    def build_index(self, papers: List[Paper]):
        """
        Build the FAISS index from paper abstracts.
        
        Args:
            papers: List of Paper objects from SemanticScholar
        """
        documents = self._create_documents(papers)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the index
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        self.vectorstore.save_local(self.index_path)
    
    def load_index(self):
        """Load the FAISS index from disk."""
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(self.index_path, self.embeddings)
        else:
            raise FileNotFoundError(f"No index found at {self.index_path}")
    
    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Search for similar papers using the FAISS index.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_index or load_index first.")
        return self.vectorstore.similarity_search(query, k=k)


class PaperSummarizer:
    """A class to summarize papers using LLMs."""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """
        Initialize the PaperSummarizer.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=0
        )
        
        # Define the summarization prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic paper summarizer. 
            Analyze the provided paper and create a comprehensive summary.
            Focus on the main contributions, methodology, results, and conclusions.
            Be precise and technical in your analysis."""),
            ("user", """Please summarize the following paper:

            {paper_content}

            Provide a structured summary with the following sections:
            1. Main Points
            2. Methodology
            3. Results
            4. Conclusions""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
    
    def _extract_main_content(self, files: Dict[str, bytes]) -> str:
        """Extract the main content from the LaTeX files."""
        parser = InTextReferenceParser()
        main_tex = parser.find_main_tex(files)
        if not main_tex:
            raise ValueError("No main.tex file found")
        
        # Convert bytes to string and clean LaTeX
        content = main_tex.decode('utf-8')
        # Remove LaTeX commands and environments
        content = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', content)
        content = re.sub(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', '', content, flags=re.DOTALL)
        return content
    
    def summarize(self, files: Dict[str, bytes]) -> PaperSummary:
        """
        Summarize a paper using the LLM.
        
        Args:
            files: Dictionary of LaTeX files
            
        Returns:
            PaperSummary object
        """
        content = self._extract_main_content(files)
        
        # Get the summary from the LLM
        result = self.chain.run(paper_content=content)
        
        # Parse the structured summary
        sections = result.split('\n\n')
        main_points = []
        methodology = ""
        results = ""
        conclusions = ""
        
        current_section = None
        for section in sections:
            if "Main Points" in section:
                current_section = "main_points"
                continue
            elif "Methodology" in section:
                current_section = "methodology"
                continue
            elif "Results" in section:
                current_section = "results"
                continue
            elif "Conclusions" in section:
                current_section = "conclusions"
                continue
            
            if current_section == "main_points":
                main_points.append(section.strip())
            elif current_section == "methodology":
                methodology = section.strip()
            elif current_section == "results":
                results = section.strip()
            elif current_section == "conclusions":
                conclusions = section.strip()
        
        return PaperSummary(
            title="",  # TODO: Extract title from LaTeX
            abstract="",  # TODO: Extract abstract from LaTeX
            main_points=main_points,
            methodology=methodology,
            results=results,
            conclusions=conclusions
        )


class PaperRelevance:
    """A class to analyze how references are cited in the original paper."""
    
    def __init__(self):
        """Initialize the PaperRelevance analyzer."""
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4-turbo-preview",
            temperature=0
        )
        
        # Define the analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing academic paper citations and their relevance.
            Given a paper's summary and how it cites another paper, analyze the relevance and relationship between these papers.
            Focus on understanding how the original paper uses and builds upon the cited work.
            
            Provide a structured analysis with the following sections:
            1. Overall Relevance Summary
            2. Key Contributions (as cited in the original paper)
            3. Methodology Mentions
            4. Results and Findings Cited
            5. Limitations Mentioned
            6. Future Work Suggested
            
            Be specific and cite the actual contexts where possible."""),
            ("user", """Original Paper Summary:
            {original_paper_summary}
            
            Cited Paper Information:
            {cited_paper_info}
            
            Citation Contexts:
            {citation_contexts}
            
            Please analyze how the original paper cites and builds upon this reference.""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
    
    def _format_citation_contexts(self, contexts: List[ReferenceContext]) -> str:
        """Format citation contexts for the prompt."""
        formatted_contexts = []
        for i, context in enumerate(contexts, 1):
            formatted_context = f"Citation {i}:\n"
            formatted_context += f"Before: {context.context_before}\n"
            formatted_context += f"Reference: {context.reference_command}\n"
            formatted_context += f"After: {context.context_after}\n"
            formatted_contexts.append(formatted_context)
        return "\n---\n".join(formatted_contexts)
    
    def _format_cited_paper_info(self, ref: ComprehensiveReference) -> str:
        """Format information about the cited paper."""
        info = []
        
        # Add BibTeX information
        if ref.bibtex_reference:
            info.append("BibTeX Information:")
            info.append(f"Title: {ref.bibtex_reference.title}")
            info.append(f"Authors: {ref.bibtex_reference.authors}")
            info.append(f"Year: {ref.bibtex_reference.year}")
            if ref.bibtex_reference.abstract:
                info.append(f"Abstract: {ref.bibtex_reference.abstract}")
        
        # Add SemanticScholar information if available
        if ref.semantic_scholar_paper:
            info.append("\nSemanticScholar Information:")
            info.append(f"Title: {ref.semantic_scholar_paper.title}")
            if ref.semantic_scholar_paper.abstract:
                info.append(f"Abstract: {ref.semantic_scholar_paper.abstract}")
        
        return "\n".join(info)
    
    def _parse_analysis_sections(self, analysis: str) -> Tuple[str, List[str], List[str], List[str], List[str], List[str]]:
        """Parse the LLM's analysis into structured sections."""
        sections = analysis.split('\n\n')
        relevance_summary = ""
        key_contributions = []
        methodology_mentions = []
        results_mentions = []
        limitations = []
        future_work = []
        
        current_section = None
        for section in sections:
            if "Overall Relevance Summary" in section:
                current_section = "relevance"
                continue
            elif "Key Contributions" in section:
                current_section = "contributions"
                continue
            elif "Methodology Mentions" in section:
                current_section = "methodology"
                continue
            elif "Results and Findings" in section:
                current_section = "results"
                continue
            elif "Limitations Mentioned" in section:
                current_section = "limitations"
                continue
            elif "Future Work" in section:
                current_section = "future_work"
                continue
            
            if current_section == "relevance":
                relevance_summary = section.strip()
            elif current_section == "contributions":
                key_contributions.append(section.strip())
            elif current_section == "methodology":
                methodology_mentions.append(section.strip())
            elif current_section == "results":
                results_mentions.append(section.strip())
            elif current_section == "limitations":
                limitations.append(section.strip())
            elif current_section == "future_work":
                future_work.append(section.strip())
        
        return relevance_summary, key_contributions, methodology_mentions, results_mentions, limitations, future_work
    
    def analyze_citations(self, original_paper_summary: PaperSummary, 
                         reference: ComprehensiveReference) -> CitationAnalysis:
        """
        Analyze how a reference is cited in the original paper.
        
        Args:
            original_paper_summary: Summary of the original paper
            reference: ComprehensiveReference object containing the reference information
            
        Returns:
            CitationAnalysis object with structured analysis
        """
        # Format the original paper summary
        summary_text = f"Title: {original_paper_summary.title}\n"
        summary_text += f"Main Points:\n" + "\n".join(f"- {point}" for point in original_paper_summary.main_points) + "\n"
        summary_text += f"Methodology: {original_paper_summary.methodology}\n"
        summary_text += f"Results: {original_paper_summary.results}\n"
        summary_text += f"Conclusions: {original_paper_summary.conclusions}"
        
        # Format the cited paper information
        cited_paper_info = self._format_cited_paper_info(reference)
        
        # Format citation contexts
        contexts_text = self._format_citation_contexts(reference.in_text_contexts)
        
        # Get analysis from LLM
        analysis = self.chain.run(
            original_paper_summary=summary_text,
            cited_paper_info=cited_paper_info,
            citation_contexts=contexts_text
        )
        
        # Parse the analysis into sections
        relevance_summary, key_contributions, methodology_mentions, results_mentions, limitations, future_work = \
            self._parse_analysis_sections(analysis)
        
        return CitationAnalysis(
            reference_key=reference.reference_key,
            reference_title=reference.bibtex_reference.title if reference.bibtex_reference else "Unknown Title",
            citation_contexts=reference.in_text_contexts,
            relevance_summary=relevance_summary,
            key_contributions=key_contributions,
            methodology_mentions=methodology_mentions,
            results_mentions=results_mentions,
            limitations_mentioned=limitations,
            future_work_mentions=future_work
        )


class AdvancedRAG:
    """A class implementing an advanced RAG pipeline with semantic search and reranking."""
    
    def __init__(self, vectorstore: PaperVectorStore):
        """
        Initialize the AdvancedRAG pipeline.
        
        Args:
            vectorstore: Initialized PaperVectorStore instance
        """
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4-turbo-preview",
            temperature=0
        )
        
        # Initialize the reranker using RankLLMRerank
        self.compressor = RankLLMRerank(
            top_n=5,
            model="gpt",
            gpt_model="gpt-4-turbo-preview"
        )
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vectorstore.vectorstore.as_retriever(
                search_kwargs={"k": 20}
            )
        )
    
    def find_similar_papers(self, query: str, k: int = 10) -> List[Document]:
        """
        Find similar papers using the advanced RAG pipeline.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents, reranked by relevance
        """
        # Use the compression retriever to get reranked results
        results = self.retriever.get_relevant_documents(query)
        return results[:k]  # Return top k results