import requests
from typing import Optional, Dict, List, Tuple
import tarfile
import io
import bibtexparser
from dataclasses import dataclass
import re
from difflib import SequenceMatcher

from papertalk.semantic_scholar import Paper


@dataclass
class ReferenceContext:
    """A class representing a reference and its surrounding context."""
    reference_key: str
    context_before: str
    context_after: str
    full_context: str
    reference_command: str  # e.g., 'cite', 'citep', 'citet', etc.


@dataclass
class Reference:
    """A class representing a single reference."""
    key: str
    entry_type: str
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    raw_entry: Optional[Dict[str, str]] = None


@dataclass
class MergedReference:
    """A class representing a merged reference from both sources."""
    reference_key: str  # BibTeX key
    bibtex_reference: Reference  # Full BibTeX reference
    semantic_scholar_paper: Optional[Paper] = None  # SemanticScholar paper if matched


@dataclass
class ComprehensiveReference:
    """A class representing a reference with all available information."""
    reference_key: str  # BibTeX key
    bibtex_reference: Reference  # Full BibTeX reference
    semantic_scholar_paper: Optional[Paper] = None  # SemanticScholar paper if matched
    in_text_contexts: List[ReferenceContext] = None  # List of in-text references with context

    def __post_init__(self):
        if self.in_text_contexts is None:
            self.in_text_contexts = []


class InTextReferenceParser:
    """A class to parse references and their context from LaTeX files."""
    
    def __init__(self, context_window: int = 2):
        """
        Initialize the InTextReferenceParser.
        
        Args:
            context_window (int): Number of sentences to include before and after reference
        """
        self.context_window = context_window
        # Common reference commands in LaTeX
        self.reference_patterns = [
            r'\\cite(?:p|t|alp|alt)?\{([^}]+)\}',  # \cite, \citep, \citet, etc.
            r'\\textcite\{([^}]+)\}',
            r'\\parencite\{([^}]+)\}',
            r'\\autocite\{([^}]+)\}',
            r'\\footcite\{([^}]+)\}',
        ]
        self.reference_regex = re.compile('|'.join(self.reference_patterns))
    
    def find_main_tex(self, files: Dict[str, bytes]) -> Optional[bytes]:
        """Find the main LaTeX file in the downloaded files."""
        # Common names for main LaTeX files
        preferred_names = ['main.tex', 'paper.tex', 'article.tex', 'manuscript.tex']
        
        for name in preferred_names:
            if name in files:
                return files[name]
        
        # Look for any .tex file
        tex_files = [f for f in files.keys() if f.endswith('.tex')]
        if tex_files:
            return files[tex_files[0]]
        
        return None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling LaTeX formatting."""
        # Remove LaTeX commands and environments
        text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', text)
        text = re.sub(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', '', text, flags=re.DOTALL)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_reference_context(self, text: str, reference_match: re.Match) -> ReferenceContext:
        """Extract context around a reference."""
        start, end = reference_match.span()
        reference_command = reference_match.group(0)
        reference_key = reference_match.group(1)
        
        # Get text before and after reference
        text_before = text[:start]
        text_after = text[end:]
        
        # Split into sentences
        sentences_before = self.split_into_sentences(text_before)
        sentences_after = self.split_into_sentences(text_after)
        
        # Get context window
        context_before = ' '.join(sentences_before[-self.context_window:])
        context_after = ' '.join(sentences_after[:self.context_window])
        
        # Combine for full context
        full_context = f"{context_before} {reference_command} {context_after}"
        
        return ReferenceContext(
            reference_key=reference_key,
            context_before=context_before,
            context_after=context_after,
            full_context=full_context,
            reference_command=reference_command
        )
    
    def parse_references(self, files: Dict[str, bytes]) -> Dict[str, List[ReferenceContext]]:
        """
        Parse references and their context from the main LaTeX file.
        
        Args:
            files (Dict[str, bytes]): Dictionary of files from PaperDownloader
            
        Returns:
            Dict[str, List[ReferenceContext]]: Dictionary mapping reference keys to their contexts
            
        Raises:
            ValueError: If no main LaTeX file is found
        """
        main_tex = self.find_main_tex(files)
        if main_tex is None:
            raise ValueError("No main LaTeX file found in the downloaded files")
        
        # Convert bytes to string
        tex_content = main_tex.decode('utf-8')
        
        # Find all references
        references: Dict[str, List[ReferenceContext]] = {}
        for match in self.reference_regex.finditer(tex_content):
            context = self.extract_reference_context(tex_content, match)
            
            # Handle multiple references in one command (e.g., \cite{key1,key2})
            keys = [k.strip() for k in context.reference_key.split(',')]
            for key in keys:
                if key not in references:
                    references[key] = []
                references[key].append(context)
        
        return references


class ReferenceParser:
    """A class to parse BibTeX files and extract reference information."""
    
    def __init__(self):
        """Initialize the ReferenceParser."""
        # Configure the parser to be more lenient and handle all cases
        self.parser = bibtexparser.bparser.BibTexParser()
        self.parser.ignore_nonstandard_types = False
        self.parser.homogenise_fields = False  # Don't modify field names
        self.parser.common_strings = False  # Don't try to resolve common strings
        self.parser.interpolate_strings = False  # Don't try to interpolate strings
    
    def clean_latex(self, text: Optional[str]) -> Optional[str]:
        """Clean LaTeX formatting from text."""
        if not text:
            return None
        # Remove LaTeX braces
        text = text.strip('{}')
        # Remove LaTeX commands
        text = text.replace('\\', '')
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text
    
    def parse_bibtex(self, bibtex_content: bytes) -> List[Reference]:
        """
        Parse a BibTeX file content and extract references.
        
        Args:
            bibtex_content (bytes): The content of the BibTeX file
            
        Returns:
            List[Reference]: List of parsed references
        """
        # Convert bytes to string
        bibtex_str = bibtex_content.decode('utf-8')
        
        # Parse the BibTeX content
        bib_database = bibtexparser.loads(bibtex_str, parser=self.parser)
        
        # Convert entries to Reference objects
        references = []
        for entry in bib_database.entries:
            # Store the raw entry for debugging or additional processing
            raw_entry = dict(entry)
            
            # Create the reference object with cleaned fields
            ref = Reference(
                key=entry.get('ID', ''),
                entry_type=entry.get('ENTRYTYPE', ''),
                title=self.clean_latex(entry.get('title')),
                authors=self.clean_latex(entry.get('author')),
                year=entry.get('year'),
                doi=entry.get('doi'),
                url=entry.get('url'),
                journal=self.clean_latex(entry.get('journal')),
                volume=entry.get('volume'),
                number=entry.get('number'),
                pages=entry.get('pages'),
                publisher=self.clean_latex(entry.get('publisher')),
                abstract=self.clean_latex(entry.get('abstract')),
                raw_entry=raw_entry
            )
            references.append(ref)
        
        return references
    
    def find_bib_file(self, files: Dict[str, bytes]) -> Optional[bytes]:
        """
        Find the BibTeX file in the downloaded files.
        
        Args:
            files (Dict[str, bytes]): Dictionary of files from PaperDownloader
            
        Returns:
            Optional[bytes]: Content of the BibTeX file if found, None otherwise
        """
        # Look for common BibTeX file names
        bib_files = [f for f in files.keys() if f.endswith('.bib')]
        
        if not bib_files:
            return None
            
        # If multiple .bib files exist, prefer references.bib or bibliography.bib
        preferred_names = ['references.bib', 'bibliography.bib']
        for name in preferred_names:
            if name in bib_files:
                return files[name]
        
        # Otherwise, return the last .bib file found
        return files[bib_files[-1]]
    
    def extract_references(self, files: Dict[str, bytes]) -> List[Reference]:
        """
        Extract references from the downloaded files.
        
        Args:
            files (Dict[str, bytes]): Dictionary of files from PaperDownloader
            
        Returns:
            List[Reference]: List of parsed references
            
        Raises:
            ValueError: If no BibTeX file is found
        """
        bib_content = self.find_bib_file(files)
        if bib_content is None:
            raise ValueError("No BibTeX file found in the downloaded files")
            
        return self.parse_bibtex(bib_content)


class PaperDownloader:
    """A class to download LaTeX source files from arXiv and store them in memory."""
    
    def __init__(self):
        """Initialize the PaperDownloader."""
        self.base_url = "https://arxiv.org/e-print/"
        
    def download_source(self, arxiv_id: str) -> Dict[str, bytes]:
        """
        Download the LaTeX source file for a given arXiv ID and store in memory.
        
        Args:
            arxiv_id (str): The arXiv ID (e.g., '2101.12345' or '2101.12345v2')
            
        Returns:
            Dict[str, bytes]: Dictionary mapping filenames to their contents
            
        Raises:
            requests.exceptions.RequestException: If download fails
            ValueError: If arxiv_id is invalid
        """
        # Clean the arXiv ID (remove version if present)
        clean_id = arxiv_id.split('v')[0]
        
        # Download the source
        url = f"{self.base_url}{clean_id}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Create an in-memory file-like object from the response content
        tar_data = io.BytesIO(response.content)
        
        # Extract the tar.gz file in memory
        file_contents = {}
        with tarfile.open(fileobj=tar_data, mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():  # Only process files, not directories
                    f = tar.extractfile(member)
                    if f is not None:
                        file_contents[member.name] = f.read()
        
        return file_contents


def similar(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def clean_title(title: str) -> str:
    """Clean a title for comparison by removing special characters and extra spaces."""
    # Remove LaTeX commands and math mode
    title = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', title)
    title = re.sub(r'\$.*?\$', '', title)
    # Remove special characters and extra spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    title = ' '.join(title.split())
    return title.lower()


def merge_references(references: List[Reference], papers: List[Paper], 
                    title_similarity_threshold: float = 0.8) -> List[MergedReference]:
    """
    Merge references from ReferenceParser with SemanticScholar papers.
    
    Args:
        references: List of references from ReferenceParser
        papers: List of papers from SemanticScholar
        title_similarity_threshold: Minimum similarity score for title matching
        
    Returns:
        List of MergedReference objects
    """
    merged_refs: List[MergedReference] = []
    matched_paper_ids = set()  # Track matched paper IDs instead of Paper objects
    
    # First pass: Try to match by identifiers
    for ref in references:
        matched = False
        best_match = None
        best_score = 0.0
        
        for paper in papers:
            # Skip already matched papers
            if paper.paper_id in matched_paper_ids:
                continue
                
            # Try to match by arXiv ID
            if ref.raw_entry and 'eprint' in ref.raw_entry:
                if paper.arxiv_id == ref.raw_entry['eprint']:
                    merged_refs.append(MergedReference(
                        reference_key=ref.key,
                        bibtex_reference=ref,
                        semantic_scholar_paper=paper
                    ))
                    matched_paper_ids.add(paper.paper_id)
                    matched = True
                    break
        
        # If no identifier match, try title matching
        if not matched:
            ref_title = clean_title(ref.title) if ref.title else ""
            
            for paper in papers:
                if paper.paper_id in matched_paper_ids:
                    continue
                    
                if paper.title:
                    paper_title = clean_title(paper.title)
                    score = similar(ref_title, paper_title)
                    
                    if score > best_score and score >= title_similarity_threshold:
                        best_score = score
                        best_match = paper
            
            if best_match:
                merged_refs.append(MergedReference(
                    reference_key=ref.key,
                    bibtex_reference=ref,
                    semantic_scholar_paper=best_match
                ))
                matched_paper_ids.add(best_match.paper_id)
            else:
                # No match found
                merged_refs.append(MergedReference(
                    reference_key=ref.key,
                    bibtex_reference=ref,
                    semantic_scholar_paper=None
                ))
    
    return merged_refs


def merge_all_references(
    bibtex_references: List[Reference],
    semantic_scholar_papers: List[Paper],
    in_text_references: Dict[str, List[ReferenceContext]],
    title_similarity_threshold: float = 0.8,
    require_all_sources: bool = False
) -> List[ComprehensiveReference]:
    """
    Merge references from all three sources: BibTeX, SemanticScholar, and in-text references.
    
    Args:
        bibtex_references: List of references from ReferenceParser
        semantic_scholar_papers: List of papers from SemanticScholar
        in_text_references: Dictionary of in-text references with context
        title_similarity_threshold: Minimum similarity score for title matching
        require_all_sources: If True, only return references that have all three sources
        
    Returns:
        List of ComprehensiveReference objects
    """
    # First merge BibTeX and SemanticScholar references
    merged_refs = merge_references(bibtex_references, semantic_scholar_papers, title_similarity_threshold)
    
    # Convert to ComprehensiveReference objects and add in-text contexts
    comprehensive_refs = []
    for merged in merged_refs:
        # Get in-text contexts for this reference key
        contexts = in_text_references.get(merged.reference_key, [])
        
        ref = ComprehensiveReference(
            reference_key=merged.reference_key,
            bibtex_reference=merged.bibtex_reference,
            semantic_scholar_paper=merged.semantic_scholar_paper,
            in_text_contexts=contexts
        )
        
        # If require_all_sources is True, only include references that have all three sources
        if require_all_sources:
            if (ref.bibtex_reference and 
                ref.semantic_scholar_paper and 
                ref.in_text_contexts):
                comprehensive_refs.append(ref)
        else:
            comprehensive_refs.append(ref)
    
    return comprehensive_refs
