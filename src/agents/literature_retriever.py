"""
Literature Retriever Agent - Multi-source research paper retrieval with Pydantic v2
Searches ArXiv, Semantic Scholar,  APIs simultaneously
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from urllib.parse import quote

# LangChain imports for agent framework (v0.3+ with Pydantic v2 support)
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

# Pydantic v2 imports (NO MORE pydantic_v1!)
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

# Setup logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """
    Data class to standardize paper information across different APIs
    
    Concept: Data Transfer Object (DTO) pattern
    - Ensures consistent data structure regardless of source API
    - Makes it easy to combine results from multiple sources
    - Using dataclass for performance (faster than Pydantic for simple structures)
    """
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    doi: Optional[str]
    arxiv_id: Optional[str]
    source: str  # Which API provided this paper
    url: str
    citations: int = 0
    venue: Optional[str] = None

class LiteratureSearchInput(BaseModel):
    """
    Pydantic v2 model for input validation
    
    Key Changes from v1 to v2:
    - Uses model_config instead of Config class
    - Field() syntax remains the same
    - Better type validation and error messages
    """
    
    # Pydantic v2 configuration
    model_config = ConfigDict(
        # Equivalent to v1's Config.validate_assignment
        validate_assignment=True,
        # Equivalent to v1's Config.str_strip_whitespace  
        str_strip_whitespace=True,
        # Better error handling
        use_enum_values=True
    )
    
    # Field definitions (syntax unchanged from v1)
    query: str = Field(
        description="Search query for research papers",
        min_length=1,
        max_length=500
    )
    max_results: int = Field(
        default=20,
        description="Maximum results per source",
        ge=1,  # Greater than or equal to 1
        le=100  # Less than or equal to 100
    )
    date_filter: Optional[str] = Field(
        default=None,
        description="Date filter (YYYY or YYYY-MM)",
        pattern=r'^\d{4}(-\d{2})?$'  # Regex validation for date format
    )

class LiteratureRetrieverAgent(BaseTool):
    """
    Multi-source literature retrieval agent with Pydantic v2 support
    
    Key Changes for Pydantic v2:
    - Updated args_schema to use Pydantic v2 model
    - Uses new validation syntax
    - Better error handling and type safety
    """
    
    name: str = "literature_retriever"
    description: str = "Searches multiple academic databases for research papers"
    args_schema: type[BaseModel] = LiteratureSearchInput   # ✅ pydantic v2 requires annotation
    
    _apis: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=lambda: {
        "arxiv": {"base_url": "http://export.arxiv.org/api/query", "rate_limit": 3.0},
        # ✅ correct S2 endpoint (GET)
        "semantic_scholar": {"base_url": "https://api.semanticscholar.org/graph/v1/paper/search", "rate_limit": 5.0},
    })
    
    def _run(
        self, 
        query: str, 
        max_results: int = 5,
        date_filter: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous entry point for the tool with Pydantic v2 validation
        
        Concept: Input Validation with Pydantic v2
        - Automatic validation of input parameters
        - Better error messages for invalid inputs
        - Type coercion and sanitization
        """
        
        # Validate inputs using Pydantic v2 model
        try:
            validated_input = LiteratureSearchInput(
                query=query,
                max_results=max_results,
                date_filter=date_filter
            )
            logger.info(f"✅ Input validation passed: {validated_input.model_dump()}")
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return []
        
        # Run async search with validated inputs
        return asyncio.run(
            self._async_search(
                validated_input.query, 
                validated_input.max_results, 
                validated_input.date_filter
            )
        )
    
    async def _async_search(
        self, 
        query: str, 
        max_results: int,
        date_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Main async search orchestrator
        
        Concept: Concurrent Processing
        - Searches all APIs simultaneously for speed
        - Uses asyncio.gather() to wait for all results
        - Much faster than sequential API calls
        """
        logger.info(f"🔍 Starting literature search for: '{query}' (max: {max_results})")
        
        # Create async HTTP session for connection pooling
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
        ) as session:
            
            # Launch all API searches concurrently
            tasks = [
                self._search_arxiv(session, query, max_results, date_filter),
                self._search_semantic_scholar(session, query, max_results, date_filter),
            ]
            
            # Wait for all searches to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate and deduplicate results
            all_papers = []
            source_counts = {}
            
            for i, result in enumerate(results):
                source_name = ['ArXiv', 'Semantic Scholar'][i]
                
                if isinstance(result, list):
                    all_papers.extend(result)
                    source_counts[source_name] = len(result)
                    logger.info(f"✅ {source_name}: {len(result)} papers")
                elif isinstance(result, Exception):
                    logger.error(f"❌ {source_name} failed: {result}")
                    source_counts[source_name] = 0
            
            logger.info(f"📊 Total papers before deduplication: {len(all_papers)}")
            
            # Remove duplicates and sort by relevance
            unique_papers = self._deduplicate_papers(all_papers)
            final_papers = unique_papers[:max_results]
            
            logger.info(f"📄 Final results: {len(final_papers)} unique papers")
            logger.info(f"📈 Source breakdown: {source_counts}")
            
            # Convert to dict format for JSON serialization
            return [paper.__dict__ for paper in final_papers]
    
    async def _search_arxiv(
        self, 
        session: aiohttp.ClientSession, 
        query: str, 
        max_results: int,
        date_filter: Optional[str] = None
    ) -> List[ResearchPaper]:
        """
        Search ArXiv API using their XML-based interface
        
        Concept: XML Processing with Error Handling
        - ArXiv returns Atom XML feeds
        - Use ElementTree for parsing structured data
        - Handle namespace prefixes for proper element access
        - Robust error handling for malformed XML
        """
        try:
            # Build ArXiv query parameters
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': min(max_results, 5),  # ArXiv limit
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            # Add date filter if specified
            if date_filter:
                # ArXiv uses submittedDate format: YYYYMMDD
                if len(date_filter) == 4:  # YYYY format
                    params['search_query'] += f' AND submittedDate:[{date_filter}0101 TO {date_filter}1231]'
                elif len(date_filter) == 7:  # YYYY-MM format
                    year, month = date_filter.split('-')
                    # Calculate last day of month (simplified)
                    last_day = '31' if month in ['01','03','05','07','08','10','12'] else '30'
                    if month == '02':
                        last_day = '28'  # Simplified (ignore leap years)
                    params['search_query'] += f' AND submittedDate:[{year}{month}01 TO {year}{month}{last_day}]'
            
            # Make HTTP request with rate limiting
            await asyncio.sleep(self._apis['arxiv']['rate_limit'])
            
            async with session.get(self._apis['arxiv']['base_url'], params=params) as response:
                if response.status != 200:
                    logger.error(f"ArXiv API error: {response.status}")
                    return []
                
                xml_content = await response.text()
                
                # Parse XML response with error handling
                try:
                    root = ET.fromstring(xml_content)
                except ET.ParseError as e:
                    logger.error(f"ArXiv XML parsing error: {e}")
                    return []
                
                # Define XML namespaces used by ArXiv
                namespaces = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                papers = []
                
                # Extract each paper entry from XML
                for entry in root.findall('atom:entry', namespaces):
                    try:
                        # Extract basic metadata with safe navigation
                        title_elem = entry.find('atom:title', namespaces)
                        title_text = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "No Title"
                        
                        # Extract authors (can be multiple)
                        author_elements = entry.findall('atom:author/atom:name', namespaces)
                        authors = [author.text.strip() for author in author_elements if author.text] or ["Unknown Author"]
                        
                        # Extract abstract/summary
                        summary_elem = entry.find('atom:summary', namespaces)
                        abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
                        
                        # Extract publication date
                        published_elem = entry.find('atom:published', namespaces)
                        pub_date = published_elem.text[:10] if published_elem is not None else ""  # Get YYYY-MM-DD part
                        
                        # Extract ArXiv ID from entry ID
                        entry_id_elem = entry.find('atom:id', namespaces)
                        if entry_id_elem is not None:
                            arxiv_id = entry_id_elem.text.split('/')[-1].replace('abs/', '')
                        else:
                            arxiv_id = ""
                        
                        # Build paper URL
                        paper_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
                        
                        # Create standardized paper object
                        paper = ResearchPaper(
                            title=title_text,
                            authors=authors,
                            abstract=abstract,
                            published_date=pub_date,
                            doi=None,  # ArXiv papers may not have DOIs initially
                            arxiv_id=arxiv_id,
                            source="ArXiv",
                            url=paper_url
                        )
                        
                        papers.append(paper)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse ArXiv entry: {e}")
                        continue
                
                logger.info(f"📚 Retrieved {len(papers)} papers from ArXiv")
                return papers
                
        except Exception as e:
            logger.error(f"❌ ArXiv search failed: {e}")
            return []
    
    
    async def _search_semantic_scholar(
        self, 
        session: aiohttp.ClientSession, 
        query: str, 
        max_results: int,
        date_filter: Optional[str] = None
    ) -> List[ResearchPaper]:
        """
        Search Semantic Scholar API
        
        Concept: Academic Search Intelligence with Rate Limiting
        - AI-powered relevance ranking
        - Rich citation and reference data
        - TL;DR summaries for papers
        - Proper rate limiting to avoid being blocked
        """
        try:
            params = {
                'query': query,
                'limit': min(max_results, 100),
                'fields': 'title,authors,abstract,year,citationCount,url,venue,tldr,externalIds'
            }
            
            # Add year filter
            if date_filter and len(date_filter) >= 4:
                year = date_filter[:4]
                params['year'] = year
            
            # Proper headers for Semantic Scholar
            headers = {
                'User-Agent': 'ResearchIntelligenceSystem/1.0',
                'Accept': 'application/json'
            }
            
            await asyncio.sleep(self._apis['semantic_scholar']['rate_limit'])
            
            async with session.get(
                self._apis['semantic_scholar']['base_url'], 
                params=params,
                headers=headers
            ) as response:
                
                if response.status != 200:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
                
                data = await response.json()
                papers = []
                
                for result in data.get('data', []):
                    try:
                        # Extract author names
                        authors = []
                        for author in result.get('authors', []):
                            author_name = author.get('name')
                            if author_name:
                                authors.append(author_name.strip())
                        
                        if not authors:
                            authors = ["Unknown Author"]
                        
                        # Use TL;DR if abstract is not available
                        abstract = result.get('abstract', '').strip()
                        if not abstract and result.get('tldr'):
                            tldr_text = result['tldr'].get('text', '')
                            if tldr_text:
                                abstract = f"TL;DR: {tldr_text}"
                        
                        # Extract external IDs for DOI
                        doi = None
                        external_ids = result.get('externalIds', {})
                        if external_ids and 'DOI' in external_ids:
                            doi = external_ids['DOI']
                        
                        paper = ResearchPaper(
                            title=result.get('title', 'No Title').strip(),
                            authors=authors,
                            abstract=abstract,
                            published_date=str(result.get('year', '')),
                            doi=doi,
                            arxiv_id=external_ids.get('ArXiv') if external_ids else None,
                            source="Semantic Scholar",
                            url=result.get('url', ''),
                            citations=result.get('citationCount', 0),
                            venue=result.get('venue')
                        )
                        
                        papers.append(paper)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse Semantic Scholar entry: {e}")
                        continue
                
                logger.info(f"📚 Retrieved {len(papers)} papers from Semantic Scholar")
                return papers
                
        except Exception as e:
            logger.error(f"❌ Semantic Scholar search failed: {e}")
            return []

    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """
        Remove duplicate papers based on title similarity and DOI matching
        
        Concept: Enhanced Deduplication Logic
        - Papers may appear in multiple databases
        - Use both title similarity and DOI matching
        - Prefer papers with more metadata and higher citation counts
        """
        if not papers:
            return []
        
        unique_papers = []
        seen_titles = set()
        seen_dois = set()
        
        # Sort by source preference and citation count
        source_priority = {
            "Semantic Scholar": 0,  # Good AI-powered insights
            "ArXiv": 1,         # Good for preprints
        }
        
        # Sort by priority, then by citations (descending)
        papers.sort(key=lambda p: (
            source_priority.get(p.source, 4), 
            -p.citations,
            -len(p.abstract)  # Prefer papers with more complete abstracts
        ))
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            # Remove common prefixes/suffixes that might vary
            normalized_title = normalized_title.replace('the ', '').replace('a ', '').replace('an ', '')
            
            # Check for DOI duplicates (most reliable)
            if paper.doi and paper.doi in seen_dois:
                continue
                
            # Check for title duplicates
            if normalized_title in seen_titles:
                continue
            
            # Add to unique set
            if paper.doi:
                seen_dois.add(paper.doi)
            seen_titles.add(normalized_title)
            unique_papers.append(paper)
        
        return unique_papers
    
literature_retriever = LiteratureRetrieverAgent()

'''
# Create instance for use in other modules
if __name__ == "__main__":
    # Instantiate the tool
    retriever = LiteratureRetrieverAgent()

    # Define a test query
    test_query = "Alpha earth"
    test_max_results = 5
    
    print(f"--- 🧪 Starting test search for: '{test_query}' ---")

    try:
        # Run the tool's main synchronous method, which is the intended entry point
        results = retriever._run(query=test_query, max_results=test_max_results)

        # Print the results in a readable format
        print(f"\n--- ✅ Found {len(results)} unique papers ---")
        for i, paper in enumerate(results):
            # The result is a dictionary because of the `paper.__dict__` conversion
            print(f"\n[{i+1}] Title: {paper['title']}")
            print(f"    Authors: {', '.join(paper['authors'][:2])}{' et al.' if len(paper['authors']) > 2 else ''}")
            print(f"    Source: {paper['source']} ({paper['published_date']})")
            print(f"    URL: {paper['url']}")
            if paper['citations'] > 0:
                print(f"    Citations: {paper['citations']}")

    except Exception as e:
        print(f"An error occurred during the test run: {e}")
'''