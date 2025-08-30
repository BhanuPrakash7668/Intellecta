"""
Analysis Agent - Processes retrieved papers and generates insights using RAG
Analyzes themes, relationships, and synthesizes key findings from research papers
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import re
import os

# LangChain imports
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Pydantic v2
from pydantic import BaseModel, Field, ConfigDict
# Local imports
from ..database.vector_store import client as chroma_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """
    Data class for analysis results
    
    Concept: Structured Analysis Output
    - Standardizes analysis results for downstream processing
    - Makes it easy to serialize and store results
    - Provides clear data structure for report generation
    """
    
    paper_id: str
    themes: List[str]
    key_concepts: List[str]
    methodology: str
    findings: List[str]
    relevance_score: float
    relationship: List[Dict[str, Any]]
    summary: str
    
    
    
class AnalysisInput(BaseModel):
    """
    Pydantic v2 model for analysis input validation
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    papers: List[Dict[str, Any]] = Field(
        description="List of research papers to analyze",
        min_length=1
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus analysis on"
    )
    analysis_depth: str = Field(
        default="medium",
        description="Analysis depth: shallow, medium, deep",
        pattern=r'^(shallow|medium|deep)$'
    )
    
class AnalysisAgent(BaseTool):
    """
    Analysis Agent for processing research papers with RAG
    
    Concept: Retrieval Augmented Generation (RAG)
    - Combines document retrieval with LLM generation
    - Creates embeddings for semantic search
    - Uses vector database for efficient similarity search
    - Generates insights based on retrieved context
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True, # Allows complex types like TextSplitter
        extra='allow'                 # Allows setting undeclared attributes
    )
    
    name: str = "analysis_agent"
    description: str = "Analyzes research papers and generates insights using RAG"
    args_schema: type[BaseModel] = AnalysisInput
    
    def __init__(self):
        """
        Initialize the Analysis Agent with RAG components
        
        Concept: RAG Pipeline Setup
        - Text splitter for chunking documents
        - Embeddings model for vector representations  
        - Vector store for similarity search
        - LLM for analysis and insight generation
        """
        super().__init__()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.embeddings = OpenAIEmbeddings(
            model = "text-embedding-3-small"
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0.1,
            max_tokens=2000
        )
        
        self.collection_name = "research_papers"
        
        self._setup_prompts()
        
    def _setup_prompts(self):
        """
        Setup analysis prompts for different tasks
        
        Concept: Prompt Engineering
        - Structured prompts for consistent analysis
        - Task-specific prompts for different analysis types
        - Clear instructions for LLM behavior
        """
        self.theme_analysis_prompt = PromptTemplate(
            input_variables = ['paper_conent', 'focus_areas'],
            template = """
            Analyze the following research paper and identify key themes and concepts.
            
            Paper Content:
            {paper_content}
            
            Focus Areas (if any): {focus_areas}
            
            Please provide:
            1. Key Themes (3-5 main themes)
            2. Core Concepts (important terms and ideas)
            3. Methodology Used
            4. Main Findings
            5. Research Significance
            
            Format your response as JSON:
            {{
                "themes": ["theme1", "theme2", ...],
                "concepts": ["concept1", "concept2", ...],
                "methodology": "description of methodology",
                "findings": "summary of main findings",
                "significance": "research significance and impact"
            }}
            """
        )
            
        self.relationship_prompt = PromptTemplate(
        input_variables=["paper1_title", "paper1_abstract", "paper2_title", "paper2_abstract"],
        template="""
        Analyze the relationship between these two research papers:
        
        Paper 1:
        Title: {paper1_title}
        Abstract: {paper1_abstract}
        
        Paper 2:
        Title: {paper2_title}
        Abstract: {paper2_abstract}
        
        Identify:
        1. Conceptual relationships (shared themes, concepts)
        2. Methodological connections (similar approaches)
        3. Complementary aspects (how they support each other)
        4. Contradictions or disagreements
        
        Provide a relationship score from 0-10 (10 being highly related).
        
        Format as a JSON object:
        {{
            "relationship_score": score,
            "relationship_type": "complementary|contradictory|similar|builds_upon",
            "shared_concepts": ["concept1", "concept2"],
            "description": "detailed relationship description"
        }}
        """
    
        )
    def _run(
        self,
        papers: List[Dict[str, Any]],
        focus_areas: Optional[List[str]] = None,
        analysis_depth: str = "medium",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous entry point for analysis
        
        Concept: Input Validation and Orchestration
        - Validates input parameters
        - Orchestrates the analysis pipeline
        - Returns structured analysis results
        """
        try:
            validated_input = AnalysisInput(
                papers=papers,
                focus_areas=focus_areas,
                analysis_depth=analysis_depth
            )
            logger.info(f"✅ Analysis input validation passed for {len(papers)} papers")
            
        except Exception as e:
            logger.error(f"❌ Analysis input validation failed: {e}")
            return []
        
        return asyncio.run(
            self._async_analyze(
                validated_input.papers,
                validated_input.focus_areas,
                validated_input.analysis_depth
            )
        )
        
    async def _async_analyze(
        self,
        papers: List[Dict[str, Any]],
        focus_areas: Optional[List[str]] = None,
        analysis_depth: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        Main async analysis orchestrator
        
        Concept: Multi-stage Analysis Pipeline
        - Document processing and embedding
        - Individual paper analysis
        - Cross-paper relationship analysis
        - Result aggregation and synthesis
        """
        logger.info(f"Starting analysis of {len(papers)} papers (depth: {analysis_depth})")
         # Step 1: Process papers into structured, chunked documents

        documents = await self._process_documents(papers)
        logger.info(f" Processed {len(documents)} document chunks")

        # Step 2: Create a searchable vector store from the documents

        vector_store = await self._setup_vector_store(documents)
        logger.info("Vector store ready")
        
        # Step 3: Analyze each paper individually

        individual_analyses = await self._analyze_individual_papers(papers, vector_store, focus_areas, analysis_depth)
        
        logger.info(f"Completed individual analysis for {len(individual_analyses)} papers")
        
        # Step 4: Analyze relationships between papers (if more than one)
        relationships = []
        if len(papers) > 1:
            relationships = await self._analyze_relationships(papers)
            logger.info(f"🔗 Identified {len(relationships)} paper relationships")
            
         # Step 5: Combine individual analyses with their corresponding relationships

        final_results = []
        
        for i, analysis in enumerate(individual_analyses):
            paper_relationships = [r for r in relationships if r.get('paper1_index') == i or r.get('paper2_index') == i]
            result = {
            **analysis,  # Unpack the individual analysis results
            'relationships': paper_relationships,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_depth': analysis_depth
            }
            final_results.append(result)
    
        logger.info(f"✅ Analysis complete: {len(final_results)} results generated")
        return final_results
    
    async def _process_documents(self, papers: List[Dict[str, Any]]) -> List[Document]:
        """
        Process papers into LangChain Document objects with chunking
        
        Concept: Document Processing for RAG
        - Convert papers to LangChain Document format
        - Chunk large documents for better retrieval
        - Add metadata for filtering and context
        """
        documents = []
        
        for i, paper in enumerate(papers):
            try:
                # Combine title, abstract, and any additional content
                content_parts = [
                    f"Title: {paper.get('title', 'No Title')}",
                    f"Authors: {', '.join(paper.get('authors', []))}",
                    f"Abstract: {paper.get('abstract', 'No abstract available')}"
                ]
                
                # Add venue and date if available
                if paper.get('venue'):
                    content_parts.append(f"Venue: {paper['venue']}")
                if paper.get('published_date'):
                    content_parts.append(f"Published: {paper['published_date']}")
                
                full_content = "\n\n".join(content_parts)
                
                # Create metadata for the document
                metadata = {
                    'paper_id': f"paper_{i}",
                    'title': paper.get('title', 'No Title'),
                    'authors': ', '.join(paper.get('authors', [])),
                    'source': paper.get('source', 'Unknown'),
                    'citations': paper.get('citations', 0),
                    'url': paper.get('url', ''),
                    'doi': paper.get('doi'),
                    'arxiv_id': paper.get('arxiv_id')
                }
                
                # Split document into chunks
                chunks = self.text_splitter.split_text(full_content)
                
                # Create Document objects for each chunk
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            'chunk_id': f"paper_{i}_chunk_{j}",
                            'chunk_index': j
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                logger.warning(f"Failed to process paper {i}: {e}")
                continue
        
        return documents
    
    async def _setup_vector_store(self, documents: list[Document]) -> Chroma:
        """
        Setup ChromaDB vector store with embeddings
        """
        try:
            # ATTEMPT TO DELETE THE COLLECTION FIRST FOR A CLEAN RUN
            logger.info(f"🧹 Clearing existing collection: {self.collection_name}")
            chroma_client.delete_collection(name=self.collection_name)
        except Exception as e:
            # This is expected if the collection doesn't exist yet
            logger.info(f"Collection '{self.collection_name}' did not exist, creating new.")

        try:
            # Create vector store from documents
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=chroma_client,
                collection_name=self.collection_name,
                persist_directory=None
            )
            logger.info(f"✅ Created vector store with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            # Your fallback logic can remain here
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=f"{self.collection_name}_temp"
            )
        return vector_store

    async def _analyze_individual_papers(
        self,
        papers: List[Dict[str, Any]],
        vector_store: Chroma,
        focus_areas: Optional[List[str]],
        analysis_depth: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze each paper individually using RAG
        
        Concept: Individual Paper Analysis
        - Uses RAG to retrieve relevant context
        - Applies LLM analysis to extract insights
        - Generates structured analysis results
        """
        analyses = []
        
        for i, paper in enumerate(papers):
            try:
                logger.info(f"🔍 Analyzing paper {i+1}/{len(papers)}: {paper.get('title', 'No Title')[:50]}...")
                
                # Prepare paper content for analysis
                paper_content = f"""
                Title: {paper.get('title', 'No Title')}
                Authors: {', '.join(paper.get('authors', []))}
                Published: {paper.get('published_date', 'Unknown')}
                Source: {paper.get('source', 'Unknown')}
                Citations: {paper.get('citations', 0)}
                
                Abstract:
                {paper.get('abstract', 'No abstract available')}
                """
                
                # Generate analysis using LLM
                focus_str = ', '.join(focus_areas) if focus_areas else 'General analysis'
                
                analysis_result = await self._generate_paper_analysis(
                    paper_content, focus_str, analysis_depth
                )
                
                # Calculate relevance score based on various factors
                relevance_score = self._calculate_relevance_score(paper, focus_areas)
                
                # Structure the analysis result
                structured_result = {
                    'paper_id': f"paper_{i}",
                    'paper_index': i,
                    'title': paper.get('title', 'No Title'),
                    'authors': paper.get('authors', []),
                    'source': paper.get('source', 'Unknown'),
                    'url': paper.get('url', ''),
                    'citations': paper.get('citations', 0),
                    'relevance_score': relevance_score,
                    **analysis_result  # Add LLM analysis results
                }
                
                analyses.append(structured_result)
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze paper {i}: {e}")
                # Add minimal result for failed analysis
                analyses.append({
                    'paper_id': f"paper_{i}",
                    'paper_index': i,
                    'title': paper.get('title', 'No Title'),
                    'error': str(e),
                    'relevance_score': 0.0
                })
        
        return analyses
    
    async def _generate_paper_analysis(
        self, 
        paper_content: str, 
        focus_areas: str,
        analysis_depth: str
    ) -> Dict[str, Any]:
        """
        Generate LLM-based analysis of a paper
        
        Concept: LLM Analysis Generation
        - Uses structured prompts for consistent results
        - Extracts themes, concepts, and insights
        - Returns JSON-formatted analysis
        """
        try:
            # Create the prompt
            prompt = self.theme_analysis_prompt.format(
                paper_content=paper_content,
                focus_areas=focus_areas
            )
            
            # Generate analysis using LLM
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            
            # Parse JSON response
            analysis_text = response.content.strip()
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis_json = json.loads(json_str)
            else:
                # Fallback parsing
                analysis_json = {
                    "themes": ["Analysis failed"],
                    "concepts": [],
                    "methodology": "Unknown",
                    "findings": "Analysis could not be completed",
                    "significance": "Unknown"
                }
            
            # Ensure required fields exist
            required_fields = ['themes', 'concepts', 'methodology', 'findings', 'significance']
            for field in required_fields:
                if field not in analysis_json:
                    analysis_json[field] = []
            
            return {
                'themes': analysis_json.get('themes', []),
                'key_concepts': analysis_json.get('concepts', []),
                'methodology': analysis_json.get('methodology', ''),
                'findings': analysis_json.get('findings', ''),
                'significance': analysis_json.get('significance', ''),
                'summary': analysis_json.get('findings', '')[:500] + '...' if len(analysis_json.get('findings', '')) > 500 else analysis_json.get('findings', '')
            }
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {e}")
            return {
                'themes': ['Analysis failed'],
                'key_concepts': [],
                'methodology': 'Unknown',
                'findings': 'Analysis could not be completed due to error',
                'significance': 'Unknown',
                'summary': 'Analysis failed'
            }
    
    def _calculate_relevance_score(
        self, 
        paper: Dict[str, Any], 
        focus_areas: Optional[List[str]]
    ) -> float:
        
        score = 0.6  # ✅ Increased base score from 0.5
        
        # Citation score (more generous)
        citations = paper.get('citations', 0)
        if citations > 0:
            # ✅ Changed scaling to give more points for fewer citations
            citation_score = min(0.3, 0.1 + (citations / 50) * 0.2)
            score += citation_score
        
        # Recency score (unchanged, but could be adjusted)
        pub_date = paper.get('published_date', '')
        if pub_date:
            try:
                if len(pub_date) >= 4:
                    year = int(pub_date[:4])
                    current_year = datetime.now().year
                    years_old = current_year - year
                    
                    if years_old <= 2:
                        score += 0.2
                    elif years_old <= 5:
                        score += 0.1
            except (ValueError, IndexError):
                pass
        
        # Focus area alignment (larger bonus)
        if focus_areas:
            content = f"{paper.get('title', '').lower()} {paper.get('abstract', '').lower()}"
            matched_areas = sum(1 for area in focus_areas if area.lower() in content)
            
            if matched_areas > 0:
                # ✅ Increased max bonus from 0.2 to 0.25
                score += min(0.25, matched_areas * 0.1)
                
        return min(1.0, score)
    
    
    async def _analyze_relationships(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze relationships between pairs of papers
        
        Concept: Cross-Paper Relationship Analysis
        - Identifies thematic connections
        - Finds methodological similarities
        - Detects citing relationships
        - Quantifies relationship strength
        """
        relationships = []
        
        # Analyze pairs of papers
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                try:
                    paper1 = papers[i]
                    paper2 = papers[j]
                    
                    # Generate relationship analysis
                    relationship = await self._analyze_paper_pair(paper1, paper2, i, j)
                    if relationship and relationship.get('relationship_score', 0) > 3:  # Only keep significant relationships
                        relationships.append(relationship)
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze relationship between papers {i} and {j}: {e}")
                    continue
        
        return relationships

    
    async def _analyze_paper_pair(
        self, 
        paper1: Dict[str, Any], 
        paper2: Dict[str, Any],
        index1: int,
        index2: int
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze relationship between two specific papers
        
        Concept: Pairwise Relationship Analysis
        - Uses LLM to identify connections
        - Generates relationship scores and types
        - Provides detailed relationship descriptions
        """
        try:
            # Create relationship analysis prompt
            prompt = self.relationship_prompt.format(
                paper1_title=paper1.get('title', 'No Title'),
                paper1_abstract=paper1.get('abstract', 'No abstract')[:1000],  # Truncate for token limits
                paper2_title=paper2.get('title', 'No Title'),
                paper2_abstract=paper2.get('abstract', 'No abstract')[:1000]
            )
            
            # Generate relationship analysis
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            
            # Parse JSON response
            analysis_text = response.content.strip()
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                relationship_json = json.loads(json_str)
                
                return {
                    'paper1_index': index1,
                    'paper2_index': index2,
                    'paper1_title': paper1.get('title', 'No Title'),
                    'paper2_title': paper2.get('title', 'No Title'),
                    'relationship_score': relationship_json.get('relationship_score', 0),
                    'relationship_type': relationship_json.get('relationship_type', 'unknown'),
                    'shared_concepts': relationship_json.get('shared_concepts', []),
                    'description': relationship_json.get('description', '')
                }
            
        except Exception as e:
            logger.warning(f"Failed to analyze paper pair: {e}")
        
        return None
'''   
if __name__ == "__main__":
    # 1. Check for OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Please set your OPENAI_API_KEY environment variable.")
    else:
        # 2. Create some sample paper data for the test
        sample_papers = [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks... We propose a new simple network architecture, the Transformer, based solely on attention mechanisms...",
                "published_date": "2017-06-12",
                "citations": 100567,
                "source": "ArXiv",
                "url": "https://arxiv.org/abs/1706.03762"
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations...",
                "published_date": "2018-10-11",
                "citations": 83456,
                "source": "ArXiv",
                "url": "https://arxiv.org/abs/1810.04805"
            }
        ]
        
        # 3. Instantiate your agent
        agent = AnalysisAgent()
        
        # 4. Define the focus of your analysis
        focus = ["transformer architecture", "attention mechanism"]
        
        print(f"\n--- 🧪 Starting Analysis Test ---")
        print(f"Focus Areas: {focus}")
        
        # 5. Run the agent's main method
        try:
            results = agent._run(papers=sample_papers, focus_areas=focus)
            
            # 6. Print the results in a readable format
            print("\n--- ✅ Analysis Complete. Results: ---")
            for result in results:
                print(f"\n📄 Paper: {result['title']}")
                print(f"   Score: {result['relevance_score']:.2f}")
                print(f"   Themes: {result.get('themes')}")
                print(f"   Findings: {result.get('findings')}")
                if result.get('relationships'):
                    print("   Relationships:")
                    for rel in result['relationships']:
                        print(f"     - Connects to '{rel['paper2_title']}' (Score: {rel['relationship_score']})")
                        print(f"       Type: {rel['relationship_type']}")

        except Exception as e:
            print(f"\n--- ❌ An error occurred during the test run: {e} ---")
            
'''          
analysis_agent = AnalysisAgent()
