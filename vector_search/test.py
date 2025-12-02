import requests
import os
from datetime import datetime
import time
import json

# Europe PMC API Configuration
EUROPEPMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
SEARCH_QUERY = "long covid AND treatment"  # Modify this to search for different topics
MAX_PAPERS = 100  # Number of papers to download per query

# Additional search queries (optional - can search multiple topics)
ADDITIONAL_QUERIES = [
    # "mRNA vaccine efficacy",
    # "diabetes management",
    # "cancer immunotherapy"
]

# Embedding configuration
EMBEDDING_MODEL = "databricks-bge-large-en"  # Databricks Foundation Model
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks


def download_medical_papers(query, output_path, max_papers=100):
    """
    Download medical research papers from Europe PMC API and save to volume
    
    Args:
        query: Search query (e.g., "long covid AND treatment")
        output_path: Path to save downloaded papers
        max_papers: Maximum number of papers to download
    
    Returns:
        Dictionary with download results
    """
    try:
        print(f"Downloading papers from Europe PMC...")
        print(f"Search Query: '{query}'")
        print(f"Max Papers: {max_papers}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Europe PMC API parameters
        base_url = EUROPEPMC_API_URL
        params = {
            'query': query,
            'resultType': 'core',  # Get full details
            'format': 'json',
            'pageSize': min(max_papers, 1000),  # Max 1000 per request
            'cursorMark': '*'  # For pagination
        }
        
        # Make API request
        print(f"Calling Europe PMC API...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract results
        results = data.get('resultList', {}).get('result', [])
        total_found = data.get('hitCount', 0)
        
        print(f"✓ API returned {len(results)} papers (total available: {total_found})")
        
        if len(results) == 0:
            print("⚠️  No papers found for this query. Try a different search term.")
            return {
                "papers_downloaded": 0,
                "query": query,
                "success": False,
                "error": "No results found"
            }
        
        # Process and save each paper
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        papers_saved = []
        
        for idx, paper in enumerate(results[:max_papers], 1):
            # Extract paper details
            paper_id = paper.get('pmcid', paper.get('pmid', f'paper_{idx}'))
            title = paper.get('title', 'No title')
            abstract = paper.get('abstractText', '')
            authors = paper.get('authorString', 'Unknown')
            journal = paper.get('journalTitle', 'Unknown')
            pub_year = paper.get('pubYear', 'Unknown')
            doi = paper.get('doi', '')
            
            # Create RTF formatted content saved as .doc (opens in Word/Pages)
            # RTF format with .doc extension - no external libraries needed
            safe_id = paper_id.replace('/', '_').replace(' ', '_')
            doc_filename = f"{output_path}/paper_{safe_id}_{timestamp}.doc"
            
            # Escape special RTF characters
            def escape_rtf(text):
                if not text:
                    return ""
                text = str(text)
                text = text.replace('\\', '\\\\')
                text = text.replace('{', '\\{')
                text = text.replace('}', '\\}')
                text = text.replace('\n', '\\par ')
                return text
            
            title_rtf = escape_rtf(title)
            authors_rtf = escape_rtf(authors)
            journal_rtf = escape_rtf(journal)
            abstract_rtf = escape_rtf(abstract)
            doi_rtf = escape_rtf(doi)
            
            rtf_content = r"""{\rtf1\ansi\deff0
{\fonttbl{\f0\fswiss Arial;}{\f1\froman Times New Roman;}}
{\colortbl;\red0\green0\blue0;\red44\green62\blue80;\red127\green140\blue141;}
\f1\fs24

{\b\fs32\cf2 """ + title_rtf + r"""}
\par\par

{\b Paper ID:} """ + escape_rtf(paper_id) + r"""
\par
{\b Authors:} """ + authors_rtf + r"""
\par
{\b Journal:} """ + journal_rtf + r"""
\par
{\b Year:} """ + escape_rtf(str(pub_year)) + r"""
\par
{\b DOI:} """ + doi_rtf + r"""
\par\par

{\b\fs28 Abstract}
\par
""" + abstract_rtf + r"""
\par\par

{\i\fs18\cf3 Downloaded from Europe PMC | Full metadata available in JSON file}
}"""
            
            # Save as .doc file (opens in Word/Pages/LibreOffice)
            with open(doc_filename, 'w', encoding='utf-8') as f:
                f.write(rtf_content)
            
            # Save raw JSON metadata
            json_filename = f"{output_path}/paper_{safe_id}_{timestamp}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(paper, f, indent=2)
            
            papers_saved.append({
                "paper_id": paper_id,
                "title": title[:100] + "..." if len(title) > 100 else title,
                "doc_file": doc_filename,
                "json_file": json_filename
            })
            
            # Progress indicator
            if idx % 10 == 0:
                print(f"  Processed {idx}/{min(len(results), max_papers)} papers...")
        
        print(f"✓ Successfully saved {len(papers_saved)} papers to volume")
        
        # Create a summary file
        summary_filename = f"{output_path}/download_summary_{timestamp}.json"
        summary = {
            "timestamp": timestamp,
            "query": query,
            "total_available": total_found,
            "papers_downloaded": len(papers_saved),
            "papers": papers_saved
        }
        
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_filename}")
        
        return {
            "query": query,
            "papers_downloaded": len(papers_saved),
            "total_available": total_found,
            "summary_file": summary_filename,
            "success": True
        }
        
    except requests.exceptions.RequestException as e:
        print(f"✗ API Error: {str(e)}")
        return {"success": False, "error": f"API Error: {str(e)}"}
    except Exception as e:
        print(f"✗ Error downloading papers: {str(e)}")
        return {"success": False, "error": str(e)}

# Download papers for primary query
print("="*80)
print("DOWNLOADING MEDICAL RESEARCH PAPERS")
print("="*80)

result = download_medical_papers(SEARCH_QUERY, 'data/', max_papers=MAX_PAPERS)
print(f"\n✓ Download complete!")
print(f"  Query: {result.get('query', 'N/A')}")
print(f"  Papers downloaded: {result.get('papers_downloaded', 0)}")
print(f"  Total available: {result.get('total_available', 0)}")

# Optional: Download papers for additional queries
if ADDITIONAL_QUERIES:
    print(f"\nDownloading {len(ADDITIONAL_QUERIES)} additional query topics...")
    for additional_query in ADDITIONAL_QUERIES:
        print(f"\n--- Query: '{additional_query}' ---")
        result_additional = download_medical_papers(additional_query, 'data/', max_papers=MAX_PAPERS)
        print(f"✓ Downloaded {result_additional.get('papers_downloaded', 0)} papers")
        time.sleep(1)  # Be nice to the API