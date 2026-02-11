from ollama import chat, Client
from ollama import ChatResponse
from collections import OrderedDict
import fitz
import re
pdf_path = "adv_res_paper.pdf"


import PyPDF2

text = ""
with open('adv_res_paper.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    
    for page in reader.pages:
      text += page.extract_text()
    # Now send this text to Llama 3.2 (text model)
# print(text)
def extract_citations_by_section(text, sections):
    """
    Extract citation blocks with preceding sentences organized by sections.
    
    Args:
        text: The full document text
        sections: Dictionary defining section structure (nested for subsections)
    
    Returns:
        Nested dictionary with citations
    """
    result = {}
    
    # Pattern to match citations like (Hendrycks et al., 2020) or (Author, 2020)
    citation_pattern = r'\([A-Z][^)]*\d{4}[a-z]?\)'
    
    def process_section_text(section_text):
      """Extract citations with context from paragraph start to next period."""
      citations_dict = {}
      
      # Remove single newline characters (keep double newlines for paragraph breaks)
      section_text = section_text.replace('\n', '')
      section_text = re.sub(r'\s+', ' ', section_text)  # Normalize whitespace
      
      # Find all citations with their positions
      for match in re.finditer(citation_pattern, section_text):
          citation_block = match.group(0)  # e.g., "(Hendrycks et al., 2020)"
          citation_start = match.start()
          citation_end = match.end()
          
          # Find the beginning of the paragraph (last period before citation)
          text_before = section_text[:citation_start]
          paragraph_start = text_before.rfind('.')
          if paragraph_start != -1:
              paragraph_start += 1  # Move past the period
          else:
              paragraph_start = 0  # Start from beginning if no period found
          
          # Find the next period after the citation
          text_after = section_text[citation_end:]
          next_period = text_after.find('.')
          
          if next_period != -1:
              # Extract from paragraph start to next period after citation
              sentence = section_text[paragraph_start:citation_end + next_period + 1].strip()
          else:
              # If no period found after, take until end of text
              sentence = section_text[paragraph_start:].strip()
          
          citations_dict[citation_block] = sentence
      
      return citations_dict
    
    def extract_section_content(text, section_name, next_section_name=None):
        """Extract text content for a specific section."""
        # Find section start
        section_start = text.find(section_name)
        if section_start == -1:
            return ""
        
        # Find section end (start of next section or end of document)
        if next_section_name:
            section_end = text.find(next_section_name, section_start + len(section_name))
            if section_end == -1:
                section_end = len(text)
        else:
            section_end = len(text)
        
        return text[section_start + len(section_name):section_end]
    
    # Process sections recursively
    def process_sections(text, sections_dict, section_names_list):
        """Recursively process sections and subsections."""
        local_result = {}
        local_content = {}
        for i, section_name in enumerate(section_names_list):
            next_section = section_names_list[i + 1] if i + 1 < len(section_names_list) else None
            section_text = extract_section_content(text, section_name, next_section)
            # section_content[section_name] = section_names
            if not section_text:
                continue
            
            # Check if this section has subsections
            subsections = sections_dict[section_name]
            
            if subsections and isinstance(subsections, dict):
                # Has subsections - process recursively
                subsection_names = list(subsections.keys())
                local_result[section_name], sub_content  = process_sections(section_text, subsections, subsection_names)
                local_content[section_name] = sub_content
            else:
                # No subsections - extract citations
                local_result[section_name] = process_section_text(section_text)
                cleaned_content = section_text.replace('\n', '')
                cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
                local_content[section_name] = cleaned_content
        
        
        return local_result, local_content
    
    section_names = list(sections.keys())
    result,content_result = process_sections(text, sections, section_names)
    
    return result, content_result


# Example usage:

# Define your section structure
sections = {
    'Introduction': None,  # No subsections
    'System Overview': None,
    'Team of Agents': None,
    'CrS-Aware Aggregation': None,
    'Learning Credibility Scores on-the-fly': {'Calculating the agent contributions': None,
                                               'Updating the CrS values':None
                                               },
    'Experiment Results': {
        'Experiments Setting': None,
        'Collaboration Setup': None,
        'Insights from Experimental Observations': {
            'Credibility Scores Drive Consistent Gains': None,
            'Reasoning vs Multi-Choice Tasks': None,
            'Model Capacity Matters But Only With Coordination': None,
            'Judge-Computed CrS Imitates the Shapley Value': None,
            'Judge Alters the Outcome' :None,
            'Topology and Link Density': None,
            'Adversary Proportion':None
        }    
    },
    'Conclusion': None,
    'Limitations' : None
}
import re
import json
from ollama import Client

def assign_importance_scores(content_dict, citations_dict, model='llama3.2', host="localhost:11434"):
    """
    Use LLM to assign hierarchical importance scores to sections, subsections, and citations.
    Handles arbitrary nesting levels.
    
    Args:
        content_dict: Dictionary with section/subsection content
        citations_dict: Dictionary with citations organized by section
        model: The LLM model to use
        host: Ollama host address
    
    Returns:
        Tuple of (citation_scores, section_scores)
        - citation_scores: Dictionary with importance scores for each citation
        - section_scores: Dictionary with importance scores for sections and subsections
    """
    client = Client(host=host)
    citation_scores = {}
    section_scores = {}
    
    # Step 1: Divide score of 1 among top-level sections
    section_names = list(content_dict.keys())
    
    system_template = """You are an expert at analyzing academic papers and determining the relative importance of different sections and citations. 
Your task is to assign importance scores that sum to exactly 1.0 (or the given total) based on the content's relevance to the paper's main contributions.
Always respond with valid JSON only, no additional text."""
    
    prompt = f"""Given the following sections from an academic paper, divide a total score of 1.0 among them based on their importance to the paper's main contribution.

Sections:
{json.dumps(section_names, indent=2)}

Respond with ONLY a JSON object in this format:
{{
  "section_name_1": 0.3,
  "section_name_2": 0.5,
  "section_name_3": 0.2
}}

Make sure the scores sum to exactly 1.0."""
    
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat(model, messages=messages)
    top_level_scores = parse_json_response(response.message.content)
    
    # Store top-level section scores
    for section_name, score in top_level_scores.items():
        section_scores[section_name] = {
            'score': score,
            'type': 'section',
            'level': 1,
            'subsections': {}
        }
    
    print("Section Scores:")
    print(json.dumps(top_level_scores, indent=2))
    
    # Step 2: For each section, recursively assign scores to subsections
    def process_section(section_name, section_content, section_score, citations, level=1, parent_ref=None):
        """Recursively process sections and subsections at any nesting level."""
        
        # Determine where to store subsection scores
        if parent_ref is None:
            # Top-level section
            current_section_ref = section_scores[section_name]
        else:
            # Subsection - parent_ref points to the parent's subsections dict
            if section_name not in parent_ref:
                parent_ref[section_name] = {
                    'score': section_score,
                    'type': 'subsection',
                    'level': level,
                    'subsections': {}
                }
            current_section_ref = parent_ref[section_name]
        
        # Check if this section has subsections
        if isinstance(section_content, dict):
            # Has potential subsections - filter out special keys
            subsection_names = [k for k in section_content.keys() if k != '_full_text']
            
            if subsection_names:
                # Ask LLM to divide score among subsections
                prompt = f"""Given the following subsections of "{section_name}", divide a total score of {section_score} among them based on their importance.

Subsections:
{json.dumps(subsection_names, indent=2)}

Respond with ONLY a JSON object in this format:
{{
  "subsection_name_1": 0.4,
  "subsection_name_2": 0.6
}}

Make sure the scores sum to exactly {section_score}."""
                system_template = """You are an expert at analyzing academic papers and determining the relative importance of different subsections and citations. 
                Your task is to assign importance scores that sum to exactly {section_score} based on the content's relevance to the paper's main contributions.
                Always respond with valid JSON only, no additional text."""            
                messages = [
                    {"role": "system", "content": system_template},
                    {"role": "user", "content": prompt}
                ]
                
                response = client.chat(model, messages=messages)
                subsection_scores_dict = parse_json_response(response.message.content)
                
                print(f"\n{'  ' * (level-1)}Subsection Scores for '{section_name}' (Level {level}):")
                print(json.dumps(subsection_scores_dict, indent=2))
                
                # Recursively process each subsection
                for subsection_name in subsection_names:
                    subsection_score = subsection_scores_dict.get(subsection_name, section_score / len(subsection_names))
                    subsection_content = section_content[subsection_name]
                    
                    # Get citations for this subsection
                    if isinstance(citations, dict):
                        subsection_citations = citations.get(subsection_name, {})
                    else:
                        subsection_citations = {}
                    
                    # Recursively process this subsection
                    # Pass current_section_ref['subsections'] as the parent for the next level
                    process_section(
                        subsection_name, 
                        subsection_content, 
                        subsection_score, 
                        subsection_citations, 
                        level + 1, 
                        current_section_ref['subsections']
                    )
            else:
                # No subsections, just full text - process citations
                assign_citation_scores(section_name, section_score, citations, level)
        else:
            # Leaf node (string content) - assign scores to citations
            assign_citation_scores(section_name, section_score, citations, level)
    
    def assign_citation_scores(section_name, section_score, section_citations, level):
        """Assign scores to citations in a section."""
        if not section_citations or not isinstance(section_citations, dict):
            return
        
        citation_blocks = list(section_citations.keys())
        
        if not citation_blocks:
            return
        
        # Count total individual citations (handling multiple citations in one block)
        total_citations = 0
        citation_counts = {}
        
        for citation_block in citation_blocks:
            # Count citations in this block
            # Pattern matches citations like (Author, 2020) or (Author1, 2020; Author2, 2021)
            num_citations = citation_block.count(';') + 1  # Simple count based on semicolons
            citation_counts[citation_block] = max(1, num_citations)
            total_citations += citation_counts[citation_block]
        
        # Distribute score uniformly among all individual citations
        score_per_citation = section_score / total_citations if total_citations > 0 else 0
        
        for citation_block in citation_blocks:
            num_in_block = citation_counts[citation_block]
            score = score_per_citation * num_in_block
            citation_scores[citation_block] = {
                'score': score,
                'section': section_name,
                'level': level,
                'context': section_citations[citation_block]
            }
        
        print(f"\n{'  ' * (level-1)}Citation Scores for '{section_name}' (Level {level}):")
        for citation_block, info in citation_scores.items():
            if info['section'] == section_name:
                print(f"{'  ' * level}{citation_block}: {info['score']:.4f}")
    
    # Process each top-level section
    for section_name in section_names:
        section_score = top_level_scores.get(section_name, 1.0 / len(section_names))
        section_content = content_dict[section_name]
        section_citations = citations_dict.get(section_name, {})
        
        process_section(section_name, section_content, section_score, section_citations, level=1, parent_ref=None)
    
    return citation_scores, section_scores


def parse_json_response(response_text):
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Remove markdown code blocks if present
    response_text = response_text.strip()
    
    # Remove ```json and ``` markers
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        response_text = response_text.replace('```json', '').replace('```', '').strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response text: {response_text}")
        return {}


def print_section_hierarchy(section_scores, indent=0):
    """Pretty print the section hierarchy with scores."""
    for section_name, section_data in section_scores.items():
        score = section_data['score']
        level = section_data['level']
        section_type = section_data['type']
        
        print(f"{'  ' * indent}{section_name} [{section_type}, Level {level}]: {score:.4f}")
        
        # Recursively print subsections
        if section_data['subsections']:
            print_section_hierarchy(section_data['subsections'], indent + 1)


# Example usage:
if __name__ == "__main__":
    # Extract citations and content
    sections = {
    'Introduction': None,
    'System Overview': None,
    'Team of Agents': None,
    'CrS-Aware Aggregation': None,
    'Learning Credibility Scores On-The-Fly': None,  # Add this
    'Experiment Results': {
        'Experiments Setting': None,
        'Collaboration Setup': None,
        'Insights from Experimental Observations': {
            'Credibility Scores Drive Consistent Gains': None,
            'Reasoning vs Multi-Choice Tasks': None,
            'Model Capacity Matters But Only With Coordination': None,
            'Judge-Computed CrS Imitates the Shapley Value': None,
            'Judge Alters the Outcome': None,
            'Topology and Link Density': None,
            'Adversary Proportion': None
        }
    },
    'Conclusion': None,
    'Limitations': None
    }

    citations, content = extract_citations_by_section(text, sections)
    
    # Assign importance scores
    citation_importance, section_importance = assign_importance_scores(content, citations)
    
    # Save citation results
    with open('citation_importance_scores.json', 'w') as f:
        json.dump(citation_importance, f, indent=2)
    
    # Save section results
    with open('section_importance_scores.json', 'w') as f:
        json.dump(section_importance, f, indent=2)
    
    print("\n" + "="*50)
    print("FINAL CITATION SCORES:")
    print("="*50)
    
    # Sort by score descending
    sorted_citations = sorted(citation_importance.items(), 
                             key=lambda x: x[1]['score'], 
                             reverse=True)
    
    for citation, info in sorted_citations[:10]:  # Top 10
        print(f"{citation}: {info['score']:.4f} (in {info['section']}, Level {info['level']})")
    
    print("\n" + "="*50)
    print("SECTION HIERARCHY WITH SCORES:")
    print("="*50)
    print_section_hierarchy(section_importance)
    
    print("\n" + "="*50)
    print("FULL SECTION SCORES JSON:")
    print("="*50)
    print(json.dumps(section_importance, indent=2))
