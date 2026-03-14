import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.core.config import settings

def get_llm():
    """
    Returns the configured LLM based on provider settings.
    """
    if settings.EMBEDDING_PROVIDER == "google":
        model_name = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )
    else:
        model_name = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

def extract_metadata_from_text(text: str) -> dict:
    """
    Use an LLM to extract the title and write a short abstract.
    We only pass the first 5000 characters to save tokens/cost, 
    since title and abstract are usually at the beginning of a paper.
    """
    llm = get_llm()
    preview_text = text[:8000]

    prompt_template = PromptTemplate.from_template(
        """You are a helpful AI assistant analyzing a scientific research paper.
Please read the beginning of the text below and extract:
1. The exactly matching Title of the paper.
2. A very brief Abstract (max 3 sentences) summarizing what the paper is about.

Text:
{text}

Output format ONLY in strict format:
Title: <title>
Abstract: <abstract>"""
    )
    
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({"text": preview_text})
        output = response.content.strip()
        
        # Parse the output
        lines = output.split('\n')
        title = "Untitled Extract"
        abstract = "Could not generate abstract."
        
        for line in lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Abstract:"):
                abstract = line.replace("Abstract:", "").strip()
                
        return {"title": title, "abstract": abstract}
        
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return {"title": "Unknown Title", "abstract": "Extraction failed."}
