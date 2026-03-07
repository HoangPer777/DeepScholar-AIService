from fastapi import APIRouter

from app.schemas.request import ResearchRequest


router = APIRouter()


@router.post("/deep-search")
async def deep_search(request: ResearchRequest):
    """
    TODO: Execute deep research workflow
    1. Parse research query
    2. Call external_search for web research
    3. Augment with internal article search
    4. Return aggregated results with sources
    """
    # TODO: Implementation
    return {"results": []}