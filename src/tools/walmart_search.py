from typing import Any, Dict, Optional
import httpx
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

async def search_walmart(
    query: str, 
    page: int = 1, 
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort: Optional[str] = None
) -> str:
    """
    Search for products on Walmart using SerpAPI.
    
    Args:
        query: Search query text
        page: Page number (default: 1)
        min_price: Minimum price filter
        max_price: Maximum price filter
        sort: Sort method (e.g. 'price_low', 'price_high', 'best_seller', etc.)
    
    Returns:
        Formatted string with search results
    """
    results = await _make_walmart_search_request(
        query=query,
        page=page,
        min_price=min_price,
        max_price=max_price,
        sort=sort
    )
    
    if not results or "organic_results" not in results:
        return "No results found or error in search."
    
    return _format_walmart_results(results)

async def _make_walmart_search_request(
    query: str,
    page: int = 1,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort: Optional[str] = None
) -> Dict[str, Any]:
    """Make a request to the SerpAPI Walmart search endpoint."""
    params = {
        "engine": "walmart",
        "query": query,
        "api_key": SERPAPI_API_KEY,
        "page": page
    }
    
    # Add optional parameters if provided
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price
    if sort is not None:
        params["sort"] = sort
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://serpapi.com/search.json", 
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error in Walmart search: {str(e)}")
            return {}

def _format_walmart_results(data: Dict[str, Any]) -> str:
    """Format Walmart search results into a readable string."""
    organic_results = data.get("organic_results", [])
    
    if not organic_results:
        return "No products found."
    
    formatted_results = []
    
    for product in organic_results[:5]:  # Show first 5 products
        price = product.get("primary_offer", {}).get("offer_price", "N/A")
        currency = product.get("primary_offer", {}).get("currency", "USD")
        rating = product.get("rating", "N/A")
        reviews = product.get("reviews", 0)
        
        formatted_product = f"""
Product: {product.get('title', 'N/A')}
Price: {price} {currency}
Rating: {rating} ({reviews} reviews)
{product.get('description', 'No description available')[:100]}...
URL: {product.get('product_page_url', 'N/A')}
"""
        formatted_results.append(formatted_product)
    
    total_results = data.get("search_information", {}).get("total_results", 0)
    result_summary = f"Found {total_results} results for '{data.get('search_parameters', {}).get('query', '')}'\n"
    
    return result_summary + "\n---\n".join(formatted_results)

if __name__ == "__main__":
    import asyncio
    
    async def test_walmart_search():
        print("Testing Walmart search...")
        
        results = await search_walmart("coffee maker")
        print(results)
        
        # Optional: test with filters
        filtered_results = await search_walmart(
            "laptop", 
            min_price=300, 
            max_price=1000,
            sort="price_low"
        )
        print("\nFiltered results:")
        print(filtered_results)
    
    asyncio.run(test_walmart_search())
