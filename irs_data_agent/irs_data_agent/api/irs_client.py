"""
Client for interacting with IRS APIs.
"""
from typing import Dict, List, Any, Optional
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class IRSClient:
    """
    Client for interacting with IRS data sources.
    This is a simplified mock implementation for demonstration purposes.
    """
    
    BASE_URL = "https://www.irs.gov/api"  # This is a mock URL
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the IRS client.
        
        Args:
            api_key: Optional API key for authenticated requests
        """
        self.api_key = api_key
        self.session = requests.Session()
        
        # For demo purposes, we'll use mock data instead of real API calls
        self.mock_data = self._initialize_mock_data()
    
    def _initialize_mock_data(self) -> Dict[str, Any]:
        """Initialize mock data for demonstration purposes."""
        current_year = datetime.now().year
        
        return {
            "forms": [
                {
                    "form_number": "1040",
                    "title": "U.S. Individual Income Tax Return",
                    "description": "Form used for personal federal income tax returns filed by citizens or residents of the United States.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/f1040.pdf",
                    "instructions_url": "https://www.irs.gov/pub/irs-pdf/i1040gi.pdf",
                    "relevance_score": 0.95
                },
                {
                    "form_number": "W-2",
                    "title": "Wage and Tax Statement",
                    "description": "Form showing wages paid to employees and taxes withheld from them.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/fw2.pdf",
                    "instructions_url": "https://www.irs.gov/pub/irs-pdf/iw2w3.pdf",
                    "relevance_score": 0.9
                },
                {
                    "form_number": "1099-MISC",
                    "title": "Miscellaneous Income",
                    "description": "Form that reports payments made in the course of a trade or business to others for services.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/f1099msc.pdf",
                    "instructions_url": "https://www.irs.gov/pub/irs-pdf/i1099msc.pdf",
                    "relevance_score": 0.85
                },
                {
                    "form_number": "1099-NEC",
                    "title": "Nonemployee Compensation",
                    "description": "Form used to report payments made to nonemployees, such as independent contractors.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/f1099nec.pdf",
                    "instructions_url": "https://www.irs.gov/pub/irs-pdf/i1099gi.pdf",
                    "relevance_score": 0.8
                },
                {
                    "form_number": "1065",
                    "title": "U.S. Return of Partnership Income",
                    "description": "Form used to report the income, gains, losses, deductions, credits, etc., from the operation of a partnership.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/f1065.pdf",
                    "instructions_url": "https://www.irs.gov/pub/irs-pdf/i1065.pdf",
                    "relevance_score": 0.75
                }
            ],
            "publications": [
                {
                    "pub_number": "17",
                    "title": "Your Federal Income Tax",
                    "description": "General guide to individual income tax returns.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/p17.pdf",
                    "relevance_score": 0.9
                },
                {
                    "pub_number": "463",
                    "title": "Travel, Gift, and Car Expenses",
                    "description": "Explains tax rules for business travel, entertainment, gifts, and transportation.",
                    "year": current_year,
                    "url": "https://www.irs.gov/pub/irs-pdf/p463.pdf",
                    "relevance_score": 0.7
                }
            ]
        }
    
    def search(self, query: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search IRS database for forms and publications matching the query.
        
        Args:
            query: Search term
            year: Optional tax year to filter results
            
        Returns:
            List of matching forms and publications
        """
        logger.info(f"Searching for: {query}, year: {year}")
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll filter our mock data
        
        results = []
        
        # Search in forms
        for form in self.mock_data["forms"]:
            if (year is None or form["year"] == year) and self._matches_query(form, query):
                results.append(form)
        
        # Search in publications
        for pub in self.mock_data["publications"]:
            if (year is None or pub["year"] == year) and self._matches_query(pub, query):
                results.append(pub)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return results
    
    def _matches_query(self, item: Dict[str, Any], query: str) -> bool:
        """Check if an item matches the search query."""
        query = query.lower()
        
        # Check various fields for matches
        if query in str(item.get("form_number", "")).lower():
            return True
        if query in str(item.get("pub_number", "")).lower():
            return True
        if query in item.get("title", "").lower():
            return True
        if query in item.get("description", "").lower():
            return True
        
        return False
    
    def get_form(self, form_number: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific IRS form by form number.
        
        Args:
            form_number: IRS form number (e.g., "1040")
            year: Optional tax year
            
        Returns:
            Form information or None if not found
        """
        logger.info(f"Retrieving form: {form_number}, year: {year}")
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll search our mock data
        
        for form in self.mock_data["forms"]:
            if form["form_number"].lower() == form_number.lower():
                if year is None or form["year"] == year:
                    return form
        
        return None
    
    def get_publication(self, pub_number: str, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific IRS publication by publication number.
        
        Args:
            pub_number: IRS publication number (e.g., "17")
            year: Optional tax year
            
        Returns:
            Publication information or None if not found
        """
        logger.info(f"Retrieving publication: {pub_number}, year: {year}")
        
        # In a real implementation, this would make an API request
        # For demo purposes, we'll search our mock data
        
        for pub in self.mock_data["publications"]:
            if pub["pub_number"] == pub_number:
                if year is None or pub["year"] == year:
                    return pub
        
        return None
    
    def list_forms(self, form_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available IRS forms, optionally filtered by type.
        
        Args:
            form_type: Optional form type to filter by
            
        Returns:
            List of forms
        """
        logger.info(f"Listing forms, type filter: {form_type}")
        
        forms = self.mock_data["forms"]
        
        if form_type:
            # This is a simplified filter - in a real implementation,
            # we would have a more sophisticated categorization
            forms = [f for f in forms if form_type.lower() in f["description"].lower()]
        
        return forms
