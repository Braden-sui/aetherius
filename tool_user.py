"""
Tool User (TU) - Sensory Apparatus
---------------------------------
The interface to external APIs and tools, allowing Aetherius to interact with the digital world.
Key responsibilities:
- Managing connections to external APIs (Google Search, Twitter)
- Securely handling credentials and API keys
- Formatting queries for external services
- Processing and structuring responses
- Handling errors and rate limits
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import requests

from google.cloud import secretmanager

# Configure logging
logger = logging.getLogger("aetherius.tool_user")

class ToolType(Enum):
    """Defines the available tool types."""
    GOOGLE_SEARCH = "google_search"
    TWITTER = "twitter"

class ToolUser:
    """
    Manages external tool interactions (Google Search, Twitter API).
    Provides a standardized interface for tool usage, credential management,
    and response processing.
    """
    
    def __init__(self, config: Dict):
        """Initialize the Tool User with configuration parameters."""
        self.config = config
        self.project_id = config.get("project_id")
        
        # Tool-specific configurations
        self.search_config = config.get("search_config", {})
        self.twitter_config = config.get("twitter_config", {})
        
        # API credentials (will be loaded from Secret Manager)
        self.credentials = {}
        
        # Rate limiting and usage tracking
        self.api_call_counts = {
            ToolType.GOOGLE_SEARCH.value: 0,
            ToolType.TWITTER.value: 0
        }
        self.last_call_times = {
            ToolType.GOOGLE_SEARCH.value: None,
            ToolType.TWITTER.value: None
        }
        
        # Load credentials (can be disabled for testing)
        if config.get("load_credentials_on_init", True):
            self._load_credentials()
            
        logger.info("Tool User initialized")
    
    def _load_credentials(self) -> None:
        """Load API credentials from Google Secret Manager."""
        try:
            # Initialize Secret Manager client
            client = secretmanager.SecretManagerServiceClient()
            
            # Credentials to load
            secret_mapping = {
                "google_search_api_key": self.search_config.get("secret_name"),
                "twitter_api_key": self.twitter_config.get("api_key_secret_name"),
                "twitter_api_secret": self.twitter_config.get("api_secret_secret_name"),
                "twitter_access_token": self.twitter_config.get("access_token_secret_name"),
                "twitter_token_secret": self.twitter_config.get("token_secret_secret_name")
            }
            
            # Load each credential if secret name is provided
            for cred_name, secret_name in secret_mapping.items():
                if secret_name:
                    # Build the resource name
                    name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
                    
                    # Access the secret version
                    response = client.access_secret_version(request={"name": name})
                    
                    # Save the secret payload
                    self.credentials[cred_name] = response.payload.data.decode("UTF-8")
                    logger.info(f"Loaded credential: {cred_name}")
                    
            logger.info("Credentials loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {str(e)}")
            # In a real implementation, we might want to raise this error
            # For now, we'll continue with missing credentials for testing
    
    def execute_tool(self, tool_type: ToolType, params: Dict) -> Dict:
        """
        Execute a tool based on the specified type and parameters.
        
        Args:
            tool_type: Type of tool to execute
            params: Tool-specific parameters
            
        Returns:
            Dict: Tool execution results
        """
        logger.info(f"Executing tool: {tool_type.value}")
        
        # Check rate limits and enforce delays if needed
        self._check_rate_limits(tool_type)
        
        # Update tracking
        self.api_call_counts[tool_type.value] += 1
        self.last_call_times[tool_type.value] = datetime.now()
        
        try:
            # Execute the appropriate tool
            if tool_type == ToolType.GOOGLE_SEARCH:
                return self._execute_search(params)
            elif tool_type == ToolType.TWITTER:
                return self._execute_twitter(params)
            else:
                return {
                    "status": "error",
                    "error": f"Unknown tool type: {tool_type.value}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "tool": tool_type.value,
                "params": params,
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_rate_limits(self, tool_type: ToolType) -> None:
        """
        Check rate limits and enforce delays if needed.
        
        Args:
            tool_type: Tool to check rate limits for
        """
        # Get the last call time for this tool
        last_call = self.last_call_times.get(tool_type.value)
        if not last_call:
            return  # First call, no need to check
            
        # Get the minimum interval for this tool
        min_interval_sec = 0
        if tool_type == ToolType.GOOGLE_SEARCH:
            min_interval_sec = self.search_config.get("min_interval_seconds", 2)
        elif tool_type == ToolType.TWITTER:
            min_interval_sec = self.twitter_config.get("min_interval_seconds", 5)
            
        # Calculate time since last call
        now = datetime.now()
        elapsed = (now - last_call).total_seconds()
        
        # If we need to wait, sleep for the required time
        if elapsed < min_interval_sec:
            wait_time = min_interval_sec - elapsed
            logger.info(f"Rate limit enforced for {tool_type.value}. Waiting {wait_time:.2f}s")
            time.sleep(wait_time)
    
    def _execute_search(self, params: Dict) -> Dict:
        """
        Execute a Google Search query.
        
        Args:
            params: Search parameters including 'query' and optional params
            
        Returns:
            Dict: Search results
        """
        query = params.get("query")
        if not query:
            return {
                "status": "error",
                "error": "Missing required parameter: query",
                "timestamp": datetime.now().isoformat()
            }
            
        # Log the query for monitoring
        logger.info(f"Executing search query: {query}")
        
        try:
            # Build the search request
            api_key = self.credentials.get("google_search_api_key")
            cx = self.search_config.get("search_engine_id")
            
            if not api_key or not cx:
                return self._simulate_search_results(query)  # For testing without credentials
                
            # Prepare request parameters
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": params.get("num_results", 5)  # Default to 5 results
            }
            
            # Execute the request
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the response
            search_data = response.json()
            
            # Format the results
            results = []
            if "items" in search_data:
                for item in search_data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "display_link": item.get("displayLink", "")
                    })
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "result_count": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search API request failed: {str(e)}")
            return {
                "status": "error",
                "error": f"Search API request failed: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Unexpected error in search execution: {str(e)}")
            return {
                "status": "error",
                "error": f"Search execution failed: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_twitter(self, params: Dict) -> Dict:
        """
        Execute a Twitter API operation.
        
        Args:
            params: Twitter parameters including 'operation' and operation-specific params
            
        Returns:
            Dict: Twitter operation results
        """
        operation = params.get("operation")
        if not operation:
            return {
                "status": "error",
                "error": "Missing required parameter: operation",
                "timestamp": datetime.now().isoformat()
            }
            
        # Log the operation for monitoring
        logger.info(f"Executing Twitter operation: {operation}")
        
        # Check API tier for rate limits
        api_tier = self.twitter_config.get("api_tier", "free")
        
        # Track rate limits based on API documentation
        rate_limits = {
            "free": {
                "search_tweets": 1,  # 1 request / 15 mins
                "get_user": 3,       # 3 requests / 15 mins
                "get_tweets": 1      # 1 request / 15 mins
            },
            "basic": {
                "search_tweets": 60,  # 60 requests / 15 mins
                "get_user": 100,      # 100 requests / 24 hours
                "get_tweets": 15      # 15 requests / 15 mins
            },
            "pro": {
                "search_tweets": 300,  # 300 requests / 15 mins
                "get_user": 900,       # 900 requests / 15 mins
                "get_tweets": 900      # 900 requests / 15 mins
            }
        }
        
        # Check if we've exceeded rate limits for this operation
        # In a real implementation, we'd maintain a counter and timestamp for each operation
        # and check against the appropriate tier's limits
        
        try:
            # In a full implementation, we would use the Twitter API v2
            # For now, we'll simulate results for demonstration
            return self._simulate_twitter_results(operation, params)
            
        except Exception as e:
            logger.error(f"Twitter API operation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "operation": operation,
                "timestamp": datetime.now().isoformat()
            }
    
    def _simulate_search_results(self, query: str) -> Dict:
        """
        Simulate search results for testing without API credentials.
        
        Args:
            query: Search query
            
        Returns:
            Dict: Simulated search results
        """
        logger.warning(f"Using simulated search results for query: {query}")
        
        # Create simulated results based on the query
        simulated_results = []
        
        if "AI" in query or "artificial intelligence" in query.lower():
            simulated_results = [
                {
                    "title": "What is Artificial Intelligence (AI)?",
                    "link": "https://www.example.com/ai-definition",
                    "snippet": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
                    "display_link": "example.com"
                },
                {
                    "title": "Latest Developments in AI Research",
                    "link": "https://www.example.com/ai-research",
                    "snippet": "Recent breakthroughs in large language models and reinforcement learning have pushed the boundaries of AI capabilities.",
                    "display_link": "example.com"
                }
            ]
        elif "ethics" in query.lower():
            simulated_results = [
                {
                    "title": "AI Ethics Principles and Frameworks",
                    "link": "https://www.example.com/ai-ethics",
                    "snippet": "Key AI ethics frameworks focus on fairness, accountability, transparency, and explainability.",
                    "display_link": "example.com"
                }
            ]
        else:
            simulated_results = [
                {
                    "title": f"Results for: {query}",
                    "link": "https://www.example.com/search",
                    "snippet": "This is a simulated search result for demonstration purposes.",
                    "display_link": "example.com"
                }
            ]
        
        return {
            "status": "success",
            "query": query,
            "results": simulated_results,
            "result_count": len(simulated_results),
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }
    
    def _simulate_twitter_results(self, operation: str, params: Dict) -> Dict:
        """
        Simulate Twitter API results for testing without API credentials.
        
        Args:
            operation: Twitter operation type
            params: Operation parameters
            
        Returns:
            Dict: Simulated Twitter API results
        """
        logger.warning(f"Using simulated Twitter results for operation: {operation}")
        
        if operation == "search_recent":
            query = params.get("query", "")
            return self._simulate_twitter_search(query)
            
        elif operation == "get_user":
            username = params.get("username", "")
            return self._simulate_twitter_user(username)
            
        elif operation == "get_user_timeline":
            user_id = params.get("user_id", "")
            return self._simulate_twitter_timeline(user_id)
            
        else:
            return {
                "status": "error",
                "error": f"Unsupported Twitter operation: {operation}",
                "timestamp": datetime.now().isoformat(),
                "simulated": True
            }
    
    def _simulate_twitter_search(self, query: str) -> Dict:
        """
        Simulate Twitter search results.
        
        Args:
            query: Search query
            
        Returns:
            Dict: Simulated search results
        """
        simulated_tweets = []
        
        # Generate simulated results based on query
        if "AI" in query or "artificial intelligence" in query.lower():
            simulated_tweets = [
                {
                    "id": "1234567890",
                    "text": "Just read a fascinating paper on the latest developments in AI reasoning capabilities. The potential applications in scientific research are mind-blowing!",
                    "created_at": "2025-04-05T10:30:00Z",
                    "author": {
                        "id": "12345",
                        "username": "ai_enthusiast",
                        "name": "AI Enthusiast"
                    },
                    "public_metrics": {
                        "retweet_count": 45,
                        "reply_count": 12,
                        "like_count": 128,
                        "quote_count": 8
                    }
                },
                {
                    "id": "1234567891",
                    "text": "AI ethics should be a mandatory subject in every computer science curriculum. We need more emphasis on responsible development.",
                    "created_at": "2025-04-05T14:22:00Z",
                    "author": {
                        "id": "12346",
                        "username": "ethicsintech",
                        "name": "Ethics in Technology"
                    },
                    "public_metrics": {
                        "retweet_count": 87,
                        "reply_count": 34,
                        "like_count": 215,
                        "quote_count": 12
                    }
                }
            ]
        elif "ethics" in query.lower():
            simulated_tweets = [
                {
                    "id": "1234567892",
                    "text": "New ethics framework proposed for large language models focuses on transparency, fairness, and accountability. An important step forward.",
                    "created_at": "2025-04-05T09:15:00Z",
                    "author": {
                        "id": "12347",
                        "username": "ethics_prof",
                        "name": "Professor of Tech Ethics"
                    },
                    "public_metrics": {
                        "retweet_count": 56,
                        "reply_count": 23,
                        "like_count": 134,
                        "quote_count": 7
                    }
                }
            ]
        else:
            simulated_tweets = [
                {
                    "id": "1234567893",
                    "text": f"This is a simulated tweet about: {query}",
                    "created_at": "2025-04-05T16:45:00Z",
                    "author": {
                        "id": "12348",
                        "username": "simulated_user",
                        "name": "Simulated User"
                    },
                    "public_metrics": {
                        "retweet_count": 12,
                        "reply_count": 5,
                        "like_count": 42,
                        "quote_count": 2
                    }
                }
            ]
        
        return {
            "status": "success",
            "operation": "search_recent",
            "query": query,
            "tweets": simulated_tweets,
            "result_count": len(simulated_tweets),
            "newest_id": simulated_tweets[0]["id"] if simulated_tweets else None,
            "oldest_id": simulated_tweets[-1]["id"] if simulated_tweets else None,
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }
    
    def _simulate_twitter_user(self, username: str) -> Dict:
        """
        Simulate Twitter user profile data.
        
        Args:
            username: Twitter username
            
        Returns:
            Dict: Simulated user data
        """
        # Create a simulated user based on the username
        user = {
            "id": f"{hash(username) % 1000000:06d}",
            "username": username,
            "name": f"{username.replace('_', ' ').title()}",
            "created_at": "2019-02-15T15:00:00Z",
            "description": f"This is a simulated Twitter profile for {username}",
            "public_metrics": {
                "followers_count": 1250,
                "following_count": 450,
                "tweet_count": 3275,
                "listed_count": 15
            },
            "verified": False,
            "protected": False,
            "location": "Internet",
            "url": f"https://example.com/{username}"
        }
        
        return {
            "status": "success",
            "operation": "get_user",
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }
    
    def _simulate_twitter_timeline(self, user_id: str) -> Dict:
        """
        Simulate a user's Twitter timeline.
        
        Args:
            user_id: Twitter user ID
            
        Returns:
            Dict: Simulated timeline data
        """
        # Create simulated timeline tweets
        timeline_tweets = [
            {
                "id": f"{user_id}01",
                "text": "This is my most recent simulated tweet. Exploring new ideas and sharing thoughts.",
                "created_at": "2025-04-05T18:30:00Z",
                "public_metrics": {
                    "retweet_count": 8,
                    "reply_count": 3,
                    "like_count": 27,
                    "quote_count": 1
                }
            },
            {
                "id": f"{user_id}02",
                "text": "Just read an interesting article about emerging technologies and their potential impact on society.",
                "created_at": "2025-04-04T14:20:00Z",
                "public_metrics": {
                    "retweet_count": 12,
                    "reply_count": 5,
                    "like_count": 45,
                    "quote_count": 2
                }
            },
            {
                "id": f"{user_id}03",
                "text": "Participated in a fascinating discussion about the future of AI and human collaboration.",
                "created_at": "2025-04-03T09:15:00Z",
                "public_metrics": {
                    "retweet_count": 15,
                    "reply_count": 7,
                    "like_count": 63,
                    "quote_count": 3
                }
            }
        ]
        
        return {
            "status": "success",
            "operation": "get_user_timeline",
            "user_id": user_id,
            "tweets": timeline_tweets,
            "result_count": len(timeline_tweets),
            "newest_id": timeline_tweets[0]["id"],
            "oldest_id": timeline_tweets[-1]["id"],
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }