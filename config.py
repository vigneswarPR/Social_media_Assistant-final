import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    INSTAGRAM_USERNAME = os.getenv('INSTAGRAM_USERNAME')
    INSTAGRAM_PASSWORD = os.getenv('INSTAGRAM_PASSWORD')
    INSTAGRAM_ACCESS_TOKEN = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

    # Social Media API Configuration
    GRAPH_API_VERSION = "v19.0"
    GRAPH_API_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/"
    
    # Instagram Configuration
    INSTAGRAM_BUSINESS_ACCOUNT_ID = os.getenv('INSTAGRAM_BUSINESS_ACCOUNT_ID')
    
    # Facebook Configuration
    FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN')
    FACEBOOK_PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
    FACEBOOK_APP_ID = os.getenv('FACEBOOK_APP_ID')
    FACEBOOK_APP_SECRET = os.getenv('FACEBOOK_APP_SECRET')
    
    # Facebook Analytics Metrics
    FACEBOOK_PAGE_METRICS = [
        'page_impressions',
        'page_engaged_users',
        'page_posts_impressions',
        'page_posts_impressions_unique',
        'page_fan_adds_unique',
        'page_fan_removes_unique',
        'page_views_total',
        'page_actions_post_reactions_total'
    ]
    
    FACEBOOK_POST_METRICS = [
        'post_impressions',
        'post_impressions_unique',
        'post_engaged_users',
        'post_reactions_by_type_total',
        'post_clicks_by_type',
        'post_video_complete_views_organic',
        'post_video_views_organic'
    ]

    @classmethod
    def validate_credentials(cls):
        """Validate that all required credentials are set"""
        missing_credentials = []
        
        # Check Instagram credentials
        if not all([cls.INSTAGRAM_USERNAME, cls.INSTAGRAM_PASSWORD, cls.INSTAGRAM_ACCESS_TOKEN]):
            missing_credentials.append("Instagram credentials (USERNAME, PASSWORD, ACCESS_TOKEN)")
            
        # Check Facebook credentials
        if not all([cls.FACEBOOK_PAGE_ACCESS_TOKEN, cls.FACEBOOK_PAGE_ID]):
            missing_credentials.append("Facebook credentials (PAGE_ACCESS_TOKEN, PAGE_ID)")
            
        if missing_credentials:
            logger.error(f"Missing required credentials: {', '.join(missing_credentials)}")
            return False
            
        return True
        
    @classmethod
    def get_platform_credentials(cls, platform):
        """Get credentials for specified platform"""
        if platform == 'instagram':
            return {
                'username': cls.INSTAGRAM_USERNAME,
                'password': cls.INSTAGRAM_PASSWORD,
                'access_token': cls.INSTAGRAM_ACCESS_TOKEN
            }
        elif platform == 'facebook':
            return {
                'page_access_token': cls.FACEBOOK_PAGE_ACCESS_TOKEN,
                'page_id': cls.FACEBOOK_PAGE_ID
            }
        else:
            raise ValueError(f"Unknown platform: {platform}")

    @classmethod
    def get_long_lived_token(cls, platform='instagram'):
        """Get a long-lived token using app secret"""
        if cls.FACEBOOK_APP_ID and cls.FACEBOOK_APP_SECRET:
            try:
                import requests
                token = cls.INSTAGRAM_ACCESS_TOKEN if platform == 'instagram' else cls.FACEBOOK_PAGE_ACCESS_TOKEN
                if not token:
                    return None
                    
                url = f"{cls.GRAPH_API_BASE_URL}oauth/access_token"
                params = {
                    "grant_type": "fb_exchange_token",
                    "client_id": cls.FACEBOOK_APP_ID,
                    "client_secret": cls.FACEBOOK_APP_SECRET,
                    "fb_exchange_token": token
                }
                response = requests.get(url, params=params)
                if response.ok:
                    return response.json().get('access_token')
            except Exception as e:
                print(f"Error exchanging token: {e}")
        return cls.INSTAGRAM_ACCESS_TOKEN if platform == 'instagram' else cls.FACEBOOK_PAGE_ACCESS_TOKEN

    @classmethod
    def validate_social_media_config(cls):
        """Validate that all required social media credentials are present"""
        required_fields = {
            'INSTAGRAM_ACCESS_TOKEN': cls.INSTAGRAM_ACCESS_TOKEN,
            'INSTAGRAM_BUSINESS_ACCOUNT_ID': cls.INSTAGRAM_BUSINESS_ACCOUNT_ID,
            'FACEBOOK_PAGE_ID': cls.FACEBOOK_PAGE_ID,
            'FACEBOOK_APP_ID': cls.FACEBOOK_APP_ID,
            'FACEBOOK_APP_SECRET': cls.FACEBOOK_APP_SECRET,
            'FACEBOOK_PAGE_ACCESS_TOKEN': cls.FACEBOOK_PAGE_ACCESS_TOKEN
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        
        if missing_fields:
            raise ValueError(
                f"Missing required social media configuration fields: {', '.join(missing_fields)}. "
                "Please check your .env file and ensure all required credentials are set."
            )
        return True