import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    INSTAGRAM_USERNAME = os.getenv('INSTAGRAM_USERNAME', 'hogistindia')
    INSTAGRAM_PASSWORD = os.getenv('INSTAGRAM_PASSWORD')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

    # Updated Instagram Graph API Configuration
    INSTAGRAM_GRAPH_API_URL = "https://graph.facebook.com/v18.0/"  # Updated to latest stable version
    INSTAGRAM_ACCESS_TOKEN = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    INSTAGRAM_BUSINESS_ACCOUNT_ID = os.getenv('INSTAGRAM_BUSINESS_ACCOUNT_ID')
    FACEBOOK_PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
    FACEBOOK_APP_ID = os.getenv('FACEBOOK_APP_ID')
    FACEBOOK_APP_SECRET = os.getenv('FACEBOOK_APP_SECRET')

    @classmethod
    def get_long_lived_token(cls):
        """Get a long-lived token using app secret"""
        if cls.FACEBOOK_APP_ID and cls.FACEBOOK_APP_SECRET and cls.INSTAGRAM_ACCESS_TOKEN:
            try:
                import requests
                url = f"{cls.INSTAGRAM_GRAPH_API_URL}oauth/access_token"
                params = {
                    "grant_type": "fb_exchange_token",
                    "client_id": cls.FACEBOOK_APP_ID,
                    "client_secret": cls.FACEBOOK_APP_SECRET,
                    "fb_exchange_token": cls.INSTAGRAM_ACCESS_TOKEN
                }
                response = requests.get(url, params=params)
                if response.ok:
                    return response.json().get('access_token')
            except Exception as e:
                print(f"Error exchanging token: {e}")
        return cls.INSTAGRAM_ACCESS_TOKEN

    @classmethod
    def validate_instagram_config(cls):
        """Validate that all required Instagram credentials are present"""
        required_fields = {
            'INSTAGRAM_ACCESS_TOKEN': cls.INSTAGRAM_ACCESS_TOKEN,
            'INSTAGRAM_BUSINESS_ACCOUNT_ID': cls.INSTAGRAM_BUSINESS_ACCOUNT_ID,
            'FACEBOOK_PAGE_ID': cls.FACEBOOK_PAGE_ID,
            'FACEBOOK_APP_ID': cls.FACEBOOK_APP_ID,
            'FACEBOOK_APP_SECRET': cls.FACEBOOK_APP_SECRET
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value]
        
        if missing_fields:
            raise ValueError(
                f"Missing required Instagram configuration fields: {', '.join(missing_fields)}. "
                "Please check your .env file and ensure all required credentials are set."
            )
        return True