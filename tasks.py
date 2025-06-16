import os
import json
import time
import re
from datetime import datetime, timedelta, timezone
import pickle
import logging
import uuid
import logging.handlers
import numpy as np
import subprocess
import tempfile
from urllib.parse import urlparse
from typing import Optional, List, Tuple, Dict, Any
import asyncio

from celery_app import celery_app
import google.generativeai as genai
from config import Config
from PIL import Image
import cv2
import requests

# Import Cloudinary for potential cleanup
import cloudinary
import cloudinary.uploader

# --- Database Setup (SQLAlchemy) ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import inspect

# Configure logging with both file and console handlers
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/instagram_automation.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # File handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/error.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Validate required configurations
def validate_config():
    required_configs = {
        'GEMINI_API_KEY': Config.GEMINI_API_KEY,
        'INSTAGRAM_ACCESS_TOKEN': Config.INSTAGRAM_ACCESS_TOKEN,
        'INSTAGRAM_BUSINESS_ACCOUNT_ID': Config.INSTAGRAM_BUSINESS_ACCOUNT_ID,
        'CLOUDINARY_CLOUD_NAME': Config.CLOUDINARY_CLOUD_NAME,
        'CLOUDINARY_API_KEY': Config.CLOUDINARY_API_KEY,
        'CLOUDINARY_API_SECRET': Config.CLOUDINARY_API_SECRET
    }
    
    missing_configs = [key for key, value in required_configs.items() if not value]
    if missing_configs:
        error_msg = f"Missing required configuration(s): {', '.join(missing_configs)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

# Validate configuration on module load
try:
    validate_config()
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    # Don't raise here, let the tasks fail individually if configs are missing

# Database file path (SQLite)
DATABASE_URL = "sqlite:///./instagram_automation.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Drop existing tables and recreate with new schema
def recreate_database():
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema updated successfully")
    except Exception as e:
        logger.error(f"Error recreating database: {e}")
        raise

# Configure Cloudinary for potential deletion later
try:
    cloudinary.config(
        cloud_name=Config.CLOUDINARY_CLOUD_NAME,
        api_key=Config.CLOUDINARY_API_KEY,
        api_secret=Config.CLOUDINARY_API_SECRET
    )
    CLOUDINARY_CONFIGURED = True
except Exception as e:
    logger.warning(f"Cloudinary not fully configured in tasks.py: {e}. Media cleanup from Cloudinary might not work.")
    CLOUDINARY_CONFIGURED = False

# Configure Gemini
try:
    if Config.GEMINI_API_KEY:
        genai.configure(api_key=Config.GEMINI_API_KEY)
    else:
        logger.warning("GEMINI_API_KEY not configured. Caption generation will not work.")
except Exception as e:
    logger.error(f"Failed to configure Gemini: {e}")

# Define SQLAlchemy models
class ScheduledPost(Base):
    __tablename__ = "scheduled_posts"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    media_urls = Column(JSON, nullable=False)  # Store array of media URLs
    media_type = Column(String, nullable=False)
    caption = Column(Text, nullable=False)
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    username = Column(String, nullable=False)
    status = Column(String, default='scheduled')
    celery_task_id = Column(String, nullable=True)
    posting_attempt_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    media_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    cloudinary_public_ids = Column(JSON, nullable=True)  # Store array of Cloudinary public IDs
    is_facebook = Column(Boolean, default=False)  # New field to distinguish Facebook posts
    platforms = Column(String, default='Instagram')  # New field to show platforms (Instagram, Facebook, or Both)
    
    def to_dict(self):
        return {
            "id": self.id,
            "media_urls": self.media_urls,
            "media_type": self.media_type,
            "caption": self.caption,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "username": self.username,
            "status": self.status,
            "celery_task_id": self.celery_task_id,
            "posting_attempt_at": self.posting_attempt_at.isoformat() if self.posting_attempt_at else None,
            "error_message": self.error_message,
            "media_id": self.media_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "cloudinary_public_ids": self.cloudinary_public_ids,
            "is_facebook": self.is_facebook,
            "platforms": self.platforms
        }



class PostStatusHistory(Base):
    __tablename__ = "post_status_history"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String, nullable=False)
    previous_status = Column(String, nullable=True)
    new_status = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
    media_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# Recreate database with new schema
recreate_database()

# Configure Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)


# --- Helper Functions for Database Interaction ---
def get_db():
    """Get database session with proper error handling"""
    db = None
    try:
        db = SessionLocal()
        return db
    except Exception as e:
        logger.error(f"Failed to create database session: {e}")
        if db:
            db.close()
        raise

def load_scheduled_posts():
    """Load all scheduled posts from the database"""
    db = SessionLocal()
    try:
        # Query all posts
        posts = db.query(ScheduledPost).all()
        
        # Convert to list of dictionaries with proper formatting
        posts_data = []
        for post in posts:
            post_dict = {
                'id': post.id,
                'media_urls': post.media_urls,
                'media_type': post.media_type,
                'caption': post.caption,
                'scheduled_time': post.scheduled_time.isoformat() if post.scheduled_time else None,
                'username': post.username,
                'status': post.status,
                'celery_task_id': post.celery_task_id,
                'posting_attempt_at': post.posting_attempt_at.isoformat() if post.posting_attempt_at else None,
                'error_message': post.error_message,
                'media_id': post.media_id,
                'created_at': post.created_at.isoformat() if post.created_at else None,
                'updated_at': post.updated_at.isoformat() if post.updated_at else None,
                'cloudinary_public_ids': post.cloudinary_public_ids,
                'is_facebook': post.is_facebook,
                'platforms': post.platforms or ('Facebook' if post.is_facebook else 'Instagram')
            }
            posts_data.append(post_dict)
            
        logger.info(f"Loaded {len(posts_data)} posts from database")
        return posts_data
        
    except Exception as e:
        logger.error(f"Error loading scheduled posts: {e}")
        return []
    finally:
        db.close()

def save_scheduled_post_to_db(post_data: dict) -> Optional[str]:
    """Save a scheduled post to the database"""
    db = SessionLocal()
    try:
        logger.info("Attempting to save post to database with data: %s", {
            k: v for k, v in post_data.items() if k != 'caption'  # Exclude caption for brevity
        })
        
        # Create new post instance
        new_post = ScheduledPost(
            id=str(uuid.uuid4()),
            media_urls=post_data['media_urls'],
            media_type=post_data['media_type'],
            caption=post_data['caption'],
            scheduled_time=post_data['scheduled_time'],
            username=post_data['username'],
            status='scheduled',
            cloudinary_public_ids=post_data['cloudinary_public_ids'],
            created_at=datetime.now(timezone.utc),
            is_facebook=post_data['is_facebook'],
            platforms=post_data.get('platforms', 'Instagram')  # Default to Instagram if not specified
        )
        
        logger.info("Created new post instance with ID: %s", new_post.id)
        
        # Add and commit
        db.add(new_post)
        logger.info("Added post to session")
        
        db.commit()
        logger.info("Committed post to database")
        
        db.refresh(new_post)
        post_id = new_post.id
        
        # Schedule the actual posting task
        eta = post_data['scheduled_time']
        if eta <= datetime.now(timezone.utc):
            eta = datetime.now(timezone.utc) + timedelta(seconds=5)
        
        logger.info("Scheduling posting task for time: %s", eta)
        
        task = _post_media_to_instagram_graph_api.apply_async(
            args=[post_id, post_data['media_urls'], post_data['media_type'], 
                  post_data['caption'], post_data['username'], post_data['is_facebook']],
            eta=eta
        )
        
        logger.info("Created Celery task with ID: %s", task.id)
        
        # Update post with task ID
        new_post.celery_task_id = task.id
        db.commit()
        logger.info("Updated post with Celery task ID")
        
        return post_id
        
    except Exception as e:
        logger.error(f"Error saving post to database: {e}", exc_info=True)
        db.rollback()
        return None
    finally:
        db.close()
        logger.info("Database connection closed")

def track_post_status_change(db: Session, post_id: str, previous_status: str, 
                           new_status: str, error_message: str = None, media_id: str = None):
    """Track post status changes in history table"""
    try:
        history = PostStatusHistory(
            post_id=post_id,
            previous_status=previous_status,
            new_status=new_status,
            error_message=error_message,
            media_id=media_id
        )
        db.add(history)
        db.commit()
        logger.info(f"Tracked status change for post {post_id}: {previous_status} -> {new_status}")
    except Exception as e:
        logger.error(f"Failed to track status change for post {post_id}: {str(e)}")
        db.rollback()

def update_post_status(db: Session, post: ScheduledPost, new_status: str, 
                      error_message: str = None, media_id: str = None):
    """Update post status with tracking"""
    try:
        previous_status = post.status
        post.status = new_status
        if error_message is not None:
            post.error_message = error_message
        if media_id is not None:
            post.media_id = media_id
        post.updated_at = datetime.now(timezone.utc)
        
        # Track the status change
        track_post_status_change(
            db=db,
            post_id=post.id,
            previous_status=previous_status,
            new_status=new_status,
            error_message=error_message,
            media_id=media_id
        )
        
        db.add(post)
        db.commit()
        logger.info(f"Updated post {post.id} status to {new_status}")
    except Exception as e:
        logger.error(f"Failed to update post {post.id} status: {str(e)}")
        db.rollback()


# --- Celery Tasks ---

@celery_app.task(bind=True, max_retries=3, default_retry_delay=300)
def generate_captions(self, media_path, media_type, style='high_engagement', custom_prompt=None, target_audience=None,
                      business_goals=None, num_variants=3):
    """Generate captions using Gemini API with proper error handling"""
    if not Config.GEMINI_API_KEY:
        error_msg = "GEMINI_API_KEY not configured. Cannot generate captions."
        logger.error(error_msg)
        raise ValueError(error_msg)

    temp_files = []  # Keep track of all temporary files
    media_inputs = []  # Keep track of all media inputs
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Handle both single media and carousel posts
        media_paths = media_path if isinstance(media_path, list) else [media_path]
        
        for path in media_paths:
            if not path:
                raise ValueError(f"Invalid media path: {path}")
                
            current_media_source = None
            temp_file_path = None
            
            try:
                # If media_path is a URL (from Cloudinary), use requests to download it temporarily
                is_url = isinstance(path, str) and (path.startswith('http://') or path.startswith('https://'))

                if is_url:
                    try:
                        logger.info(f"Downloading media from URL: {path}")
                        response = requests.get(path, stream=True, timeout=30)
                        response.raise_for_status()

                        # Create a temporary file
                        temp_file_ext = os.path.splitext(path)[1] or '.jpg'  # Default to .jpg if no extension
                        temp_file_path = f"temp_media_{uuid.uuid4()}{temp_file_ext}"
                        with open(temp_file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:  # filter out keep-alive new chunks
                                    f.write(chunk)
                        current_media_source = temp_file_path
                        if temp_file_path:  # Only append if successfully created
                            temp_files.append(temp_file_path)
                        logger.info(f"Media downloaded to temporary file: {temp_file_path}")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Error downloading media from URL {path}: {e}")
                        raise
                    except Exception as e:
                        logger.error(f"Error handling downloaded media from URL {path}: {e}")
                        raise
                else:
                    current_media_source = path
                    if not os.path.exists(current_media_source):
                        raise ValueError(f"Media file not found: {current_media_source}")

                # Read media file based on type
                if media_type == 'image' or media_type == 'carousel':
                    try:
                        image = Image.open(current_media_source)
                        media_inputs.append(image)
                    except Exception as e:
                        logger.error(f"Error opening image {current_media_source}: {e}")
                        raise
                elif media_type in ['video', 'reel']:
                    try:
                        cap = cv2.VideoCapture(current_media_source)
                        if not cap.isOpened():
                            raise ValueError(f"Could not open video file: {current_media_source}")
                        ret, frame = cap.read()
                        if not ret:
                            raise ValueError(f"Could not read frame from {current_media_source}")
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        media_inputs.append(Image.fromarray(frame_rgb))
                        cap.release()
                    except Exception as e:
                        logger.error(f"Error processing video {current_media_source}: {e}")
                        raise
                else:
                    raise ValueError(f"Unsupported media type: {media_type}")
            except Exception as e:
                logger.error(f"Error processing media file {path}: {str(e)}")
                # Clean up the current temp file if there was an error
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info(f"Cleaned up temporary file after processing error: {temp_file_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")
                raise

        if not media_inputs:
            raise ValueError("No media could be processed for caption generation")

        # Customize prompt based on media type and number of images
        if media_type == 'carousel':
            base_prompt = f"""
            Analyze these {len(media_inputs)} images for an Instagram carousel post and create {num_variants} engaging captions.
            
            Key requirements for Carousel captions:
            1. Create a compelling hook that makes users want to swipe through all images
            2. Reference the journey/story through the images
            3. Use emojis to guide users (e.g., "→ Swipe to see more!")
            4. Include a clear call-to-action to save or share
            5. Add relevant hashtags
            6. Keep users engaged throughout the carousel
            7. Encourage discussion in comments
            
            Make the caption engaging and optimized for carousel post discovery.
            """
        elif media_type == 'reel':
            base_prompt = f"""
            Analyze this Instagram Reel thumbnail and create {num_variants} engaging captions.
            This is a Reel (short-form vertical video) that needs to be especially catchy and engaging.
            
            Key requirements for Reel captions:
            1. Start with a strong hook to grab attention
            2. Keep it concise but impactful
            3. Use emojis strategically
            4. Include 2-3 relevant trending Reels hashtags
            5. Add a call-to-action (e.g., "Watch till the end!", "Save for later!")
            6. Encourage engagement (comments, shares, saves)
            7. Match the energy and style of trending Reels
            """
        else:
            base_prompt = f"""
            Analyze this image and create {num_variants} engaging social media captions.
            Focus on creating attention-grabbing, authentic, and engaging content.
            """

        style_prompts = {
            'high_engagement': """
                Focus on:
                - Hook in the first line
                - Emotional connection
                - Call-to-action
                - Relevant hashtags
                - Question to drive comments
                """,
            'story_style': """
                Include:
                - Personal narrative
                - Behind-the-scenes feel
                - Authentic voice
                - Relatable moments
                """,
            'viral_potential': """
                Focus on:
                - Trending topics
                - Shareable content
                - Humor or surprise
                - Current events tie-in
                """,
            'targeted': f"""
                Target Audience: {target_audience or 'general'}
                Business Goals: {business_goals or 'N/A'}
                Focus on tailoring the message specifically to this audience and goals.
                """,
            'A/B Test': f"""
                Generate {num_variants} distinct captions suitable for A/B testing. Each caption should offer a different approach or emphasis while conveying the main message.
                """,
            'custom': custom_prompt or "Generate creative captions for this media."
        }

        prompt = base_prompt + style_prompts.get(style, style_prompts['high_engagement'])

        prompt += """
        Return the captions as a JSON array, like this:
        ```json
        {
          "captions": [
            {"text": "Caption 1 text", "engagement_score": 85, "hashtags": "#example #hashtags"},
            {"text": "Caption 2 text", "engagement_score": 90, "hashtags": "#another #set"}
          ]
        }
        ```
        """

        # For carousel posts, send all images to Gemini
        try:
            if media_type == 'carousel':
                response = model.generate_content([prompt] + media_inputs)
            else:
                response = model.generate_content([prompt, media_inputs[0]])
            
            response_text = response.text

            # Try to parse JSON response
            try:
                json_match = re.search(r'```json\n(.*)\n```', response_text, re.DOTALL)
                if json_match:
                    captions_data = json.loads(json_match.group(1))
                else:
                    captions_data = json.loads(response_text)

                if isinstance(captions_data, dict) and 'captions' in captions_data:
                    return captions_data

            except json.JSONDecodeError:
                # Fallback to text parsing if JSON parsing fails
                captions = []
                lines = response_text.split('\n')
                current_caption = []
                engagement_score = 85

                for line in lines:
                    line = line.strip()
                    if line and not line.lower().startswith(('json', '```')):
                        if re.match(r'^\d+\.', line) or line.startswith('- ') or line.startswith('* '):
                            if current_caption:
                                captions.append({
                                    'text': '\n'.join(current_caption).strip(),
                                    'engagement_score': min(engagement_score, 100),
                                    'hashtags': '#carousel #swiperight' if media_type == 'carousel' else 
                                              '#reels #trending' if media_type == 'reel' else '#content'
                                })
                                engagement_score += 5
                            current_caption = [re.sub(r'^\d+\.\s*|^- \s*|^\* \s*', '', line)]
                        else:
                            current_caption.append(line)

                if current_caption:
                    captions.append({
                        'text': '\n'.join(current_caption).strip(),
                        'engagement_score': min(engagement_score, 100),
                        'hashtags': '#carousel #swiperight' if media_type == 'carousel' else 
                                  '#reels #trending' if media_type == 'reel' else '#content'
                    })

                return {'captions': captions[:num_variants] if captions else [{
                    'text': response_text.strip(),
                    'engagement_score': 75,
                    'hashtags': '#carousel' if media_type == 'carousel' else 
                              '#reels' if media_type == 'reel' else '#content'
                }]}

        except Exception as e:
            logger.error(f"Error generating captions with Gemini: {e}")
            raise

    except Exception as e:
        logger.exception(f"Error in caption generation: {e}")
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            return {'error': f"Failed to generate captions after multiple retries: {str(e)}"}
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temporary file in final cleanup: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
                    continue


@celery_app.task
def schedule_instagram_post(media_url: str, media_type: str, caption: str, scheduled_time_str: str, username: str,
                          cloudinary_public_id: str = None, is_facebook: bool = False):
    """Schedule a post for Instagram/Facebook"""
    try:
        # Convert media_url and cloudinary_public_id to lists if they're not already
        media_urls = media_url if isinstance(media_url, list) else [media_url]
        cloudinary_ids = cloudinary_public_id if isinstance(cloudinary_public_id, list) else [cloudinary_public_id]
        
        # Parse scheduled time
        scheduled_time = datetime.fromisoformat(scheduled_time_str)
        
        # Determine platform based on is_facebook flag
        platforms = "Facebook" if is_facebook else "Instagram"
        
        # Create post data
        post_data = {
            'media_urls': media_urls,
            'media_type': media_type,
            'caption': caption,
            'scheduled_time': scheduled_time,
            'username': username,
            'cloudinary_public_ids': cloudinary_ids,
            'is_facebook': is_facebook,
            'platforms': platforms
        }
        
        # Save to database
        post_id = save_scheduled_post_to_db(post_data)
        
        if not post_id:
            logger.error("Failed to save post to database")
            return None
            
        logger.info(f"Successfully scheduled post with ID: {post_id}")
        return post_id
        
    except Exception as e:
        logger.error(f"Error scheduling post: {e}")
        return None


def verify_instagram_post(media_id: str, access_token: str, max_retries: int = 3, retry_delay: int = 10) -> bool:
    """
    Verify if a post exists and is visible on Instagram with retries.
    
    Args:
        media_id: The Instagram media ID to verify
        access_token: Instagram Graph API access token
        max_retries: Maximum number of verification attempts
        retry_delay: Delay in seconds between retries
    """
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    params = {
        "fields": "id,media_type,media_url,permalink,thumbnail_url",
        "access_token": access_token
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Verification attempt {attempt + 1} for media {media_id}")
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'error' in data:
                error_msg = data['error'].get('message', 'Unknown error')
                logger.error(f"Error verifying media {media_id} (attempt {attempt + 1}): {error_msg}")
                if 'code' in data['error'] and data['error']['code'] == 100:  # Not Found
                    time.sleep(retry_delay)
                    continue
                return False
            
            # If we get here and have an ID, the post exists and is accessible
            if 'id' in data:
                logger.info(f"Successfully verified media {media_id} on attempt {attempt + 1}")
                logger.info(f"Media Type: {data.get('media_type')}")
                logger.info(f"Permalink: {data.get('permalink')}")
                return True
            else:
                logger.error(f"Media {media_id} response missing ID field: {data}")
                return False
            
        except Exception as e:
            logger.error(f"Error during verification attempt {attempt + 1} for media {media_id}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
    
    logger.error(f"Failed to verify media {media_id} after {max_retries} attempts")
    return False

def backup_post_data(post: ScheduledPost) -> bool:
    """Create a backup of post data before any status changes"""
    try:
        if not os.path.exists('backups'):
            os.makedirs('backups')
            
        backup_file = f"backups/post_{post.id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        backup_data = post.to_dict()
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
            
        logger.info(f"Created backup for post {post.id} at {backup_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create backup for post {post.id}: {str(e)}")
        return False

@celery_app.task
def monitor_instagram_post_status():
    """
    Monitors 'posting_in_progress' scheduled posts and updates their status based on Celery task results.
    Also handles actual posting for 'scheduled' posts when their time comes.
    """
    db = SessionLocal()
    try:
        current_time = datetime.now(timezone.utc)
        logger.info(f"[{current_time}] Monitoring scheduled posts...")

        # Find posts that are 'scheduled' and whose scheduled_time has passed
        posts_to_process = db.query(ScheduledPost).filter(
            ScheduledPost.status == 'scheduled',
            ScheduledPost.scheduled_time <= current_time
        ).all()

        for post in posts_to_process:
            logger.info(f"Initiating post attempt for scheduled post {post.id} (Media Type: {post.media_type}).")
            
            # Create backup before status change
            backup_post_data(post)
            
            # Update status with tracking
            update_post_status(
                db=db,
                post=post,
                new_status='posting_in_progress'
            )
            post.posting_attempt_at = current_time
            db.add(post)
            db.commit()

            try:
                task = _post_media_to_instagram_graph_api.delay(
                    post.id,
                    post.media_urls,
                    post.media_type,
                    post.caption,
                    post.username
                )
                post.celery_task_id = task.id
                db.add(post)
                db.commit()
                logger.info(f"Post {post.id} enqueued for posting with Celery task ID: {task.id}")
            except Exception as e:
                error_msg = f"Failed to enqueue posting task for post {post.id}: {str(e)}"
                logger.error(error_msg)
                update_post_status(
                    db=db,
                    post=post,
                    new_status='failed_to_enqueue',
                    error_message=error_msg
                )

        # Monitor posts that are 'posting_in_progress'
        in_progress_posts = db.query(ScheduledPost).filter(
            ScheduledPost.status == 'posting_in_progress'
        ).all()

        for post in in_progress_posts:
            if not post.celery_task_id:
                logger.warning(f"Post {post.id} is 'posting_in_progress' but has no celery_task_id. Marking as failed.")
                update_post_status(
                    db=db,
                    post=post,
                    new_status='failed',
                    error_message="No Celery task ID found for in-progress post."
                )
                continue

            async_result = celery_app.AsyncResult(post.celery_task_id)

            if async_result.ready():
                if async_result.successful():
                    result = async_result.result
                    media_id = result.get('media_id')
                    
                    # Verify the post exists on Instagram with retries
                    if media_id:
                        verification_success = verify_instagram_post(
                            media_id, 
                            Config.INSTAGRAM_ACCESS_TOKEN,
                            max_retries=5,  # Try up to 5 times
                            retry_delay=15   # Wait 15 seconds between attempts
                        )
                        
                        if verification_success:
                            logger.info(f"Post {post.id} (Task {post.celery_task_id}) successful and verified. Media ID: {media_id}")
                            update_post_status(
                                db=db,
                                post=post,
                                new_status='posted',
                                media_id=media_id
                            )
                            
                            # Create backup after successful posting
                            backup_post_data(post)

                            # Delete media from Cloudinary after successful post
                            if CLOUDINARY_CONFIGURED and post.cloudinary_public_ids:
                                try:
                                    for public_id in post.cloudinary_public_ids:
                                        resource_type = "image" if post.media_type == "image" else "video"
                                        cloudinary.uploader.destroy(public_id, resource_type=resource_type)
                                        logger.info(f"Deleted media {public_id} from Cloudinary.")
                                except Exception as e:
                                    logger.warning(f"Failed to delete media from Cloudinary: {e}")
                            else:
                                logger.info(f"Skipping Cloudinary deletion for {post.id}. Configured: {CLOUDINARY_CONFIGURED}, Public IDs: {post.cloudinary_public_ids}")
                        else:
                            logger.error(f"Post {post.id} verification failed after multiple attempts")
                            update_post_status(
                                db=db,
                                post=post,
                                new_status='verification_failed',
                                error_message="Post could not be verified on Instagram after multiple attempts"
                            )
                    else:
                        logger.error(f"Post {post.id} has no media_id in result: {result}")
                        update_post_status(
                            db=db,
                            post=post,
                            new_status='failed',
                            error_message="No media ID returned from Instagram"
                        )

                    # Clean up posted task
                    if post.celery_task_id:
                        celery_app.control.revoke(post.celery_task_id, terminate=True)
                        logger.info(f"Cleaned up posted task {post.celery_task_id}")

                else:
                    error_msg = f"Post {post.id} (Task {post.celery_task_id}) failed: {async_result.result}"
                    logger.error(error_msg)
                    update_post_status(
                        db=db,
                        post=post,
                        new_status='failed',
                        error_message=str(async_result.result)
                    )
                    
                    # Create backup after failure
                    backup_post_data(post)
                    
                    # Clean up failed task
                    try:
                        async_result.forget()
                        logger.info(f"Cleaned up failed task {post.celery_task_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up task {post.celery_task_id}: {e}")

            elif async_result.state in ['PENDING', 'STARTED']:
                # Check for timeout on in-progress posts
                if post.posting_attempt_at:
                    # Ensure posting_attempt_at is timezone-aware
                    if post.posting_attempt_at.tzinfo is None:
                        post.posting_attempt_at = post.posting_attempt_at.replace(tzinfo=timezone.utc)
                        logger.info(f"Added UTC timezone to naive posting_attempt_at for post {post.id}")
                    
                    time_elapsed = current_time - post.posting_attempt_at
                    if time_elapsed > timedelta(minutes=10):
                        error_msg = f"Post {post.id} (Task {post.celery_task_id}) timed out after 10 minutes."
                        logger.warning(error_msg)
                        update_post_status(
                            db=db,
                            post=post,
                            new_status='failed',
                            error_message=error_msg + " (Task timeout or stuck)"
                        )

        db.commit()
        logger.info(f"[{current_time}] Finished monitoring posts.")
        return {'status': 'ok', 'monitored_count': len(in_progress_posts)}

    except Exception as e:
        db.rollback()
        logger.exception(f"Error in monitor_instagram_post_status: {e}")
        return {'error': str(e)}
    finally:
        db.close()


def validate_video_for_instagram(video_url):
    """
    Validates video meets Instagram Reels requirements before upload.
    Returns (is_valid, error_message)
    """
    try:
        # Download video to temporary file if it's a URL
        temp_file = None
        if video_url.startswith(('http://', 'https://')):
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            temp_file = f"temp_video_{uuid.uuid4()}.mp4"
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            video_path = temp_file
        else:
            video_path = video_url

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video file"

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Log video properties
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, {duration:.2f}s")
        
        # Check video requirements
        if width == 0 or height == 0:
            return False, "Invalid video dimensions"
            
        aspect_ratio = height / width if width > 0 else 0
        if not (1.6 <= aspect_ratio <= 1.91):  # 9:16 is 1.77
            return False, f"Invalid aspect ratio {aspect_ratio:.2f} (should be between 1.6 and 1.91)"
            
        if duration < 3:
            return False, f"Video too short ({duration:.1f}s, minimum 3s)"
        if duration > 90:
            return False, f"Video too long ({duration:.1f}s, maximum 90s)"
            
        if fps < 23 or fps > 60:
            return False, f"Invalid frame rate ({fps}fps, should be between 23-60fps)"
            
        # Check resolution
        min_dimension = min(width, height)
        max_dimension = max(width, height)
        if min_dimension < 720 or max_dimension < 1280:
            return False, f"Resolution too low ({width}x{height}, minimum 720p)"
            
        # Get video codec info
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        logger.info(f"Video codec: {codec}")
        
        cap.release()
        
        # Clean up temp file if created
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
            
        return True, "Video meets Instagram requirements"
        
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        return False, f"Error validating video: {str(e)}"

def verify_facebook_permissions(access_token: str) -> bool:
    """Verify that the access token has the required permissions for Facebook posting"""
    try:
        # First check the token permissions
        response = requests.get(
            'https://graph.facebook.com/v19.0/me/permissions',
            params={'access_token': access_token}
        )

        if not response.ok:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            logger.error(f"Failed to verify permissions: {error_message}")
            logger.error("Please ensure your access token includes all required permissions")
            return False

        permissions = response.json().get('data', [])
        required_permissions = {
            'pages_show_list',
            'pages_read_engagement',
            'pages_manage_posts',
            'publish_to_groups',
            'pages_manage_metadata'  # Added this permission
        }

        granted_permissions = {
            perm['permission'] 
            for perm in permissions 
            if perm.get('status') == 'granted'
        }

        missing_permissions = required_permissions - granted_permissions
        if missing_permissions:
            logger.error(f"Missing permissions: {', '.join(missing_permissions)}")
            logger.error("Please ensure you have granted these permissions in your Facebook Developer App")
            return False

        # Now check page access
        page_response = requests.get(
            'https://graph.facebook.com/v19.0/me/accounts',
            params={'access_token': access_token}
        )

        if not page_response.ok:
            error_data = page_response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            logger.error(f"Failed to verify page access: {error_message}")
            return False

        pages = page_response.json().get('data', [])
        target_page = None
        for page in pages:
            if page['id'] == Config.FACEBOOK_PAGE_ID:
                target_page = page
                break

        if not target_page:
            logger.error(f"Facebook Page ID {Config.FACEBOOK_PAGE_ID} not found in accessible pages")
            logger.error("Please ensure:")
            logger.error("1. The Facebook Page ID is correct")
            logger.error("2. You have admin access to the Facebook Page")
            logger.error("3. The page is connected to your Instagram Professional Account")
            return False

        # Check page permissions
        required_tasks = {'CREATE_CONTENT', 'MODERATE', 'MANAGE'}
        page_tasks = set(target_page.get('tasks', []))
        missing_tasks = required_tasks - page_tasks

        if missing_tasks:
            logger.error(f"Missing page permissions: {', '.join(missing_tasks)}")
            logger.error("Please ensure you have the following roles on the Facebook Page:")
            logger.error("- Admin")
            logger.error("- Editor")
            logger.error("Or at minimum, permissions to:")
            logger.error("- Create content")
            logger.error("- Moderate content")
            logger.error("- Manage page settings")
            return False

        logger.info("Successfully verified all Facebook permissions and page access")
        return True

    except Exception as e:
        logger.error(f"Error verifying permissions: {str(e)}")
        return False

def get_facebook_page_access_token() -> Optional[str]:
    """Get access token for Facebook posting (uses Instagram access token)"""
    try:
        # We use the Instagram access token directly for Facebook
        if not Config.INSTAGRAM_ACCESS_TOKEN:
            logger.error("Instagram access token not configured")
            return None
            
        if not Config.FACEBOOK_PAGE_ID:
            logger.error("Facebook Page ID not configured")
            return None

        # Verify the token has required permissions
        if not verify_facebook_permissions(Config.INSTAGRAM_ACCESS_TOKEN):
            logger.error("Token is missing required permissions")
            return None

        logger.info("Using Instagram access token for Facebook posting")
        return Config.INSTAGRAM_ACCESS_TOKEN

    except Exception as e:
        logger.error(f"Error getting access token: {str(e)}")
        return None

def wait_for_media_ready(creation_id, access_token, max_retries=30, retry_delay=10):
    """Wait for media to be ready before publishing"""
    base_url = "https://graph.facebook.com/v18.0"
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{base_url}/{creation_id}",
                params={
                    'access_token': access_token,
                    'fields': 'status_code,status'
                }
            )
            if response.ok:
                data = response.json()
                status_code = data.get('status_code')
                status = data.get('status')
                
                logger.info(f"Media status check attempt {attempt + 1}/{max_retries}: status_code={status_code}, status={status}")
                
                if status_code == 'FINISHED':
                    logger.info(f"Media {creation_id} is ready for publishing")
                    return True
                elif status_code in ['ERROR', 'EXPIRED']:
                    logger.error(f"Media {creation_id} failed with status_code={status_code}")
                    return False
                elif status_code == 'IN_PROGRESS':
                    logger.info(f"Media {creation_id} is still processing, waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
            else:
                logger.warning(f"Failed to check media status: {response.text}")
            
            # Increase delay for subsequent retries
            current_delay = retry_delay * (attempt + 1)
            logger.info(f"Waiting {current_delay} seconds before next attempt...")
            time.sleep(current_delay)
        except Exception as e:
            logger.error(f"Error checking media status: {e}")
            time.sleep(retry_delay)
            continue
    
    logger.error(f"Media {creation_id} failed to become ready after {max_retries} attempts")
    return False

def publish_media(creation_id, access_token, account_id, base_url):
    """Helper function to publish media with retries"""
    logger.info(f"Attempting to publish media with creation_id: {creation_id}")
    
    max_publish_retries = 5
    for publish_attempt in range(max_publish_retries):
        try:
            logger.info(f"Publish attempt {publish_attempt + 1}/{max_publish_retries}")
            
            # First check if media is ready
            status_response = requests.get(
                f"{base_url}/{creation_id}",
                params={
                    'access_token': access_token,
                    'fields': 'status_code'
                },
                timeout=30
            )
            
            if not status_response.ok:
                logger.warning(f"Failed to check media status: {status_response.text}")
                if publish_attempt < max_publish_retries - 1:
                    time.sleep(10)
                    continue
                return None
            
            status = status_response.json().get('status_code')
            logger.info(f"Media status before publishing: {status}")
            
            if status != 'FINISHED':
                if publish_attempt < max_publish_retries - 1:
                    logger.warning(f"Media not ready (status: {status}), waiting before retry")
                    time.sleep(15)
                    continue
                logger.error(f"Media failed to become ready after {max_publish_retries} attempts")
                return None

            # Media is ready, attempt to publish
            response = requests.post(
                f"{base_url}/{account_id}/media_publish",
                params={
                    'access_token': access_token,
                    'creation_id': creation_id
                },
                timeout=30
            )
            
            logger.info(f"Publish response status code: {response.status_code}")
            logger.info(f"Publish response content: {response.text}")
            
            if response.ok:
                logger.info(f"Successfully published media with creation_id: {creation_id}")
                return response
            
            error_data = response.json().get('error', {})
            error_code = error_data.get('code')
            
            # Handle specific error codes
            if error_code == 9007:  # Media ID is not available
                if publish_attempt < max_publish_retries - 1:
                    logger.warning("Media not ready for publishing, waiting longer...")
                    time.sleep(20)  # Wait longer for media processing
                    continue
            
            logger.warning(f"Failed to publish: {response.text}")
            if publish_attempt < max_publish_retries - 1:
                time.sleep(10)
                continue
            
            return response
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout during publish attempt {publish_attempt + 1}")
            if publish_attempt < max_publish_retries - 1:
                time.sleep(10)
                continue
            return None
            
        except Exception as e:
            logger.error(f"Error during publish attempt {publish_attempt + 1}: {str(e)}")
            if publish_attempt < max_publish_retries - 1:
                time.sleep(10)
                continue
            return None
    
    return None

async def post_media_to_facebook(media_urls: List[str], media_type: str, caption: str) -> Tuple[bool, str]:
    """Post media to Facebook"""
    try:
        if media_type == 'carousel':
            return await post_carousel_to_facebook(media_urls, caption)
        else:
            # Get the page access token
            access_token = Config.INSTAGRAM_ACCESS_TOKEN
            if not access_token:
                return False, "Instagram access token not configured"

            # First, get the page access token
            page_response = requests.get(
                'https://graph.facebook.com/v19.0/me/accounts',
                params={'access_token': access_token}
            )

            if not page_response.ok:
                error_data = page_response.json()
                return False, f"Failed to get page access: {error_data.get('error', {}).get('message', 'Unknown error')}"

            pages = page_response.json().get('data', [])
            page_token = None
            for page in pages:
                if page['id'] == Config.FACEBOOK_PAGE_ID:
                    page_token = page.get('access_token')
                    break

            if not page_token:
                return False, f"Could not find access token for page {Config.FACEBOOK_PAGE_ID}"

            # For reels, we need to use chunked upload
            if media_type == 'reel':
                try:
                    # Download video to temporary file
                    response = requests.get(media_urls[0], stream=True)
                    response.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                temp_file.write(chunk)
                        temp_file_path = temp_file.name

                    # Get video properties
                    cap = cv2.VideoCapture(temp_file_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    file_size = os.path.getsize(temp_file_path)
                    cap.release()

                    logger.info(f"Video properties - Width: {width}, Height: {height}, Duration: {duration}s, Size: {file_size} bytes")

                    # Initialize upload session
                    init_data = {
                        'access_token': page_token,
                        'upload_phase': 'start',
                        'file_size': file_size,
                        'content_category': 'OTHER'
                    }

                    logger.info("Initializing Facebook reel upload session...")
                    init_response = requests.post(
                        f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}/videos',
                        data=init_data
                    )
                    
                    if not init_response.ok:
                        error_data = init_response.json()
                        error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                        logger.error(f"Failed to initialize upload: {error_msg}")
                        logger.error(f"Full error response: {error_data}")
                        return False, f"Failed to initialize upload: {error_msg}"
                    
                    response_data = init_response.json()
                    logger.info(f"Upload session response: {response_data}")
                    
                    # Get both video_id and upload_session_id from response
                    video_id = response_data.get('video_id')
                    upload_session_id = response_data.get('upload_session_id')
                    
                    if not video_id or not upload_session_id:
                        logger.error(f"Missing required IDs in response: {response_data}")
                        return False, "Failed to get upload session ID or video ID"
                    
                    # Upload video in chunks
                    chunk_size = 4 * 1024 * 1024  # 4MB chunks
                    with open(temp_file_path, 'rb') as video_file:
                        chunk_number = 0
                        while True:
                            chunk = video_file.read(chunk_size)
                            if not chunk:
                                break
                            
                            start_offset = chunk_number * chunk_size
                            logger.debug(f"Uploading chunk {chunk_number}, offset: {start_offset}")
                            
                            # Upload chunk
                            files = {
                                'video_file_chunk': (
                                    'chunk',
                                    chunk,
                                    'application/octet-stream'
                                )
                            }
                            
                            chunk_data = {
                                'access_token': page_token,
                                'upload_phase': 'transfer',
                                'start_offset': start_offset,
                                'upload_session_id': upload_session_id
                            }
                            
                            chunk_response = requests.post(
                                f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}/videos',
                                data=chunk_data,
                                files=files
                            )
                            
                            if not chunk_response.ok:
                                error_data = chunk_response.json()
                                logger.error(f"Chunk upload failed: {error_data}")
                                return False, f"Failed to upload chunk {chunk_number}: {error_data.get('error', {}).get('message', 'Unknown error')}"
                            
                            chunk_number += 1
                    
                    # Finish upload with metadata
                    finish_data = {
                        'access_token': page_token,
                        'upload_phase': 'finish',
                        'upload_session_id': upload_session_id,
                        'description': caption,
                        'content_category': 'OTHER'
                    }

                    logger.info("Finishing Facebook reel upload...")
                    finish_response = requests.post(
                        f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}/videos',
                        data=finish_data
                    )
                    
                    if not finish_response.ok:
                        error_data = finish_response.json()
                        logger.error(f"Failed to finish upload: {error_data}")
                        return False, f"Failed to finish upload: {error_data.get('error', {}).get('message', 'Unknown error')}"
                    
                    finish_data = finish_response.json()
                    logger.info(f"Upload finished successfully: {finish_data}")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file: {e}")
                    
                    return True, finish_data.get('id')
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error during video upload: {e}")
                    return False, f"Failed to download video: {str(e)}"
                except Exception as e:
                    logger.error(f"Unexpected error during video upload: {e}")
                    return False, f"Error during video upload: {str(e)}"
                finally:
                    # Ensure temporary file is deleted
                    try:
                        if 'temp_file_path' in locals():
                            os.unlink(temp_file_path)
                    except Exception:
                        pass
            else:
                # For photos, use simple upload
                post_data = {
                    'url': media_urls[0],
                    'caption': caption,
                    'access_token': page_token
                }

                response = requests.post(
                    f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}/photos',
                    data=post_data
                )

                if not response.ok:
                    error_data = response.json()
                    return False, f"Failed to post to Facebook: {error_data.get('error', {}).get('message', 'Unknown error')}"

                return True, response.json().get('id')

    except Exception as e:
        return False, f"Error posting to Facebook: {str(e)}"

async def post_carousel_to_facebook(media_urls: List[str], caption: str) -> Tuple[bool, str]:
    """Post carousel media to Facebook as individual photos"""
    try:
        # Get the page access token
        access_token = Config.INSTAGRAM_ACCESS_TOKEN
        if not access_token:
            return False, "Instagram access token not configured"

        # First, get the page access token
        page_response = requests.get(
            'https://graph.facebook.com/v19.0/me/accounts',
            params={'access_token': access_token}
        )

        if not page_response.ok:
            error_data = page_response.json()
            return False, f"Failed to get page access: {error_data.get('error', {}).get('message', 'Unknown error')}"

        pages = page_response.json().get('data', [])
        page_token = None
        for page in pages:
            if page['id'] == Config.FACEBOOK_PAGE_ID:
                page_token = page.get('access_token')
                break

        if not page_token:
            return False, f"Could not find access token for page {Config.FACEBOOK_PAGE_ID}"

        # Post first image with caption
        first_post_data = {
            'url': media_urls[0],
            'caption': caption,
            'access_token': page_token
        }
        
        response = requests.post(
            f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}/photos',
            data=first_post_data
        )
        
        if not response.ok:
            error_data = response.json()
            logger.error("Failed to post first image to Facebook: %s", error_data)
            return False, f"Failed to post to Facebook: {error_data.get('error', {}).get('message', 'Unknown error')}"
        
        first_post_id = response.json().get('id')
        
        # Post remaining images without caption
        for media_url in media_urls[1:]:
            post_data = {
                'url': media_url,
                'access_token': page_token
            }
            
            response = requests.post(
                f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}/photos',
                data=post_data
            )
            
            if not response.ok:
                logger.warning(f"Failed to post additional image in carousel: {response.json()}")
                # Continue with remaining images even if one fails
                continue
        
        return True, first_post_id
        
    except Exception as e:
        logger.error("Error posting carousel to Facebook: %s", str(e))
        return False, f"Error posting carousel to Facebook: {str(e)}"

@celery_app.task
def _post_media_to_instagram_graph_api(post_id: str, media_urls: List[str], media_type: str, 
                                     caption: str, username: str, is_facebook: bool = False):
    """Post media to Instagram/Facebook using the Graph API"""
    db = SessionLocal()
    try:
        post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
        if not post:
            logger.error(f"Post {post_id} not found in database")
            return

        post.status = 'posting_in_progress'
        post.posting_attempt_at = datetime.now(timezone.utc)
        db.commit()

        success = False
        error_message = None
        media_id = None

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            if is_facebook:
                success, result = loop.run_until_complete(
                    post_media_to_facebook(media_urls, media_type, caption)
                )
            else:
                # Instagram posting logic
                if media_type == 'carousel':
                    success, result = loop.run_until_complete(
                        post_carousel_to_instagram(media_urls, caption)
                    )
                elif media_type == 'reel':
                    success, result = loop.run_until_complete(
                        post_reel_to_instagram(media_urls[0], caption)
                    )
                else:
                    success, result = loop.run_until_complete(
                        post_photo_to_instagram(media_urls[0], caption)
                    )

            loop.close()

            if success:
                media_id = result
            else:
                error_message = result

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error posting to {'Facebook' if is_facebook else 'Instagram'}: {e}")

        # Update post status
        post.status = 'posted' if success else 'failed'
        post.error_message = error_message
        post.media_id = media_id
        post.updated_at = datetime.now(timezone.utc)
        db.commit()

        if success and post.cloudinary_public_ids:
            cleanup_cloudinary_media.delay(post.cloudinary_public_ids)

    except Exception as e:
        logger.error(f"Error in posting task: {e}")
        try:
            post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
            if post:
                post.status = 'failed'
                post.error_message = str(e)
                post.updated_at = datetime.now(timezone.utc)
                db.commit()
        except Exception as inner_e:
            logger.error(f"Error updating post status: {inner_e}")
    finally:
        db.close()


# Celery Beat tasks (scheduling)
celery_app.conf.beat_schedule = {
    'monitor-instagram-posts-every-30-seconds': {
        'task': 'tasks.monitor_instagram_post_status',
        'schedule': timedelta(seconds=30),
        'args': (),
    },
}

# Make sure all required functions are explicitly exported
__all__ = [
    'generate_captions',
    'schedule_instagram_post', 
    'load_scheduled_posts',
    'monitor_instagram_post_status',
    '_post_media_to_instagram_graph_api',
    'get_db',
    'ScheduledPost',
    'PostStatusHistory'
]

def preprocess_video_for_instagram(video_url):
    """Preprocess video to meet Instagram requirements"""
    logger.info(f"Preprocessing video from URL: {video_url}")
    
    temp_dir = None
    input_file = None
    output_file = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download video to temporary file
        input_file = os.path.join(temp_dir, 'input.mp4')
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(input_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        
        # Get video properties
        cap = cv2.VideoCapture(input_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate target dimensions for 9:16 aspect ratio
        target_width = 1080
        target_height = 1920
        
        # Create output filename
        output_file = os.path.join(temp_dir, 'output.mp4')
        
        # Construct FFmpeg command
        command = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',  # Use H.264 codec
            '-c:a', 'aac',      # Use AAC codec
            '-b:v', '3500k',    # Video bitrate
            '-b:a', '128k',     # Audio bitrate
            '-r', '30',         # Frame rate
            '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2',
            '-movflags', '+faststart',
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        # Run FFmpeg
        logger.info("Running FFmpeg command...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
        
        # Upload processed video to Cloudinary
        logger.info("Uploading processed video to Cloudinary...")
        upload_result = cloudinary.uploader.upload_large(
            output_file,
            resource_type="video",
            folder="instagram_posts"
        )
        
        # Clean up temporary files
        os.remove(input_file)
        os.remove(output_file)
        os.rmdir(temp_dir)
        
        return upload_result['secure_url']
        
    except Exception as e:
        logger.error(f"Error preprocessing video: {str(e)}")
        return None
        
    finally:
        # Clean up temporary files if they exist
        try:
            if input_file and os.path.exists(input_file):
                os.remove(input_file)
            if output_file and os.path.exists(output_file):
                os.remove(output_file)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

def verify_database():
    """Verify database setup and create tables if they don't exist"""
    try:
        # Create engine
        engine = create_engine('sqlite:///social_media_assistant.db')
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Verify tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info("Database tables: %s", tables)
        
        if 'scheduled_posts' in tables:
            # Get column information
            columns = inspector.get_columns('scheduled_posts')
            logger.info("Scheduled posts table columns: %s", 
                       [f"{col['name']} ({col['type']})" for col in columns])
            return True
        else:
            logger.error("scheduled_posts table not found in database")
            return False
            
    except Exception as e:
        logger.error("Error verifying database: %s", str(e), exc_info=True)
        return False

# Call verify_database when module is imported
verify_database()

async def post_photo_to_instagram(media_url: str, caption: str) -> Tuple[bool, str]:
    """Post a single photo to Instagram"""
    try:
        # Get Instagram account ID
        instagram_account_id = get_instagram_account_id()
        if not instagram_account_id:
            return False, "Failed to get Instagram account ID"

        # Create container
        container_response = requests.post(
            f'https://graph.facebook.com/v19.0/{instagram_account_id}/media',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'image_url': media_url,
                'caption': caption
            }
        )

        if not container_response.ok:
            error_data = container_response.json()
            return False, f"Failed to create media container: {error_data.get('error', {}).get('message', 'Unknown error')}"

        container_id = container_response.json().get('id')

        # Publish the container
        publish_response = requests.post(
            f'https://graph.facebook.com/v19.0/{instagram_account_id}/media_publish',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'creation_id': container_id
            }
        )

        if not publish_response.ok:
            error_data = publish_response.json()
            return False, f"Failed to publish media: {error_data.get('error', {}).get('message', 'Unknown error')}"

        return True, publish_response.json().get('id')

    except Exception as e:
        return False, f"Error posting photo to Instagram: {str(e)}"

async def post_carousel_to_instagram(media_urls: List[str], caption: str) -> Tuple[bool, str]:
    """Post carousel media to Instagram"""
    try:
        instagram_account_id = get_instagram_account_id()
        if not instagram_account_id:
            return False, "Failed to get Instagram account ID"

        # Create containers for each media
        children = []
        for media_url in media_urls:
            try:
                container_response = requests.post(
                    f'https://graph.facebook.com/v19.0/{instagram_account_id}/media',
                    params={
                        'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                        'image_url': media_url,
                        'is_carousel_item': 'true'
                    }
                )

                if not container_response.ok:
                    error_data = container_response.json()
                    return False, f"Failed to create media container: {error_data.get('error', {}).get('message', 'Unknown error')}"

                children.append(container_response.json().get('id'))
            except Exception as e:
                return False, f"Error creating carousel item: {str(e)}"

        # Create carousel container
        carousel_response = requests.post(
            f'https://graph.facebook.com/v19.0/{instagram_account_id}/media',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'media_type': 'CAROUSEL',
                'caption': caption,
                'children': ','.join(children)
            }
        )

        if not carousel_response.ok:
            error_data = carousel_response.json()
            return False, f"Failed to create carousel container: {error_data.get('error', {}).get('message', 'Unknown error')}"

        carousel_container_id = carousel_response.json().get('id')

        # Publish the carousel
        response = requests.post(
            f'https://graph.facebook.com/v19.0/{instagram_account_id}/media_publish',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'creation_id': carousel_container_id
            }
        )

        if not response.ok:
            error_data = response.json()
            return False, f"Failed to publish carousel: {error_data.get('error', {}).get('message', 'Unknown error')}"

        return True, response.json().get('id')

    except Exception as e:
        return False, f"Error posting carousel to Instagram: {str(e)}"

async def post_reel_to_instagram(media_url: str, caption: str) -> Tuple[bool, str]:
    """Post a reel to Instagram"""
    try:
        instagram_account_id = get_instagram_account_id()
        if not instagram_account_id:
            return False, "Failed to get Instagram account ID"

        # Create container for reel
        container_response = requests.post(
            f'https://graph.facebook.com/v19.0/{instagram_account_id}/media',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'media_type': 'REELS',
                'video_url': media_url,
                'caption': caption
            }
        )

        if not container_response.ok:
            error_data = container_response.json()
            return False, f"Failed to create reel container: {error_data.get('error', {}).get('message', 'Unknown error')}"

        container_id = container_response.json().get('id')

        # Check status and wait for video processing
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            try:
                status_response = requests.get(
                    f'https://graph.facebook.com/v19.0/{container_id}',
                    params={'access_token': Config.INSTAGRAM_ACCESS_TOKEN, 'fields': 'status_code'}
                )

                if not status_response.ok:
                    error_data = status_response.json()
                    return False, f"Failed to check reel status: {error_data.get('error', {}).get('message', 'Unknown error')}"

                status = status_response.json().get('status_code')
                if status == 'FINISHED':
                    break
                elif status in ['ERROR', 'EXPIRED']:
                    return False, f"Reel processing failed with status: {status}"

                await asyncio.sleep(5)  # Wait 5 seconds before checking again
                attempt += 1
            except Exception as e:
                logger.error(f"Error checking reel status: {e}")
                await asyncio.sleep(5)
                attempt += 1
                continue

        if attempt >= max_attempts:
            return False, "Timeout waiting for reel processing"

        # Publish the reel
        response = requests.post(
            f'https://graph.facebook.com/v19.0/{instagram_account_id}/media_publish',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'creation_id': container_id
            }
        )

        if not response.ok:
            error_data = response.json()
            return False, f"Failed to publish reel: {error_data.get('error', {}).get('message', 'Unknown error')}"

        return True, response.json().get('id')
            
    except Exception as e:
        return False, f"Error posting reel to Instagram: {str(e)}"

@celery_app.task
def cleanup_cloudinary_media(public_ids: List[str]):
    """Clean up media files from Cloudinary"""
    try:
        if not CLOUDINARY_CONFIGURED:
            logger.warning("Cloudinary not configured, skipping media cleanup")
            return
            
        for public_id in public_ids:
            try:
                result = cloudinary.uploader.destroy(public_id)
                if result.get('result') == 'ok':
                    logger.info(f"Successfully deleted media {public_id} from Cloudinary")
                else:
                    logger.warning(f"Failed to delete media {public_id} from Cloudinary: {result}")
            except Exception as e:
                logger.error(f"Error deleting media {public_id} from Cloudinary: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error cleaning up Cloudinary media: {e}")

def get_instagram_account_id() -> Optional[str]:
    """Get Instagram account ID from Facebook Page ID"""
    try:
        response = requests.get(
            f'https://graph.facebook.com/v19.0/{Config.FACEBOOK_PAGE_ID}',
            params={
                'access_token': Config.INSTAGRAM_ACCESS_TOKEN,
                'fields': 'instagram_business_account'
            }
        )

        if not response.ok:
            logger.error("Failed to get Instagram account ID: %s", response.json())
            return None

        data = response.json()
        return data.get('instagram_business_account', {}).get('id')

    except Exception as e:
        logger.error("Error getting Instagram account ID: %s", str(e))
        return None