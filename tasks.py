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
            "cloudinary_public_ids": self.cloudinary_public_ids
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


# Create database tables
Base.metadata.create_all(bind=engine)

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
        posts = db.query(ScheduledPost).order_by(ScheduledPost.scheduled_time.asc()).all()
        return [post.to_dict() for post in posts]
    except Exception as e:
        logger.error(f"Error loading scheduled posts from DB: {e}")
        return []
    finally:
        db.close()

def save_scheduled_post_to_db(post_data: dict):
    db = SessionLocal()
    try:
        # Ensure media_urls and cloudinary_public_ids are lists
        media_urls = post_data['media_urls'] if isinstance(post_data['media_urls'], list) else [post_data['media_urls']]
        cloudinary_public_ids = post_data.get('cloudinary_public_ids', [])
        if not isinstance(cloudinary_public_ids, list):
            cloudinary_public_ids = [cloudinary_public_ids] if cloudinary_public_ids else []

        # Start a transaction
        db.begin()

        new_post = ScheduledPost(
            id=post_data.get('id', str(uuid.uuid4())),
            media_urls=media_urls,
            media_type=post_data['media_type'],
            caption=post_data['caption'],
            scheduled_time=post_data['scheduled_time'],
            username=post_data['username'],
            status=post_data.get('status', 'scheduled'),
            cloudinary_public_ids=cloudinary_public_ids,
            created_at=datetime.now(timezone.utc)
        )
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        logger.info(f"Scheduled post {new_post.id} saved to DB with {len(media_urls)} media items.")
        return new_post
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving scheduled post to DB: {e}")
        raise
    finally:
        db.close()

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
                            cloudinary_public_id: str = None):
    """Schedule a post for Instagram. For carousel posts, media_url and cloudinary_public_id should be lists."""
    try:
        scheduled_time = datetime.fromisoformat(scheduled_time_str).astimezone(timezone.utc)
        
        # Ensure media_url and cloudinary_public_id are lists for carousel posts
        if media_type == 'carousel':
            if not isinstance(media_url, list) or not isinstance(cloudinary_public_id, list):
                raise ValueError("Carousel posts require lists of media URLs and Cloudinary public IDs")
            if len(media_url) < 2 or len(media_url) > 20:
                raise ValueError("Carousel posts must have between 2 and 20 images")
            if len(media_url) != len(cloudinary_public_id):
                raise ValueError("Number of media URLs and Cloudinary public IDs must match")
        else:
            # Convert single items to lists for consistent handling
            media_url = [media_url] if isinstance(media_url, str) else media_url
            cloudinary_public_id = [cloudinary_public_id] if isinstance(cloudinary_public_id, str) else cloudinary_public_id

        post_data = {
            'id': str(uuid.uuid4()),
            'media_urls': media_url if isinstance(media_url, list) else [media_url],
            'media_type': media_type,
            'caption': caption,
            'scheduled_time': scheduled_time,
            'username': username,
            'status': 'scheduled',
            'cloudinary_public_ids': cloudinary_public_id if isinstance(cloudinary_public_id, list) else [cloudinary_public_id],
        }
        new_post = save_scheduled_post_to_db(post_data)
        logger.info(f"Post {new_post.id} scheduled for {scheduled_time} by {username}.")
        return {'success': True, 'post_id': new_post.id}
    except Exception as e:
        logger.exception(f"Error scheduling Instagram post: {e}")
        return {'error': str(e)}


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

                    # Clean up completed task
                    try:
                        async_result.forget()
                        logger.info(f"Cleaned up completed task {post.celery_task_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up task {post.celery_task_id}: {e}")

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

@celery_app.task(bind=True, max_retries=5, default_retry_delay=60)
def _post_media_to_instagram_graph_api(self, post_id: str, media_urls: list, media_type: str, caption: str, username: str):
    """
    Internal task to post media to Instagram using the Graph API.
    Supports:
    - Single Image posts
    - Carousel posts (2-20 images)
    - Reels (video must meet Instagram's requirements):
        * MP4 container with H.264 codec
        * AAC audio codec
        * 9:16 aspect ratio (1080x1920 recommended)
        * 30fps
        * 3-90 seconds duration
        * Maximum file size: 650MB
        * Minimum resolution: 720p
    """
    logger.info(f"Attempting to post media for post {post_id}")
    db = SessionLocal()
    
    try:
        access_token = Config.INSTAGRAM_ACCESS_TOKEN
        business_account_id = Config.INSTAGRAM_BUSINESS_ACCOUNT_ID
        graph_api_base = "https://graph.facebook.com/v19.0/"

        if not access_token or not business_account_id:
            raise ValueError("Instagram Graph API Access Token or Business Account ID not configured.")

        # Handle Reels posting
        if media_type.lower() == 'reel':
            # Get the first media URL if it's a list
            media_url = media_urls[0] if isinstance(media_urls, list) else media_urls
            
            # Create container for reel with specific parameters
            container_payload = {
                "media_type": "REELS",
                "video_url": media_url,
                "caption": caption,
                "share_to_feed": "true",
                "access_token": access_token
            }

            # Add retry logic for container creation
            max_retries = 3
            retry_delay = 5
            last_error = None

            for attempt in range(max_retries):
                try:
                    logger.info(f"Creating reel container (attempt {attempt + 1})")
                    
                    # First, check if the URL is accessible
                    url_check = requests.head(media_url)
                    if url_check.status_code != 200:
                        raise ValueError(f"Video URL is not accessible (status code: {url_check.status_code})")

                    # Create the container
                    container_response = requests.post(
                        f"{graph_api_base}{business_account_id}/media",
                        data=container_payload
                    )
                    
                    # Log the full response for debugging
                    try:
                        response_data = container_response.json()
                        logger.info(f"Container creation response: {response_data}")
                        
                        if 'error' in response_data:
                            error_msg = response_data['error'].get('message', 'Unknown error')
                            error_code = response_data['error'].get('code')
                            logger.error(f"Error creating container: {error_msg} (Code: {error_code})")
                            
                            # Handle specific error codes
                            if error_code == 2207026:
                                error_detail = """
Your video does not meet Instagram's requirements. Please ensure your video meets ALL of these specifications:

Required Format:
- Container: MP4
- Video Codec: H.264
- Audio Codec: AAC
- Aspect Ratio: 9:16 (1080x1920 recommended)
- Frame Rate: 30fps
- Duration: 3-90 seconds
- Maximum File Size: 650MB
- Resolution: Minimum 720p
- Bitrate: Recommended 3500kbps

You can use tools like Adobe Premiere, Final Cut Pro, or FFmpeg to format your video correctly.
Example FFmpeg command:
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -b:a 128k -vf "scale=1080:1920:force_original_aspect_ratio=decrease" -r 30 output.mp4
"""
                                logger.error(f"Instagram Error 2207026: {error_detail}")
                                return {'error': f"Instagram Error 2207026: {error_detail}"}
                            elif error_code == 190:
                                raise ValueError("Invalid access token. Please check your credentials.")
                            else:
                                raise ValueError(f"Container creation error: {error_msg}")
                    except json.JSONDecodeError:
                        logger.warning("Could not parse response as JSON")
                        logger.info(f"Raw response: {container_response.text}")

                    container_response.raise_for_status()
                    container_data = response_data

                    if 'id' not in container_data:
                        raise ValueError("No container ID in response")

                    container_id = container_data['id']
                    logger.info(f"Successfully created reel container with ID: {container_id}")

                    # Wait for a moment to ensure the container is ready
                    time.sleep(5)

                    # Publish the container with enhanced error handling
                    publish_payload = {
                        "creation_id": container_id,
                        "access_token": access_token
                    }

                    # Add retry logic for publishing
                    publish_max_retries = 3
                    publish_retry_delay = 10
                    publish_attempt = 0
                    last_error = None
                    
                    while publish_attempt < publish_max_retries:
                        try:
                            publish_attempt += 1
                            logger.info(f"Attempting to publish container (attempt {publish_attempt})")
                            
                            # Check container status before publishing
                            status_response = requests.get(
                                f"{graph_api_base}{container_id}",
                                params={"fields": "status_code,status", "access_token": access_token}
                            )
                            status_data = status_response.json()
                            
                            if 'error' in status_data:
                                raise ValueError(f"Error checking container status: {status_data['error'].get('message')}")
                            
                            status = status_data.get('status_code')
                            if status != 'FINISHED':
                                logger.warning(f"Container not ready for publishing. Status: {status}")
                                if publish_attempt < publish_max_retries:
                                    time.sleep(publish_retry_delay)
                                    continue
                                raise ValueError(f"Container failed to process. Final status: {status}")

                            # Attempt to publish
                            publish_response = requests.post(
                                f"{graph_api_base}{business_account_id}/media_publish",
                                data=publish_payload
                            )
                            
                            if publish_response.status_code == 400:
                                error_data = publish_response.json()
                                error_message = error_data.get('error', {}).get('message', 'Unknown error')
                                logger.error(f"Publishing failed with 400 error: {error_message}")
                                
                                if 'Media ID not found' in error_message or 'not ready' in error_message.lower():
                                    if publish_attempt < publish_max_retries:
                                        logger.info(f"Retrying publish after delay ({publish_retry_delay}s)")
                                        time.sleep(publish_retry_delay)
                                        continue
                                
                                raise ValueError(f"Publishing failed: {error_message}")
                            
                            publish_response.raise_for_status()
                            publish_data = publish_response.json()

                            if 'id' in publish_data:
                                logger.info(f"Successfully published media with ID: {publish_data['id']}")
                                return {'success': True, 'media_id': publish_data['id']}
                            else:
                                raise ValueError("No media ID in publish response")

                        except (requests.exceptions.RequestException, ValueError) as e:
                            last_error = e
                            logger.error(f"Error on publish attempt {publish_attempt}: {str(e)}")
                            if publish_attempt < publish_max_retries:
                                time.sleep(publish_retry_delay)
                                continue
                            if publish_attempt == publish_max_retries:
                                raise ValueError(f"Failed to publish media after {publish_max_retries} attempts. Last error: {str(last_error)}")

                except requests.exceptions.RequestException as e:
                    last_error = e
                    logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise
                except ValueError as e:
                    last_error = e
                    logger.error(f"Value error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise
                except Exception as e:
                    last_error = e
                    logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise

            # If we get here, all retries failed
            raise ValueError(f"Failed to create and publish reel after {max_retries} attempts. Last error: {str(last_error)}")

        # Handle Image posting
        elif media_type.lower() == 'image':
            media_url = media_urls[0] if isinstance(media_urls, list) else media_urls
            
            # Create container for image
            container_payload = {
                "media_type": "IMAGE",
                "image_url": media_url,
                "caption": caption,
                "access_token": access_token
            }

            # Create container
            container_response = requests.post(
                f"{graph_api_base}{business_account_id}/media",
                data=container_payload
            )
            container_response.raise_for_status()
            container_data = container_response.json()

            if 'id' not in container_data:
                raise ValueError("No container ID in response")

            container_id = container_data['id']
            logger.info(f"Successfully created image container with ID: {container_id}")

            # Wait for a moment to ensure the container is ready
            time.sleep(5)

            # Publish the container
            publish_payload = {
                "creation_id": container_id,
                "access_token": access_token
            }

            publish_response = requests.post(
                f"{graph_api_base}{business_account_id}/media_publish",
                data=publish_payload
            )
            publish_response.raise_for_status()
            publish_data = publish_response.json()

            if 'id' in publish_data:
                logger.info(f"Successfully published image with ID: {publish_data['id']}")
                return {'success': True, 'media_id': publish_data['id']}
            else:
                raise ValueError("No media ID in publish response")

        # Handle Carousel posting
        elif media_type.lower() == 'carousel':
            if not isinstance(media_urls, list):
                raise ValueError("Carousel posts require a list of media URLs")
            if len(media_urls) < 2 or len(media_urls) > 20:
                raise ValueError("Carousel posts must have between 2 and 20 images")

            # Create containers for each image
            container_ids = []
            for idx, media_url in enumerate(media_urls, 1):
                logger.info(f"Creating container for carousel item {idx}")
                
                # Create container for each image
                container_payload = {
                    "media_type": "IMAGE",
                    "image_url": media_url,
                    "is_carousel_item": "true",
                    "access_token": access_token
                }

                container_response = requests.post(
                    f"{graph_api_base}{business_account_id}/media",
                    data=container_payload
                )
                container_response.raise_for_status()
                container_data = container_response.json()
                
                logger.info(f"API Response for item {idx}: {container_data}")
                
                if 'id' not in container_data:
                    raise ValueError(f"No container ID in response for item {idx}")
                
                container_ids.append(container_data['id'])
                logger.info(f"Successfully created container for carousel item {idx} with ID: {container_data['id']}")
                
                # Small delay between container creations
                time.sleep(1)

            # Create the carousel container
            logger.info(f"Creating carousel container with {len(container_ids)} items")
            carousel_payload = {
                "media_type": "CAROUSEL",
                "caption": caption,
                "children": ",".join(container_ids),
                "access_token": access_token
            }

            carousel_response = requests.post(
                f"{graph_api_base}{business_account_id}/media",
                data=carousel_payload
            )
            carousel_response.raise_for_status()
            carousel_data = carousel_response.json()
            
            logger.info(f"Carousel container response: {carousel_data}")
            
            if 'id' not in carousel_data:
                raise ValueError("No container ID in carousel response")

            carousel_container_id = carousel_data['id']
            logger.info(f"Successfully created carousel container with ID: {carousel_container_id}")

            # Wait for a moment to ensure the container is ready
            time.sleep(5)

            # Publish the carousel
            logger.info("Publishing carousel post")
            publish_payload = {
                "creation_id": carousel_container_id,
                "access_token": access_token
            }

            publish_response = requests.post(
                f"{graph_api_base}{business_account_id}/media_publish",
                data=publish_payload
            )
            publish_response.raise_for_status()
            publish_data = publish_response.json()
            
            logger.info(f"Successfully published carousel post. Response: {publish_data}")

            if 'id' in publish_data:
                return {'success': True, 'media_id': publish_data['id']}
            else:
                raise ValueError("No media ID in publish response")

        else:
            raise ValueError(f"Unsupported media_type: {media_type}")

    except requests.exceptions.RequestException as e:
        error_msg = f"Instagram Graph API request failed: {str(e)}"
        logger.error(error_msg)
        try:
            self.retry(exc=e, countdown=60)
        except self.MaxRetriesExceededError:
            return {'error': error_msg + " (Max retries exceeded)"}
    except Exception as e:
        error_msg = f"Error posting to Instagram: {str(e)}"
        logger.error(error_msg)
        return {'error': error_msg}
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
    """
    Preprocesses video to meet Instagram Reels requirements.
    Returns the path to the processed video file.
    """
    try:
        logger.info(f"Starting video preprocessing for URL: {video_url}")
        
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp()
        
        # Download video if it's a URL
        if video_url.startswith(('http://', 'https://')):
            input_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.mp4")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            input_path = video_url

        # Get video info using ffprobe
        probe_cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            input_path
        ]
        
        probe_output = subprocess.check_output(probe_cmd).decode('utf-8')
        video_info = json.loads(probe_output)
        
        # Log original video properties
        logger.info(f"Original video info: {json.dumps(video_info, indent=2)}")
        
        # Get video stream info
        video_stream = next((s for s in video_info['streams'] if s['codec_type'] == 'video'), None)
        if not video_stream:
            raise ValueError("No video stream found")
            
        # Calculate target dimensions maintaining aspect ratio
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Target 9:16 aspect ratio (1080x1920)
        target_width = 1080
        target_height = 1920
        
        # Calculate padding for letterboxing/pillarboxing
        if (height/width) > (16/9):  # Too tall
            new_height = int((width * 16) / 9)
            pad_top = int((height - new_height) / 2)
            pad_bottom = height - new_height - pad_top
            pad_left = 0
            pad_right = 0
        else:  # Too wide
            new_width = int((height * 9) / 16)
            pad_left = int((width - new_width) / 2)
            pad_right = width - new_width - pad_left
            pad_top = 0
            pad_bottom = 0
            
        output_path = os.path.join(temp_dir, f"processed_{uuid.uuid4()}.mp4")
        
        # Construct FFmpeg command for processing
        ffmpeg_cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'pad=width={width + pad_left + pad_right}:height={height + pad_top + pad_bottom}:x={pad_left}:y={pad_top}:color=black',
            '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease',
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'medium',  # Balance between speed and quality
            '-crf', '23',  # Constant Rate Factor (18-28 is visually lossless)
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '128k',  # Audio bitrate
            '-r', '30',  # 30fps
            '-movflags', '+faststart',  # Enable fast start for web playback
            '-y',  # Overwrite output file if exists
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        # Run FFmpeg
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            raise ValueError(f"FFmpeg processing failed: {stderr.decode()}")
            
        # Verify the processed file
        if not os.path.exists(output_path):
            raise ValueError("FFmpeg did not create output file")
            
        # Get processed video info
        probe_cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            output_path
        ]
        
        probe_output = subprocess.check_output(probe_cmd).decode('utf-8')
        processed_info = json.loads(probe_output)
        logger.info(f"Processed video info: {json.dumps(processed_info, indent=2)}")
        
        # Upload to Cloudinary
        if CLOUDINARY_CONFIGURED:
            try:
                upload_result = cloudinary.uploader.upload(
                    output_path,
                    resource_type="video",
                    folder="instagram_reels"
                )
                logger.info(f"Uploaded processed video to Cloudinary: {upload_result['secure_url']}")
                
                # Clean up local files
                os.remove(input_path)
                os.remove(output_path)
                os.rmdir(temp_dir)
                
                return upload_result['secure_url']
            except Exception as e:
                logger.error(f"Cloudinary upload failed: {e}")
                raise
        else:
            logger.warning("Cloudinary not configured, returning local file path")
            return output_path
            
    except Exception as e:
        logger.error(f"Video preprocessing failed: {e}")
        # Clean up any remaining temporary files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
        raise