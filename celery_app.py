from celery import Celery
from datetime import timedelta
from config import Config
import os

# Create uploads directory if it doesn't exist
# It's good practice to ensure this exists before any tasks try to write to it.
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize Celery
celery_app = Celery(
    'social_media_assistant',
    broker=Config.REDIS_URL,
    backend=Config.REDIS_URL,
    # This tells Celery to look for tasks in the 'tasks' module (e.g., tasks.py)
    # Make sure your main_tasks.py is named tasks.py or adjust this line.
    include=['tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',        # All internal times in Celery will be UTC
    enable_utc=True,       # Ensures UTC is used for schedules and timestamps

    # Define Celery Beat schedules
    beat_schedule={
        'check-scheduled-posts-every-minute': {
            'task': 'tasks.check_and_post_scheduled', # Assuming the task is in tasks.py
            'schedule': timedelta(minutes=1),         # Check every minute
            'args': (),                               # No arguments for this task
        },
        'monitor-instagram-post-status-every-5-minutes': {
            'task': 'tasks.monitor_instagram_post_status', # Assuming the task is in tasks.py
            'schedule': timedelta(minutes=5),               # Monitor status every 5 minutes
            'args': (),                                     # No arguments for this task
        },
    }
)