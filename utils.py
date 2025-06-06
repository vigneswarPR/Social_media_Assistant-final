import os
import tempfile
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return None

def get_media_type(file_path):
    """Determine media type based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.gif']:
        return 'image'
    elif ext in ['.mp4', '.mov']:
        return 'video'
    else:
        return None 