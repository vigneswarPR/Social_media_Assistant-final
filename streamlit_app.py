import google.generativeai as genai
import streamlit as st
import json
from datetime import datetime, time, timezone, timedelta
import pandas as pd
from PIL import Image
import calendar
from io import StringIO
import os
from tasks import generate_captions, schedule_instagram_post, load_scheduled_posts
from utils import save_uploaded_file, get_media_type
from config import Config
import uuid
import re
import cloudinary
import cloudinary.uploader
import requests
import cv2
import logging
import tempfile
import pytz
import matplotlib.pyplot as plt
from collections import defaultdict
from dateutil import parser
from instagram_analytics import InstagramAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Streamlit layout
st.set_page_config(layout="wide", page_title="AI Social Media")

# Initialize session state
session_defaults = {
    'uploaded_media_path': None,
    'current_media_type': None,
    'generated_captions': [],
    'selected_caption_text': "",
    'caption_style': 'high_engagement',
    'generated_calendar_text': None,
    'calendar_dataframe': None,
    'cloudinary_public_id': None,
    'post_scheduled_success': False
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Success message after scheduling
if st.session_state['post_scheduled_success']:
    st.success("✅ Post scheduled successfully and all fields have been reset.")
    st.session_state['post_scheduled_success'] = False

# Page Title
st.title("📸 AI Social Media Assistant 🎥")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Post Scheduler", "Content Calendar", "Scheduled Posts", "Analytics"])

# Configure Cloudinary
try:
    cloudinary.config(
        cloud_name=Config.CLOUDINARY_CLOUD_NAME,
        api_key=Config.CLOUDINARY_API_KEY,
        api_secret=Config.CLOUDINARY_API_SECRET
    )
except Exception as e:
    st.error(f"Cloudinary configuration error: {e}")
    st.stop()

# Post Scheduler Page
if page == "Post Scheduler":
    st.header("1. Upload Your Media")
    
    # Media type selection
    media_type_choice = st.radio(
        "What type of content are you uploading?",
        ["Image", "Carousel", "Reel"],
        help="Select the type of content you want to upload. For Reels, video will be resized to 9:16. For Carousel, you can upload 2-20 images."
    )
    
    # File type restrictions based on selection
    if media_type_choice == "Image":
        allowed_types = ["png", "jpg", "jpeg", "gif"]
        help_text = "Upload an image (PNG, JPG, JPEG, or GIF)"
        st.session_state['current_media_type'] = "image"  # Set lowercase media type
    elif media_type_choice == "Carousel":
        allowed_types = ["png", "jpg", "jpeg"]
        help_text = "Upload 2-20 images for your carousel (PNG, JPG, JPEG)"
        st.session_state['current_media_type'] = "carousel"  # Set lowercase media type
        st.info("📌 For carousel posts:\n"
                "- You must upload between 2 and 20 images\n"
                "- All images must be in PNG, JPG, or JPEG format\n"
                "- Each image should be less than 10MB")
    else:  # Reel
        allowed_types = ["mp4", "mov"]
        help_text = "Upload a video (MP4 or MOV) that meets Instagram Reels requirements:\n"
        st.session_state['current_media_type'] = "reel"  # Set lowercase media type
        st.warning("""
        ⚠️ Instagram Reels Requirements:
        
        Your video MUST meet these specifications:
        1. Format: MP4 with H.264 codec
        2. Audio: AAC codec
        3. Aspect Ratio: 9:16 (1080x1920 recommended)
        4. Frame Rate: 30fps
        5. Duration: 3-90 seconds
        6. Maximum File Size: 650MB
        7. Resolution: Minimum 720p
        8. Bitrate: Recommended 3500kbps
        
        Common Issues:
        - Wrong aspect ratio (must be vertical 9:16)
        - Incorrect video codec (must be H.264)
        - Wrong audio codec (must be AAC)
        - Frame rate too high or low (must be 30fps)
        - Duration too short or long (must be 3-90 seconds)
        
        💡 Tip: Use video editing software like Adobe Premiere, Final Cut Pro, or FFmpeg to format your video correctly.
        """)

    # Initialize temp_files list before file upload
    temp_files = []
    cloudinary_uploads = []

    uploaded_files = st.file_uploader(
        f"Choose your {media_type_choice.lower()}", 
        type=allowed_types, 
        help=help_text,
        accept_multiple_files=(media_type_choice == "Carousel")
    )

    if uploaded_files:
        try:
            # Convert single file to list for consistent handling
            files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            
            # Enhanced carousel validation with clear error messages
            if media_type_choice == "Carousel":
                logger.info(f"Starting carousel validation for {len(files_list)} files")
                if not files_list:
                    logger.warning("No files uploaded for carousel")
                    st.error("❌ No files were uploaded. Please select some images.")
                    st.stop()
                elif len(files_list) == 1:
                    logger.warning("Only one file uploaded for carousel")
                    st.error("❌ Carousel posts require at least 2 images. You uploaded only 1 image.")
                    st.info("💡 Tip: Hold Ctrl/Cmd while selecting files to choose multiple images.")
                    st.stop()
                elif len(files_list) > 20:
                    logger.warning(f"Too many files uploaded for carousel: {len(files_list)}")
                    st.error(f"❌ Carousel posts can have maximum 20 images. You uploaded {len(files_list)} images.")
                    st.info("💡 Please remove some images and try again.")
                    st.stop()
                
                # Validate file sizes
                oversized_files = [f.name for f in files_list if f.size > 10 * 1024 * 1024]  # 10MB limit
                if oversized_files:
                    logger.warning(f"Oversized files detected: {oversized_files}")
                    st.error("❌ Some files exceed the 10MB size limit:")
                    for fname in oversized_files:
                        st.write(f"- {fname}")
                    st.info("💡 Please optimize your images and try again.")
                    st.stop()

                # Validate aspect ratios immediately
                logger.info("Starting aspect ratio validation")
                st.write("🔍 Checking image aspect ratios...")
                invalid_images = []
                
                for idx, file in enumerate(files_list, 1):
                    try:
                        logger.info(f"Validating image {idx}/{len(files_list)}: {file.name}")
                        # Create a temporary file to check the image
                        temp_path = save_uploaded_file(file)
                        if temp_path:
                            temp_files.append(temp_path)  # Track for cleanup
                            logger.debug(f"Created temporary file: {temp_path}")
                            
                            with Image.open(temp_path) as img:
                                width, height = img.size
                                aspect_ratio = width / height
                                logger.debug(f"Image dimensions: {width}x{height}, ratio: {aspect_ratio:.2f}")
                                
                                # Define acceptable ratio ranges (with some tolerance)
                                SQUARE_RATIO = 1.0  # 1:1
                                PORTRAIT_RATIO = 0.8  # 4:5
                                TOLERANCE = 0.02  # 2% tolerance for more flexibility
                                
                                # Check if the aspect ratio matches either 1:1 or 4:5
                                is_square = abs(aspect_ratio - SQUARE_RATIO) <= TOLERANCE
                                is_portrait = abs(aspect_ratio - PORTRAIT_RATIO) <= TOLERANCE
                                
                                # Display current image's ratio (for debugging)
                                st.write(f"Checking image {idx}: {file.name}")
                                st.write(f"Dimensions: {width}x{height}")
                                st.write(f"Aspect ratio: {aspect_ratio:.3f}")
                                
                                if not (is_square or is_portrait):
                                    logger.warning(f"Invalid aspect ratio for {file.name}: {aspect_ratio:.3f}")
                                    st.warning(f"""
                                    ⚠️ Invalid aspect ratio detected:
                                    - Current ratio: {aspect_ratio:.3f}
                                    - Required ratios: 1.0 (square) or 0.8 (portrait)
                                    - Tolerance: ±{TOLERANCE*100}%
                                    """)
                                    
                                    invalid_images.append({
                                        'name': file.name,
                                        'dimensions': f"{width}x{height}",
                                        'ratio': f"{aspect_ratio:.3f}",
                                        'index': idx,
                                        'square_diff': abs(aspect_ratio - SQUARE_RATIO),
                                        'portrait_diff': abs(aspect_ratio - PORTRAIT_RATIO)
                                    })
                                else:
                                    ratio_type = "square (1:1)" if is_square else "portrait (4:5)"
                                    st.success(f"✅ Valid {ratio_type} ratio")
                                
                    except Exception as e:
                        logger.error(f"Error processing image {file.name}: {str(e)}", exc_info=True)
                        st.error(f"Error processing {file.name}: {str(e)}")
                        invalid_images.append({
                            'name': file.name,
                            'error': str(e),
                            'index': idx
                        })

                if invalid_images:
                    logger.warning(f"Found {len(invalid_images)} invalid images")
                    st.error("❌ Some images have invalid aspect ratios for Instagram carousel:")
                    
                    # Create a more detailed error message
                    st.markdown("""
                    ### Invalid Images Detected
                    
                    Instagram requires carousel images to be either:
                    - Square (1:1 ratio = 1.0)
                    - Portrait (4:5 ratio = 0.8)
                    """)
                    
                    for img in invalid_images:
                        if 'dimensions' in img:
                            st.markdown(f"""
                            **Image {img['index']}: {img['name']}**
                            - Dimensions: {img['dimensions']}
                            - Current ratio: {img['ratio']}
                            - Difference from square: {img['square_diff']:.3f}
                            - Difference from portrait: {img['portrait_diff']:.3f}
                            """)
                        else:
                            st.error(f"Image {img['index']}: {img['name']} (Error: {img['error']})")
                    
                    st.info("""
                    💡 To fix this:
                    1. Resize your images to either:
                       - Square: 1080x1080 pixels (recommended)
                       - Portrait: 1080x1350 pixels (recommended)
                    2. Make sure the aspect ratio is exactly 1:1 (square) or 4:5 (portrait)
                    3. Upload the resized images
                    """)
                    
                    # Clean up any temporary files
                    logger.info("Cleaning up temporary files")
                    for temp_file in temp_files:
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                                logger.debug(f"Cleaned up temporary file: {temp_file}")
                            except Exception as e:
                                logger.error(f"Failed to clean up temporary file {temp_file}: {e}", exc_info=True)
                    st.stop()
                
                logger.info(f"Successfully validated {len(files_list)} images for carousel post")
                st.success(f"✅ Successfully validated {len(files_list)} images for your carousel post.")

            for idx, uploaded_file in enumerate(files_list):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        temp_files.append(file_path)
                    
                    # Determine media type
                    detected_media_type = "reel" if media_type_choice == "Reel" else "carousel" if media_type_choice == "Carousel" else "image"
                    
                    # Upload to Cloudinary
                    upload_result = cloudinary.uploader.upload(
                        file_path,
                        folder="instagram_posts",
                        resource_type="video" if detected_media_type == "reel" else "image"
                    )

                    if 'secure_url' in upload_result:
                        cloudinary_uploads.append({
                            'url': upload_result['secure_url'],
                            'public_id': upload_result['public_id']
                        })
                        
                        # Preview the uploaded media
                        if detected_media_type == "image" or detected_media_type == "carousel":
                            st.image(upload_result['secure_url'], caption=f"Image {idx + 1}", use_column_width=True)
                        else:
                            st.video(upload_result['secure_url'])

                except Exception as e:
                    st.error(f"Error uploading file {uploaded_file.name}: {str(e)}")
                    continue

            if cloudinary_uploads:
                st.session_state['uploaded_media_path'] = [upload['url'] for upload in cloudinary_uploads]
                st.session_state['cloudinary_public_id'] = [upload['public_id'] for upload in cloudinary_uploads]
                st.session_state['current_media_type'] = detected_media_type
                st.success(f"{media_type_choice} uploaded successfully!")
        
        except Exception as e:
            st.error(f"Upload error: {e}")
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        st.warning(f"Failed to clean up temporary file: {e}")
                try:
                    os.rmdir(os.path.dirname(temp_file))
                except:
                    pass

    st.header("2. Generate AI Captions")
    if st.session_state['uploaded_media_path']:
        col1, col2 = st.columns(2)
        with col1:
            st.session_state['caption_style'] = st.selectbox("Caption Style:", ['high_engagement', 'story_style', 'viral_potential', 'targeted', 'A/B Test', 'custom'], index=0)
            num_variants = st.slider("Caption Variants", 1, 5, 3)
        with col2:
            custom_prompt = st.text_area("Custom Prompt:") if st.session_state['caption_style'] == 'custom' else None
            target_audience = st.text_input("Target Audience") if st.session_state['caption_style'] == 'targeted' else None
            business_goals = st.text_input("Business Goals") if st.session_state['caption_style'] == 'targeted' else None

        if st.button("Generate Captions"):
            with st.spinner("Generating captions..."):
                try:
                    task = generate_captions.delay(
                        st.session_state['uploaded_media_path'],
                        st.session_state['current_media_type'],
                        st.session_state['caption_style'],
                        custom_prompt,
                        target_audience,
                        business_goals,
                        num_variants
                    )
                    result = task.get(timeout=180)
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.session_state['generated_captions'] = result['captions']
                        st.session_state['selected_caption_text'] = result['captions'][0]['text'] if result['captions'] else ""
                        st.success("Captions ready!")
                except Exception as e:
                    st.error(f"Caption generation failed: {e}")

    if st.session_state['generated_captions']:
        st.header("3. Select and Edit Caption")
        previews = [f"Caption {i+1}: {c['text'][:60]}..." for i, c in enumerate(st.session_state['generated_captions'])]
        selected_index = st.radio("Choose a caption:", range(len(previews)), format_func=lambda i: previews[i])
        selected_caption = st.session_state['generated_captions'][selected_index]['text']
        st.session_state['selected_caption_text'] = st.text_area("Edit Caption", value=selected_caption, height=150)

    if st.session_state['selected_caption_text']:
        st.header("4. Schedule Post")
        col1, col2 = st.columns(2)
        
        # Get current time in IST using pytz
        IST = pytz.timezone('Asia/Kolkata')
        current_time_ist = datetime.now(IST)
        
        # Initialize schedule time in session state if not present
        if 'schedule_date' not in st.session_state:
            st.session_state['schedule_date'] = current_time_ist.date()
        if 'schedule_time' not in st.session_state:
            st.session_state['schedule_time'] = current_time_ist.time()
        
        # Use session state for date and time, updating only when user changes them
        schedule_date = col1.date_input(
            "Post Date",
            value=st.session_state['schedule_date'],
            key='date_input'
        )
        schedule_time = col2.time_input(
            "Post Time",
            value=st.session_state['schedule_time'],
            key='time_input'
        )
        
        # Update session state with new values
        st.session_state['schedule_date'] = schedule_date
        st.session_state['schedule_time'] = schedule_time
        
        # Add timezone info
        st.info("⏰ All times are in Indian Standard Time (IST / UTC+5:30)")
        
        post_now = st.checkbox("Post immediately")
        
        if st.button("Schedule Post"):
            try:
                # Ensure we have both media URLs and Cloudinary public IDs
                if not st.session_state.get('uploaded_media_path') or not st.session_state.get('cloudinary_public_id'):
                    st.error("❌ Missing media information. Please upload your media files first.")
                    st.stop()

                # Convert to lists if not already
                media_urls = st.session_state['uploaded_media_path']
                cloudinary_ids = st.session_state['cloudinary_public_id']
                
                if not isinstance(media_urls, list):
                    media_urls = [media_urls]
                if not isinstance(cloudinary_ids, list):
                    cloudinary_ids = [cloudinary_ids]

                # Convert local schedule time to UTC for storage
                if post_now:
                    scheduled_datetime = datetime.now(timezone.utc)
                    local_dt_ist = datetime.now(IST)
                else:
                    # Combine date and time and localize to IST using pytz
                    naive_dt = datetime.combine(st.session_state['schedule_date'], st.session_state['schedule_time'])
                    local_dt_ist = IST.localize(naive_dt)
                    # Convert to UTC for storage
                    scheduled_datetime = local_dt_ist.astimezone(pytz.UTC)
                
                # Use isoformat() with proper timezone handling
                scheduled_datetime_str = scheduled_datetime.replace(microsecond=0).isoformat()
                
                # Schedule the post
                task = schedule_instagram_post.delay(
                    media_urls,
                    st.session_state['current_media_type'],
                    st.session_state['selected_caption_text'],
                    scheduled_datetime_str,
                    Config.INSTAGRAM_USERNAME,
                    cloudinary_ids
                )
                
                result = task.get(timeout=180)
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.session_state['post_scheduled_success'] = True
                    # Reset session state except for schedule date/time
                    for k in ['uploaded_media_path', 'current_media_type', 'generated_captions', 
                             'selected_caption_text', 'cloudinary_public_id']:
                        st.session_state[k] = session_defaults[k]
                    
                    # Format the scheduled time for display
                    scheduled_time_str = local_dt_ist.strftime('%Y-%m-%d %I:%M %p IST')
                    st.success(f"✅ Post scheduled successfully for {scheduled_time_str}!")
                    st.info("You can view and manage it in the Scheduled Posts tab.")
            except Exception as e:
                st.error(f"Scheduling failed: {e}")

# Add logic for "Content Calendar" and "Scheduled Posts" pages here as needed

elif page == "Content Calendar":
    st.title("🗓️ AI-Generated Content Calendar for Food Delivery Service")
    st.write("Generate a customized content calendar for your South Indian and North Indian food delivery service.")

    col1, col2 = st.columns(2)
    with col1:
        current_year = datetime.now().year
        current_month = datetime.now().month
        selected_month = st.selectbox(
            "Select Month:",
            options=range(1, 13),
            format_func=lambda x: calendar.month_name[x],
            index=current_month - 1,
            key="calendar_month_select"
        )
    selected_year = st.selectbox(
        "Select Year:",
        options=range(current_year, current_year + 2),
        index=0,
        key="calendar_year_select"
    )

    with col2:
        num_days = st.number_input(
            "Number of days to generate calendar for:",
            min_value=7,
            max_value=31,
            value=14,
            key="calendar_num_days"
        )
    food_style = st.multiselect(
        "Select Food Style(s):",
        options=["South Indian", "North Indian", "Both"],
        default=["Both"],
        key="calendar_food_style"
    )
    promotion_focus = st.text_area(
        "Any specific promotions or themes to focus on?",
        "e.g., 'summer specials', 'healthy options', 'festive deals'",
        key="calendar_promotion_focus"
    )

    if st.button("Generate Content Calendar", type="primary"):
        if not Config.GEMINI_API_KEY:
            st.error("Please set your GEMINI_API_KEY in the .env file or as an environment variable.")
        else:
            with st.spinner("Generating content calendar with AI..."):
                try:
                    # Construct prompt for content calendar generation
                    food_style_str = ", ".join(food_style)
                    prompt = f"""
                    Generate a {num_days}-day social media content calendar for a food delivery service focusing on {food_style_str} cuisine for {calendar.month_name[selected_month]}, {selected_year}.
                    Include topics, post ideas, and suggested Instagram features (e.g., Reel, Story, Carousel, Static Post).
                    Consider the following:
                    - Promotions/Themes: {promotion_focus if promotion_focus else 'None'}
                    - Include a mix of engaging, informative, and promotional content.
                    - Suggest relevant hashtags.
                    - Structure the output as a JSON array of daily entries, with each entry having 'date', 'topic', 'post_idea', 'instagram_feature', 'hashtags'.
                    - Example format:
                    ```json
                    [
                      {{
                        "date": "YYYY-MM-DD",
                        "topic": "Topic Name",
                        "post_idea": "Detailed post description",
                        "instagram_feature": "Reel/Story/Carousel/Static Post",
                        "hashtags": "#food #delivery #delicious"
                      }}
                    ]
                    ```
                    Ensure the dates are correct for the specified month and year, starting from the 1st of the month.
                    """

                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(prompt)
                    response_text = response.text

                    try:
                        json_match = re.search(r'```json\n(.*)\n```', response_text, re.DOTALL)
                        if json_match:
                            calendar_data = json.loads(json_match.group(1))
                        else:
                            calendar_data = json.loads(response_text)

                        if isinstance(calendar_data, list):
                            st.session_state['generated_calendar_text'] = response_text
                            st.session_state['calendar_dataframe'] = pd.DataFrame(calendar_data)
                            st.success("Content calendar generated successfully!")
                        else:
                            st.error("AI generated an invalid calendar format. Please try again.")
                            st.session_state['generated_calendar_text'] = response_text
                            st.session_state['calendar_dataframe'] = None

                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse AI response as JSON: {e}. Raw response: {response_text}")
                        st.session_state['generated_calendar_text'] = response_text
                        st.session_state['calendar_dataframe'] = None

                except Exception as e:
                    st.error(f"Error generating content calendar: {str(e)}")

    if st.session_state['calendar_dataframe'] is not None:
        st.subheader("Generated Content Calendar")
        st.dataframe(st.session_state['calendar_dataframe'], use_container_width=True)

        st.download_button(
            label="Download Calendar as CSV",
            data=st.session_state['calendar_dataframe'].to_csv(index=False).encode('utf-8'),
            file_name=f"content_calendar_{calendar.month_name[selected_month]}_{selected_year}.csv",
            mime="text/csv",
            key="download_calendar_csv"
        )
        st.info("You can further refine the calendar by editing the downloaded CSV or adjusting your prompt.")

elif page == "Scheduled Posts":
    st.header("📋 Your Scheduled Posts")
    scheduled_posts = load_scheduled_posts()

    if scheduled_posts:
        # Sort posts by scheduled time
        scheduled_posts.sort(key=lambda x: x['scheduled_time'])

        df = pd.DataFrame(scheduled_posts)
        
        # Define IST timezone
        IST = pytz.timezone('Asia/Kolkata')

        # Convert datetime columns to IST
        datetime_columns = ['scheduled_time', 'created_at', 'updated_at', 'posting_attempt_at']
        for col in datetime_columns:
            if col in df.columns and df[col].notna().any():
                try:
                    # Convert to pandas datetime with coerce option to handle invalid dates
                    df[col] = pd.to_datetime(df[col], format='ISO8601', errors='coerce')
                    
                    # Check if the column has any timezone info
                    if df[col].dt.tz is None:
                        # If timezone naive, localize to UTC first
                        df[f'{col}_ist'] = df[col].dt.tz_localize('UTC').dt.tz_convert(IST)
                    else:
                        # If already timezone aware, just convert
                        df[f'{col}_ist'] = df[col].dt.tz_convert(IST)
                    
                    # Format for display
                    df[f'{col}_display'] = df[f'{col}_ist'].dt.strftime('%Y-%m-%d %I:%M %p IST')
                except Exception as e:
                    st.error(f"Error processing {col}: {str(e)}")
                    st.write("Data sample:", df[col].head())

        # Create display DataFrame with formatted columns
        df_display = pd.DataFrame({
            'Scheduled Time (IST)': df['scheduled_time_display'],
            'Caption': df['caption'],
            'Media Type': df['media_type'],
            'Status': df['status'],
            'Error Message': df['error_message'],
            'Last Update': df.get('updated_at_display', ''),
            'Posting Attempt': df.get('posting_attempt_at_display', '')
        })

        status_filter = st.selectbox("Filter by Status:", ['All', 'scheduled', 'posting_in_progress', 'posted', 'failed'])

        if status_filter != 'All':
            df_display = df_display[df_display['Status'] == status_filter]

        st.dataframe(df_display, use_container_width=True)

        st.info(
            """
            📅 All times are shown in Indian Standard Time (IST / UTC+5:30)
            
            Status meanings:
            - 'scheduled': Waiting for scheduled time
            - 'posting_in_progress': Currently being posted
            - 'posted': Successfully published to Instagram
            - 'failed': Error occurred during posting
            """
        )
        st.warning("Media files for 'posted' items are automatically deleted from storage to save space.")

        # Option to clear all scheduled posts (for development/testing)
        st.markdown("---")
        st.subheader("Danger Zone")
        if st.button("Clear All Scheduled Posts (DANGER!)"):
            try:
                with open('scheduled_posts.json', 'w') as f:
                    json.dump([], f)
                st.success("All scheduled posts cleared. Refreshing page...")
            except Exception as e:
                st.error(f"Error clearing posts: {str(e)}")
    else:
        st.info("No scheduled posts found. Schedule some posts using the 'Post Scheduler'!")

elif page == "Analytics":
    st.title("📊 Instagram Analytics Dashboard")
    
    # Initialize InstagramAnalytics if access token is available
    if not Config.INSTAGRAM_ACCESS_TOKEN:
        st.error("⚠️ Instagram access token not configured. Please set up your credentials first.")
        st.stop()
        
    try:
        analytics = InstagramAnalytics(Config.INSTAGRAM_ACCESS_TOKEN)
        
        # Use st.cache_data to prevent recomputing on every rerun
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def get_instagram_id(page_id):
            return analytics.get_instagram_account_id(page_id)
            
        @st.cache_data(ttl=300)
        def get_analytics_data(instagram_id):
            return analytics.get_media_insights(instagram_id)
            
        # Create progress container
        progress_container = st.empty()
        
        with progress_container:
            with st.spinner("🔄 Loading Instagram analytics..."):
                instagram_id = get_instagram_id(Config.FACEBOOK_PAGE_ID)
                posts = get_analytics_data(instagram_id)
        
        if not posts:
            st.warning("No posts data available. Showing sample data for demonstration.")
        
        # Create tabs for different analytics sections
        tab1, tab2, tab3 = st.tabs(["📈 Growth Metrics", "⏰ Best Time to Post", "🎯 Post Performance"])
        
        with tab1:
            st.header("Account Growth")
            
            # Make the graph section full width
            # Follower Growth Chart
            dates, counts = analytics.get_follower_count_trend(instagram_id)
            if dates and counts:
                # Create a larger figure with better proportions
                fig = plt.figure(figsize=(16, 8))
                
                # Plot with enhanced styling
                plt.plot(dates, counts, marker='o', linestyle='-', linewidth=2.5, 
                        color='#1f77b4', markersize=8)
                
                # Add grid and styling
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.title("Follower Growth Trend (Last 7 Days)", fontsize=16, pad=20)
                plt.xlabel("Date", fontsize=12, labelpad=10)
                plt.ylabel("Total Followers", fontsize=12, labelpad=10)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')
                
                # Add data points annotation
                for i, (date, count) in enumerate(zip(dates, counts)):
                    plt.annotate(f'{count:,}', 
                               (date, count),
                               textcoords="offset points",
                               xytext=(0,10),
                               ha='center',
                               fontsize=10)
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                # Use the full width of the page
                st.pyplot(fig, use_container_width=True)
                plt.close()
                
                # Show metrics in columns below the graph
                col1, col2, col3 = st.columns(3)
                
                # Calculate metrics
                growth = counts[-1] - counts[0]
                growth_rate = ((counts[-1] / counts[0]) - 1) * 100 if counts[0] > 0 else 0
                daily_avg = growth / len(counts) if len(counts) > 1 else 0
                
                with col1:
                    st.metric(
                        "Total Growth (7 Days)", 
                        f"{growth:+,d}",
                        f"{growth_rate:.1f}%"
                    )
                with col2:
                    st.metric(
                        "Current Followers",
                        f"{counts[-1]:,d}",
                        f"Started at {counts[0]:,d}"
                    )
                with col3:
                    st.metric(
                        "Average Daily Growth",
                        f"{daily_avg:+.1f}",
                        f"{(daily_avg/counts[0]*100):.2f}% per day" if counts[0] > 0 else "N/A"
                    )
        
        with tab2:
            st.header("Optimal Posting Times")
            
            if posts:
                # Convert posts to JSON string for caching
                posts_json = json.dumps(posts)
                best_times = analytics.get_best_times(posts_json)
                
                # Create engagement heatmap
                hours = list(range(24))
                values = [best_times[h]['engagement_rate'] for h in hours]
                
                fig = plt.figure(figsize=(12, 4))
                plt.bar(hours, values, color='#2ecc71')
                plt.title("Average Engagement Rate by Hour")
                plt.xlabel("Hour of Day (24-hour format)")
                plt.ylabel("Engagement Rate (%)")
                plt.xticks(hours)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Show best posting times with more detail
                st.subheader("🎯 Recommended Posting Times")
                sorted_hours = sorted(best_times.items(), 
                                   key=lambda x: (x[1]['engagement_rate'], x[1]['post_count']), 
                                   reverse=True)
                
                top_hours = st.slider("Number of top hours to show", 3, 8, 5)
                
                for i, (hour, data) in enumerate(sorted_hours[:top_hours], 1):
                    with st.expander(
                        f"#{i} - {hour:02d}:00 - {(hour+1):02d}:00 "
                        f"(Engagement: {data['engagement_rate']:.1f}%)"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Posts at this hour", data['post_count'])
                            st.metric("Avg. Engagement", f"{data['avg_engagement']:,d}")
                        with col2:
                            st.metric("Avg. Reach", f"{data['avg_reach']:,d}")
                            st.metric("Success Rate", f"{data['engagement_rate']:.1f}%")
        
        with tab3:
            st.header("Post Performance Analysis")
            
            if posts:
                # Calculate overall engagement metrics
                engagement_rates = [post['engagement_rate'] for post in posts]
                avg_rate = sum(engagement_rates) / len(engagement_rates)
                
                # Show overall metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Posts Analyzed", len(posts))
                with col2:
                    st.metric("Avg. Engagement Rate", f"{avg_rate:.1f}%")
                with col3:
                    st.metric("Best Performing Post", f"{max(engagement_rates):.1f}%")
                
                # Show detailed engagement metrics
                st.subheader("Engagement Breakdown")
                total_likes = sum(post['likes'] for post in posts)
                total_comments = sum(post['comments'] for post in posts)
                total_saves = sum(post['saved'] for post in posts)
                total_reach = sum(post['reach'] for post in posts)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Likes", f"{total_likes:,d}")
                with col2:
                    st.metric("Total Comments", f"{total_comments:,d}")
                with col3:
                    st.metric("Total Saves", f"{total_saves:,d}")
                with col4:
                    st.metric("Total Reach", f"{total_reach:,d}")
                
                # Create performance visualizations
                st.subheader("Performance Analysis")
                
                tab_hist, tab_trend = st.tabs(["📊 Distribution", "📈 Trends"])
                
                with tab_hist:
                    fig = plt.figure(figsize=(10, 4))
                    plt.hist(engagement_rates, bins=20, color='#3498db', alpha=0.7)
                    plt.title("Engagement Rate Distribution")
                    plt.xlabel("Engagement Rate (%)")
                    plt.ylabel("Number of Posts")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with tab_trend:
                    fig = plt.figure(figsize=(12, 6))
                    
                    # Plot engagement rate trend
                    plt.subplot(2, 1, 1)
                    timestamps = [parser.isoparse(post['timestamp']) for post in posts]
                    plt.plot(timestamps, engagement_rates, marker='o', 
                            linestyle='-', color='#e74c3c', alpha=0.7,
                            label='Engagement Rate (%)')
                    plt.title("Engagement Rate Trend")
                    plt.ylabel("Engagement Rate (%)")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Plot individual metrics
                    plt.subplot(2, 1, 2)
                    plt.plot(timestamps, [post['likes'] for post in posts], 
                            marker='o', label='Likes', alpha=0.7)
                    plt.plot(timestamps, [post['comments'] for post in posts], 
                            marker='s', label='Comments', alpha=0.7)
                    plt.plot(timestamps, [post['saved'] for post in posts], 
                            marker='^', label='Saves', alpha=0.7)
                    plt.xlabel("Post Date")
                    plt.ylabel("Count")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Show top performing posts
                st.subheader("Top Performing Posts")
                top_posts = sorted(posts, key=lambda x: x['engagement_rate'], reverse=True)
                
                num_top_posts = st.slider("Number of top posts to show", 3, 10, 5)
                
                for i, post in enumerate(top_posts[:num_top_posts], 1):
                    with st.expander(
                        f"#{i} - Posted on {parser.isoparse(post['timestamp']).strftime('%Y-%m-%d %H:%M')} "
                        f"(Engagement: {post['engagement_rate']:.1f}%)"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Likes", f"{post['likes']:,d}")
                            st.metric("Comments", f"{post['comments']:,d}")
                        with col2:
                            st.metric("Saves", f"{post['saved']:,d}")
                            st.metric("Reach", f"{post['reach']:,d}")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
        st.info("Please check your Instagram API credentials and try again.")

