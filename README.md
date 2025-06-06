To run this application locally, you need to have **Redis** installed and running, and then start **Celery Worker**, **Celery Beat**, and the **Streamlit app** in separate terminal windows.

**1. Install Redis:**
   - On macOS: `brew install redis`
   - On Linux: `sudo apt-get install redis-server`
   - On Windows: Follow instructions [here](https://redis.io/docs/getting-started/installation/install-redis-on-windows/)

**2. Start Redis Server:**
   - Open a terminal and run: `redis-server`

**3. Install Python dependencies:**
   - `pip install -r requirements.txt`

**4. Create a `.env` file:**
   - Create a file named `.env` in the root directory of your project.
   - Add your API keys and Instagram credentials (replace with your actual values):
     ```
     GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
     INSTAGRAM_USERNAME='your_instagram_username'
     INSTAGRAM_PASSWORD='your_instagram_password'
     INSTAGRAM_ACCESS_TOKEN='YOUR_INSTAGRAM_GRAPH_API_ACCESS_TOKEN'
     INSTAGRAM_BUSINESS_ACCOUNT_ID='YOUR_INSTAGRAM_BUSINESS_ACCOUNT_ID'
     CLOUDINARY_CLOUD_NAME='your_cloudinary_cloud_name'
     CLOUDINARY_API_KEY='your_cloudinary_api_key'
     CLOUDINARY_API_SECRET='your_cloudinary_api_secret'
     ```
   - **Important:** Ensure your Instagram account is a Professional Account (Creator or Business) and linked to a Facebook Page to use the Graph API. Obtain `INSTAGRAM_ACCESS_TOKEN` and `INSTAGRAM_BUSINESS_ACCOUNT_ID` via the [Facebook Graph API Explorer](https://developers.facebook.com/tools/explorer/).

**5. Start Celery Worker:**
   - Open a **new** terminal and run:
     `celery -A tasks worker --loglevel=info`

**6. Start Celery Beat (Scheduler):**
   - Open another **new** terminal and run:
     `celery -A tasks beat --loglevel=info --schedule=/tmp/celerybeat-schedule`
     (On Windows, `/tmp/celerybeat-schedule` might need to be a valid path like `C:\temp\celerybeat-schedule`)

**7. Start Streamlit App:**
   - Open a final **new** terminal (or use the one where you installed dependencies) and run:
     `streamlit run streamlit_app.py`

This will open the application in your web browser. Ensure all four terminals are running concurrently for full functionality.
"""
