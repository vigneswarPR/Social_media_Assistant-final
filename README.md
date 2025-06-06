This project is a Python-based social media assistant designed to provide actionable insights into Instagram account performance. Leveraging the Instagram Graph API, it helps content creators and businesses understand their audience better and optimize their posting strategies, specifically by identifying the best times to post for maximum reach and engagement.

Features:
Instagram Business Account Integration: Connects securely to your Instagram Business or Creator account via a Facebook Page.
Audience Activity Analysis: Fetches and visualizes data on when your followers are most active online, hour by hour.
Post Engagement Analysis: Analyzes the performance of your recent Instagram posts (likes, comments, shares, saves, impressions, reach, video views) to determine average engagement rates per hour.
Intelligent Posting Time Recommendations: Combines follower activity data with historical post engagement to recommend the top hours for posting, providing a weighted score for each.
Visualizations: Generates clear and stylish charts using matplotlib and seaborn to illustrate follower activity and engagement trends throughout the day.
Comprehensive Analytics: Provides a foundation for extending to other analytics like follower demographics and overall account insights (though the primary focus currently is posting times).
How it Works
The core of this assistant involves:

Authentication: Uses a Facebook Page Access Token to securely access the Instagram Graph API.
Data Retrieval:
It first identifies your Instagram Business Account ID associated with your Facebook Page.
It then queries the API for online_followers lifetime data, giving an hourly breakdown of audience presence.
Concurrently, it fetches metadata and insights for your most recent [e.g., 50] posts (media items), gathering metrics like likes, comments, impressions, and reach.
Data Processing & Analysis:
It normalizes both follower online activity and post engagement scores.
These normalized scores are then combined using a weighted average (currently 60% for online followers and 40% for average post engagement) to calculate a "combined score" for each hour of the day.
The hours are then ranked based on this combined score to identify the most opportune posting windows.
Reporting:
Visual charts provide a quick graphical overview.
Detailed text output in the console lists the recommended best posting times, making it easy to schedule content strategically.



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
