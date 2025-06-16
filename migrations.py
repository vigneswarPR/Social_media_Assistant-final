from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Boolean, JSON, Text
from config import Config
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def run_migrations():
    """Run database migrations"""
    try:
        # Create engine
        engine = create_engine('sqlite:///social_media_assistant.db')
        
        # Create MetaData instance
        metadata = MetaData()
        
        # Define the table
        scheduled_posts = Table(
            'scheduled_posts', metadata,
            Column('id', String, primary_key=True),
            Column('media_urls', JSON, nullable=False),
            Column('media_type', String, nullable=False),
            Column('caption', Text, nullable=False),
            Column('scheduled_time', DateTime(timezone=True), nullable=False),
            Column('username', String, nullable=False),
            Column('status', String, default='scheduled'),
            Column('celery_task_id', String, nullable=True),
            Column('posting_attempt_at', DateTime(timezone=True), nullable=True),
            Column('error_message', Text, nullable=True),
            Column('media_id', String, nullable=True),
            Column('created_at', DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
            Column('updated_at', DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
            Column('cloudinary_public_ids', JSON, nullable=True),
            Column('is_facebook', Boolean, default=False),
            Column('platforms', String, default='Instagram')
        )
        
        # Create table if it doesn't exist
        metadata.create_all(engine)
        logger.info("Ensured scheduled_posts table exists")
        
        # Add platforms column if it doesn't exist
        with engine.connect() as conn:
            # Check if platforms column exists
            result = conn.execute(text("PRAGMA table_info(scheduled_posts)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'platforms' not in columns:
                logger.info("Adding 'platforms' column to scheduled_posts table")
                conn.execute(text("""
                    ALTER TABLE scheduled_posts 
                    ADD COLUMN platforms TEXT DEFAULT 'Instagram'
                """))
                conn.commit()
                logger.info("Successfully added 'platforms' column")
                
                # Update existing records based on is_facebook field
                logger.info("Updating existing records with platform information")
                conn.execute(text("""
                    UPDATE scheduled_posts 
                    SET platforms = CASE 
                        WHEN is_facebook = 1 THEN 'Facebook'
                        ELSE 'Instagram'
                    END
                """))
                conn.commit()
                logger.info("Successfully updated existing records")
            
        logger.info("Migrations completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        return False

if __name__ == "__main__":
    run_migrations() 