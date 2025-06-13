// Jenkinsfile
// This file defines the CI/CD pipeline for your social media assistant application.

pipeline {
    agent any

    environment {
        // Internal Redis URL for inter-container communication within Docker Compose network.
        REDIS_URL_FOR_CONTAINERS = 'redis://redis:6379/0'
    }

    stages {
        stage('Cleanup Previous Runs') {
            steps {
                script {
                    echo 'Cleaning up any existing Docker Compose services...'
                    sh 'docker rm -f social_media_assistant_redis || true'
                    sh 'docker rm -f social_media_assistant_streamlit || true'
                    sh 'docker rm -f social_media_assistant_celery_worker || true'
                    sh 'docker rm -f social_media_assistant_celery_beat || true'
                    sh 'docker compose down --volumes --remove-orphans || true'
                    echo 'Cleanup complete.'
                }
            }
        }

        stage('Build Docker Images') {
            steps {
                script {
                    echo "Building Docker images for all services defined in docker-compose.yml..."
                    // Note: You might still see warnings here if your Dockerfile or docker-compose.yml build section
                    // tries to use these variables during build, but they are generally harmless for `docker compose build`.
                    sh 'docker compose build'
                    echo "Docker images built successfully."
                }
            }
        }

        stage('Deploy Local (with Secrets)') {
            steps {
                script {
                    echo "Starting Docker Compose services with injected credentials..."
                    withCredentials([
                        string(credentialsId: 'gemini-api-key', variable: 'GEMINI_API_KEY'),
                        string(credentialsId: 'instagram-access-token', variable: 'INSTAGRAM_ACCESS_TOKEN'),
                        string(credentialsId: 'instagram-business-account-id', variable: 'INSTAGRAM_BUSINESS_ACCOUNT_ID'),
                        string(credentialsId: 'facebook-page-id', variable: 'FACEBOOK_PAGE_ID'),
                        string(credentialsId: 'facebook-app-id', variable: 'FACEBOOK_APP_ID'),
                        string(credentialsId: 'facebook-app-secret', variable: 'FACEBOOK_APP_SECRET'),
                        string(credentialsId: 'cloudinary-cloud-name', variable: 'CLOUDINARY_CLOUD_NAME'),
                        string(credentialsId: 'cloudinary-api-key', variable: 'CLOUDINARY_API_KEY'),
                        string(credentialsId: 'cloudinary-api-secret', variable: 'CLOUDINARY_API_SECRET')
                    ]) {
                        // --- START OF CRITICAL FIX ---
                        // Dynamically create a .env file with the secrets
                        // This file will be read by `docker compose up -d`
                        sh '''
                            echo "GEMINI_API_KEY=${GEMINI_API_KEY}" > .env
                            echo "INSTAGRAM_ACCESS_TOKEN=${INSTAGRAM_ACCESS_TOKEN}" >> .env
                            echo "INSTAGRAM_BUSINESS_ACCOUNT_ID=${INSTAGRAM_BUSINESS_ACCOUNT_ID}" >> .env
                            echo "FACEBOOK_PAGE_ID=${FACEBOOK_PAGE_ID}" >> .env
                            echo "FACEBOOK_APP_ID=${FACEBOOK_APP_ID}" >> .env
                            echo "FACEBOOK_APP_SECRET=${FACEBOOK_APP_SECRET}" >> .env
                            echo "CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME}" >> .env
                            echo "CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY}" >> .env
                            echo "CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET}" >> .env
                            echo "REDIS_URL=${REDIS_URL_FOR_CONTAINERS}" >> .env

                            echo "DEBUG: .env file created in workspace (for debugging, will be removed later)"
                            cat .env # For debugging purposes, remove in production
                            echo "--- END OF .env debug ---"

                            docker compose up -d # Docker Compose will automatically read the .env file

                            # Clean up the .env file immediately for security
                            rm .env
                        '''
                        // --- END OF CRITICAL FIX ---

                        echo "Services started. Streamlit app should be available at http://localhost:8501 on the Jenkins host."
                        sh 'sleep 10'
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'No post-build cleanup for always block (services kept running).'
        }
        success {
            echo 'Pipeline finished successfully! Services should remain running.'
        }
        failure {
            echo 'Pipeline failed! Please check the console output for error details.'
        }
        unstable {
            echo 'Pipeline finished with some issues (e.g., skipped tests).'
        }
        changed {
            echo 'Pipeline status changed.'
        }
    }
}