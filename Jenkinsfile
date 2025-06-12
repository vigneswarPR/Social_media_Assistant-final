// Jenkinsfile
// This file defines the CI/CD pipeline for your social media assistant application.

pipeline {
    // Defines where the pipeline will run. 'agent any' means Jenkins can run it on any available agent.
    // For more advanced setups, you might specify a Docker agent or a specific label.
    agent any

    // Define global environment variables for the pipeline.
    // Sensitive information (like API keys) will be injected securely via 'withCredentials' later.
    // This section can be used for non-sensitive, static variables.
    environment {
        // Internal Redis URL for inter-container communication within Docker Compose network.
        // This should generally remain 'redis://redis:6379/0' within the Docker Compose environment.
        REDIS_URL_FOR_CONTAINERS = 'redis://redis:6379/0'
    }

    // Define the stages of your pipeline.
    stages {
        // Add a cleanup stage at the very beginning to ensure a clean Docker environment
        // before starting any new services. This helps prevent conflicts from previous
        // failed or incomplete runs.
        stage('Cleanup Previous Runs') {
            steps {
                script {
                    echo 'Cleaning up any existing Docker Compose services...'
                    // Force remove the named Redis container if it exists from a previous run (even if stopped)
                    // This is added to prevent "Conflict" errors during `docker compose run`
                    sh 'docker rm -f social_media_assistant_redis || true'
                    // Stop and remove all services, networks, and volumes from previous runs.
                    // '|| true' ensures the pipeline doesn't fail if there's nothing to remove.
                    sh 'docker compose down --volumes --remove-orphans || true'
                    echo 'Cleanup complete.'
                }
            }
        }

        // The "Declarative: Checkout SCM" stage handled by Jenkins itself at the start
        // already fetches your code. This 'Checkout Source Code' stage is redundant and
        // was causing issues by trying to fetch 'master' branch. It's now removed.
        // stage('Checkout Source Code') {
        //     steps {
        //         echo "Checking out source code from Git repository..."
        //         git 'https://github.com/vigneswarPR/Social_media_Assistant-final.git'
        //         echo "Source code checked out."
        //     }
        // }

        stage('Build Docker Images') {
            steps {
                script {
                    echo "Building Docker images for all services defined in docker-compose.yml..."
                    // The 'docker compose build' command uses your Dockerfile to build images
                    // for all services (streamlit, celery_worker, celery_beat).
                    sh 'docker compose build'
                    echo "Docker images built successfully."
                }
            }
        }

        // The 'Run Tests' stage has been commented out as requested.
        /*
        stage('Run Tests') {
            steps {
                script {
                    echo "Running tests within the 'streamlit' service container..."
                    // Securely inject credentials required for tests (e.g., API keys, user credentials).
                    // This ensures the Streamlit container has access to necessary environment variables.
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
                        // This command runs 'pytest' inside the 'streamlit' container.
                        // '--rm' ensures the temporary container is removed after the tests.
                        // Environment variables from 'withCredentials' and global 'environment' are passed with -e.
                        // Using triple single quotes and ${VARIABLE} for robust parsing.
                        sh '''
                            docker compose run --rm \\
                                -e GEMINI_API_KEY=${GEMINI_API_KEY} \\
                                -e INSTAGRAM_ACCESS_TOKEN=${INSTAGRAM_ACCESS_TOKEN} \\
                                -e INSTAGRAM_BUSINESS_ACCOUNT_ID=${INSTAGRAM_BUSINESS_ACCOUNT_ID} \\
                                -e FACEBOOK_PAGE_ID=${FACEBOOK_PAGE_ID} \\
                                -e FACEBOOK_APP_ID=${FACEBOOK_APP_ID} \\
                                -e FACEBOOK_APP_SECRET=${FACEBOOK_APP_SECRET} \\
                                -e CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME} \\
                                -e CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY} \\
                                -e CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET} \\
                                -e REDIS_URL=${REDIS_URL_FOR_CONTAINERS} \\
                                streamlit /usr/local/bin/python -m pytest
                        '''
                    }
                    echo "Tests completed."
                }
            }
        }
        */

        stage('Deploy Local (with Secrets)') {
            steps {
                script {
                    echo "Starting Docker Compose services with injected credentials..."
                    // The 'withCredentials' block securely fetches secrets from Jenkins Credentials.
                    // The 'variable' name (e.g., 'GEMINI_API_KEY') MUST match the environment variable name
                    // your Python application (via os.getenv()) and Docker Compose's '-e' flag expect.
                    // The 'credentialsId' MUST match the ID you gave the secret in Jenkins (e.g., 'gemini-api-key').
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
                        // FIX: Removed the -e flags for `docker compose up`.
                        // Environment variables passed via `withCredentials` are already available
                        // in the shell environment and will be picked up by Docker Compose,
                        // especially since your `docker-compose.yml` uses `env_file: .env`.
                        sh '''
                            docker compose up -d
                        '''
                        echo "Services started. Streamlit app should be available at http://localhost:8501 on the Jenkins host."
                        // Optional: Add a short delay to allow services to fully initialize before finishing the stage.
                        sh 'sleep 10'
                    }
                }
            }
        }

        // Optional: Add more stages here as needed for your CI/CD process, e.g.:
        // stage('Push to Docker Registry') { ... }
        // stage('Deploy to Staging/Production') { ... }
    }

    // Post-build actions: These steps run after all stages are complete, regardless of success or failure.
    post {
        always {
            echo "Stopping and removing Docker Compose services after pipeline run..."
            // This command stops containers and removes networks, volumes, and images created by 'docker compose up'.
            // This ensures a clean slate for the next build.
            sh 'docker compose down'
        }
        success {
            echo 'Pipeline finished successfully!'
        }
        failure {
            echo 'Pipeline failed! Please check the console output for error details.'
        }
        unstable {
            echo 'Pipeline finished with some issues (e.g., skipped tests).'
        }
        changed {
            // Runs only if the build status has changed from the previous build
            echo 'Pipeline status changed.'
        }
    }
}