#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Auto Data Analyst - Google Cloud Deploy${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}gcloud CLI not found. Please install it first:${NC}"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Step 1: Authenticate
echo -e "${BLUE}Step 1: Authenticating with Google Cloud...${NC}"
gcloud auth login

# Step 2: Set project
echo -e "\n${BLUE}Step 2: Setting up project...${NC}"
read -p "Enter your Google Cloud Project ID: " PROJECT_ID
gcloud config set project $PROJECT_ID

# Step 3: Enable APIs
echo -e "\n${BLUE}Step 3: Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Step 4: Build image
echo -e "\n${BLUE}Step 4: Building Docker image...${NC}"
gcloud builds submit --tag gcr.io/$PROJECT_ID/auto-data-analyst

# Step 5: Deploy to Cloud Run
echo -e "\n${BLUE}Step 5: Deploying to Cloud Run...${NC}"
gcloud run deploy auto-data-analyst \
  --image gcr.io/$PROJECT_ID/auto-data-analyst \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --no-allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=$GROQ_API_KEY

echo -e "\n${GREEN}✓ Deployment complete!${NC}"
echo -e "${GREEN}Your app is now live on Google Cloud Run${NC}\n"
