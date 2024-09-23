PROJ=$(gcloud config get-value project)
PROJ_NUMBER=$(gcloud projects list \
--filter="$(gcloud config get-value project)" \
--format="value(PROJECT_NUMBER)")
SERVICE_ACCOUNT="cloud-run-manager@scg-l200-genai2.iam.gserviceaccount.com"
LOCATION="us-west1"
IMG_TAG="gcr.io/$PROJ/travel-chatbot"
SECRET_NAME="travelbot-endpoint-id"

gcloud run deploy "travel-chatbot"  \
    --project "$PROJ"    \
    --set-env-vars "PROJECT_ID=$PROJ,LOCATION=$LOCATION,PROJECT_NUMBER=$PROJ_NUMBER"     \
    --set-secrets "ENDPOINT_ID=projects/$PROJ_NUMBER/secrets/$SECRET_NAME:1"     \
    --image "$IMG_TAG:latest"    \
    --platform "managed"  \
    --port 8080           \
    --region "$LOCATION"    \
    --allow-unauthenticated    \
    --service-account "$SERVICE_ACCOUNT"

