WEEK="5"
PROJ=$(gcloud config get-value project)
gcloud builds submit --tag "gcr.io/$PROJ/travel-chatbot-$WEEK"
