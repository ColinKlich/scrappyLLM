# The URL of the raw text file on GitHub
import urllib
import urllib.request
import kagglehub

downloadpath = "./data"

# This dataset contains simple dialogs for chatbot training
downloadpath = kagglehub.dataset_download("mukulhase/simple-chatbot-dataset")
