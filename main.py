import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from locations import locations_in_trinidad  # Import the list of locations

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Labeling function based on keywords
def label_shooting(text):
    text = text.lower()
    fatal_keywords = ['fatal', 'died', 'killed', 'dead', 'death']
    non_fatal_keywords = ['injured', 'hospitalized', 'wounded', 'shot']

    if any(keyword in text for keyword in fatal_keywords):
        return 'fatal shooting'
    elif any(keyword in text for keyword in non_fatal_keywords):
        return 'non-fatal shooting'
    else:
        return 'non-shooting'

# Location extraction function
def extract_location(text, locations_list):
    text = text.lower()  # Ensure text is lowercase for matching
    for location in locations_list:
        if location in text:
            return location.capitalize()  # Return the location if found
    return 'Unknown'  # Return 'Unknown' if no location is found

# Load your dataset
data = pd.read_csv('crime_csv.csv')

# Ensure the dataset contains the 'article' column
if 'article' not in data.columns:
    raise ValueError("Dataset must contain 'article' column")

# Handle missing or non-string values before applying preprocessing
# Convert non-string or NaN values to empty strings
data['article'] = data['article'].fillna('').astype(str)

# Preprocess the text
data['cleaned_text'] = data['article'].apply(preprocess_text)

# Create the 'label' column based on keyword matching
data['label'] = data['article'].apply(label_shooting)

# Add the 'location' column based on the label
data['location'] = data.apply(
    lambda row: extract_location(row['cleaned_text'], locations_in_trinidad) 
    if row['label'] in ['fatal shooting', 'non-fatal shooting'] 
    else 'N/A', axis=1
)

# Save the dataset with labels and locations to a new CSV file
data.to_csv('labeled_crime_articles_with_locations.csv', index=False)
print("Dataset with labels and locations saved to 'labeled_crime_articles_with_locations.csv'")
