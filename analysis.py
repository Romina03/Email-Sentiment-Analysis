import csv
from flair.data import Sentence
from flair.nn import Classifier

tagger = Classifier.load('sentiment')

# Input and output files
input_file = "emails.txt"  
output_file = "sentiment_analysis.csv"

# Read emails from the input file
with open(input_file, "r", encoding="utf-8") as f:
    emails = f.read().strip().split("\n\n")  

# Process each email
results = []
for email in emails:
    # Extract the first word of the email as stakeholder, removing any colon at the end
    first_word = email.split()[0].rstrip(":") if email else ""  
    
    # Remove the first word from the email content
    email_content = " ".join(email.split()[1:])  # Join all words except the first one
    
    sentence = Sentence(email_content)
    tagger.predict(sentence)
    sentiment = sentence.labels[0].value  # Get sentiment label
    confidence = sentence.labels[0].score  # Get confidence score
    
    # Apply condition: if confidence is less than 0.55, set sentiment to neutral
    if confidence < 0.55:
        sentiment = "NEUTRAL"
    
    results.append([first_word, email_content, sentiment, confidence])

# Write results to the output file
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Stakeholder", "Email", "Sentiment", "Confidence"])
    writer.writerows(results)

print(f"Sentiment analysis complete. Results saved in {output_file}")



