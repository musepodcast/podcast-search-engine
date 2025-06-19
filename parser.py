import feedparser

feed_url = 'https://open.spotify.com/episode/0uIMGfE0VMkzWuWJDzfkAb?si=424728c264aa42d2'

# Iterate over each entry (episode) in the feed
for entry in feed.entries:
    # Print episode title and publication date
    print(f"Title: {entry.title}")
    print(f"Published: {entry.published}")

    # Extract the audio file URL from the 'enclosures' field
    if 'enclosures' in entry and len(entry.enclosures) > 0:
        audio_url = entry.enclosures[0].href
        print(f"Audio URL: {audio_url}\n")
    else:
        print("No audio URL found for this episode.\n")