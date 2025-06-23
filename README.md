# Muse Podcast Search Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Website:** https://musepodcast.com

---

## 🚀 About

Muse is a free, GPU-powered podcast search engine. Tired of scouring episode show notes for bits of information, we built a platform where **you** can:

- 🔍 **Search any podcast** by keyword (titles, descriptions, or transcript content)  
- 📄 View **full, time-stamped transcripts** (speaker-diarized and formatted)  
- ✍️ Read **AI-generated chapter summaries**  
- 🎧 Click through to listen at the exact moment a topic is discussed  
- 🆕 Discover **new channels** or episodes based on your interests  

> **Currently optimized for desktop browsers.**  
> Mobile-friendly support and native apps are coming soon!

---

## 🌟 Key Features

- **Account Signup & Management**  
  Create a free account to save searches, bookmark episodes, and track listening history.

- **Transcript Generation**  
  – Ingests RSS feeds from any podcast.  
  – Uses OpenAI Whisper Turbo on an NVIDIA GPU for fast, high-quality transcription.  
  – Diarizes speakers and splits into readable paragraphs with timestamps.

- **Chapter Summaries**  
  Generates concise summaries of each transcript segment using FalconsAI text-summarization.  

- **Full-Text Search**  
  – Powered by Elasticsearch for lightning-fast, fuzzy-match queries.  
  – Filters by episode title, description, channel metadata, or transcript content.

- **Flexible Playback**  
  Jump directly to the moment a keyword appears in the audio player on the website.

---

## 🏗️ Architecture & Tech Stack

| Layer               | Technology                                  |
|---------------------|---------------------------------------------|
| **Backend**         | Python, Django                              |
| **Transcription**   | OpenAI Whisper Turbo (GPU-accelerated)      |
| **Summarization**   | FalconsAI / text_summarization              |
| **Database**        | PostgreSQL                                  |
| **Search Engine**   | Elasticsearch                               |
| **Frontend**        | Django templates, HTML, CSS, JavaScript     |
| **Deployment**      | Docker, NVIDIA GPU passthrough, Ubuntu VM   |

---
