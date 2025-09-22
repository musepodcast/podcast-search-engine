# Muse Podcast Search Engine

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Website:** https://musepodcast.com

**Donations:** [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R6R01HY3FF)
---

## 🚀 About

Muse is a free, AI-powered podcast search engine. Tired of scouring episode show notes for bits of information, we built a platform where **you** can:

- 🔍 **Search any podcast** by keyword (titles, descriptions, or transcript content)  
- 📄 View **full, time-stamped transcripts** (speaker-diarized and formatted)  
- ✍️ Read **AI-generated chapter summaries**
- 🌐 Find podcasts in **17 different languages** natively transcribed or AI translated  
- 🎧 Click through to listen at the exact moment a topic is discussed  
- 🆕 Discover **new channels** or episodes based on your interests  

> **Currently optimized for desktop browsers, tablets, foldables, and smartphones.**  
> Community contribution and native apps are coming soon!

---

## 🌟 Key Features

- **Account Signup & Management**  
  - Create a free account to save searches, bookmark episodes, and track listening history.

- **Transcript Generation**  
  – Ingests RSS feeds from any podcast.  
  – Uses OpenAI Whisper Turbo on an NVIDIA GPU for fast, high-quality transcription.  
  – Diarizes speakers and splits into readable paragraphs with timestamps.

- **Chapter Summaries**  
  - Generates concise summaries of each transcript segment using FalconsAI text-summarization.  

- **Transcript Translations**
  - Takes podcasts and translates them into 17 different languages using nllb-200-distilled-600M translation model.

- **Full-Text Search**  
  – Powered by Elasticsearch for lightning-fast, fuzzy-match queries.  
  – Filters by episode title, description, channel metadata, or transcript content.

- **Flexible Playback**  
  - Jump directly to the moment a keyword appears in the audio player on the website.

---

## 🏗️ Architecture & Tech Stack

| Layer                   | Technology                                  |
|-------------------------|---------------------------------------------|
| **Backend**             | Python, Django                              |
| **Transcription**       | OpenAI Whisper Turbo (GPU-accelerated)      |
| **Summarization**       | FalconsAI / text_summarization              |
| **Translation**         | Facebook / nllb-200-distilled-600M          |
| **Connect Pooling**     | PgBouncer                                   |
| **Database**            | PostgreSQL, MariaDB                         |
| **Search Engine**       | Elasticsearch                               |
| **Frontend**            | Django templates, HTML, CSS, JavaScript     |
| **Deployment**          | Docker, Ubuntu VM, Windows host             |

---
