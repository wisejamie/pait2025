# Tutor App: Adaptive Quiz Generator for Complex Texts

This app turns difficult documents into adaptive, interactive quizzes designed to help users understand complex material through guided question-based learning.

## 🚀 MVP Features

- Upload or paste a document
- GPT-based section detection
- Automatic multiple-choice question generation
- Adaptive quiz experience with feedback
- Summary of performance and weak topics

## 🛠 Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** Next.js (React)
- **LLM:** GPT-4 via OpenAI API
- **Storage:** To be determined (likely SQLite or Supabase for MVP)

## 📁 Project Structure

```
.
├── LICENSE                 # Project license
├── README.md               # Project overview and setup instructions
├── backend/                # Backend code (FastAPI)
│   ├── app/                # Main FastAPI application logic
│   └── tests/              # Unit and integration tests for backend
├── frontend/               # Frontend code (Next.js)
│   ├── components/         # React UI components
│   └── pages/              # Next.js page routes
├── docs/                   # Internal documentation (timeline, planning notes) 
```

## ✅ Getting Started

1. Clone the repo  
   `git clone https://github.com/your-username/tutor-app.git`

2. Set up a Python virtual environment  
   `python -m venv venv && source venv/bin/activate`

3. Install backend dependencies  
   `pip install -r backend/requirements.txt` *(once backend is scaffolded)*

## 📌 Goals

- Make scientific and academic texts more accessible
- Support self-paced learning with structured guidance
- Launch MVP by early July 2025

## 📝 Notes

- This README will be updated as features are added.
- Development begins June 9, 2025.
- Frontend and deployment setup to come later.
