# Personal AI Tutor (PAIT)

An AI-powered tutor that helps users **understand difficult documents** through summaries, guided conversations, and adaptive quizzes.  
The app combines the simplicity of the original 2024 PAIT conversational model with the more polished full-stack web app developed in 2025.

---

## ✨ Core Features

- **Upload PDFs or text** → choose between two processing modes:
  - **Basic mode (recommended):**  
    Fast, reliable text extraction for any PDF. Produces conversational summaries and supports guided Q&A (like the 2024 version).
  - **Advanced mode:**  
    Extracts cleaned, displayable Markdown from simple PDFs. Supports section navigation, readable document text, and quizzes.
- **GPT-based section detection** (Basic = raw anchor text, Advanced = structured Markdown).
- **Tutor interactions:**  
  - Ask free-form questions  
  - Guided open-ended question testing (3-question pipeline)  
  - Section exploration (“zoom in / out”)  
- **Quiz generation** from document or sections, with performance feedback.
- **Caching** for summaries and micro-summaries to speed up repeat visits.

---

## 🛠 Tech Stack

- **Backend:** FastAPI (Python 3.9)  
- **Frontend:** React (Vite, TailwindCSS)  
- **LLM:** GPT-4.1-nano (via OpenAI API)  
- **Storage:** In-memory during dev; SQLite planned for persistence  
- **Deployment:** Vercel (frontend) + Render (backend) for MVP

---

## 📝 Notes

- When first using the app, please allow a few minutes for the backend to load. We use a free Render instance that spins down after inactivity.
- **Advanced mode limitation**: it does not reliably extract correctly formatted text from PDFs that use complex layouts (e.g., multi-column, heavy figures).
