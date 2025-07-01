import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Documents from "./pages/Documents";
import DocumentDetail from "./pages/DocumentDetail";
import SectionQuizPreview from "./pages/SectionQuizPreview";
import QuizView from "./pages/QuizView";
import QuizSummary from "./pages/QuizSummary";

function App() {
  return (
    <Router>
      <div>
        <nav className="bg-gray-800 text-white p-4 mb-6">
          <div className="container mx-auto">
            <Link to="/" className="font-semibold text-lg hover:underline">
              🧠 AI Tutor
            </Link>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Documents />} />
          <Route path="/documents/:id" element={<DocumentDetail />} />
          <Route
            path="/documents/:docId/sections/:sectionId"
            element={<SectionQuizPreview />}
          />
          <Route path="/quiz/:sessionId" element={<QuizView />} />
          <Route path="/quiz/:sessionId/summary" element={<QuizSummary />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
