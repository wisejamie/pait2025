// import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
// import Documents from "./pages/Documents";
// import DocumentDetail from "./pages/DocumentDetail";
// import SectionQuizPreview from "./pages/SectionQuizPreview";

// function App() {
//   return (
//     <Router>
//       <Routes>
//         <Route path="/" element={<div className="p-6">🏠 Home Page</div>} />
//         <Route path="/documents" element={<Documents />} />
//         <Route path="/documents/:id" element={<DocumentDetail />} />
//         <Route
//           path="/documents/:docId/sections/:sectionId"
//           element={<SectionQuizPreview />}
//         />
//       </Routes>
//     </Router>
//   );
// }

// export default App;

import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Documents from "./pages/Documents";
import DocumentDetail from "./pages/DocumentDetail";
import SectionQuizPreview from "./pages/SectionQuizPreview";

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
        </Routes>
      </div>
    </Router>
  );
}

export default App;
