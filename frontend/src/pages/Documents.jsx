import { useEffect, useState } from "react";
import { fetchDocuments } from "../services/api";
import { Link } from "react-router-dom";

export default function Documents() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchDocuments();
        console.log("Fetched documents:", data);
        setDocuments(data); // or data.documents if needed
      } catch (err) {
        console.error("Failed to fetch documents:", err);
      } finally {
        setLoading(false); // <-- make sure this always runs
      }
    };

    load();
  }, []);

  if (loading) return <div className="p-6">Loading documents...</div>;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">📄 Documents</h1>
      {documents.length === 0 ? (
        <p>No documents found.</p>
      ) : (
        <ul className="space-y-4">
          {documents.map((doc) => (
            <li key={doc.document_id} className="p-4 bg-white shadow rounded">
              <Link
                to={`/documents/${doc.document_id}`}
                className="text-blue-600 hover:underline"
              >
                {doc.title || "Untitled Document"}
              </Link>
              <p className="text-sm text-gray-500">
                Uploaded on: {new Date(doc.upload_time).toLocaleString()}
              </p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
