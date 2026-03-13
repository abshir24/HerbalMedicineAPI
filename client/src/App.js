import React, { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query) return;

    setLoading(true);

    try {
      // const res = await axios.post("http://localhost:5000/query", {
      //   question: query,
      // });

      setResponse("This is a dummy response will be replaced once api is active." + query);
    } catch (error) {
      console.error(error);
      setResponse("Something went wrong.");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-green-50 flex flex-col items-center">

      {/* Header */}
      <header className="w-full bg-green-700 text-white py-6 shadow-md">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-3xl font-bold">
            🌿 Herbal Medicine Assistant
          </h1>
          <p className="text-green-100 mt-2">
            Ask about herbs, nutrition, and natural remedies
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex flex-col items-center w-full max-w-3xl p-6">

        {/* Search Card */}
        <div className="w-full bg-white shadow-lg rounded-xl p-6 mb-6">

          <input
            type="text"
            placeholder="Example: What herbs help with digestion?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full border border-gray-300 rounded-lg p-3 mb-4 focus:outline-none focus:ring-2 focus:ring-green-500"
          />

          <button
            onClick={handleSearch}
            className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition"
          >
            {loading ? "Thinking..." : "Ask the Herbal Assistant"}
          </button>

        </div>

        {/* Response Panel */}
        {response && (
          <div className="w-full bg-white shadow-lg rounded-xl p-6">
            <h2 className="text-xl font-semibold text-green-700 mb-3">
              Response
            </h2>

            <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
              {response}
            </p>
          </div>
        )}

      </main>

    </div>
  );
}

export default App;