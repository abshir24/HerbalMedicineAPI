import { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const handleSearch = async () => {
    try {
      
      // const res = await axios.post("http://localhost:5000/query", {
      //   question: query
      // });

      setResponse("This is a dummy response until open ai api is connected to app. " + query);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="min-h-screen bg-green-50 flex flex-col items-center p-10">

      <h1 className="text-4xl font-bold text-green-800 mb-8">
        Herbal Medicine Assistant
      </h1>

      <div className="bg-white shadow-lg rounded-xl p-6 w-full max-w-xl">

        <input
          type="text"
          placeholder="Ask about herbs, remedies, or nutrition..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full border rounded-lg p-3 mb-4 focus:outline-none focus:ring-2 focus:ring-green-500"
        />

        <button
          onClick={handleSearch}
          className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 transition"
        >
          Ask
        </button>

      </div>

      {response && (
        <div className="mt-8 bg-white shadow-md rounded-xl p-6 max-w-xl w-full">
          <h2 className="text-xl font-semibold mb-2 text-green-700">
            Response
          </h2>

          <p className="text-gray-700 whitespace-pre-wrap">
            {response}
          </p>
        </div>
      )}

    </div>
  );
}

export default App;