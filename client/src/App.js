import React, { useState } from "react";
import QueryBox from "./components/QueryBox";
import ResponseCard from "./components/ResponseCard";
import { queryHerbalAPI } from "./services/api";

function App() {
  const [response, setResponse] = useState(null);

  const handleQuery = async (query) => {
    try {
      const data = await queryHerbalAPI(query);
      setResponse(data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Herbal Medicine AI</h1>

      <QueryBox onSubmit={handleQuery} />

      <ResponseCard response={response} />
    </div>
  );
}

export default App;