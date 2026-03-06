import React, { useState } from "react";

function QueryBox({ onSubmit }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(query);
    setQuery("");
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask about herbal medicine..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button type="submit">Ask</button>
    </form>
  );
}

export default QueryBox;