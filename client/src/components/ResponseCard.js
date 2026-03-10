import React from "react";

function ResponseCard({ response }) {
  if (!response) return null;

  return (
    <div>
      <h2>Answer</h2>
      <p>{response.main_answer}</p>

      <h3>Additional Notes</h3>
      <p>{response.additional_notes}</p>
    </div>
  );
}

export default ResponseCard;