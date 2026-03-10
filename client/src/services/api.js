import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8080"
});

export const queryHerbalAPI = async (query) => {
  const response = await API.post("/query", {
    query: query
  });

  return response.data;
};