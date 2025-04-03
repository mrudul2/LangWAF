import { useState } from "react";
import ModelResults from "./ModelResults";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./Navbar";

function App() {
  const [count, setCount] = useState(0);

  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/final-model-results" element={<ModelResults />} />
      </Routes>
    </Router>
  );
}

export default App;
