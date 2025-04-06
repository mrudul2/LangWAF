import { useState } from "react";
import ModelResults from "./ModelResults";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./Navbar";
import Model1Res from "./Model1Res";
import Model2Res from "./Model2Res";

function App() {
  const [count, setCount] = useState(0);

  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/final-model-results" element={<ModelResults />} />
        <Route path="/model1-results" element={<Model1Res />} />
        <Route path="/model2-results" element={<Model2Res />} />
      </Routes>
    </Router>
  );
}

export default App;
