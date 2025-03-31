import React, { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./ModelResults.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ModelResults = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/model-results")
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load results");
        return response.json();
      })
      .then((json) => {
        // Ensure all values are in percentage format
        json.accuracy = json.accuracy > 1 ? json.accuracy : json.accuracy * 100;
        json.precision =
          json.precision > 1 ? json.precision : json.precision * 100;
        json.recall = json.recall > 1 ? json.recall : json.recall * 100;
        json.f1_score = json.f1_score > 1 ? json.f1_score : json.f1_score * 100;

        setData(json);
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="error-message">{error}</div>;
  if (!data) return <div className="loading-message">Loading...</div>;

  // Data for Bar Chart
  const metricsData = {
    labels: ["Accuracy", "Precision", "Recall", "F1 Score"],
    datasets: [
      {
        label: "Performance Metrics",
        data: [data.accuracy, data.precision, data.recall, data.f1_score],
        backgroundColor: ["#e6e6fa", "#8a2be2", "#9370db", "#ba55d3"],
        borderRadius: 5,
      },
    ],
  };

  const metricsOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: "Model Performance",
        color: "#ffffff",
        font: { size: 16 },
      },
      tooltip: {
        bodyFont: { size: 14 },
        titleFont: { size: 14 },
      },
    },
    scales: {
      x: {
        ticks: { color: "#ffffff", font: { size: 14 } },
      },
      y: {
        beginAtZero: true,
        max: 100,
        ticks: { color: "#ffffff", font: { size: 14 } },
      },
    },
  };

  return (
    <div className="model-results-container">
      <div className="model-results-grid">
        {/* Left Card: Model Metrics */}
        <div className="model-card">
          <h2>Model Performance</h2>
          <Bar data={metricsData} options={metricsOptions} />
        </div>

        {/* Right Card: Confusion Matrix */}
        <div className="model-card">
          <h2>Confusion Matrix</h2>
          <table className="confusion-matrix-table">
            <thead>
              <tr>
                <th></th>
                <th>Predicted 0</th>
                <th>Predicted 1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <strong>Actual 0</strong>
                </td>
                <td className="matrix-cell">{data.confusion_matrix[0][0]}</td>
                <td className="matrix-cell">{data.confusion_matrix[0][1]}</td>
              </tr>
              <tr>
                <td>
                  <strong>Actual 1</strong>
                </td>
                <td className="matrix-cell">{data.confusion_matrix[1][0]}</td>
                <td className="matrix-cell">{data.confusion_matrix[1][1]}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ModelResults;
