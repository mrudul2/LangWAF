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
import "./Model1Res.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Model1Res = () => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/model1-comparison-summary")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch model1 results");
        return res.json();
      })
      .then((json) => {
        setData(json);
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <div className="error-message">{error}</div>;
  if (!data) return <div className="loading-message">Loading...</div>;

  const metricLabels = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1_score: "F1 Score",
  };

  const formatModelName = (key) => {
    const map = {
      svm: "SVM",
      svm_prob: "SVM Prob",
      logistic_regression: "Logistic Regression",
      random_forest: "Random Forest",
    };
    return map[key] || key;
  };

  const chartOptions = (title) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: title,
        color: "#ffffff",
        font: { size: 20, weight: "bold" },
      },
    },
    scales: {
      x: {
        ticks: {
          color: "#ffffff",
          font: { size: 14, weight: "bold" },
        },
        grid: {
          color: "rgba(255, 255, 255, 0.1)",
        },
      },
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          color: "#ffffff",
          font: { size: 14, weight: "bold" },
          stepSize: 10,
          callback: (value) => value + "%",
        },
        grid: {
          color: "rgba(255, 255, 255, 0.1)",
        },
      },
    },
  });

  const metricColors = {
    accuracy: "#6495ed",
    precision: "#ff69b4",
    recall: "#3cb371",
    f1_score: "#ffa500",
  };

  const createChartDataForModel = (modelKey) => {
    const modelMetrics = data[modelKey];
    return {
      labels: Object.keys(metricLabels).map((metric) => metricLabels[metric]),
      datasets: [
        {
          label: formatModelName(modelKey),
          data: Object.keys(metricLabels).map((metric) => {
            const val = modelMetrics?.[metric];
            return val > 1 ? val : val * 100;
          }),
          backgroundColor: Object.keys(metricLabels).map(
            (metric) => metricColors[metric]
          ),
          borderRadius: 5,
        },
      ],
    };
  };

  const modelKeys = ["svm", "svm_prob", "logistic_regression", "random_forest"];

  return (
    <div className="model-results-container">
      <div className="model-results-grid">
        {modelKeys.map((model) => (
          <div className="model-card" key={model}>
            <Bar
              data={createChartDataForModel(model)}
              options={chartOptions(`${formatModelName(model)} Metrics`)}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default Model1Res;
