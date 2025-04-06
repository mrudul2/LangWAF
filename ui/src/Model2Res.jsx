import React, { useEffect, useState } from "react";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend, Title } from "chart.js";
import "./Model2Res.css";

ChartJS.register(ArcElement, Tooltip, Legend, Title);

const Model2Charts = () => {
  const [languageData, setLanguageData] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/model2-language-summary")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch data");
        return res.json();
      })
      .then((json) => setLanguageData(json))
      .catch((err) => setError(err.message));
  }, []);

  const preferredOrder = ["javascript", "sql", "text", "html"];

  const renderPieChart = (lang) => {
    const stats = languageData[lang];
    if (!stats) return null;

    const pieData = {
      labels: ["Safe", "Unsafe"],
      datasets: [
        {
          data: [stats.safe, stats.unsafe],
          backgroundColor: ["#4caf50", "#f44336"],
          borderWidth: 1,
        },
      ],
    };

    return (
      <div key={lang} className="pie-chart">
        <Pie
          data={pieData}
          options={{
            plugins: {
              title: {
                display: true,
                text: lang.toUpperCase(),
                font: { size: 18 },
              },
            },
          }}
        />
      </div>
    );
  };

  if (error) return <div>Error: {error}</div>;
  if (Object.keys(languageData).length === 0) return <div>Loading...</div>;

  const sortedLanguages = Object.entries(languageData)
    .sort(([, a], [, b]) => b.safe + b.unsafe - (a.safe + a.unsafe))
    .slice(0, 6)
    .map(([lang]) => lang);

  return (
    <div className="chart-grid">
      {sortedLanguages.map((lang) => renderPieChart(lang))}
    </div>
  );
};

export default Model2Charts;
