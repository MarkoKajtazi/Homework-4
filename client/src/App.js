import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";

// Register Chart.js components and plugins
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    zoomPlugin
);

const App = () => {
    const [companies, setCompanies] = useState([]);
    const [selectedCompany, setSelectedCompany] = useState("");
    const [transactions, setTransactions] = useState([]);
    const [filteredTransactions, setFilteredTransactions] = useState([]);
    // State for predictions; if no prediction info is returned for an issuer, this stays empty.
    const [predictions, setPredictions] = useState([]);
    const [fromDate, setFromDate] = useState("");
    const [toDate, setToDate] = useState("");
    const [visibleDatasets, setVisibleDatasets] = useState({
        sma20: true,
        sma50: true,
        ema20: true,
        ema50: true,
    });

    useEffect(() => {
        fetch("http://localhost:8080/api/companies")
            .then((response) => response.json())
            .then((data) => setCompanies(data));
    }, []);

    const handleFetchTransactions = () => {
        if (selectedCompany) {
            // Clear previous predictions whenever a new fetch is initiated.
            setPredictions([]);

            // First, fetch the transactions for the selected company.
            fetch(`http://localhost:8080/api/transactions/${selectedCompany}`)
                .then((response) => response.json())
                .then((data) => {
                    const sortedTransactions = data.sort(
                        (a, b) => new Date(a.date) - new Date(b.date)
                    );

                    const filtered =
                        fromDate && toDate
                            ? sortedTransactions.filter((transaction) => {
                                const transactionDate = new Date(transaction.date);
                                return (
                                    transactionDate >= new Date(fromDate) &&
                                    transactionDate <= new Date(toDate)
                                );
                            })
                            : sortedTransactions;

                    setFilteredTransactions(filtered);
                    setTransactions(sortedTransactions);

                    // Then, fetch predictions for the selected company.
                    fetch(`http://localhost:8080/api/predict/${selectedCompany}`)
                        .then((response) => response.json())
                        .then((data) => {
                            // Expected format: ["6512.61056907475","6442.734447076917"]
                            // If no prediction info is available for this issuer, data should be an empty array.
                            setPredictions(data);
                        })
                        .catch((error) =>
                            console.error("Error fetching predictions:", error)
                        );
                });
        } else {
            alert("Please select a company.");
        }
    };

    const handleReset = () => {
        setFromDate("");
        setToDate("");
        setFilteredTransactions(transactions);
        // Also reset predictions.
        setPredictions([]);
    };

    const toggleDataset = (dataset) => {
        setVisibleDatasets((prev) => ({
            ...prev,
            [dataset]: !prev[dataset],
        }));
    };

    // Helper function to format dates.
    const formatDate = (dateInput) => {
        const d = new Date(dateInput);
        const day = d.getDate().toString().padStart(2, "0");
        const month = (d.getMonth() + 1).toString().padStart(2, "0");
        const year = d.getFullYear();
        return `${day}.${month}.${year}`;
    };

    // Build labels for the chart.
    const transactionLabels = filteredTransactions.map((transaction) =>
        formatDate(transaction.date)
    );

    // Only add prediction labels if prediction data exists.
    let predictedLabels = [];
    if (filteredTransactions.length > 0 && predictions.length > 0) {
        const lastDate = new Date(
            filteredTransactions[filteredTransactions.length - 1].date
        );
        predictedLabels = predictions.map((_, index) => {
            const d = new Date(lastDate);
            d.setDate(d.getDate() + index + 1);
            return formatDate(d);
        });
    }
    const labels = [...transactionLabels, ...predictedLabels];

    // Function to pad historical data with nulls for the prediction period.
    const padData = (dataArray) => [
        ...dataArray,
        ...new Array(predictedLabels.length).fill(null),
    ];

    // Determine prediction line color.
    // If prediction info exists and the first prediction is higher than the last historical price, use green.
    let predColor = "red";
    if (predictions.length > 0 && filteredTransactions.length > 0) {
        const lastPrice = parseFloat(
            filteredTransactions[filteredTransactions.length - 1].lastPrice
        );
        const firstPrediction = parseFloat(predictions[0]);
        if (firstPrediction > lastPrice) {
            predColor = "green";
        }
    }

    const chartData = {
        labels,
        datasets: [
            {
                label: "Transaction Price",
                data: padData(
                    filteredTransactions.map((transaction) =>
                        parseFloat(transaction.lastPrice)
                    )
                ),
                borderColor: "rgba(75,192,192,1)",
                backgroundColor: "rgba(75,192,192,0.2)",
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: "rgba(75,192,192,1)",
                fill: false,
            },
            visibleDatasets.sma20 && {
                label: "SMA 20",
                data: padData(
                    filteredTransactions.map((transaction) =>
                        parseFloat(transaction.sma20)
                    )
                ),
                borderColor: "rgba(255,99,132,1)",
                backgroundColor: "rgba(255,99,132,0.2)",
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: "rgba(255,99,132,1)",
                fill: false,
            },
            visibleDatasets.sma50 && {
                label: "SMA 50",
                data: padData(
                    filteredTransactions.map((transaction) =>
                        parseFloat(transaction.sma50)
                    )
                ),
                borderColor: "rgba(153,102,255,1)",
                backgroundColor: "rgba(153,102,255,0.2)",
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: "rgba(153,102,255,1)",
                fill: false,
            },
            visibleDatasets.ema20 && {
                label: "EMA 20",
                data: padData(
                    filteredTransactions.map((transaction) =>
                        parseFloat(transaction.ema20)
                    )
                ),
                borderColor: "rgba(54,162,235,1)",
                backgroundColor: "rgba(54,162,235,0.2)",
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: "rgba(54,162,235,1)",
                fill: false,
            },
            visibleDatasets.ema50 && {
                label: "EMA 50",
                data: padData(
                    filteredTransactions.map((transaction) =>
                        parseFloat(transaction.ema50)
                    )
                ),
                borderColor: "rgba(255,159,64,1)",
                backgroundColor: "rgba(255,159,64,0.2)",
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: "rgba(255,159,64,1)",
                fill: false,
            },
            // Add the prediction dataset only if prediction data exists.
            predictions.length > 0 && {
                label: "Predicted Price",
                data: [
                    ...new Array(filteredTransactions.length).fill(null),
                    ...predictions.map((p) => parseFloat(p)),
                ],
                borderColor: predColor,
                backgroundColor: predColor,
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: predColor,
                fill: false,
            },
        ].filter(Boolean),
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: "top",
            },
            title: {
                display: true,
                text: `Transactions for ${selectedCompany}`,
                color: "white",
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: "x",
                },
                zoom: {
                    wheel: {
                        enabled: true,
                    },
                    pinch: {
                        enabled: true,
                    },
                    mode: "x",
                },
            },
        },
        elements: {
            line: {
                tension: 0.2,
            },
        },
        scales: {
            x: {
                grid: {
                    color: "rgba(255,255,255,0.1)",
                },
                ticks: {
                    color: "white",
                },
            },
            y: {
                grid: {
                    color: "rgba(255,255,255,0.1)",
                },
                ticks: {
                    color: "white",
                },
            },
        },
    };

    return (
        <div style={{ display: "flex", height: "100vh", backgroundColor: "#333" }}>
            <div
                style={{
                    width: "300px",
                    backgroundColor: "white",
                    padding: "20px",
                    boxSizing: "border-box",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "flex-start",
                    alignItems: "flex-start",
                    height: "100vh",
                }}
            >
                <h1 style={{ color: "#333" }}>Macedonian Stocks Demo</h1>
                <div>
                    <select
                        name="companies"
                        id="companies"
                        onChange={(e) => setSelectedCompany(e.target.value)}
                        style={{
                            marginBottom: "10px",
                            padding: "10px",
                            fontSize: "16px",
                        }}
                    >
                        <option value="">Select a Company</option>
                        {companies.map((company, index) => (
                            <option key={index} value={company}>
                                {company}
                            </option>
                        ))}
                    </select>
                </div>
                <div>
                    <label>
                        From:{" "}
                        <input
                            type="date"
                            value={fromDate}
                            onChange={(e) => setFromDate(e.target.value)}
                            style={{ marginLeft: "10px", marginBottom: "10px" }}
                        />
                    </label>
                </div>
                <div style={{ marginTop: "10px" }}>
                    <label>
                        To:{" "}
                        <input
                            type="date"
                            value={toDate}
                            onChange={(e) => setToDate(e.target.value)}
                            style={{ marginLeft: "10px" }}
                        />
                    </label>
                </div>
                <button
                    onClick={handleFetchTransactions}
                    style={{
                        marginTop: "20px",
                        backgroundColor: "#333",
                        color: "white",
                        padding: "10px 20px",
                        fontSize: "16px",
                        border: "none",
                        cursor: "pointer",
                    }}
                    disabled={!selectedCompany}
                >
                    View Transactions
                </button>
                <button
                    onClick={handleReset}
                    style={{
                        marginTop: "10px",
                        backgroundColor: "#555",
                        color: "white",
                        padding: "10px 20px",
                        fontSize: "16px",
                        border: "none",
                        cursor: "pointer",
                    }}
                >
                    Reset
                </button>
            </div>
            <div
                style={{
                    flex: 1,
                    padding: "20px",
                    boxSizing: "border-box",
                    height: "100vh",
                }}
            >
                <div
                    style={{
                        marginBottom: "20px",
                        display: "flex",
                        justifyContent: "space-around",
                        alignItems: "center",
                        flexWrap: "wrap",
                    }}
                >
                    {["sma20", "sma50", "ema20", "ema50"].map((dataset) => (
                        <label
                            key={dataset}
                            style={{
                                color: "black",
                                marginRight: "10px",
                                display: "flex",
                                alignItems: "center",
                            }}
                        >
                            <input
                                type="checkbox"
                                checked={visibleDatasets[dataset]}
                                onChange={() => toggleDataset(dataset)}
                                style={{ marginRight: "5px" }}
                            />
                            {dataset.toUpperCase()}
                        </label>
                    ))}
                </div>
                {labels.length > 0 ? (
                    <Line data={chartData} options={chartOptions} />
                ) : (
                    <p style={{ color: "white" }}>No transactions found.</p>
                )}
            </div>
        </div>
    );
};

export default App;
