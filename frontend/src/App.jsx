// file: frontend/src/App.jsx
import React from "react";
import PredictForm from "./PredictForm";
import CMCMarketData from "./CMCMarketData";
import { Container } from "react-bootstrap";

function App() {
  return (
    <>
      {/* HERO SECTION */}
      <div className="hero-section">
        <h1>Crypto Forecasting Capstone</h1>
        <p>Accurate price predictions & real-time market data</p>
      </div>

      {/* Main container for cards */}
      <Container className="app-container">
        <PredictForm />
        <CMCMarketData />
      </Container>
    </>
  );
}

export default App;
