// file: frontend/src/PredictForm.jsx
import React, { useState } from "react";
import { Card, Form, Button } from "react-bootstrap";

function PredictForm() {
  const [coin, setCoin] = useState("BTC");
  const [horizon, setHorizon] = useState("7");
  const [result, setResult] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setResult("");
    setErrorMsg("");

    try {
      const resp = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ coin, horizon })
      });
      const data = await resp.json();
      if (!resp.ok) {
        setErrorMsg(data.error || data.error_msg || "Request failed");
        console.log("Debug stdout:", data.stdout);
        console.log("Debug stderr:", data.stderr);
      } else {
        if (data.status === "ok") {
          // success
          setResult(`Predicted Price = $${data.predicted_price.toFixed(4)}`);
        } else {
          setErrorMsg("Error: " + JSON.stringify(data));
        }
      }
    } catch (err) {
      setErrorMsg(err.message || "Something went wrong");
    }
  }

  return (
    <Card>
      <Card.Body>
        <Card.Title>Crypto Prediction</Card.Title>
        <Form onSubmit={handleSubmit}>
          <Form.Group className="mb-3">
            <Form.Label>Coin</Form.Label>
            <Form.Select value={coin} onChange={(e) => setCoin(e.target.value)}>
              <option value="ADA">ADA</option>
              <option value="AVAX">AVAX</option>
              <option value="BCH">BCH</option>
              <option value="BNB">BNB</option>
              <option value="BTC">BTC</option>
              <option value="DOGE">DOGE</option>
              <option value="DOT">DOT</option>
              <option value="ETH">ETH</option>
              <option value="LEO">LEO</option>
              <option value="LINK">LINK</option>
              <option value="LTC">LTC</option>
              <option value="MATIC">MATIC</option>
              <option value="NEAR">NEAR</option>
              <option value="SHIB">SHIB</option>
              <option value="SOL">SOL</option>
              <option value="TON">TON</option>
              <option value="TRX">TRX</option>
              <option value="UNI">UNI</option>
              <option value="XRP">XRP</option>
            </Form.Select>
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>Predicted Days</Form.Label>
            <Form.Select
              value={horizon}
              onChange={(e) => setHorizon(e.target.value)}
            >
              <option value="1">1</option>
              <option value="7">7</option>
              <option value="30">30</option>
              <option value="90">90</option>
            </Form.Select>
          </Form.Group>

          <Button variant="primary" type="submit">
            Predict
          </Button>
        </Form>

        {result && <p className="result-msg">{result}</p>}
        {errorMsg && <p className="error-msg">{errorMsg}</p>}
      </Card.Body>
    </Card>
  );
}

export default PredictForm;
