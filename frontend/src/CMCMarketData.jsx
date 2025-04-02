// file: frontend/src/CMCMarketData.jsx
import React, { useState, useEffect } from "react";
import { Card, Table } from "react-bootstrap";

function CMCMarketData() {
  const [coins, setCoins] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("http://localhost:5000/api/market_data?limit=50")
      .then((resp) => {
        if (!resp.ok) {
          throw new Error("Failed to fetch /api/market_data");
        }
        return resp.json();
      })
      .then((data) => {
        if (data.status === "ok") {
          setCoins(data.coins || []);
        } else {
          setError(data.msg || "Error returned from server");
        }
      })
      .catch((err) => {
        setError(err.message);
      });
  }, []);

  return (
    <Card>
      <Card.Body>
        <Card.Title>CMC Top 50 Listing</Card.Title>
        {error && <p className="error-msg">{error}</p>}

        <Table bordered hover responsive className="cmc-table mt-3">
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Symbol</th>
              <th>Price (USD)</th>
              <th>Market Cap</th>
            </tr>
          </thead>
          <tbody>
            {coins.map((coin, idx) => {
              const quote = coin.quote.USD;
              return (
                <tr key={coin.id}>
                  <td>{idx + 1}</td>
                  <td>{coin.name}</td>
                  <td>{coin.symbol}</td>
                  <td>${quote.price.toFixed(2)}</td>
                  <td>${quote.market_cap.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      </Card.Body>
    </Card>
  );
}

export default CMCMarketData;
