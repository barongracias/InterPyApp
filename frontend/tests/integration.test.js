import assert from "node:assert";
import http from "node:http";
import { once } from "node:events";
import { describe, test } from "node:test";

// Simple mock backend to mirror the endpoints the UI calls
const createMockServer = () =>
  http.createServer((req, res) => {
    const { url, method } = req;
    res.setHeader("Content-Type", "application/json");

    if (url === "/health" && method === "GET") {
      res.writeHead(200);
      return res.end(JSON.stringify({ status: "ok" }));
    }
    if (url === "/upload" && method === "POST") {
      res.writeHead(200);
      return res.end(
        JSON.stringify({
          message: "File uploaded successfully",
          path: "dataset.pkl",
          stored_filename: "mock_dataset.pkl",
          original_filename: "dataset.pkl",
          stats: { rows: 10, features: 5, x_min: [0, 0, 0, 0, 0], x_max: [1, 1, 1, 1, 1], y_min: 0, y_max: 1 },
        })
      );
    }
    if (url === "/train" && method === "POST") {
      res.writeHead(200);
      return res.end(
        JSON.stringify({
          message: "Training completed successfully.",
          model_type: "numpy",
          train_loss_start: 0.5,
          train_loss_end: 0.1,
          val_loss_start: 0.6,
          val_loss_end: 0.12,
          plots: ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"],
          artifacts: ["model_weights.npz", "normalisation_values.npz", "model_metadata.json"],
        })
      );
    }
    if (url === "/predict" && method === "POST") {
      res.writeHead(200);
      return res.end(JSON.stringify({ y_pred: [[0.42]] }));
    }
    res.writeHead(404);
    res.end(JSON.stringify({ error: "not found" }));
  });

const buildFormData = () => {
  const fd = new FormData();
  fd.append("pkl_filename", "dataset.pkl");
  fd.append("hidden_sizes", "4,2");
  fd.append("Lambda", "0.01");
  fd.append("epochs", "5");
  fd.append("learning_rate", "0.01");
  fd.append("train_val_split", "0.8");
  fd.append("beta1", "0.9");
  fd.append("beta2", "0.999");
  fd.append("epsilon", "1e-8");
  return fd;
};

const withServer = async (handler) => {
  const server = createMockServer();
  server.listen(0);
  await once(server, "listening");
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;
  try {
    await handler(base);
  } finally {
    server.close();
  }
};

describe("frontend mock integration", () => {
  test("health, upload, train, predict flow works against mock backend", async () => {
    await withServer(async (base) => {
      const health = await fetch(`${base}/health`).then((r) => r.json());
      assert.equal(health.status, "ok");

      const uploadRes = await fetch(`${base}/upload`, { method: "POST", body: new FormData() });
      assert.equal(uploadRes.status, 200);
      const upload = await uploadRes.json();
      assert.equal(upload.message, "File uploaded successfully");
      assert.equal(upload.stats.features, 5);

      const trainRes = await fetch(`${base}/train`, { method: "POST", body: buildFormData() });
      assert.equal(trainRes.status, 200);
      const train = await trainRes.json();
      assert.ok(Array.isArray(train.plots));
      assert.ok(Array.isArray(train.artifacts));

      const predictRes = await fetch(`${base}/predict`, { method: "POST", body: new FormData() });
      assert.equal(predictRes.status, 200);
      const predict = await predictRes.json();
      assert.ok(Array.isArray(predict.y_pred));
    });
  });
});
