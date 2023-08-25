import { check } from 'k6';
import http from 'k6/http';
import { Trend, Counter } from 'k6/metrics';

// Define configurations
const host = __ENV.HOST || 'http://127.0.0.1:8080';

// Define the metrics
const totalTime = new Trend('total_time', true);
const validationTime = new Trend('validation_time', true);
const queueTime = new Trend('queue_time', true);
const inferenceTime = new Trend('inference_time', true);
const timePerToken = new Trend('time_per_token', true);
const generatedTokens = new Counter('generated_tokens');

export const options = {
  thresholds: {
    http_req_failed: ['rate==0'],
  },
  scenarios: {
    load_test: {
      executor: 'constant-vus',
      duration: '60s',
      vus: 140,
    },
  },
};

export default function () {
  // Create Body 
  const payload = {
    inputs: "\n    def test():\n        x=1+1\n        assert x ",
    parameters: {
      max_new_tokens: 60,
      details: true
    },
  };

  const headers = { 'Content-Type': 'application/json' };
  const res = http.post("http://127.0.0.1:8080/generate", JSON.stringify(payload), {
    headers
  });

  check(res, {
    'Post status is 200': (r) => res.status === 200,
  });

  if (res.status === 200) {
    totalTime.add(res.headers["X-Total-Time"]);
    validationTime.add(res.headers["X-Validation-Time"]);
    queueTime.add(res.headers["X-Queue-Time"]);
    inferenceTime.add(res.headers["X-Inference-Time"]);
    timePerToken.add(res.headers["X-Time-Per-Token"]);
    generatedTokens.add(res.json().details.generated_tokens);
  }
}