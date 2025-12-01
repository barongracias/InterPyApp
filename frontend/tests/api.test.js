import assert from 'node:assert';

const backend = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const runIntegration = process.env.RUN_INTEGRATION_TESTS === '1';

if (!runIntegration) {
  console.log('Skipping backend integration test (set RUN_INTEGRATION_TESTS=1 to enable).');
} else {
  // Simple integration check: backend health endpoint
  async function fetchJson(url) {
    const res = await fetch(url);
    const data = await res.json();
    return { status: res.status, data };
  }

  await (async () => {
    const { status, data } = await fetchJson(`${backend}/health`);
    assert.equal(status, 200, 'health status code');
    assert.equal(data.status, 'ok', 'health status payload');
  })();
}
