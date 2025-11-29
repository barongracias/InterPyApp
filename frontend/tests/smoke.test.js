const test = require("node:test");
const assert = require("node:assert/strict");
const React = require("react");
const ReactDOMServer = require("react-dom/server");

test("react renders a basic component", () => {
  const Component = () => React.createElement("div", null, "smoke-ok");
  const html = ReactDOMServer.renderToString(React.createElement(Component));
  assert.ok(html.includes("smoke-ok"));
});
