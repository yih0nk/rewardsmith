const fs = require('node:fs');
const { createRequire } = require('node:module');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const webRequire = createRequire(require.resolve('../web/package.json'));
const ts = webRequire('typescript');
const React = webRequire('react');
const { renderToStaticMarkup } = webRequire('react-dom/server');

for (const ext of ['.ts', '.tsx']) {
  require.extensions[ext] = (module, filename) => {
    const result = ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: {
      module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true,
    }});
    module._compile(result.outputText, filename);
  };
}
const Detail = require('../web/src/components/RequestDetail.tsx').default;
const candidate = (id, score, pass) => ({id, idx: 0, reward_src: '', diff: null,
  status: 'scored', score, critique: '', metrics: {}, video_path: null,
  check_result: {pass, clauses: []}});
const detail = {
  id: 'ranking', text: 'walk backwards', status: 'done', best_score: 7,
  best_candidate_id: 2, succeeded: 0, events: [], rounds: [
    {id: 1, n: 1, status: 'done', best_candidate_id: 1, candidates: [candidate(1, 9, false)]},
    {id: 2, n: 2, status: 'done', best_candidate_id: 2, candidates: [candidate(2, 7, true)]},
  ],
};
const render = (data) => renderToStaticMarkup(React.createElement(Detail, {detail: data, health: null}));

test('dashboard uses the backend winner even when a failed candidate scores higher', () => {
  const html = render(detail);
  assert.match(html, /summary-best[\s\S]*?7\.0[\s\S]*?Round 2/);
  assert.match(html, /Round 2 · Candidate 1/);
  assert.match(html, /Success check: passed/);
  assert.match(html, /Success criteria not met/);
});

test('legacy responses without an explicit winner use pass-first ranking', () => {
  const html = render({...detail, best_candidate_id: undefined});
  assert.match(html, /summary-best[\s\S]*?7\.0[\s\S]*?Round 2/);
});

test('degraded checker is displayed as unverified', () => {
  assert.match(render({...detail, success_check_error: 'timeout'}), /Success could not be verified/);
});
