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


test('scene success criteria show measured rates, exact requirements and failures', () => {
  const SuccessCheck = require('../web/src/components/SuccessCheck.tsx').default;
  const html = renderToStaticMarkup(React.createElement(SuccessCheck, {check: {pass: false, clauses: [
    {metric: 'target_reached', op: '>=', value: 0.8, actual: 0.75, pass: false},
    {metric: 'target_fell', op: '<=', value: 0.1, actual: 0.0, pass: true},
    {metric: 'target_final_dist', op: '<', value: 0.5, actual: null, pass: false},
  ]}}));
  assert.match(html, /Reached target/);
  assert.match(html, /75%/);
  assert.match(html, /&gt;= 80%/);
  assert.match(html, /&lt;= 10%/);
  assert.match(html, /Not measured/);
  assert.match(html, /Fail/);
});


test('advanced outcomes display readable stages and episode success rates', () => {
  const SuccessCheck = require('../web/src/components/SuccessCheck.tsx').default;
  const html = renderToStaticMarkup(React.createElement(SuccessCheck, {check: {pass: true, clauses: [
    {metric: 'sequence:climb:step>stand_on:step@2', op: '>=', value: 0.8, actual: 0.875, pass: true},
    {metric: 'all:push:box:goal&avoid:obstacle', op: '>=', value: 0.8, actual: 0.9, pass: true},
  ]}}));
  assert.match(html, /Climb onto step, then Stand still on step for 2 s/);
  assert.match(html, /Push box into goal and Avoid obstacle with progress/);
  assert.match(html, /87.5%/);
  assert.match(html, /&gt;= 80%/);
});

test('training budget control converts millions to steps and can restore automatic mode', () => {
  const Options = require('../web/src/components/MessageOptions.tsx').default;
  const changes = [];
  const props = {value: {mode: 'effort', model: 'claude-opus-5-5', training_steps: 20000000},
    onChange: value => changes.push(value), models: ['claude-opus-5-5'], idPrefix: 'task'};
  const tree = Options(props);
  const find = node => {
    if (node?.props?.type === 'number') return node;
    for (const child of React.Children.toArray(node?.props?.children)) {
      const result = find(child);
      if (result) return result;
    }
  };
  const input = find(tree);
  assert.equal(input.props.value, 20);
  assert.equal(input.props.min, 1);
  assert.equal(input.props.max, 100);
  input.props.onChange({target: {value: '50'}});
  input.props.onChange({target: {value: ''}});
  assert.equal(changes[0].training_steps, 50000000);
  assert.equal(changes[1].training_steps, undefined);
  assert.equal(changes[0].model, props.value.model);
  const html = renderToStaticMarkup(React.createElement(Options, props));
  assert.match(html, /Training steps \(millions\)/);
  assert.match(html, /Per candidate/);
  assert.match(html, /Training budget/);
  assert.match(html, /20M/);
  assert.match(html, /Effort trains four candidates/);
  const automatic = renderToStaticMarkup(React.createElement(Options, {
    ...props, value: {...props.value, mode: 'instant', training_steps: undefined},
  }));
  assert.match(automatic, /Auto/);
  assert.match(automatic, /Instant trains one candidate/);
  assert.doesNotMatch(automatic, /Effort trains four candidates/);
});
