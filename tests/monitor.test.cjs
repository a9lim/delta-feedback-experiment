// Numerical and lifecycle contracts for the actual monitor scripts; no browser
// or npm dependencies. UI layout and controls are checked in the real browser.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const shared = path.resolve(root, '../transformer_experiments/monitor');
const html = fs.readFileSync(path.join(root, 'monitor/index.html'), 'utf8');
const inline = html.slice(html.indexOf("'use strict';"), html.lastIndexOf('</script>'));
const plain = (value) => JSON.parse(JSON.stringify(value));

function monitor() {
  const context = vm.createContext({
    console, location: { pathname: '/delta-feedback/', hash: '' },
    document: { documentElement: {} },
    getComputedStyle: () => ({ getPropertyValue: () => '#888888' }),
  });
  vm.runInContext(fs.readFileSync(path.join(shared, 'vendor/uPlot.iife.min.js'), 'utf8'), context);
  // Expose ingestion from the chassis closure without changing its behavior.
  vm.runInContext(fs.readFileSync(path.join(shared, 'chassis.js'), 'utf8').replace('return { configure,', 'return { ingest, configure,'), context);
  vm.runInContext(inline.replace('\nsetupSnapshotControls();\nM.start();', ''), context);
  vm.runInContext("M.configure({onStep, onEval, onRecord, metaEvents: ['run', 'schedule']});", context);
  const api = vm.runInContext('({M, onRecord, onEval, snapshot, reconcileSnapshot, orderedSites, SITES, profileOf, expertBanks, heatmapHTML, biasBound, nullRange, seriesOf, tokenTotals})', context);
  api.read = (tag, log) => {
    api.M.ingest(api.M.runFor(tag), log);
    api.M.rebuildAll();
    api.M.state.selected = tag;
    return api.M.viewOf(tag);
  };
  return api;
}
const line = (event, fields) => `${event} | ${Object.entries(fields).map(([key, value]) => `${key}=${value}`).join(' | ')}\n`;
const setup = (tag, condition = 'fl') => line('run', {tag, condition, batch_rows: 2, seq_len: 8, layers: 16, routing_block_size: 4});
const route = (step, site, nul, extras = {}) => line('route', {step: `${step}/100`, site, null: nul, seed: 0.2, max: 0.4, head_js: 0.1, n: 4, null_rms: 0, ...extras});
const expert = (step, site, extras = {}) => line('expert', {step: `${step}/100`, site, used: 2, entropy: 0.6, expert0: 0.25, expert1: 0.75, bias0: -0.1, bias1: 0.1, ...extras});
const step = (n, extras = {}) => line('step', {step: `${n}/100`, k: 2, r: 1, ...extras});

test('eval slider uses sorted unique addresses, including partial eval batches', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + route(20, 'L0.attn', 0.2) + expert(10, 'L0.experts') + line('eval', {step: '20/100', val: 6}) + line('eval', {step: '30/100', val: 5}));
  a.reconcileSnapshot(view);
  assert.deepEqual(plain(a.snapshot.steps), [10, 20, 30]);
  assert.equal(a.snapshot.current, 30);
  assert.equal(a.snapshot.step, null);
  assert.deepEqual(plain(a.orderedSites([view], 'route')), []);
});

test('latest follows appends; pinned step survives appends and refresh', () => {
  const a = monitor();
  let view = a.read('a', setup('a') + route(10, 'L0.attn', 0.1));
  a.reconcileSnapshot(view);
  view = a.read('a', route(20, 'L0.attn', 0.2));
  a.reconcileSnapshot(view);
  assert.equal(a.snapshot.current, 20);
  a.snapshot.step = 10;
  view = a.read('a', route(30, 'L0.attn', 0.3));
  a.reconcileSnapshot(view);
  assert.equal(a.snapshot.current, 10);
  assert.deepEqual(plain(a.snapshot.steps), [10, 20, 30]);
  a.snapshot.step = null;
  a.reconcileSnapshot(view);
  assert.equal(a.snapshot.current, 30);
});

test('switching run resets to latest; zero and single eval are valid', () => {
  const a = monitor();
  a.reconcileSnapshot(a.read('a', setup('a') + route(10, 'L0.attn', 0.1)));
  a.snapshot.step = 10;
  a.reconcileSnapshot(a.read('b', setup('b')));
  assert.equal(a.snapshot.current, null);
  a.reconcileSnapshot(a.read('b', route(0, 'L0.attn', 0)));
  assert.equal(a.snapshot.current, 0);
  assert.equal(a.snapshot.step, null);
});

test('resume folding removes future route and expert snapshots and clamps selection', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + step(1) + route(1, 'L0.attn', 0.1) + expert(1, 'L0.experts') + step(2) + route(2, 'L0.attn', 0.9) + expert(2, 'L0.experts'));
  a.reconcileSnapshot(view); a.snapshot.step = 2;
  const resumed = a.read('a', step(2));
  a.reconcileSnapshot(resumed);
  assert.equal(a.snapshot.current, 1);
  assert.deepEqual(plain(a.snapshot.steps), [1]);
  assert.equal(resumed.evals.get('expert|L0.experts').length, 1);
});

test('fork snapshots inherit only the parent prefix', () => {
  const a = monitor();
  a.read('parent', setup('parent') + step(1) + route(1, 'L0.attn', 0.1) + step(2) + route(2, 'L0.attn', 0.9));
  const child = a.read('child', setup('child') + line('fork', {step: '1/100', path: 'runs/parent.pt.1'}) + step(2) + route(2, 'L0.attn', 0.2));
  a.reconcileSnapshot(child);
  assert.deepEqual(plain(child.evals.get('route|L0.attn').map((r) => r.nul)), [0.1, 0.2]);
  assert.deepEqual(plain(a.snapshot.steps), [1, 2]);
});

test('profiles use exact step in every run; missing overlays stay empty', () => {
  const a = monitor();
  const primary = a.read('a', setup('a') + route(10, 'L0.attn', 0.1) + route(20, 'L0.attn', 0.2));
  const overlay = a.read('b', setup('b') + route(10, 'L0.attn', 0.9));
  a.reconcileSnapshot(primary);
  a.SITES.push(...a.orderedSites([primary, overlay], 'route'));
  assert.deepEqual(plain(a.profileOf(primary)['p-site-mass']), [[0], [0.2], [0.2], [null]]);
  assert.deepEqual(plain(a.profileOf(overlay)['p-site-mass']), [[0], [null], [null], [null]]);
  assert.deepEqual(plain(a.profileOf(primary)['p-site-sharp'][3]), [0.25]);
});

test('execution order handles repeated core layers, coda, payload and changing sites', () => {
  const a = monitor();
  const names = ['payload', 'L12.attn', 'L4i2.mlp', 'L4i1.mlp', 'L0.mlp', 'L4i1.attn', 'L0.attn'];
  const view = a.read('a', setup('a') + names.map((name) => route(10, name, 0.2)).join('') + route(20, 'L0.attn', 0.3));
  a.reconcileSnapshot(view); a.snapshot.step = 10; a.reconcileSnapshot(view);
  assert.deepEqual(plain(a.orderedSites([view], 'route').map((s) => s.name)), ['L0.attn', 'L0.mlp', 'L4i1.attn', 'L4i1.mlp', 'L4i2.mlp', 'L12.attn', 'payload']);
  a.snapshot.step = 20; a.reconcileSnapshot(view);
  assert.deepEqual(plain(a.orderedSites([view], 'route').map((s) => s.name)), ['L0.attn']);
});

test('expert vectors use numeric ids and preserve missing versus zero', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + line('expert', {step: '10/100', site: 'L0.experts', expert10: 0.5, expert2: 0, expert0: 0.5, bias10: 0, bias0: -0.1}));
  const e = view.evals.get('expert|L0.experts')[0];
  assert.equal(e.loads.length, 11);
  assert.equal(e.loads[1], null);
  assert.equal(e.loads[2], 0);
  assert.equal(e.loads[10], 0.5);
  assert.equal(e.biases[10], 0);
});

test('biases collapse shared iteration banks; assignment sites remain separate', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + ['L4i1.experts', 'L4i2.experts', 'L12.experts', 'mtp.experts'].map((s) => expert(10, s)).join(''));
  a.reconcileSnapshot(view);
  const sites = a.orderedSites([view], 'expert');
  assert.equal(sites.length, 4);
  assert.deepEqual(plain(a.expertBanks(sites).map((s) => s.name)), ['L4.experts', 'L12.experts', 'mtp.experts']);
});

test('heatmaps expose values and one keyboard entry, with stable scales across evals', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + expert(10, 'L0.experts', {bias0: 0, bias1: 0}) + expert(20, 'L0.experts', {bias0: -0.25, bias1: 0.25}));
  a.reconcileSnapshot(view); a.snapshot.step = 10; a.reconcileSnapshot(view);
  const sites = a.orderedSites([view], 'expert');
  const loads = a.heatmapHTML(view, sites, 'loads');
  assert.match(loads, /25.00% of assignments · 0.5× uniform/);
  assert.match(loads, /gate entropy 0.6 nats/);
  assert.equal((loads.match(/tabindex="0"/g) || []).length, 1);
  assert.equal(a.biasBound(view), 0.25);
  const biases = a.heatmapHTML(view, sites, 'biases');
  assert.match(biases, /-0.25000/);
  assert.match(biases, /bias 0.000000/);
  assert.doesNotMatch(biases, /NaN|Infinity/);
});

test('unobserved heatmaps and zero biases do not invent measurements', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + expert(10, 'L0.experts', {bias0: 0, bias1: 0}));
  a.reconcileSnapshot(view);
  assert.match(a.heatmapHTML(view, [], 'loads'), /No expert observations/);
  const biases = a.heatmapHTML(view, a.orderedSites([view], 'expert'), 'biases');
  assert.doesNotMatch(biases, /NaN|Infinity/);
  assert.match(biases, /--s1\) 0%/);
});

test('raw training traces and feedback gap use CE, not weighted total loss', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + step(1, {loss: 20, pass1: 5, ntp: 9.5, mtp: 7, k: 2, r: 3}) + step(2, {loss: 8, pass1: 4, ntp: 4, mtp: 6, k: 1, r: 1}));
  const series = a.seriesOf(view);
  assert.deepEqual(plain(series['p-nll']), [[1, 2], [20, 8], [5, 4], [9.5, 4], [7, 6]]);
  assert.deepEqual(plain(series['p-gap'][2]), [-0.5, null]);
  assert.deepEqual(plain(series['p-recurrence']), [[1, 2], [2, 1], [3, 1]]);
});

test('token totals account for actual recurrence and reject incomplete histories', () => {
  const a = monitor();
  let view = a.read('a', setup('a') + step(1, {k: 2, r: 1}) + step(2, {k: 3, r: 2}));
  assert.deepEqual(plain(a.tokenTotals(view)), {predicted: 32, passes: 80, cells: 416});
  view = a.read('a', step(4));
  assert.equal(a.tokenTotals(view), null);
});
