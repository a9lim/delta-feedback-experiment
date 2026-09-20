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
  const scaleSteps = { checked: false };
  const context = vm.createContext({
    console, location: { pathname: '/delta-feedback/', hash: '' },
    document: { documentElement: {}, getElementById: (id) => id === 'scalesteps' ? scaleSteps : null },
    getComputedStyle: () => ({ getPropertyValue: () => '#888888' }),
  });
  vm.runInContext(fs.readFileSync(path.join(shared, 'vendor/uPlot.iife.min.js'), 'utf8'), context);
  // Expose ingestion from the chassis closure without changing its behavior.
  vm.runInContext(fs.readFileSync(path.join(shared, 'chassis.js'), 'utf8').replace('return { configure,', 'return { ingest, newRun, configure,'), context);
  vm.runInContext(inline.replace('\nsetupSnapshotControls();\nM.start();', ''), context);
  vm.runInContext("M.configure({onStep, onEval, onRecord, metaEvents: ['run', 'schedule']});", context);
  const api = vm.runInContext('({M, onRecord, onEval, snapshot, reconcileSnapshot, orderedSites, SITES, profileOf, heatmapHTML, biasBound, nullRange, seriesOf, stepScale, stepDecorations, tokenTotals, downstream, taskRecords, baselineColumns, resultsFor, resultSelection, resultColumns, resultsHTML, resultCell, tightLogRange})', context);
  api.scaleSteps = scaleSteps;
  api.read = (tag, log) => {
    api.M.ingest(api.M.runFor(tag), log);
    api.M.rebuildAll();
    api.M.state.selected = tag;
    return api.M.viewOf(tag);
  };
  api.save = (file) => {
    api.M.state.results.files = api.M.state.results.files.filter((old) => old.file !== file.file).concat(file);
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

test('execution order follows layers, then the payload, and tracks changing sites', () => {
  const a = monitor();
  const names = ['payload', 'L12.attn', 'L4.mlp', 'L0.mlp', 'L4.attn', 'L0.attn'];
  const view = a.read('a', setup('a') + names.map((name) => route(10, name, 0.2)).join('') + route(20, 'L0.attn', 0.3));
  a.reconcileSnapshot(view); a.snapshot.step = 10; a.reconcileSnapshot(view);
  assert.deepEqual(plain(a.orderedSites([view], 'route').map((s) => s.name)), ['L0.attn', 'L0.mlp', 'L4.attn', 'L4.mlp', 'L12.attn', 'payload']);
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

test('expert sites order by layer with the auxiliary bank last', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + ['mtp.experts', 'L12.experts', 'L4.experts'].map((s) => expert(10, s)).join(''));
  a.reconcileSnapshot(view);
  assert.deepEqual(plain(a.orderedSites([view], 'expert').map((s) => s.name)), ['L4.experts', 'L12.experts', 'mtp.experts']);
});

test('depth and eval records read out per column at the evaluation count', () => {
  const a = monitor();
  const view = a.read('a', setup('a') + line('eval', {step: '10/100', val: 6, val_fused: 5.5, val_one: 5.8, val_mtp: 7})
    + line('depth', {step: '10/100', r_eval: 2, r_max: 3, loss_one: 5.8, loss_eval: 6, loss_max: 5.9, upd_max: 0.3}));
  const series = a.seriesOf(view);
  assert.deepEqual(plain(series['p-val']), [[10], [6], [5.5], [5.8]]);
  assert.deepEqual(plain(series['p-depth']), [[10], [5.8], [6], [5.9]]);
  assert.deepEqual(plain(series['p-depthupd']), [[10], [0.3]]);
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
  const view = a.read('a', setup('a') + step(1, {loss: 20, pass1: 5, ntp: 9.5, mtp: 7, k: 2, r: 1}) + step(2, {loss: 8, pass1: 4, ntp: 4, mtp: 6, k: 1, r: 1})
    + step(3, {loss: 30, pass1: 5, ntp: 19, mtp: 7, k: 2, r: 2}));
  const series = a.seriesOf(view);
  assert.deepEqual(plain(series['p-nll']), [[1, 2, 3], [20, 8, 30], [5, 4, 5], [9.5, 4, 19], [7, 6, 7]]);
  // The train gain reads only single-column feedback steps; looped steps
  // fold the loop blocks into the combined CE.
  assert.deepEqual(plain(series['p-gap'][2]), [-0.5, null, null]);
  assert.deepEqual(plain(series['p-recurrence']), [[1, 2, 3], [2, 1, 2], [1, 1, 2]]);
});

test('token totals account for every column of every pass and reject incomplete histories', () => {
  const a = monitor();
  let view = a.read('a', setup('a') + step(1, {k: 2, r: 1}) + step(2, {k: 3, r: 2}));
  assert.deepEqual(plain(a.tokenTotals(view)), {predicted: 32, passes: 80, cells: 512});
  view = a.read('a', step(4));
  assert.equal(a.tokenTotals(view), null);
});

const task = (metrics = {acc: {mean: 0.3326, se: 0.01}, acc_norm: {mean: 0.3969, se: 0.01}}, n = 10042) => ({n, metrics, meta: {revision: 'test', max_len: 1024}});
const result = (tag, overrides = {}, tasks = {hellaswag: task()}) => {
  const meta = {tag, step: 9142, mode: 'standard', passes: 0, ...overrides};
  return {file: `figures/downstream-${tag}/${meta.mode}${meta.passes}.${meta.step}.json`, meta, tasks};
};
const baseline = (model = 'org/reference', tasks = {hellaswag: task()}) => ({file: `figures/baseline/${model}.json`, meta: {model, dtype: 'float32'}, tasks});

test('saved JSON supplies results without eval logs or fork inheritance', () => {
  const a = monitor();
  a.read('parent', setup('parent') + line('downstream', {step: '9142/9142', mode: 'standard', passes: 0, task: 'hellaswag', n: 10042, acc: 0.99}));
  assert.deepEqual(plain(a.resultsFor('parent')), []);
  a.save(result('parent'));
  assert.equal(a.resultsFor('parent')[0].metrics.acc.mean, 0.3326);
  a.read('child', setup('child') + line('fork', {step: '9142/10000', path: 'runs/parent.pt.9142'}));
  assert.deepEqual(plain(a.resultsFor('child')), []);
  a.M.state.results.files = [];
  assert.deepEqual(plain(a.resultsFor('parent')), []);
});

test('JSON replacements separate modes, passes and steps and preserve checkpoint selection', () => {
  const a = monitor();
  a.save(result('a'));
  a.save(result('a', {}, {hellaswag: task({acc: {mean: 0.4, se: 0.02}})}));
  a.save(result('a', {mode: 'fused', passes: 1}));
  a.save(result('a', {mode: 'fused', passes: 2}));
  a.save(result('a', {step: 4000}));
  const results = a.resultsFor('a');
  assert.equal(results.length, 4);
  assert.equal(results[0].metrics.acc.mean, 0.4);
  assert.deepEqual(plain(a.resultSelection('a', results)), {steps: [4000, 9142], selected: null, current: 9142});
  a.downstream.selection.set('a', 4000);
  assert.equal(a.resultSelection('a', results).current, 4000);
  assert.equal(a.resultSelection('a', results.filter((r) => r.step === 9142)).current, 9142);
});

test('JSON task metrics preserve zero and precision, translate perplexity, and reject malformed scores', () => {
  const a = monitor();
  a.save(result('a', {}, {lambada_openai: task({perplexity: {mean: 15.81234, se: 0.2}, acc: {mean: 0, se: 0}, acc_norm: {mean: null}}, 5153)}));
  assert.deepEqual(plain(a.resultsFor('a')[0].metrics), {acc: {mean: 0, se: 0}, ppl: {mean: 15.81234, se: 0.2}});
  for (const overrides of [{step: 'bad'}, {passes: -1}, {mode: 'bogus'}, {passes: null}, {step: false}]) {
    const b = monitor(); b.save(result('a', overrides));
    assert.deepEqual(plain(b.resultsFor('a')), []);
  }
  for (const raw of [task(undefined, 0), task(undefined, '4oops'), task({acc: {mean: 'nan'}, acc_norm: {mean: Infinity}}), null]) {
    assert.deepEqual(plain(a.taskRecords({tasks: {bad: raw}})), []);
  }
});

test('baselines appear alongside runs and remain when no selected run has saved results', () => {
  const a = monitor();
  a.save(result('a'));
  a.save(baseline());
  a.save(baseline('org/second', {piqa: task({acc: {mean: 0.8}}, 1838)}));
  const groups = [{tag: 'a', current: 9142, color: '#888', records: a.resultsFor('a')}];
  const columns = [...a.resultColumns(groups), ...a.baselineColumns()], html = a.resultsHTML(columns);
  assert.equal(columns.length, 3);
  assert.match(html, /org\/reference<small>Baseline<\/small>/);
  assert.match(html, /80.00%/);
  assert.match(html, /aria-label="No result"/);
  assert.match(html, /n=1,838/);
  assert.doesNotMatch(html, /Δ|Baseline · step|NaN|Infinity|undefined/);
  assert.equal(a.resultsFor('org/reference').length, 0);
  assert.match(a.resultsHTML(a.baselineColumns()), /org\/reference/);
});

test('cross-run table aligns tasks and modes and keeps missing scores', () => {
  const a = monitor();
  a.save(result('a'));
  a.save(result('a', {mode: 'fused', passes: 2}));
  a.save(result('b'));
  a.save(result('b', {mode: 'fused', passes: 1}, {piqa: task({acc: {mean: 0.7}}, 1838)}));
  const groups = ['a', 'b', 'no-json'].map((tag) => ({tag, current: 9142, color: '#888', records: a.resultsFor(tag)}));
  const columns = a.resultColumns(groups), html = a.resultsHTML(columns);
  assert.equal(columns.length, 4);
  assert.doesNotMatch(html, /Δ|no-json/);
  assert.match(html, /aria-label="No result"/);
  assert.match(html, /n=1,838/);
  assert.match(a.resultCell({n: 32, metrics: {acc: {mean: 0.4}}}, 'acc'), /title="Accuracy ↑ · n=32">40.00%<\/div>/);
  assert.match(a.resultCell({n: 32, metrics: {ppl: {mean: 15.8}}}, 'ppl'), />15.80<\/div>/);
});

test('eval table escapes JSON-provided task, model and run labels', () => {
  const a = monitor();
  a.save(result('a', {}, {'<script>alert(1)</script>': task()}));
  a.save(baseline('<img src=x onerror=alert(1)>'));
  const columns = [...a.resultColumns([{tag: '<img src=x>', current: 9142, records: a.resultsFor('a'), color: '#888'}]), ...a.baselineColumns()];
  const html = a.resultsHTML(columns);
  assert.doesNotMatch(html, /<script>|<img/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /&lt;img/);
});

test('log axis has finite positive bounds for empty, singleton and normal ranges', () => {
  const a = monitor();
  for (const [lo, hi] of [[null, null], [0, 0], [1, 1], [100, 100], [1, 10000]]) {
    const range = a.tightLogRange(null, lo, hi);
    assert.ok(range.every(Number.isFinite));
    assert.ok(range[0] > 0 && range[1] > range[0]);
    if (lo > 0) assert.ok(range[0] <= lo && range[1] >= hi);
  }
});

test('scaled trajectories align equal token budgets while preserving fourfold longer runs', () => {
  const a = monitor();
  const make = (tag, dim, params, end) => a.read(tag,
    line('run', { condition: 'f', dim, params, vocab_size: 8, layers: 2, expert_width: 3, expert_routed: 3, expert_top_k: 1, batch_rows: 2, seq_len: 5 }) +
    line('step', { step: `${end / 2}/${end}`, loss: 4, ntp: 3, pass1: 2, k: 2 }) +
    line('step', { step: `${end}/${end}`, loss: 3, ntp: 2, pass1: 1, k: 3 }) +
    line('eval', { step: `${end}/${end}`, val: 2 }));
  // Reference active counts are 4,000 and 8,000; each update predicts 10 tokens.
  const views = [make('screen-25', 2, 4124, 10000), make('bridge-25', 4, 8248, 20000), make('screen-100', 2, 4124, 40000)];
  const original = views.map(a.seriesOf);
  assert.deepEqual(original.map((s) => s['p-nll'][0].at(-1)), [10000, 20000, 40000]);
  a.scaleSteps.checked = true;
  const scaled = views.map(a.seriesOf);
  assert.deepEqual(scaled.map((s) => s['p-nll'][0].at(-1)), [25, 25, 100]);
  for (let i = 0; i < views.length; i++) for (const id of Object.keys(original[i])) {
    assert.deepEqual(plain(scaled[i][id].slice(1)), plain(original[i][id].slice(1)), `${id} values unchanged`);
    assert.deepEqual(plain(scaled[i][id][0]), plain(original[i][id][0].map((x) => x * a.stepScale(views[i]))), `${id} addresses scaled`);
  }
  const joined = a.M.join(scaled.map((s) => s['p-val']));
  assert.deepEqual(plain(joined), [[25, 100], [2, null], [null, null], [null, null], [2, null], [null, null], [null, null], [null, 2], [null, null], [null, null]]);
  a.scaleSteps.checked = false;
  assert.deepEqual(views.map((v) => plain(a.seriesOf(v))), original.map(plain));
});

test('step scaling matches the schedule denominator for every current scale and condition', () => {
  const { execFileSync } = require('node:child_process');
  const geometries = JSON.parse(execFileSync('python', ['-c', `
import json
import torch
from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.train import parse_run_args, model_fields, reference_active
records = []
for scale in ('screen', 'bridge', 'flagship', 'extension'):
    args = parse_run_args(['monitor-check', '--scale', scale])
    active = reference_active(args)
    for condition in ('f', 'l', 'fl'):
        with torch.device('meta'):
            model = DeltaModel(condition_config(condition, **model_fields(args)))
        cfg = model.cfg
        records.append(dict(condition=condition, params=sum(p.numel() for p in model.parameters()),
            dim=cfg.dim, vocab_size=cfg.vocab_size, layers=cfg.layers, expert_width=cfg.expert_intermediate,
            expert_routed=cfg.num_routed_experts, expert_top_k=cfg.experts_per_token,
            batch_rows=args.batch_rows, seq_len=args.seq_len, active=active, scale=scale))
print(json.dumps(records))
`], { cwd: root, encoding: 'utf8' }));
  const a = monitor();
  a.scaleSteps.checked = true;
  for (const record of geometries) {
    const view = a.read(`${record.scale}-${record.condition}`, line('run', record));
    assert.equal(a.stepScale(view), record.batch_rows * record.seq_len / record.active);
    record.batch_rows *= 2;
    const largerBatch = a.read(view.run.tag, line('run', record));
    assert.equal(a.stepScale(largerBatch), record.batch_rows * record.seq_len / record.active);
  }
});

test('scaled bands, fork/recurrence lines and checkpoints preserve raw snapshot selection', () => {
  const a = monitor();
  const view = a.read('a', line('run', { condition: 'fl', params: 4126, dim: 2, vocab_size: 8, layers: 2, expert_width: 3, expert_routed: 3, expert_top_k: 1, batch_rows: 2, seq_len: 5 }) +
    line('schedule', { warmup_steps: 10, preheat_steps: 0, heat_steps: 70, cooldown_steps: 20, recurrence_boundary: 60 }) +
    line('fork', { step: '40/100', path: 'runs/parent.pt.40' }) + step(100) + route(80, 'L0.attn', 0.2) +
    line('checkpoint', { step: '80/100', kind: 'snapshot' }));
  a.reconcileSnapshot(view);
  a.snapshot.step = 80;
  const raw = a.stepDecorations(view), profile = plain(a.profileOf(view));
  a.scaleSteps.checked = true;
  const factor = a.stepScale(view), scaled = a.stepDecorations(view);
  assert.deepEqual(plain(scaled.bands), plain(raw.bands.map((b) => ({ ...b, from: b.from * factor, to: b.to * factor }))));
  assert.deepEqual(plain(scaled.forks.map((f) => f.step)), [0.1, 0.15]);
  assert.deepEqual(plain(scaled.marks), [0.2]);
  a.reconcileSnapshot(view);
  assert.equal(a.snapshot.current, 80);
  assert.equal(view.snapshots[0], 80);
  assert.deepEqual(plain(a.profileOf(view)), profile);
});

test('missing geometry never mixes raw steps into scaled trajectories', () => {
  const a = monitor(), view = a.read('a', setup('a') + step(1, { loss: 2 }));
  assert.equal(a.stepScale(view), 1);
  a.scaleSteps.checked = true;
  assert.equal(a.stepScale(view), null);
  assert.ok(Object.values(a.seriesOf(view)).every((data) => data.every((col) => col.length === 0)));
  assert.deepEqual(plain(a.stepDecorations(view)), { bands: [], forks: [], marks: [] });
});
