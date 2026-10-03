/** Noninteractive diagnostic using Asuna's installed DSH adapter and assembler.
 * Uses synthetic histories and a local fixture tool, never Asuna sessions/tools.
 * No listener or proxy is created: both providers call the shared Flash endpoint.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values: args } = parseArgs({ options: {
  asuna: { type: 'string', default: 'C:/workspace/asuna_cognition_core_v2' },
  fixture: { type: 'string', default: 'benchmarks/raw/qwen38-strata/2026-10-03/cache/parking-8k' },
  output: { type: 'string', default: 'benchmarks/raw/qwen38-strata/2026-10-03/cache/asuna-adapter-fidelity' },
  'base-url': { type: 'string', default: 'http://127.0.0.1:1919/v1' },
  rounds: { type: 'string', default: '6' },
  schedule: { type: 'string', default: 'interleaved' },
  'run-id': { type: 'string', default: 'dsh-replay-fidelity' },
  'engine-log': { type: 'string', default: 'benchmarks/raw/qwen38-strata/server/engine-1919.log' },
  'strip-replay': { type: 'boolean', default: false },
} });
const require = createRequire(join(resolve(args.asuna), 'package.json'));
const installed = name => import(pathToFileURL(name.startsWith('@earendil-works/pi-ai/')
  ? join(args.asuna, 'node_modules/@earendil-works/pi-ai/dist', name.slice('@earendil-works/pi-ai/'.length) + '.js')
  : require.resolve(name)).href);
const { PiAiAdapter } = await installed('@deepseek-ai/dsh-llm-pi-ai');
const { BlockAssembler, createSystemMessage, createToolResultMessage } = await installed('@deepseek-ai/dsh-llm');
const { openAICompletionsApi } = await installed('@earendil-works/pi-ai/api/openai-completions.lazy');
const api = openAICompletionsApi();
const { InMemoryCredentialStore } = await installed('@earendil-works/pi-ai/auth/credential-store');
const { defaultProviderAuthContext } = await installed('@earendil-works/pi-ai/auth/context');
const yaml = await installed('yaml');
const patch = yaml.parse(await readFile(join(args.asuna, '.runtime/adr008/home/profiles/asuna-native/cordis.patch.yml'), 'utf8'));
const saved = patch.find(item => item.id === 'llm-pi-ai').config.providers;
await mkdir(args.output, { recursive: true });
const save = (name, value) => writeFile(join(args.output, name), JSON.stringify(value, null, 2) + '\n');
const report = { started_at: new Date().toISOString(), base_url: args['base-url'],
  scope: 'Installed DSH PiAiAdapter + BlockAssembler, synthetic histories, in-memory credentials; no live Asuna session',
  schedule: args.schedule, strip_replay_negative_control: args['strip-replay'], cases: [], replay_degradations: [] };
report.adapter_version = JSON.parse(await readFile(require.resolve('@deepseek-ai/dsh-llm-pi-ai/package.json'), 'utf8')).version;
report.pi_ai_version = JSON.parse(await readFile(join(args.asuna, 'node_modules/@earendil-works/pi-ai/package.json'), 'utf8')).version;
const logOffset = (await readFile(args['engine-log'])).length;
let currentLabel;
const profiles = new Map();
// Reconstruct the public resolved-profile constructor input from the saved
// route. The unexported catalog resolver is not patched or copied into Asuna.
for (const provider of ['asuna-character', 'asuna-action']) {
  const config = saved[provider];
  assert.equal(config.api, 'openai-completions');
  const entry = config.models[0];
  const thinkingLevelMap = Object.fromEntries(['off','minimal','low','medium','high','xhigh','max']
    .flatMap(level => entry.reasoningEfforts?.[level] === null ? [] : [[level, entry.reasoningEfforts?.[level] ?? null]]));
  const model = { id: entry.id, name: entry.name || entry.id, provider, api: config.api,
    baseUrl: args['base-url'], contextWindow: entry.contextWindow, maxTokens: entry.maxTokens,
    input: entry.input, reasoning: true, thinkingLevelMap,
    compat: { ...config.compat, ...entry.compat }, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const instrument = options => ({ ...options, onPayload: async payload => { await save(currentLabel + '.request.json', payload); } });
  const piProvider = { id: provider, name: provider, baseUrl: args['base-url'], getModels: () => [model],
    auth: { apiKey: { name: provider, resolve: async ({credential}) => ({auth: {apiKey: credential?.key}, source: provider}) } },
    stream: (m, c, o) => api.stream(m, c, instrument(o)),
    streamSimple: (m, c, o) => api.streamSimple(m, c, instrument(o)) };
  profiles.set(provider, { provider, displayName: provider, piProvider, modelErrors: new Map(),
    configuredMaxTokens: new Map([[entry.id, entry.maxTokens]]), timeoutMs: config.timeoutMs,
    streamIdleTimeoutMs: config.streamIdleTimeoutMs });
  report[provider] = { saved_base_url: config.baseURL, tested_base_url: args['base-url'], model };
}
const adapter = new PiAiAdapter({ profiles: () => profiles,
  resolveApiKey: async () => process.env.STRATA_API_KEY || 'local-diagnostic',
  auth: { credentials: new InMemoryCredentialStore(), authContext: defaultProviderAuthContext() },
  onReplayDegrade: detail => report.replay_degradations.push(detail) });
const histories = {};
let tools;
for (const agent of ['A','B']) {
  const fixture = JSON.parse(await readFile(join(args.fixture,
    agent === 'A' ? 'a1-tool.request.json' : 'b1-tool-switch.request.json'), 'utf8'));
  tools = fixture.tools.map(t => t.function);
  histories[agent] = [createSystemMessage(fixture.messages[0].content + ' DSH adapter diagnostic ' + args['run-id'] + '.'),
    {role: 'user', content: [{type: 'text', text: fixture.messages[1].content}]}];
}
const route = agent => agent === 'A' ? 'asuna-character' : 'asuna-action';
async function call(agent, label, toolExpected) {
  currentLabel = label;
  const provider = route(agent), model = report[provider].model.id;
  const start = performance.now(), assembler = new BlockAssembler();
  let firstToken;
  for await (const chunk of adapter.stream({ provider, model, messages: histories[agent], tools,
    reasoningEffort: 'high', maxTokens: 2048, temperature: 0, sessionId: 'strata-probe-' + agent })) {
    if (firstToken === undefined && ['text-delta','reasoning-delta','tool-call-delta'].includes(chunk.type)) firstToken = performance.now();
    assembler.push(chunk);
  }
  // Match dsh-agent-loop's completed-message commit, including replay metadata.
  const message = assembler.message({provider, model,
    ...(assembler.replayState === undefined ? {} : {replayState: assembler.replayState})});
  assert.ok(message.source.replayState, 'The installed adapter must retain reasoning/tool replay metadata');
  const record = {label, wall_s: (performance.now()-start)/1000, ttft_s: (firstToken-start)/1000,
    usage: assembler.usage, finish: assembler.finish, message};
  report.cases.push(record);
  await save('report.json', report);
  console.log(JSON.stringify({label, wall_s: record.wall_s, ttft_s: record.ttft_s,
    usage: record.usage, finish: record.finish, blocks: message.content.map(b => b.type)}));
  assert.equal(message.content.filter(b => b.type === 'tool-call').length, toolExpected ? 1 : 0);
  // A JSON round-trip models durable storage without mutating Asuna's sessions.
  const durable = JSON.parse(JSON.stringify(message));
  if (args['strip-replay']) delete durable.source.replayState;
  histories[agent].push(durable);
  return message;
}
async function result(agent, round) {
  const tool = histories[agent].at(-1).content.find(b => b.type === 'tool-call');
  assert.equal(tool.name, 'lookup_probe_value');
  assert.deepEqual(JSON.parse(tool.arguments), {agent, round});
  const value = `${agent}-round-${round}-verified`;
  histories[agent].push(createToolResultMessage({callId: tool.id, isError: false,
    content: [{type: 'text', text: JSON.stringify({agent, round, value})}]}));
  const response = await call(agent, `${agent}${round}-result`, false);
  const text = response.content.filter(b => b.type === 'text').map(b => b.text).join('');
  assert.ok(text.includes(value));
  assert.ok(text.includes(agent === 'A' ? 'cobalt-falcon-174' : 'amber-otter-826'));
}
function nextRound(agent, round) {
  if (round>1) histories[agent].push({role: 'user', content: [{type: 'text',
    text: `Round ${round}: call lookup_probe_value for agent ${agent} and round ${round}.`}]});
}
assert.ok(['interleaved', 'parent-idle'].includes(args.schedule));
if (args.schedule === 'parent-idle') {
  await call('A', 'A1-tool', true);
  for (let round=1; round<=Number(args.rounds); round++) {
    nextRound('B', round);
    await call('B', `B${round}-tool`, true);
    await result('B', round);
  }
  await result('A', 1);
} else {
  for (let round=1; round<=Number(args.rounds); round++) {
    for (const agent of ['A','B']) {
      nextRound(agent, round);
      await call(agent, `${agent}${round}-tool`, true);
    }
    // Each result returns to an agent whose state was displaced by the other.
    for (const agent of ['A','B']) await result(agent, round);
  }
}
assert.equal(report.replay_degradations.length, 0);
report.passed = true;
await save('report.json', report);
await writeFile(join(args.output, 'engine.log'), (await readFile(args['engine-log'])).subarray(logOffset));
for (const agent of ['A','B']) await save(agent + '.history.json', histories[agent]);
