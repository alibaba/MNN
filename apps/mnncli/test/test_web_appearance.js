const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
const start = html.indexOf("  const appearanceKey = ");
const end = html.indexOf('  const queryKey = ', start);
const handlersStart = html.indexOf("  $('appearanceButton').onclick=");
const handlersEnd = html.indexOf("  function renderAgentSettings()", handlersStart);
assert(start >= 0 && end > start && handlersStart >= 0 && handlersEnd > handlersStart);

function createPage(storage, dark = false) {
  const themes = ['system', 'light', 'dark'].map(value => ({dataset: {themeChoice: value}, setAttribute(k, v) { this[k] = v; }}));
  const accents = ['violet', 'blue', 'green', 'orange'].map(value => ({dataset: {accentChoice: value}, setAttribute(k, v) { this[k] = v; }}));
  const nodes = Object.fromEntries(['appearanceButton', 'appearanceDialog', 'closeAppearance', 'doneAppearance', 'resetAppearance', 'sidebar'].map(key => [key, {}]));
  nodes.sidebar.classList = {remove() {}};
  nodes.appearanceDialog.showModal = function () { this.open = true; };
  nodes.appearanceDialog.close = function () { this.open = false; };
  const system = {matches: dark, addEventListener(_, handler) { this.change = handler; }};
  const root = {dataset: {}};
  const document = {documentElement: root, querySelectorAll(selector) {
    if (selector === '[data-theme-choice]') return themes;
    if (selector === '[data-accent-choice]') return accents;
    throw Error('Unexpected selector: ' + selector);
  }};
  const messages = [];
  const window = {matchMedia: () => system, addEventListener(_, handler) { this.storageChange = handler; },
    webkit: {messageHandlers: {appearance: {postMessage: value => messages.push(value)}}}};
  const localStorage = {getItem: key => storage[key] ?? null, setItem: (key, value) => { storage[key] = value; }};
  vm.runInNewContext(html.slice(start, end) + html.slice(handlersStart, handlersEnd),
    {document, window, localStorage, $: id => nodes[id]});
  return {themes, accents, nodes, system, root, window, messages};
}

const storage = {};
let page = createPage(storage);
assert.strictEqual(page.root.dataset.theme, 'light');
assert.strictEqual(page.root.dataset.accent, 'violet');
assert.strictEqual(page.messages.at(-1).mode, 'system');
assert.strictEqual(page.messages.at(-1).resolved, 'light');
assert.strictEqual(page.themes[0]['aria-pressed'], 'true');
page.nodes.appearanceButton.onclick();
assert.strictEqual(page.nodes.appearanceDialog.open, true);
page.themes[2].onclick();
page.accents[1].onclick();
assert.strictEqual(page.root.dataset.theme, 'dark');
assert.strictEqual(page.root.dataset.accent, 'blue');
assert.strictEqual(page.messages.at(-1).mode, 'dark');
page = createPage(storage);
assert.strictEqual(page.root.dataset.theme, 'dark');
assert.strictEqual(page.accents[1]['aria-pressed'], 'true');
page.nodes.resetAppearance.onclick();
assert.strictEqual(page.root.dataset.theme, 'light');
assert.strictEqual(page.root.dataset.accent, 'violet');
page.system.matches = true;
page.system.change();
assert.strictEqual(page.root.dataset.theme, 'dark');
page.themes[1].onclick();
page.system.change();
assert.strictEqual(page.root.dataset.theme, 'light');
storage['mnn-studio-appearance-v1'] = '{invalid';
page.window.storageChange({key: 'mnn-studio-appearance-v1'});
assert.strictEqual(page.root.dataset.theme, 'dark');
assert.strictEqual(page.root.dataset.accent, 'violet');
page.nodes.doneAppearance.onclick();
assert.strictEqual(page.nodes.appearanceDialog.open, false);
console.log('Web appearance settings: PASS');
