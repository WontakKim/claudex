// Runs the supplied production probes with inert browser collaborators only.
const fs = require('node:fs');
const vm = require('node:vm');

async function download(input) {
  const observation = {requests: [], reads: 0, cancellations: 0, pendingTimers: 0};
  const context = vm.createContext({
    location: {origin: input.pageOrigin},
    Uint8Array,
    btoa: (binary) => Buffer.from(binary, 'binary').toString('base64'),
    AbortController: class {
      signal = {aborted: false};
      abort() {
        this.signal.aborted = true;
        observation.cancellations += 1;
      }
    },
    setTimeout: () => { observation.pendingTimers += 1; return 1; },
    clearTimeout: () => { observation.pendingTimers -= 1; },
    fetch: async (url, options) => {
      observation.requests.push({url, redirect: options.redirect,
        credentials: options.credentials, headers: options.headers});
      if (input.redirect && options.redirect === 'error') {
        throw new TypeError('Redirect refused');
      }
      let nextChunk = 0;
      let isCancelled = false;
      const cancel = async () => {
        isCancelled = true;
        observation.cancellations += 1;
      };
      return {
        status: 200,
        headers: new Headers(input.responseHeaders),
        redirected: Boolean(input.redirect),
        url,
        body: {cancel, getReader: () => ({cancel, read: async () => {
          observation.reads += 1;
          if (options.signal.aborted) throw new Error('Read after abort');
          if (isCancelled) return {done: true};
          if (nextChunk === input.chunks.length) return {done: true};
          return {done: false, value: new Uint8Array(input.chunks[nextChunk++])};
        }})},
      };
    },
  });
  const probe = vm.runInContext(`(${input.source})`, context, {timeout: 1000});
  observation.result = await probe(input.args);
  return observation;
}

class Element {
  constructor(fixture, parentElement = null) {
    this.attributes = fixture.attributes || {};
    this.text = fixture.text || '';
    this.parentElement = parentElement;
    this.children = (fixture.children || []).map((child) => new Element(child, this));
  }
  get textContent() {
    return this.text + this.children.map((child) => child.textContent).join('');
  }
  getAttribute(name) { return this.attributes[name] ?? null; }
  hasAttribute(name) { return Object.hasOwn(this.attributes, name); }
  matches(selector) {
    return selector.split(',').some((part) => {
      const match = part.trim().match(/^\[([^\s$=\]]+)(?:(\$?=)["']([^"']*)["'])?\]$/);
      if (!match) throw new Error(`Unsupported fixture selector: ${part}`);
      const [, name, operator, value] = match;
      if (!this.hasAttribute(name)) return false;
      return !operator || (operator === '$='
        ? this.attributes[name].endsWith(value) : this.attributes[name] === value);
    });
  }
  closest(selector) {
    for (let node = this; node; node = node.parentElement) {
      if (node.matches(selector)) return node;
    }
    return null;
  }
  querySelectorAll(selector) {
    return this.children.flatMap((child) => [
      ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
}

function roles(input) {
  const context = vm.createContext({document: new Element(input.tree)});
  const results = {};
  for (const [name, source] of Object.entries(input.sources)) {
    const probe = vm.runInContext(`(${source})`, context, {timeout: 1000});
    results[name] = probe(input.args);
  }
  return results;
}

(async () => {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  if (input.mode === 'download') return download(input);
  if (input.mode === 'roles') return roles(input);
  throw new Error(`Unknown probe mode: ${input.mode}`);
})().then((result) => console.log(JSON.stringify(result))).catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
