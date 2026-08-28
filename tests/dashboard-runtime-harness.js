"use strict";

const fs = require("fs");
const vm = require("vm");

const [javascriptPath, htmlPath] = process.argv.slice(2);
if (!javascriptPath || !htmlPath) {
  throw new Error("dashboard JavaScript and HTML paths are required");
}

class FakeClassList {
  constructor(element) {
    this.element = element;
  }

  values() {
    return new Set(this.element.className.split(/\s+/).filter(Boolean));
  }

  write(values) {
    this.element.className = Array.from(values).join(" ");
  }

  add(...names) {
    const values = this.values();
    names.forEach((name) => values.add(name));
    this.write(values);
  }

  remove(...names) {
    const values = this.values();
    names.forEach((name) => values.delete(name));
    this.write(values);
  }

  toggle(name, force) {
    const values = this.values();
    const shouldAdd = force === undefined ? !values.has(name) : Boolean(force);
    if (shouldAdd) values.add(name);
    else values.delete(name);
    this.write(values);
    return shouldAdd;
  }

  contains(name) {
    return this.values().has(name);
  }
}

class FakeElement {
  constructor(document, tagName = "div") {
    this.ownerDocument = document;
    this.ownerDocument.allElements.push(this);
    this.tagName = tagName.toUpperCase();
    this._id = "";
    this._innerHTML = "";
    this.className = "";
    this.classList = new FakeClassList(this);
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.children = [];
    this.parentNode = null;
    this.textContent = "";
    this.title = "";
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.checked = false;
    this.isConnected = true;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this.clientWidth = 1000;
    this.offsetTop = 0;
    this.offsetHeight = 40;
  }

  set id(value) {
    this._id = String(value);
    if (this._id) this.ownerDocument.elements.set(this._id, this);
  }

  get id() {
    return this._id;
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this.ownerDocument.registerIds(this._innerHTML);
  }

  get innerHTML() {
    return this._innerHTML;
  }

  insertAdjacentHTML(_position, value) {
    this.innerHTML = this._innerHTML + String(value);
  }

  appendChild(child) {
    child.parentNode = this;
    child.isConnected = true;
    this.children.push(child);
    if (child.id) this.ownerDocument.elements.set(child.id, child);
    return child;
  }

  remove() {
    this.isConnected = false;
    if (this.id) this.ownerDocument.elements.delete(this.id);
  }

  addEventListener() {}
  focus() {}
  click() {}
  closest() { return null; }
  querySelector() { return null; }
  querySelectorAll() { return []; }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "id") this.id = value;
    if (name === "class") this.className = String(value);
  }

  getAttribute(name) {
    if (name === "id") return this.id || null;
    if (name === "class") return this.className || null;
    return Object.prototype.hasOwnProperty.call(this.attributes, name)
      ? this.attributes[name]
      : null;
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  getBoundingClientRect() {
    return {left: 0, right: 0, top: 0, bottom: 0, width: 1000, height: 700};
  }
}

class FakeDocument {
  constructor(html) {
    this.elements = new Map();
    this.fallbacks = new Map();
    this.allElements = [];
    this.body = new FakeElement(this, "body");
    this.registerIds(html);
  }

  registerIds(html) {
    for (const match of String(html).matchAll(/\bid="([^"]+)"/g)) {
      if (!this.elements.has(match[1])) {
        const element = new FakeElement(this);
        element.id = match[1];
      }
    }
  }

  getElementById(id) {
    return this.elements.get(String(id)) || null;
  }

  createElement(tagName) {
    return new FakeElement(this, tagName);
  }

  createElementNS(_namespace, tagName) {
    return new FakeElement(this, tagName);
  }

  querySelector(selector) {
    if (selector === ".applybar") {
      if (!this.fallbacks.has(selector)) {
        this.fallbacks.set(selector, new FakeElement(this));
      }
      return this.fallbacks.get(selector);
    }
    return null;
  }

  querySelectorAll() { return []; }
  addEventListener() {}
  elementFromPoint() { return null; }

  snapshot() {
    return this.allElements.map((element) => [
      element.id,
      element.className,
      element.innerHTML,
      element.textContent,
      element.title,
      element.value,
      JSON.stringify(element.attributes),
      JSON.stringify(element.dataset),
      JSON.stringify(element.style),
    ].join("|")).join("\n");
  }
}

function createStorageRecorder() {
  const writes = [];
  return {
    writes,
    storage: {
      setItem(key, value) { writes.push([String(key), String(value)]); },
      getItem() { return null; },
      removeItem() {},
      clear() {},
    },
  };
}

function statusText(document, name) {
  const box = document.getElementById(`${name}-stat`);
  const line = document.getElementById(`${name}-statline`);
  if (!box || !line) throw new Error(`missing rendered status elements for ${name}`);
  return {
    className: box.className,
    text: line.innerHTML.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim(),
    title: box.title,
  };
}

async function main() {
  const html = fs.readFileSync(htmlPath, "utf8");
  const document = new FakeDocument(html);
  const localStorageRecorder = createStorageRecorder();
  const sessionStorageRecorder = createStorageRecorder();
  const rawFetchRequests = [];
  const consoleCalls = [];
  const consoleRecorder = {};
  for (const level of ["log", "info", "warn", "error", "debug"]) {
    consoleRecorder[level] = (...args) => consoleCalls.push([level, ...args]);
  }

  const context = {
    document,
    location: {host: "127.0.0.1:8787", hash: ""},
    history: {replaceState() {}},
    navigator: {clipboard: {writeText: async () => {}}},
    CSS: {escape: (value) => String(value)},
    performance: {now: () => 0},
    window: {addEventListener() {}, prompt: () => null},
    localStorage: localStorageRecorder.storage,
    sessionStorage: sessionStorageRecorder.storage,
    console: consoleRecorder,
    fetch: async (url, options) => {
      rawFetchRequests.push({url: String(url), options: options || null});
      return {ok: true, status: 200, json: async () => ({})};
    },
    setTimeout: () => 1,
    clearTimeout() {},
    setInterval: () => 1,
    clearInterval() {},
    Date,
    JSON,
    Math,
    Promise,
    RegExp,
    String,
    Object,
    Array,
    Number,
    Boolean,
    encodeURIComponent,
  };
  vm.createContext(context);

  let source = fs.readFileSync(javascriptPath, "utf8");
  if (!/boot\(\);\s*$/.test(source)) {
    throw new Error("dashboard JavaScript no longer ends with boot();");
  }
  source = source.replace(/boot\(\);\s*$/, "");
  vm.runInContext(source, context, {filename: javascriptPath});

  const requests = [];
  context.jfetch = (url, options) => {
    requests.push({url, options: options || null});
    if (url.startsWith("/admin/providers/custom/")) {
      return Promise.resolve({ok: true, status: 200, body: {models: ["suggested-model"]}});
    }
    if (url === "/admin/usage") {
      const unavailable = {
        status: "unavailable",
        error: "generic usage unavailable",
        updated_at: 1,
      };
      return Promise.resolve({
        ok: true,
        status: 200,
        body: {
          claude: unavailable,
          codex: unavailable,
          kimi: unavailable,
          grok: unavailable,
        },
      });
    }
    return Promise.resolve({ok: false, status: 500, body: {}});
  };

  const credentialMarker = "safe-synthetic-credential";
  const configuredName = "openai-by-name";
  const connectedName = "anthropic-by-name";
  const unusedName = "unused-provider";
  const errorName = "error-provider";
  context.configureCustomProviders([
    {
      name: connectedName,
      family: "anthropic_compatible",
      wire_kind: "responses",
      catalog_available: true,
      api_key: credentialMarker,
    },
    {
      name: configuredName,
      family: "openai_compatible",
      wire_kind: "anthropic_messages",
      catalog_available: false,
      api_key: credentialMarker,
    },
    {
      name: unusedName,
      family: "anthropic_compatible",
      wire_kind: "responses",
      catalog_available: true,
    },
    {
      name: errorName,
      family: "openai_compatible",
      wire_kind: "anthropic_messages",
      catalog_available: false,
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();

  context.renderProviderCards({
    providers: {
      codex: {status: "ok", auth_mode: "api_key"},
      kimi: {status: "ok"},
      grok: {status: "ok", auth_mode: "api_key"},
      [configuredName]: {status: "ok", required: true},
      [connectedName]: {status: "ok", required: true},
      [unusedName]: {
        status: "error",
        required: false,
        detail: "generic optional check failed",
      },
      [errorName]: {
        status: "error",
        required: true,
        detail: "generic binding failed",
      },
    },
  });

  context.addProvider = configuredName;
  context.renderAddForm();
  const manualInput = document.getElementById("add-model");
  const manualInputAvailable = manualInput !== null && !manualInput.disabled;
  manualInput.value = "manual-model-alpha";
  context.addTargetNode();

  context.fetchUsage();
  await Promise.resolve();
  await Promise.resolve();

  const gptProSessionStates = {};
  context.renderGptProSession({
    exists: true, has_auth_cookie: true, expired: false, valid: true,
    expires_in_seconds: 8 * 24 * 60 * 60,
  });
  gptProSessionStates.valid = statusText(document, "gptpro-session");
  context.renderGptProSession({
    exists: true, has_auth_cookie: true, expired: false, valid: true,
    expires_in_seconds: 7 * 24 * 60 * 60,
  });
  gptProSessionStates.expiring = statusText(document, "gptpro-session");
  context.renderGptProSession({
    exists: true, has_auth_cookie: true, expired: true, valid: false,
    expires_in_seconds: 0,
  });
  gptProSessionStates.expired = statusText(document, "gptpro-session");
  context.renderGptProSession({
    exists: false, has_auth_cookie: false, expired: null, valid: false,
    expires_in_seconds: null,
  });
  gptProSessionStates.missing = statusText(document, "gptpro-session");

  const mcpInfoStates = {};
  context.renderMcpInfo({
    endpoint: "http://127.0.0.1:8787/mcp",
    auth_required: false,
  });
  mcpInfoStates.open = {
    command: document.getElementById("mcp-connect-command").textContent,
    endpoint: document.getElementById("mcp-endpoint").textContent,
    authHintHidden: document.getElementById("mcp-auth-hint").hidden,
  };
  context.renderMcpInfo({
    endpoint: "http://127.0.0.1:9000/mcp",
    auth_required: true,
  });
  mcpInfoStates.authenticated = {
    command: document.getElementById("mcp-connect-command").textContent,
    endpoint: document.getElementById("mcp-endpoint").textContent,
    authHintHidden: document.getElementById("mcp-auth-hint").hidden,
  };

  let gptProSessionRefreshes = 0;
  context.fetchGptProSession = () => { gptProSessionRefreshes += 1; };
  const gptProLoginStates = {};
  const captureGptProLogin = () => ({
    buttonText: document.getElementById("gptpro-login-btn").textContent,
    buttonDisabled: document.getElementById("gptpro-login-btn").disabled,
    detail: document.getElementById("gptpro-login-detail").textContent,
    polling: Boolean(context.loginPolling),
  });
  context.renderGptProLogin({status: "idle"});
  gptProLoginStates.idle = captureGptProLogin();
  context.renderGptProLogin({
    status: "running",
    detail: "sign in to ChatGPT in the opened browser",
    output: "",
    error: null,
  });
  gptProLoginStates.running = captureGptProLogin();
  context.renderGptProLogin({
    status: "succeeded",
    detail: "verifying the saved ChatGPT session",
    output: "saved and verified the gptpro session\n",
    error: null,
  });
  gptProLoginStates.terminal = captureGptProLogin();
  gptProLoginStates.sessionRefreshes = gptProSessionRefreshes;

  const gptProDoctorStates = {};
  context.renderGptProDoctor({ok: true, exit_code: 0, output: "doctor passed\n"});
  gptProDoctorStates.passed = {
    className: document.getElementById("gptpro-doctor-output").className,
    text: document.getElementById("gptpro-doctor-output").textContent,
    hidden: document.getElementById("gptpro-doctor-output").hidden,
  };
  context.renderGptProDoctor({ok: false, exit_code: 1, output: "doctor failed\n"});
  gptProDoctorStates.failed = {
    className: document.getElementById("gptpro-doctor-output").className,
    text: document.getElementById("gptpro-doctor-output").textContent,
    hidden: document.getElementById("gptpro-doctor-output").hidden,
  };

  const requestSnapshot = JSON.stringify(requests);
  const domSnapshot = document.snapshot();
  const credentialLeak = [
    requestSnapshot,
    JSON.stringify(rawFetchRequests),
    JSON.stringify(context.location),
    domSnapshot,
    JSON.stringify(localStorageRecorder.writes),
    JSON.stringify(sessionStorageRecorder.writes),
    JSON.stringify(consoleCalls),
  ].some((snapshot) => snapshot.includes(credentialMarker));

  const cards = {
    configured: document.getElementById(`card-${configuredName}`).innerHTML,
    connected: document.getElementById(`card-${connectedName}`).innerHTML,
  };

  const output = {credentialLeak};
  if (!credentialLeak) {
    Object.assign(output, {
      names: {configuredName, connectedName, unusedName, errorName},
      jfetchRequests: requests.map((request) => ({
        url: String(request.url),
        options: request.options,
      })),
      rawFetchRequests,
      statuses: {
        configured: statusText(document, configuredName),
        connected: statusText(document, connectedName),
        unused: statusText(document, unusedName),
        error: statusText(document, errorName),
      },
      cards,
      gptProSessionStates,
      mcpInfoStates,
      gptProLoginStates,
      gptProDoctorStates,
      manual: {
        providerButtonPresent: document.getElementById("add-prov").innerHTML.includes(configuredName),
        modelInputAvailable: manualInputAvailable,
        targetAccepted: context.addedTargets.includes(`${configuredName}:manual-model-alpha`),
        inputCleared: manualInput.value === "",
      },
      storageWrites: {
        local: localStorageRecorder.writes,
        session: sessionStorageRecorder.writes,
      },
      consoleCalls,
    });
  }
  process.stdout.write(JSON.stringify(output));
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
