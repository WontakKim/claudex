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
    this._children = [];
    this.listeners = new Map();
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
    this.offsetWidth = 1000;
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
    this._innerHTML += String(value);
    this.ownerDocument.registerIds(this._innerHTML);
  }

  appendChild(child) {
    child.parentNode = this;
    child.isConnected = true;
    this._children.push(child);
    if (child.id) this.ownerDocument.elements.set(child.id, child);
    return child;
  }

  remove() {
    this.isConnected = false;
    if (this.id) this.ownerDocument.elements.delete(this.id);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  dispatchEvent(event) {
    if (!event || !event.type) throw new Error("event type is required");
    if (!event.target) event.target = this;
    event.currentTarget = this;
    event.defaultPrevented = Boolean(event.defaultPrevented);
    event.propagationStopped = Boolean(event.propagationStopped);
    event.preventDefault = event.preventDefault || function() { this.defaultPrevented = true; };
    event.stopPropagation = event.stopPropagation || function() { this.propagationStopped = true; };
    (this.listeners.get(event.type) || []).slice().forEach((listener) => {
      listener.call(this, event);
    });
    if (event.bubbles && !event.propagationStopped && this.parentNode) {
      this.parentNode.dispatchEvent(event);
    }
    return !event.defaultPrevented;
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  click() {
    if (!this.disabled) this.dispatchEvent({type: "click", bubbles: true});
  }

  closest(selector) {
    let element = this;
    while (element) {
      if (selector === "button" && element.tagName === "BUTTON") return element;
      if (selector === '[role="option"]' && element.attributes
        && element.attributes.role === "option") return element;
      element = element.parentNode;
    }
    return null;
  }

  querySelector() { return null; }
  querySelectorAll() { return []; }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "id") this.id = value;
    if (name === "class") this.className = String(value);
    if (name.startsWith("data-")) {
      const key = name.slice(5).replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
      this.dataset[key] = String(value);
    }
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
    return {left: 0, top: 0, right: 1000, bottom: 700, width: 1000, height: 700};
  }

  // The quick-add chooser measures itself against the board and the apply
  // bar; these layout helpers exist so production code can call them as-is.
  scrollIntoView() {}

  select() {}

  replaceChildren(...children) {
    this._children = [];
    children.forEach((child) => this.appendChild(child));
  }
  // Real browsers expose an HTMLCollection (indexable + length, no forEach).
  get children() {
    const list = this._children;
    return new Proxy(
      {},
      {
        get: (target, prop) => {
          if (prop === "length") return list.length;
          if (prop === "item") return (i) => list[i] || null;
          if (typeof prop === "string" && /^[0-9]+$/.test(prop)) return list[Number(prop)];
          return undefined;
        },
      }
    );
  }

  contains(candidate) {
    let element = candidate;
    while (element) {
      if (element === this) return true;
      element = element.parentNode;
    }
    return false;
  }
}

class FakeDocument {
  constructor(html) {
    this.elements = new Map();
    this.fallbacks = new Map();
    this.allElements = [];
    this.body = new FakeElement(this, "body");
    this.activeElement = this.body;
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

  querySelectorAll() {
    return [];
  }
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

function createDeferred() {
  let resolve;
  const promise = new Promise((resolvePromise) => { resolve = resolvePromise; });
  return {promise, resolve};
}

function requireQuickAddElements(document) {
  for (const elementId of [
    "node-add", "model-picker", "provider-select", "model-query",
    "picker-state", "model-options", "picker-close",
  ]) {
    if (!document.getElementById(elementId)) {
      throw new Error(`missing approved quick-add element #${elementId}`);
    }
  }
}

function optionId(element) {
  return element.children[0] ? element.children[0].textContent : "";
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
    window: {addEventListener() {}, prompt: () => null, innerWidth: 1280, innerHeight: 800},
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
  const credentialMarker = "safe-synthetic-credential";
  const configuredName = "openai-by-name";
  const connectedName = "anthropic-by-name";
  const unusedName = "unused-provider";
  const errorName = "error-provider";
  const connectedCatalog = createDeferred();
  context.jfetch = (url, options) => {
    requests.push({url, options: options || null});
    if (url === `/admin/providers/custom/${connectedName}/models`) {
      return connectedCatalog.promise;
    }
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

  requireQuickAddElements(document);
  const boardEl = document.getElementById("board");
  const nodeAddButton = document.getElementById("node-add");
  const modelPicker = document.getElementById("model-picker");
  const providerSelect = document.getElementById("provider-select");
  const modelQuery = document.getElementById("model-query");
  const pickerState = document.getElementById("picker-state");
  const modelOptions = document.getElementById("model-options");
  const pickerClose = document.getElementById("picker-close");
  const applyBar = document.querySelector(".applybar");
  // Realistic client rects so the fixed-position chooser geometry code runs
  // against meaningful numbers; the apply bar collapses to zero height so it
  // never clips the bounds while its visibility is CSS-driven.
  boardEl.getBoundingClientRect = () => ({left: 0, top: 0, right: 1000, bottom: 700, width: 1000, height: 700});
  nodeAddButton.getBoundingClientRect = () => ({left: 880, top: 640, right: 990, bottom: 672, width: 110, height: 32});
  applyBar.getBoundingClientRect = () => ({left: 0, top: 700, right: 1000, bottom: 700, width: 1000, height: 0});
  const suggestionIds = () => Array.from(modelOptions.children).map(optionId);

  context.DIR.locked = true;
  context.renderChrome();
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

  const lockedControls = {
    addButtonDisabled: nodeAddButton.disabled,
    providerSelectDisabled: providerSelect.disabled,
    modelQueryDisabled: modelQuery.disabled,
    pickerCloseDisabled: pickerClose.disabled,
  };
  nodeAddButton.click();
  const nativeLockGuards = {
    pickerHidden: modelPicker.hidden,
    targetsEmpty: context.addedTargets.length === 0,
  };
  boardEl.dispatchEvent({type: "dblclick", clientY: 300});
  const forcedDblclickHidden = modelPicker.hidden;
  boardEl.dispatchEvent({type: "contextmenu", clientY: 300});
  const forcedContextmenuHidden = modelPicker.hidden;
  boardEl.dispatchEvent({type: "keydown", key: "A", shiftKey: true});
  const forcedShortcutHidden = modelPicker.hidden;
  const forcedLockGuards = {
    dblclickHidden: forcedDblclickHidden,
    contextmenuHidden: forcedContextmenuHidden,
    shortcutHidden: forcedShortcutHidden,
    targetsEmpty: context.addedTargets.length === 0,
  };

  context.DIR.locked = false;
  context.renderChrome();
  const unlockedControls = {
    addButtonEnabled: !nodeAddButton.disabled,
    providerSelectEnabled: !providerSelect.disabled,
    modelQueryEnabled: !modelQuery.disabled,
    pickerCloseEnabled: !pickerClose.disabled,
  };

  nodeAddButton.click();
  const opened = {
    pickerHidden: modelPicker.hidden,
    entry: modelPicker.dataset.entry,
    queryEmpty: modelQuery.value === "",
    queryFocused: document.activeElement === modelQuery,
    stateHidden: pickerState.hidden,
  };

  // Bottom-anchor regression: the browser paints a fractional box height
  // (248.5) while offsetHeight reports the rounded integer (249). Anchoring
  // on the rounded integer compounds rounding into a wider-than-contract
  // gap; placement must use the actual layout box. The stub reports the
  // painted rect from the assigned style, as a real layout engine would.
  modelPicker.offsetWidth = 713;
  modelPicker.offsetHeight = 249;
  modelPicker.getBoundingClientRect = () => {
    const l = parseFloat(modelPicker.style.left) || 0;
    const t = parseFloat(modelPicker.style.top) || 0;
    return {left: l, top: t, right: l + 713, bottom: t + 248.5, width: 713, height: 248.5};
  };
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  const fractionalGeometry = {
    styleTop: modelPicker.style.top,
    paintedBottom: (parseFloat(modelPicker.style.top) || 0) + 248.5,
    gap: 640 - ((parseFloat(modelPicker.style.top) || 0) + 248.5),
    gapExactlyEight: Math.abs(640 - ((parseFloat(modelPicker.style.top) || 0) + 248.5) - 8) < 1e-6,
    rightEdgeDelta: 990 - ((parseFloat(modelPicker.style.left) || 0) + 713),
  };

  modelQuery.value = "stale-provider-value";
  providerSelect.value = connectedName;
  providerSelect.dispatchEvent({type: "change", bubbles: true});
  const providerSwitch = {
    selectedProvider: context.addProvider,
    queryCleared: modelQuery.value === "",
    queryFocused: document.activeElement === modelQuery,
    suggestionsBefore: suggestionIds(),
  };
  connectedCatalog.resolve({
    ok: true,
    status: 200,
    body: {models: ["late-model", "late:model:variant"]},
  });
  await Promise.resolve();
  await Promise.resolve();
  const catalogAfterResolution = {
    selectedProvider: context.addProvider,
    suggestions: suggestionIds(),
    stateHidden: pickerState.hidden,
  };

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

  context.DIR.LIVE = {sonnet: "codex:existing-model"};
  context.DIR.mapping = {sonnet: "codex:existing-model"};
  context.DIR.sources = [];
  context.DIR.targets = [];
  context.addedTargets = [];
  context.render();

  providerSelect.value = configuredName;
  providerSelect.dispatchEvent({type: "change", bubbles: true});
  const cataloglessState = {
    stateHidden: pickerState.hidden,
    stateText: pickerState.textContent,
  };
  const manualQueryAvailable = !modelQuery.disabled;
  const manualProviderOptionPresent = Array.from(providerSelect.children)
    .some((option) => option.value === configuredName);

  modelQuery.value = "   ";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const blankIgnored = context.addedTargets.length === 0;

  const colonTarget = `${configuredName}:manual:model:alpha`;
  modelQuery.value = "manual:model:alpha";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const afterManualEnter = context.addedTargets.slice();
  const closedAfterCommit = modelPicker.hidden;

  const enterTarget = `${configuredName}:manual-enter-beta`;
  nodeAddButton.click();
  modelQuery.value = "manual-enter-beta";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const afterEnter = context.addedTargets.slice();

  nodeAddButton.click();
  modelQuery.value = "manual:model:alpha";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const afterDuplicate = context.addedTargets.slice();
  const duplicateStaysOpen = !modelPicker.hidden;

  modelQuery.value = "manual-ime";
  modelQuery.dispatchEvent({type: "compositionstart", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const composingBlocked = context.addedTargets.length === afterDuplicate.length;
  modelQuery.dispatchEvent({type: "compositionend", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", isComposing: true});
  const isComposingBlocked = context.addedTargets.length === afterDuplicate.length;
  modelQuery.value = "manual-tab";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Tab", bubbles: true});
  const tabNeverAdds = context.addedTargets.length === afterDuplicate.length;

  // Arrow navigation moves the active suggestion and Enter commits it.
  providerSelect.value = connectedName;
  providerSelect.dispatchEvent({type: "change", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "ArrowDown", bubbles: true});
  const arrowActiveDescendant = modelQuery.getAttribute("aria-activedescendant");
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const arrowCommitTarget = context.addedTargets[context.addedTargets.length - 1];

  // Entry-path exclusions: nodes never invoke blank-canvas creation, and the
  // native menu stays available on them.
  document.getElementById("layer").parentNode = boardEl;
  const nodeStub = document.createElement("div");
  nodeStub.className = "node tgt";
  document.getElementById("layer").appendChild(nodeStub);
  nodeStub.dispatchEvent({type: "dblclick", bubbles: true});
  const nodeDblclickHidden = modelPicker.hidden;
  const nodeContextmenuNative = nodeStub.dispatchEvent({type: "contextmenu", bubbles: true});

  // Contextual staging: reset the draft through the real Discard path, then
  // capture the invocation y through the live pan and zoom (clientY 320 with
  // pan.y 20 and zoom 2 lands at graph y 150); a second node near the same
  // lane is nudged clear of the first.
  document.getElementById("discardbtn").click();
  context.pan = {x: 40, y: 20};
  context.zoom = 2;
  const blankContextmenuPrevented = !boardEl.dispatchEvent({type: "contextmenu", clientY: 320});
  const contextmenuEntry = modelPicker.dataset.entry;
  modelQuery.value = "manual-at-y";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const contextualFirst = context.DIR.targets.map(({id, y}) => ({id, y}));
  boardEl.dispatchEvent({type: "dblclick", clientY: 330});
  const dblclickEntry = modelPicker.dataset.entry;
  modelQuery.value = "manual-second-y";
  modelQuery.dispatchEvent({type: "input", bubbles: true});
  modelQuery.dispatchEvent({type: "keydown", key: "Enter", bubbles: true});
  const contextualSecondY = context.DIR.targets
    .find((target) => target.id === `${connectedName}:manual-second-y`)?.y;
  context.pan = {x: 0, y: 0};
  context.zoom = 1;

  // Right-button pointer starts never begin a pan.
  context.panning = null;
  boardEl.dispatchEvent({type: "pointerdown", button: 2});
  const rightButtonNoPan = context.panning === null;
  boardEl.dispatchEvent({type: "pointerdown", button: 0});
  const leftButtonPan = context.panning !== null;
  context.panning = null;

  const stagingState = {
    selectedProvider: context.addProvider,
    blankIgnored,
    afterManualEnter,
    closedAfterCommit,
    afterEnter,
    afterDuplicate,
    duplicateStaysOpen,
    exactTargets: context.DIR.targets.map((target) => ({id: target.id, y: target.y})),
    mapping: Object.assign({}, context.DIR.mapping),
    counts: context.draftCounts(),
    isDirty: applyBar.classList.contains("dirty"),
    colonTarget,
    enterTarget,
  };

  context.DIR.mapping.sonnet = colonTarget;
  context.render();
  const dirtyBeforeDiscard = {
    mapping: Object.assign({}, context.DIR.mapping),
    counts: context.draftCounts(),
    isDirty: applyBar.classList.contains("dirty"),
  };
  document.getElementById("discardbtn").click();
  const discardedState = {
    mapping: Object.assign({}, context.DIR.mapping),
    addedTargets: context.addedTargets.slice(),
    targets: context.DIR.targets.map((target) => ({id: target.id, y: target.y})),
    counts: context.draftCounts(),
    isDirty: applyBar.classList.contains("dirty"),
  };

  // Leaving the Router tab closes the chooser, and a late catalog callback
  // for a still-selected provider must not reopen it.
  nodeAddButton.click();
  context.setTab("settings");
  const closedOnTabLeave = modelPicker.hidden;
  context.configureCustomProviders([
    {
      name: connectedName,
      family: "anthropic_compatible",
      wire_kind: "responses",
      catalog_available: true,
    },
  ]);
  await Promise.resolve();
  await Promise.resolve();
  const lateCallbackNoReopen = modelPicker.hidden;
  context.setTab("map");

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

  const mcpConnectStates = {};
  context.renderMcpConnect({ok: true, exit_code: 0, output: "added\n"});
  mcpConnectStates.passed = {
    className: document.getElementById("mcp-connect-result").className,
    text: document.getElementById("mcp-connect-result").textContent,
    hidden: document.getElementById("mcp-connect-result").hidden,
  };
  context.renderMcpConnect({ok: false, exit_code: 1, output: "registration failed\n"});
  mcpConnectStates.failed = {
    className: document.getElementById("mcp-connect-result").className,
    text: document.getElementById("mcp-connect-result").textContent,
    hidden: document.getElementById("mcp-connect-result").hidden,
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
    status: "running",
    detail: "no compatible browser found; installing Playwright Chromium",
    output: "",
    error: null,
  });
  gptProLoginStates.installing = captureGptProLogin();
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
      mcpConnectStates,
      gptProLoginStates,
      gptProDoctorStates,
      picker: {
        lockedControls,
        nativeLockGuards,
        forcedLockGuards,
        unlockedControls,
        opened,
        fractionalGeometry,
        providerSwitch,
        catalogAfterResolution,
        cataloglessState,
        stagingState,
        contextual: {
          contextmenuEntry,
          first: contextualFirst,
          dblclickEntry,
          secondId: `${connectedName}:manual-second-y`,
          secondY: contextualSecondY,
        },
        guards: {
          nodeDblclickHidden,
          nodeContextmenuNative,
          blankContextmenuPrevented,
          rightButtonNoPan,
          leftButtonPan,
        },
        ime: {
          composingBlocked,
          isComposingBlocked,
          tabNeverAdds,
        },
        arrows: {
          activeDescendant: arrowActiveDescendant,
          commitTarget: arrowCommitTarget,
        },
        tabLeave: {
          closedOnTabLeave,
          lateCallbackNoReopen,
        },
        dirtyBeforeDiscard,
        discardedState,
      },
      manual: {
        providerOptionPresent: manualProviderOptionPresent,
        modelQueryAvailable: manualQueryAvailable,
        targetAccepted: afterManualEnter.includes(colonTarget),
        stateLineShown: !cataloglessState.stateHidden,
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
