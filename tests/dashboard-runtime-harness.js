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
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return {promise, resolve, reject};
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

function codexCapabilitiesEnvelope(tier = null, models = {}) {
  return {ok: true, status: 200, body: {
    configured_tier: tier, env_locked: false, status: "ok", reason: null, fetched_at: 1790000000, models,
  }};
}

function localTokenRejection(message = "Missing or invalid bearer token") {
  return {error: {message, type: "authentication_error", param: null, code: "invalid_api_key"}};
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

  const initialLocalHeroHtml = document.getElementById("local-body").innerHTML;
  const requests = [];
  const credentialMarker = "safe-synthetic-credential";
  const configuredName = "openai-by-name";
  const connectedName = "anthropic-by-name";
  const unusedName = "unused-provider";
  const errorName = "error-provider";
  const connectedCatalog = createDeferred();
  const dashboardJfetch = context.jfetch;
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

  const routingEnvelopeValidation = {
    included: context.isRoutingEnvelope({mode: "balanced", env_locked: false, include_local_login: true}),
    excluded: context.isRoutingEnvelope({mode: "balanced", env_locked: false, include_local_login: false}),
    missing: context.isRoutingEnvelope({mode: "balanced", env_locked: false}),
    invalid: context.isRoutingEnvelope({mode: "balanced", env_locked: false, include_local_login: "false"}),
  };
  const localHeroStates = {};
  context.ACCT.local = {email: "local@example.com", organizationName: "Local organization"};
  context.ACCT.localLoaded = true;
  context.ACCT.localUsage = {status: "error"};
  context.renderLocalHero();
  localHeroStates.unknown = document.getElementById("local-body").innerHTML;
  for (const mode of ["disabled", "fallback", "balanced"]) {
    for (const includeLocalLogin of [true, false]) {
      context.renderRoutingState({mode, env_locked: false, include_local_login: includeLocalLogin});
      localHeroStates[`${mode}-${includeLocalLogin}`] = document.getElementById("local-body").innerHTML;
    }
  }
  context.renderRoutingState({mode: "balanced", env_locked: false, include_local_login: true});
  context.ROUTING.draft = "disabled";
  context.renderRouting();
  context.renderLocalHero();
  localHeroStates.unappliedDraft = document.getElementById("local-body").innerHTML;
  context.ACCT.local = null;
  context.renderLocalHero();
  localHeroStates.noLoginIncluded = document.getElementById("local-body").innerHTML;
  context.renderRoutingState({mode: "fallback", env_locked: false, include_local_login: true});
  localHeroStates.noLoginExcluded = document.getElementById("local-body").innerHTML;

  const localHeroLoadOrder = {};
  const previousJfetch = context.jfetch;
  const pendingLocalLogin = createDeferred();
  context.ACCT.local = null;
  context.ACCT.localLoaded = false;
  document.getElementById("local-body").innerHTML = initialLocalHeroHtml;
  localHeroLoadOrder.initial = document.getElementById("local-body").innerHTML;
  context.renderRoutingState({mode: "balanced", env_locked: false, include_local_login: true});
  localHeroLoadOrder.routingFirst = document.getElementById("local-body").innerHTML;
  context.jfetch = (url) => {
    if (url === "/admin/providers/claude/local") return pendingLocalLogin.promise;
    return Promise.resolve({ok: true, body: {accounts: [], members: []}});
  };
  context.fetchAccounts();
  await new Promise((resolve) => setImmediate(resolve));
  localHeroLoadOrder.pendingLocal = document.getElementById("local-body").innerHTML;
  pendingLocalLogin.resolve({ok: true, body: {local: {email: "loaded@example.com"}}});
  await new Promise((resolve) => setImmediate(resolve));
  localHeroLoadOrder.loaded = document.getElementById("local-body").innerHTML;
  context.renderRoutingState({mode: "disabled", env_locked: false, include_local_login: true});
  localHeroLoadOrder.repaintAfterSuccess = document.getElementById("local-body").innerHTML;
  context.ACCT.localLoaded = false;
  context.jfetch = (url) => Promise.resolve(url === "/admin/providers/claude/local"
    ?{ok: false, status: 503, body: {error: {message: "local login unavailable"}}}
    :{ok: true, body: {accounts: [], members: []}});
  context.fetchAccounts();
  await new Promise((resolve) => setImmediate(resolve));
  context.renderRoutingState({mode: "balanced", env_locked: false, include_local_login: true});
  localHeroLoadOrder.repaintAfterLocalError = document.getElementById("local-body").innerHTML;
  context.jfetch = previousJfetch;

  const gptProConcurrencyStates = {};
  const concurrencyJfetch = context.jfetch;
  const concurrencySelect = document.getElementById("gptpro-concurrency");
  const concurrencyRequests = [];
  const concurrencyResponses = [];
  const flushConcurrency = () => new Promise((resolve) => setImmediate(resolve));
  const concurrencyEnvelope = (limit = 3, locked = false) => ({
    ok: true, status: 200, body: {max_concurrent_asks: limit, env_locked: locked},
  });
  context.jfetch = (url, options) => {
    if (url !== "/admin/settings/gptpro") {
      return Promise.resolve({ok: false, status: 500, body: {}});
    }
    concurrencyRequests.push({url, options: options || null});
    if (!concurrencyResponses.length) throw new Error("unexpected concurrency request");
    return Promise.resolve(concurrencyResponses.shift());
  };
  const captureConcurrency = () => ({
    value: concurrencySelect.value,
    selectDisabled: concurrencySelect.disabled,
    locked: document.getElementById("gptpro-concurrency-card").classList.contains("locked"),
    env: document.getElementById("gptpro-concurrency-lock-env").textContent,
    status: document.getElementById("gptpro-concurrency-status").textContent,
    live: context.GPTPRO_CONCURRENCY.live,
    draft: context.GPTPRO_CONCURRENCY.draft,
    hasLoaded: context.GPTPRO_CONCURRENCY.hasLoaded,
    isSaving: context.GPTPRO_CONCURRENCY.isSaving,
    message: context.GPTPRO_CONCURRENCY.message,
    messageKind: context.GPTPRO_CONCURRENCY.messageKind,
  });
  const changeConcurrency = (value) => {
    concurrencySelect.value = String(value);
    concurrencySelect.dispatchEvent({type: "change"});
  };
  const loadConcurrency = async (limit = 3, locked = false) => {
    concurrencyResponses.push(concurrencyEnvelope(limit, locked));
    await context.fetchGptProConcurrency();
  };
  const captureIgnoredChanges = (values) => {
    const before = captureConcurrency();
    const requestCount = concurrencyRequests.length;
    const states = values.map((value) => {
      changeConcurrency(value);
      return captureConcurrency();
    });
    return {before, states, requestsAdded: concurrencyRequests.length - requestCount};
  };

  gptProConcurrencyStates.noApply = !document.getElementById("gptpro-concurrency-apply");
  gptProConcurrencyStates.initial = captureConcurrency();
  gptProConcurrencyStates.unloadedGuard = captureIgnoredChanges(["5", ""]);
  const initialGet = createDeferred();
  concurrencyResponses.push(initialGet.promise);
  context.setTab("mcp");
  gptProConcurrencyStates.loading = captureConcurrency();
  initialGet.resolve({ok: false, status: 503, body: {}});
  await flushConcurrency();
  gptProConcurrencyStates.startupUnavailable = captureConcurrency();
  gptProConcurrencyStates.startupUnavailableGuard = captureIgnoredChanges(["5"]);
  await loadConcurrency();
  gptProConcurrencyStates.loaded = captureConcurrency();
  gptProConcurrencyStates.invalidGuard = captureIgnoredChanges(["", "0", "11", "3.5", "bad", "3"]);

  const firstSave = createDeferred();
  concurrencyResponses.push(firstSave.promise);
  const beforeSave = concurrencyRequests.length;
  changeConcurrency("5");
  gptProConcurrencyStates.pending = captureConcurrency();
  gptProConcurrencyStates.immediateRequests = concurrencyRequests.slice(beforeSave);
  gptProConcurrencyStates.busyGuard = captureIgnoredChanges(["6", "7"]);
  const beforeSuppressedGet = concurrencyRequests.length;
  await context.fetchGptProConcurrency();
  gptProConcurrencyStates.suppressedGetRequests = concurrencyRequests.length - beforeSuppressedGet;
  firstSave.resolve(concurrencyEnvelope(4));
  await flushConcurrency();
  gptProConcurrencyStates.saved = captureConcurrency();
  concurrencyResponses.push(concurrencyEnvelope(2, true));
  await context.saveGptProConcurrency(6);
  gptProConcurrencyStates.savedLocked = captureConcurrency();
  await loadConcurrency(3, true);
  gptProConcurrencyStates.envLocked = captureConcurrency();
  gptProConcurrencyStates.lockedGuard = captureIgnoredChanges(["5"]);
  await loadConcurrency();

  const malformedBodies = [
    null, {}, {max_concurrent_asks: 0, env_locked: false},
    {max_concurrent_asks: 11, env_locked: false},
    {max_concurrent_asks: 3.5, env_locked: false},
    {max_concurrent_asks: "3", env_locked: false},
    {max_concurrent_asks: 3}, {max_concurrent_asks: 3, env_locked: "false"},
  ];
  gptProConcurrencyStates.validation = [
    concurrencyEnvelope(1).body, concurrencyEnvelope(10, true).body,
    ...malformedBodies,
  ].map((body) => context.isGptProConcurrencyEnvelope(body));
  gptProConcurrencyStates.saveErrors = [];
  for (const failure of [
    {ok: false, status: 500, body: {error: {message: "save unavailable"}}},
    ...malformedBodies.map((body) => ({ok: true, status: 200, body})),
    "network",
  ]) {
    await loadConcurrency();
    const failedPut = createDeferred();
    concurrencyResponses.push(failedPut.promise);
    const savePromise = context.saveGptProConcurrency(5);
    if (!savePromise || typeof savePromise.then !== "function") {
      throw new Error("saveGptProConcurrency must return its full promise");
    }
    if (failure === "network") failedPut.reject(new Error("PUT disconnected"));
    else failedPut.resolve(failure);
    await savePromise;
    const failed = captureConcurrency();
    concurrencyResponses.push(concurrencyEnvelope(5));
    changeConcurrency("5");
    await flushConcurrency();
    gptProConcurrencyStates.saveErrors.push({failed, retried: captureConcurrency()});
  }

  gptProConcurrencyStates.getErrors = [];
  for (const failure of [
    {ok: false, status: 503, body: {}},
    ...malformedBodies.map((body) => ({ok: true, status: 200, body})), "network",
  ]) {
    await loadConcurrency();
    const failedGet = createDeferred();
    concurrencyResponses.push(failedGet.promise);
    const getPromise = context.fetchGptProConcurrency();
    if (!getPromise || typeof getPromise.then !== "function") {
      throw new Error("fetchGptProConcurrency must return a promise");
    }
    if (failure === "network") failedGet.reject(new Error("GET disconnected"));
    else failedGet.resolve(failure);
    await getPromise;
    const unavailable = captureConcurrency();
    const toast = document.getElementById("toast").innerHTML;
    const guard = captureIgnoredChanges(["5"]);
    await loadConcurrency(2);
    gptProConcurrencyStates.getErrors.push({unavailable, toast, guard, recovered: captureConcurrency()});
  }

  gptProConcurrencyStates.staleGets = [];
  for (const outcome of ["success", "error", "network"]) {
    for (const settleDuringSave of [true, false]) {
      await loadConcurrency();
      const staleGet = createDeferred();
      concurrencyResponses.push(staleGet.promise);
      const getPromise = context.fetchGptProConcurrency();
      const pendingPut = createDeferred();
      concurrencyResponses.push(pendingPut.promise);
      const generationBeforeSave = context.GPTPRO_CONCURRENCY.requestGeneration;
      const savePromise = context.saveGptProConcurrency(5);
      const generationAfterSave = context.GPTPRO_CONCURRENCY.requestGeneration;
      if (!settleDuringSave) {
        pendingPut.resolve(concurrencyEnvelope(4));
        await savePromise;
      }
      const before = captureConcurrency();
      const toastBefore = document.getElementById("toast").innerHTML;
      if (outcome === "network") staleGet.reject(new Error("stale GET disconnected"));
      else staleGet.resolve(outcome === "success"
        ? concurrencyEnvelope(9, true) : {ok: false, status: 500, body: {}});
      await getPromise;
      const after = captureConcurrency();
      const toastAfter = document.getElementById("toast").innerHTML;
      if (settleDuringSave) {
        pendingPut.resolve(concurrencyEnvelope(4));
        await savePromise;
      }
      gptProConcurrencyStates.staleGets.push({
        outcome, settleDuringSave, before, after, toastBefore, toastAfter,
        generationBeforeSave, generationAfterSave, saved: captureConcurrency(),
      });
    }
  }

  gptProConcurrencyStates.conflicts = [];
  for (const outcome of ["unlocked", "locked", "error", "malformed", "network"]) {
    await loadConcurrency();
    const conflictingPut = createDeferred();
    const refresh = createDeferred();
    concurrencyResponses.push(conflictingPut.promise, refresh.promise);
    const requestStart = concurrencyRequests.length;
    let saveSettled = false;
    const savePromise = context.saveGptProConcurrency(5);
    const observedSave = savePromise.then(() => { saveSettled = true; });
    conflictingPut.resolve({ok: false, status: 409, body: concurrencyEnvelope(9).body});
    await flushConcurrency();
    const pendingRefresh = captureConcurrency();
    const settledBeforeRefresh = saveSettled;
    const guard = captureIgnoredChanges(["6"]);
    await context.fetchGptProConcurrency();
    const requestsBeforeRefresh = concurrencyRequests.slice(requestStart);
    if (outcome === "network") refresh.reject(new Error("refresh disconnected"));
    else if (outcome === "error") refresh.resolve({ok: false, status: 500, body: {}});
    else if (outcome === "malformed") refresh.resolve({ok: true, status: 200, body: {}});
    else refresh.resolve(concurrencyEnvelope(2, outcome === "locked"));
    await observedSave;
    const refreshed = captureConcurrency();
    await loadConcurrency(7);
    gptProConcurrencyStates.conflicts.push({
      outcome, pendingRefresh, settledBeforeRefresh, guard, requestsBeforeRefresh,
      refreshed, saveSettled, recovered: captureConcurrency(),
    });
  }

  // Exercise the real jfetch wrapper, not just the request-shape mock.
  const concurrencyFetch = context.fetch;
  const concurrencyAuthRequests = [];
  context.jfetch = dashboardJfetch;
  context.localToken = "concurrency-test-token";
  context.fetch = async (url, options) => {
    concurrencyAuthRequests.push({url, options: options || null});
    return {ok: true, status: 200, json: async () => concurrencyEnvelope(6).body};
  };
  await context.saveGptProConcurrency(6);
  gptProConcurrencyStates.authRequests = concurrencyAuthRequests;
  context.localToken = null;
  context.fetch = concurrencyFetch;
  context.jfetch = concurrencyJfetch;

  const feedbackDocument = new FakeDocument(html);
  const feedbackRequests = [];
  const feedbackContext = Object.assign({}, context, {document: feedbackDocument});
  vm.createContext(feedbackContext);
  vm.runInContext(source, feedbackContext, {filename: javascriptPath});
  const captureToast = () => {
    const element = feedbackDocument.getElementById("toast");
    return {role: element.getAttribute("role"), live: element.getAttribute("aria-live"),
      atomic: element.getAttribute("aria-atomic"), html: element.innerHTML};
  };
  feedbackContext.showToast("<b>First failure</b>", true);
  const alertToast = captureToast();
  feedbackContext.showToast("Saved", false);
  const statusToast = captureToast();

  feedbackContext.LOGIN.attemptId = "feedback-attempt";
  feedbackContext.renderLoginModal({status: "awaiting-browser", url: "https://example.com/sign-in"});
  feedbackDocument.getElementById("login-code-input").value = "synthetic-sign-in-code";
  feedbackContext.jfetch = (url, options) => {
    feedbackRequests.push({url, options});
    return Promise.resolve({ok: false, status: 400, body: {error: {message: "<invalid code>"}}});
  };
  feedbackContext.submitLoginCode();
  await new Promise((resolve) => setImmediate(resolve));
  const codeError = feedbackDocument.getElementById("login-code-error");
  const rejectedCode = {text: codeError.textContent, hidden: codeError.hidden, toast: captureToast()};
  feedbackDocument.getElementById("toast").remove();
  const retainedCodeError = codeError.textContent;
  feedbackContext.jfetch = (url, options) => {
    feedbackRequests.push({url, options});
    return Promise.reject(new Error("code response lost"));
  };
  feedbackContext.submitLoginCode();
  await new Promise((resolve) => setImmediate(resolve));
  const disconnectedCode = {text: codeError.textContent, hidden: codeError.hidden};
  feedbackContext.jfetch = (url, options) => {
    feedbackRequests.push({url, options});
    return Promise.resolve({ok: true, status: 200, body: {}});
  };
  feedbackContext.submitLoginCode();
  await new Promise((resolve) => setImmediate(resolve));
  const acceptedCode = {text: codeError.textContent, hidden: codeError.hidden};
  feedbackContext.LOGIN.attemptId = "new-feedback-attempt";
  const staleCodeResponse = createDeferred();
  feedbackContext.jfetch = () => staleCodeResponse.promise;
  feedbackContext.submitLoginCode();
  feedbackContext.LOGIN.attemptId = "newer-feedback-attempt";
  staleCodeResponse.resolve({ok: false, status: 400, body: {error: {message: "stale code error"}}});
  await new Promise((resolve) => setImmediate(resolve));
  const staleCodeIgnored = codeError.hidden && codeError.textContent === "";

  feedbackContext.renderGptProConcurrencyState(concurrencyEnvelope(3).body);
  let persistedLimit = 3;
  const lostResponseRequests = [];
  feedbackContext.jfetch = (url, options) => {
    lostResponseRequests.push({url, options});
    persistedLimit = JSON.parse(options.body).max_concurrent_asks;
    return Promise.reject(new Error("response lost after persistence"));
  };
  await feedbackContext.saveGptProConcurrency(5);
  const responseLoss = {persistedLimit, displayedLimit: feedbackDocument.getElementById("gptpro-concurrency").value,
    message: feedbackDocument.getElementById("gptpro-concurrency-status").textContent, requests: lostResponseRequests};
  const account = {id: "copy-account", email: "copy@example.com", state: "ready"};
  const servingCopy = {};
  feedbackContext.ACCT.serving = account.id;
  servingCopy.pinned = feedbackContext.acctRowHtml(account);
  feedbackContext.ACCT.serving = null;
  servingCopy.unpinned = feedbackContext.acctRowHtml(account);
  feedbackContext.renderUsageProvider("codex", {status: "ok", session: {used_percent: 5, window_minutes: 60}});
  const unknownDurationUsage = feedbackDocument.getElementById("usage-body-codex").innerHTML;
  const operatorFeedback = {alertToast, statusToast, rejectedCode, retainedCodeError, disconnectedCode,
    acceptedCode, staleCodeIgnored, codeRequests: feedbackRequests, responseLoss, servingCopy, unknownDurationUsage};

  const usageRendering = {plans: [], windows: {}};
  for (const provider of ["codex", "claude", "kimi", "grok"]) {
    for (const planType of ["pro", "prolite", "promax", "plus", "<future-plan>", "PRO", "constructor", "toString", "__proto__", "", null]) {
      const data = {status: "ok", plan_type: planType, updated_at: 1};
      feedbackContext.renderUsageProvider(provider, data);
      usageRendering.plans.push({provider, planType, rawPlanType: data.plan_type,
        html: feedbackDocument.getElementById(`usage-plan-${provider}`).innerHTML});
    }
  }
  for (const [name, data] of Object.entries({
    weeklyOnly: {status: "ok", plan_type: "promax", session: null,
      weekly: {used_percent: 37, window_minutes: 10080, resets_at: 1754600000},
      reset_credits_available: null, updated_at: 1},
    noWindows: {status: "ok", plan_type: "promax", session: null, weekly: null, updated_at: 1},
  })) {
    feedbackContext.renderUsageProvider("codex", data);
    usageRendering.windows[name] = feedbackDocument.getElementById("usage-body-codex").innerHTML;
  }

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

  context.jfetch = (url) => {
    if (url === "/admin/providers/codex/models") {
      return Promise.resolve({ok: false, status: 502, body: {error: {message: "<broken Codex version>"}}});
    }
    if (url === "/api/hello") {
      return Promise.resolve({ok: true, body: {local_auth_required: false}});
    }
    if (url === "/health") {
      return Promise.resolve({ok: true, body: {status: "ok", providers: {
        codex: {status: "ok"}, kimi: {status: "ok"}, grok: {status: "ok"},
      }}});
    }
    if (url === "/admin/settings/mapping") {
      return Promise.resolve({ok: true, body: {model_map: {}, custom_providers: []}});
    }
    if (url === "/admin/settings/codex") {
      return Promise.resolve({ok: true, body: {env_locked: false, service_tier: null}});
    }
    if (url === "/admin/providers/codex/capabilities") return Promise.resolve(codexCapabilitiesEnvelope());
    if (url === "/admin/settings/compaction") {
      return Promise.resolve({ok: true, body: {env_locked: false, model: null}});
    }
    if (url === "/admin/providers/claude/pool/routing") {
      return Promise.resolve({ok: true, body: {env_locked: false, mode: "disabled", include_local_login: true}});
    }
    return Promise.resolve({ok: true, body: {models: [], choices: []}});
  };
  context.boot();
  await new Promise((resolve) => setImmediate(resolve));
  const toast = document.getElementById("toast");
  const codexCatalogBoot = {
    className: toast ? toast.className : "",
    html: toast ? toast.innerHTML : "",
    suggestions: Array.from(context.CATALOG.codex),
  };

  const healthAuthenticationBoot = [];
  for (const firstToken of ["dashboard-local-token", "wrong-token", null]) {
    const authDocument = new FakeDocument(html);
    const authRequests = [];
    let promptCount = 0;
    const authContext = Object.assign({}, context, {
      document: authDocument,
      window: Object.assign({}, context.window, {prompt: () => {
        promptCount += 1;
        if (promptCount === 1) return firstToken;
        return promptCount === 2 ? "dashboard-local-token" : null;
      }}),
      fetch: async (url, options) => {
        const hasValidToken = !!(options && options.headers &&
          options.headers.Authorization === "Bearer dashboard-local-token");
        authRequests.push({url, hasValidToken});
        let status = 200;
        let body = {};
        if (url === "/api/hello") body = {local_auth_required: true};
        else if (url.startsWith("/admin/") && !hasValidToken) {
          status = 401;
          body = localTokenRejection();
        } else if (url === "/health") {
          body = {status: "ok", providers: {
            codex: {status: "ok", auth_mode: "chatgpt"},
            kimi: {status: "ok", required: false},
            grok: {status: "ok", required: false, auth_mode: "oauth"},
          }};
          if (hasValidToken) {
            body.providers.codex.email = "codex-identity@example.com";
            body.providers.kimi.account = "kimi-account-identity";
            body.providers.grok.account = "grok-identity@example.com";
          }
        } else if (url === "/admin/settings/mapping") {
          body = {model_map: {}, custom_providers: []};
        } else if (url === "/admin/settings/codex") {
          body = {env_locked: false, service_tier: null};
        } else if (url === "/admin/providers/codex/capabilities") {
          body = codexCapabilitiesEnvelope().body;
        } else if (url === "/admin/settings/compaction") {
          body = {env_locked: false, model: null};
        } else if (url === "/admin/providers/claude/pool/routing") {
          body = {env_locked: false, mode: "disabled", include_local_login: true};
        } else body = {models: [], choices: []};
        return {ok: status === 200, status, json: async () => body};
      },
    });
    vm.createContext(authContext);
    vm.runInContext(source, authContext, {filename: javascriptPath});
    authContext.boot();
    await new Promise((resolve) => setImmediate(resolve));
    healthAuthenticationBoot.push({
      firstTokenWasValid: firstToken === "dashboard-local-token",
      promptCount,
      adminRequests: authRequests.filter((request) => request.url.startsWith("/admin/")),
      healthRequests: authRequests.filter((request) => request.url === "/health"),
      helloHasToken: authRequests.find((request) => request.url === "/api/hello").hasValidToken,
      identities: {
        codex: authDocument.getElementById("codex-statline").innerHTML,
        kimi: authDocument.getElementById("kimi-statline").innerHTML,
        grok: authDocument.getElementById("grok-statline").innerHTML,
      },
    });
  }

  const codexDocument = new FakeDocument(html);
  const codexContext = Object.assign({}, context, {document: codexDocument});
  vm.createContext(codexContext);
  vm.runInContext(source, codexContext, {filename: javascriptPath});
  const codexRequests = [];
  let persistedCodexTier = null;
  codexContext.jfetch = (url, options) => {
    codexRequests.push({url, options: options || null});
    if (url === "/admin/providers/codex/capabilities") return Promise.resolve(codexCapabilitiesEnvelope(persistedCodexTier));
    if (options) persistedCodexTier = JSON.parse(options.body).service_tier;
    return Promise.resolve({ok: true, status: 200, body: {service_tier: persistedCodexTier, env_locked: false}});
  };
  const codexSelect = codexDocument.getElementById("codex-speed");
  const captureCodexSelection = () => ({
    value: codexSelect ? codexSelect.value : null,
    live: codexContext.CODEX.serviceTier,
    draft: codexContext.CODEX.draft,
    applyDisabled: codexDocument.getElementById("codex-apply").disabled,
  });
  const changeCodexSpeed = (tier) => {
    const input = codexSelect || codexDocument.getElementById("codex-fast");
    input.value = tier || "standard";
    input.checked = tier === "fast";
    input.dispatchEvent({type: "change"});
  };
  const flushCodex = () => new Promise((resolve) => setImmediate(resolve));
  const captureCodexSettings = () => ({
    ...captureCodexSelection(),
    selectDisabled: codexSelect ? codexSelect.disabled : false,
    locked: codexContext.CODEX.locked,
    hasLoaded: !!codexContext.CODEX.hasLoaded,
    isSaving: !!codexContext.CODEX.isSaving,
    status: codexDocument.getElementById("codex-settings-status")?.textContent || "",
    toast: codexDocument.getElementById("toast")?.innerHTML || "",
  });
  const codexSettingsStates = {initial: captureCodexSettings(), locks: [], saveErrors: [], loadErrors: [], conflicts: []};
  codexContext.renderCodexState({service_tier: null, env_locked: false});
  const codexSpeedStates = {
    selectPresent: !!codexSelect,
    validEnvelopes: [null, "fast", "ultrafast"].map((tier) =>
      codexContext.isCodexEnvelope({service_tier: tier, env_locked: false})),
    invalidEnvelopes: [{}, {service_tier: "priority", env_locked: false},
      {service_tier: "ultrafast"}, {service_tier: false, env_locked: false}]
      .map((body) => codexContext.isCodexEnvelope(body)),
    initial: captureCodexSelection(),
    selections: [],
  };
  for (const tier of ["fast", "ultrafast", null]) {
    changeCodexSpeed(tier);
    const draft = captureCodexSelection();
    const start = codexRequests.length;
    codexDocument.getElementById("codex-apply").click();
    await flushCodex();
    const request = codexRequests.slice(start).find((item) => item.options);
    codexSpeedStates.selections.push({draft, saved: captureCodexSelection(),
      request: request ? JSON.parse(request.options.body) : null});
  }
  codexContext.renderCodexState({service_tier: "ultrafast", env_locked: false});
  codexSpeedStates.reloaded = captureCodexSelection();
  codexSpeedStates.nonSettingsRequests = codexRequests.filter((item) => item.url !== "/admin/settings/codex");

  const codexResponses = [];
  codexContext.jfetch = (url, options) => {
    if (url === "/admin/providers/codex/capabilities") return Promise.resolve(codexCapabilitiesEnvelope(codexContext.CODEX.serviceTier));
    codexRequests.push({url, options: options || null});
    if (!codexResponses.length) return Promise.resolve({ok: false, status: 500, body: {error: {message: "unexpected Codex settings request"}}});
    return Promise.resolve(codexResponses.shift());
  };
  const codexEnvelope = (tier, locked = false) => ({ok: true, status: 200,
    body: {service_tier: tier, env_locked: locked}});
  const reloadCodexSettings = async () => {
    codexDocument.getElementById("codex-reload")?.click();
    await flushCodex();
  };
  for (const tier of [null, "fast", "ultrafast"]) {
    codexContext.renderCodexState(codexEnvelope(tier, true).body);
    const start = codexRequests.length;
    changeCodexSpeed(tier === "ultrafast" ? null : "ultrafast");
    codexContext.applyCodex();
    await flushCodex();
    codexSettingsStates.locks.push({tier, state: captureCodexSettings(), requests: codexRequests.slice(start)});
  }
  codexContext.renderCodexState(codexEnvelope("fast").body);
  const invalidStart = codexRequests.length;
  changeCodexSpeed("priority");
  codexSettingsStates.invalidChange = {state: captureCodexSettings(), requests: codexRequests.slice(invalidStart)};
  codexContext.renderCodexState(codexEnvelope("fast").body);
  changeCodexSpeed("ultrafast");
  const pendingCodexSave = createDeferred();
  codexResponses.push(pendingCodexSave.promise);
  const pendingStart = codexRequests.length;
  codexDocument.getElementById("codex-apply").click();
  const pendingState = captureCodexSettings();
  changeCodexSpeed(null);
  codexContext.applyCodex();
  await reloadCodexSettings();
  const afterForcedEvents = captureCodexSettings();
  pendingCodexSave.resolve(codexEnvelope("ultrafast"));
  await flushCodex();
  codexSettingsStates.pendingSave = {pendingState, afterForcedEvents,
    requests: codexRequests.slice(pendingStart), saved: captureCodexSettings()};
  for (const outcome of ["error", "malformed", "missing", "network"]) {
    codexResponses.length = 0;
    codexContext.renderCodexState(codexEnvelope("fast").body);
    changeCodexSpeed("ultrafast");
    const deferred = createDeferred();
    codexResponses.push(deferred.promise);
    codexDocument.getElementById("codex-apply").click();
    if (outcome === "network") deferred.reject(new Error("settings response lost"));
    else if (outcome === "error") deferred.resolve({ok: false, status: 500, body: {error: {message: "<save failed>"}}});
    else if (outcome === "missing") deferred.resolve({ok: true, status: 200, body: {}});
    else deferred.resolve(codexEnvelope("priority"));
    await flushCodex();
    const failed = captureCodexSettings();
    codexResponses.push(codexEnvelope("ultrafast"));
    await reloadCodexSettings();
    codexSettingsStates.saveErrors.push({outcome, failed, recovered: captureCodexSettings()});
  }
  for (const outcome of ["error", "malformed", "network"]) {
    codexResponses.length = 0;
    codexContext.renderCodexState(codexEnvelope("fast").body);
    const deferred = createDeferred();
    codexResponses.push(deferred.promise);
    const reload = reloadCodexSettings();
    if (outcome === "network") deferred.reject(new Error("settings load disconnected"));
    else if (outcome === "error") deferred.resolve({ok: false, status: 503, body: {error: {message: "<load failed>"}}});
    else deferred.resolve({ok: true, status: 200, body: {service_tier: "ultrafast"}});
    await reload;
    const failed = captureCodexSettings();
    const guardStart = codexRequests.length;
    changeCodexSpeed("ultrafast");
    codexContext.applyCodex();
    const guardRequests = codexRequests.slice(guardStart);
    codexResponses.length = 0;
    codexResponses.push(codexEnvelope("ultrafast"));
    await reloadCodexSettings();
    codexSettingsStates.loadErrors.push({outcome, failed, guardRequests, recovered: captureCodexSettings()});
  }
  for (const outcome of ["unlocked", "locked", "malformed", "error", "network"]) {
    codexResponses.length = 0;
    codexContext.renderCodexState(codexEnvelope("fast").body);
    changeCodexSpeed("ultrafast");
    const refresh = createDeferred();
    codexResponses.push({ok: false, status: 409, body: {service_tier: "ultrafast", env_locked: false}}, refresh.promise);
    const start = codexRequests.length;
    codexDocument.getElementById("codex-apply").click();
    await flushCodex();
    const pendingRefresh = captureCodexSettings();
    changeCodexSpeed(null);
    codexContext.applyCodex();
    await reloadCodexSettings();
    const beforeRefreshRequests = codexRequests.slice(start);
    if (outcome === "network") refresh.reject(new Error("conflict refresh disconnected"));
    else if (outcome === "error") refresh.resolve({ok: false, status: 503, body: {}});
    else if (outcome === "malformed") refresh.resolve({ok: true, status: 200, body: {}});
    else refresh.resolve(codexEnvelope(null, outcome === "locked"));
    await flushCodex();
    const refreshed = captureCodexSettings();
    codexResponses.push(codexEnvelope("ultrafast"));
    await reloadCodexSettings();
    codexSettingsStates.conflicts.push({outcome, pendingRefresh, beforeRefreshRequests,
      refreshed, recovered: captureCodexSettings()});
  }

  const capabilitiesDocument = new FakeDocument(html);
  const capabilitiesContext = Object.assign({}, context, {document: capabilitiesDocument});
  vm.createContext(capabilitiesContext);
  vm.runInContext(source, capabilitiesContext, {filename: javascriptPath});
  capabilitiesContext.DIR.LIVE = {opus: "codex:both", sonnet: "codex:fast-only",
    haiku: "codex:manual:future<script>", fable: "codex:partial", external: "kimi:kimi-model"};
  capabilitiesContext.DIR.mapping = {...capabilitiesContext.DIR.LIVE, sonnet: "codex:draft-only"};
  capabilitiesContext.addedTargets = ["codex:staged-only"];
  capabilitiesContext.renderCodexState(codexEnvelope("ultrafast").body);
  const capabilitiesSnapshot = {
    configured_tier: "ultrafast", env_locked: false, status: "ok", reason: null, fetched_at: 1790000000,
    models: {
      both: {supported_tiers: ["standard", "fast", "ultrafast"], advertised_tier_ids: ["priority", "ultrafast", "future-tier"],
        status: "supported", reason: null, applicable_tier: "ultrafast", wire_tier: "ultrafast"},
      "fast-only": {supported_tiers: ["standard", "fast"], advertised_tier_ids: ["priority"],
        status: "unsupported", reason: "Catalog does not advertise ultrafast for this model.", applicable_tier: null, wire_tier: null},
      "manual:future<script>": {supported_tiers: null, advertised_tier_ids: null,
        status: "unknown", reason: "Model is not listed in the authenticated catalog.", applicable_tier: null, wire_tier: null},
      partial: {supported_tiers: ["standard", "fast"], advertised_tier_ids: ["priority"],
        status: "unknown", reason: "Model tier metadata is incomplete; Ultrafast support is unknown.", applicable_tier: null, wire_tier: null},
      unrelated: {supported_tiers: ["standard", "ultrafast"], advertised_tier_ids: ["ultrafast"],
        status: "supported", reason: null, applicable_tier: "ultrafast", wire_tier: "ultrafast"},
    },
  };
  const captureCapabilities = () => ({
    header: capabilitiesDocument.getElementById("codex-capabilities-status")?.textContent || "",
    time: capabilitiesDocument.getElementById("codex-capabilities-time")?.textContent || "",
    models: capabilitiesDocument.getElementById("codex-capabilities-models")?.innerHTML || "",
    settings: capabilitiesDocument.getElementById("codex-settings-status")?.textContent || "",
  });
  const showCapabilities = (body) => {
    if (typeof capabilitiesContext.renderCodexCapabilitiesState === "function") {
      capabilitiesContext.renderCodexCapabilitiesState(body);
    }
    return captureCapabilities();
  };
  const codexCapabilitiesStates = {
    initial: captureCapabilities(),
    mixed: showCapabilities(capabilitiesSnapshot),
    expectedFetchedTime: new Date(1790000000 * 1000).toLocaleString(),
  };
  const capabilitiesSelect = capabilitiesDocument.getElementById("codex-speed");
  capabilitiesSelect.value = "fast";
  capabilitiesSelect.dispatchEvent({type: "change"});
  codexCapabilitiesStates.draftSelection = captureCapabilities();
  codexCapabilitiesStates.stale = showCapabilities({...capabilitiesSnapshot,
    status: "stale", reason: "Catalog refresh failed: <busy>. Using last-known metadata."});
  codexCapabilitiesStates.unknown = showCapabilities({...capabilitiesSnapshot,
    status: "unknown", reason: "Authentication unavailable.", fetched_at: null,
    models: Object.fromEntries(Object.keys(capabilitiesSnapshot.models).map((model) => [model,
      {supported_tiers: null, advertised_tier_ids: null, status: "unknown",
        reason: "Current account metadata unavailable.", applicable_tier: null, wire_tier: null}]))});
  capabilitiesContext.DIR.LIVE = {sonnet: "kimi:only-non-codex"};
  codexCapabilitiesStates.noMappings = showCapabilities(capabilitiesSnapshot);

  const metadataDocument = new FakeDocument(html);
  const metadataRequests = [];
  let metadataTier = "ultrafast";
  let metadataMapping = {opus: "codex:both", sonnet: "codex:fast-only"};
  let metadataUpgraded = false;
  const metadataBody = () => {
    const body = JSON.parse(JSON.stringify(capabilitiesSnapshot));
    body.configured_tier = metadataTier;
    for (const [model, entry] of Object.entries(body.models)) {
      if (model === "fast-only" && metadataUpgraded) {
        entry.supported_tiers = ["standard", "fast", "ultrafast"];
        entry.advertised_tier_ids = ["priority", "ultrafast"];
      }
      const supported = metadataTier === null || entry.supported_tiers?.includes(metadataTier);
      if (supported) Object.assign(entry, {status: "supported", reason: null,
        applicable_tier: metadataTier, wire_tier: metadataTier === "fast" ? "priority" : metadataTier});
      else Object.assign(entry, {applicable_tier: null, wire_tier: null,
        status: entry.status === "unknown" ? "unknown" : "unsupported",
        reason: entry.status === "unknown" ? entry.reason : "Catalog does not advertise the configured tier for this model."});
    }
    if (metadataUpgraded) body.fetched_at += 60;
    return body;
  };
  const metadataEnvironment = Object.assign({}, context, {document: metadataDocument,
    location: {host: "127.0.0.1:8787", hash: "#map"},
    jfetch: (url, options) => {
      metadataRequests.push({url, options: options || null});
      if (url === "/api/hello") return Promise.resolve({ok: true, body: {local_auth_required: false}});
      if (url === "/health") return Promise.resolve({ok: true, body: {status: "ok", providers: {codex: {status: "ok"}}}});
      if (url === "/admin/settings/mapping") {
        if (options) metadataMapping = JSON.parse(options.body).model_map;
        return Promise.resolve({ok: true, body: {model_map: metadataMapping, custom_providers: [], env_locked: {}}});
      }
      if (url === "/admin/settings/codex") {
        if (options) metadataTier = JSON.parse(options.body).service_tier;
        return Promise.resolve(codexEnvelope(metadataTier));
      }
      if (url.startsWith("/admin/providers/codex/capabilities")) {
        if (options) metadataUpgraded = true;
        return Promise.resolve({ok: true, status: 200, body: metadataBody()});
      }
      if (url === "/admin/settings/compaction") return Promise.resolve({ok: true, body: {model: null, env_locked: false}});
      if (url === "/admin/providers/claude/pool/routing") return Promise.resolve({ok: true, body: {mode: "disabled", env_locked: false, include_local_login: true}});
      return Promise.resolve({ok: true, body: {models: [], log_level: "INFO"}});
    },
  });
  const metadataContext = Object.assign({}, metadataEnvironment);
  vm.createContext(metadataContext);
  vm.runInContext(source, metadataContext, {filename: javascriptPath});
  const metadataRealJfetch = metadataContext.jfetch;
  metadataContext.jfetch = metadataEnvironment.jfetch;
  const captureMetadata = () => ({
    header: metadataDocument.getElementById("codex-capabilities-status")?.textContent || "",
    models: metadataDocument.getElementById("codex-capabilities-models")?.innerHTML || "",
    time: metadataDocument.getElementById("codex-capabilities-time")?.textContent || "",
    value: metadataDocument.getElementById("codex-speed").value,
  });
  metadataContext.boot();
  await flushCodex();
  const codexMetadataStates = {boot: captureMetadata(), bootRequests: metadataRequests.slice()};
  const metadataSelect = metadataDocument.getElementById("codex-speed");
  const selectionStart = metadataRequests.length;
  metadataSelect.value = "fast";
  metadataSelect.dispatchEvent({type: "change"});
  codexMetadataStates.draft = {state: captureMetadata(), requests: metadataRequests.slice(selectionStart)};
  const applyStart = metadataRequests.length;
  await metadataContext.applyCodex();
  await flushCodex();
  codexMetadataStates.saved = {state: captureMetadata(), requests: metadataRequests.slice(applyStart)};
  const mappingStart = metadataRequests.length;
  metadataContext.DIR.mapping.haiku = "codex:unrelated";
  metadataContext.apply();
  await flushCodex();
  codexMetadataStates.mappingSaved = {state: captureMetadata(), requests: metadataRequests.slice(mappingStart)};
  metadataSelect.value = "ultrafast";
  metadataSelect.dispatchEvent({type: "change"});
  await metadataContext.applyCodex();
  await flushCodex();
  const refreshStart = metadataRequests.length;
  metadataDocument.getElementById("codex-capabilities-refresh")?.click();
  await flushCodex();
  codexMetadataStates.refreshed = {state: captureMetadata(), requests: metadataRequests.slice(refreshStart)};
  codexMetadataStates.allRequests = metadataRequests.slice();
  codexMetadataStates.race = null;
  codexMetadataStates.authRequests = [];
  if (typeof metadataContext.fetchCodexCapabilities === "function") {
    const obsolete = createDeferred();
    const savedMetadataJfetch = metadataContext.jfetch;
    metadataContext.jfetch = () => obsolete.promise;
    const oldRefresh = metadataContext.fetchCodexCapabilities(true);
    metadataContext.jfetch = () => Promise.resolve({ok: true, status: 200,
      body: {...metadataBody(), reason: "new current snapshot", fetched_at: 1790000200}});
    await metadataContext.fetchCodexCapabilities();
    obsolete.resolve({ok: true, status: 200, body: {...capabilitiesSnapshot,
      configured_tier: "fast", reason: "obsolete snapshot", env_locked: true}});
    await oldRefresh;
    codexMetadataStates.race = {state: captureMetadata(), locked: metadataContext.CODEX.locked};
    metadataContext.jfetch = savedMetadataJfetch;
    metadataContext.localToken = "metadata-test-token";
    metadataContext.fetch = async (url, options) => {
      codexMetadataStates.authRequests.push({url, options});
      return {ok: true, status: 200, json: async () => metadataBody()};
    };
    metadataContext.jfetch = metadataRealJfetch;
    await metadataContext.fetchCodexCapabilities(true);
  }

  const codexMetadataFailureStates = {scenarios: [], pending: null, missingModel: null, changedSetting: null};
  capabilitiesContext.DIR.LIVE = {opus: "codex:both", haiku: "codex:omitted-from-response"};
  capabilitiesContext.DIR.mapping = {...capabilitiesContext.DIR.LIVE};
  capabilitiesContext.renderCodexState(codexEnvelope("ultrafast").body);
  codexMetadataFailureStates.missingModel = showCapabilities(capabilitiesSnapshot);
  if (typeof capabilitiesContext.fetchCodexCapabilities === "function") {
    const malformedSnapshots = [
      ["empty", {}], ["null", null],
      ["configured-tier", {...capabilitiesSnapshot, configured_tier: "priority"}],
      ["env-lock", {...capabilitiesSnapshot, env_locked: "false"}],
      ["top-status", {...capabilitiesSnapshot, status: "connected"}],
      ["top-reason", {...capabilitiesSnapshot, reason: 42}],
      ["timestamp", {...capabilitiesSnapshot, fetched_at: "yesterday"}],
      ["timestamp-range", {...capabilitiesSnapshot, fetched_at: 1e20}],
      ["models-array", {...capabilitiesSnapshot, models: []}],
      ["model-null", {...capabilitiesSnapshot, models: {both: null}}],
      ["supported-tiers", {...capabilitiesSnapshot, models: {both: {...capabilitiesSnapshot.models.both, supported_tiers: ["standard", "warp"]}}}],
      ["advertised-ids", {...capabilitiesSnapshot, models: {both: {...capabilitiesSnapshot.models.both, advertised_tier_ids: [42]}}}],
      ["model-status", {...capabilitiesSnapshot, models: {both: {...capabilitiesSnapshot.models.both, status: "current"}}}],
      ["model-reason", {...capabilitiesSnapshot, models: {both: {...capabilitiesSnapshot.models.both, reason: {error: "bad"}}}}],
      ["model-wire", {...capabilitiesSnapshot, models: {both: {...capabilitiesSnapshot.models.both, wire_tier: "priority"}}}],
      ["model-applicability", {...capabilitiesSnapshot, models: {both: {...capabilitiesSnapshot.models.both, applicable_tier: "fast", wire_tier: "priority"}}}],
    ];
    const outcomes = [["http", {ok: false, status: 503, body: {error: {message: "<catalog unavailable>"}}}],
      ["network", null], ...malformedSnapshots.map(([name, body]) => [name, {ok: true, status: 200, body}])];
    for (const [outcome, response] of outcomes) {
      showCapabilities(capabilitiesSnapshot);
      const failureRequests = [];
      capabilitiesContext.jfetch = (url, options) => {
        failureRequests.push({url, options: options || null});
        return outcome === "network" ? Promise.reject(new Error("metadata disconnected")) : Promise.resolve(response);
      };
      await capabilitiesContext.fetchCodexCapabilities(true);
      const failed = captureCapabilities();
      const editingEnabled = !capabilitiesDocument.getElementById("codex-speed").disabled;
      const refreshEnabled = !capabilitiesDocument.getElementById("codex-capabilities-refresh").disabled;
      capabilitiesContext.jfetch = (url, options) => {
        failureRequests.push({url, options: options || null});
        return Promise.resolve({ok: true, status: 200, body: capabilitiesSnapshot});
      };
      await capabilitiesContext.fetchCodexCapabilities(true);
      codexMetadataFailureStates.scenarios.push({outcome, failed, editingEnabled, refreshEnabled,
        recovered: captureCapabilities(), requests: failureRequests});
    }
    const pendingMetadata = createDeferred();
    const pendingMetadataRequests = [];
    capabilitiesContext.jfetch = (url, options) => {
      pendingMetadataRequests.push({url, options: options || null});
      return pendingMetadata.promise;
    };
    const refresh = capabilitiesContext.fetchCodexCapabilities(true);
    const during = {state: captureCapabilities(), disabled: capabilitiesDocument.getElementById("codex-capabilities-refresh").disabled,
      busy: capabilitiesDocument.getElementById("codex-capabilities").getAttribute("aria-busy")};
    capabilitiesDocument.getElementById("codex-capabilities-refresh").click();
    pendingMetadata.resolve({ok: true, status: 200, body: {...capabilitiesSnapshot, status: "stale", reason: "Same-account fetch failed; using cached support."}});
    await refresh;
    codexMetadataFailureStates.pending = {during, after: captureCapabilities(), requests: pendingMetadataRequests};

    showCapabilities(capabilitiesSnapshot);
    capabilitiesContext.renderCodexState(codexEnvelope("ultrafast").body);
    const changedMetadata = createDeferred();
    capabilitiesContext.jfetch = (url, options) => url === "/admin/settings/codex"
      ? Promise.resolve(codexEnvelope(JSON.parse(options.body).service_tier)) : changedMetadata.promise;
    capabilitiesSelect.value = "fast";
    capabilitiesSelect.dispatchEvent({type: "change"});
    await capabilitiesContext.applyCodex();
    codexMetadataFailureStates.changedSetting = captureCapabilities();
    changedMetadata.resolve({ok: true, status: 200, body: {...capabilitiesSnapshot, configured_tier: "fast",
      models: {both: {...capabilitiesSnapshot.models.both, applicable_tier: "fast", wire_tier: "priority"}}}});
    await flushCodex();
    codexMetadataFailureStates.changedSettingResolved = captureCapabilities();
  }

  const speedDocument = new FakeDocument(html);
  const speedContext = Object.assign({}, context, {document: speedDocument});
  vm.createContext(speedContext);
  vm.runInContext(source, speedContext, {filename: javascriptPath});
  const speedRealJfetch = speedContext.jfetch;
  const speedRequests = [];
  const speedResponses = [];
  speedContext.jfetch = (url, options) => {
    speedRequests.push({url, options: options || null});
    return Promise.resolve(speedResponses.shift() || {ok: false, status: 500, body: {}});
  };
  speedContext.renderCodexState(codexEnvelope(null).body);
  const speedSelect = speedDocument.getElementById("codex-speed");
  const speedInput = speedDocument.getElementById("ct-in");
  const speedButton = speedDocument.getElementById("codex-speed-test");
  const captureSpeedTest = () => ({
    html: speedDocument.getElementById("codex-speed-result")?.innerHTML || "",
    hidden: speedDocument.getElementById("codex-speed-result")?.hidden ?? true,
    role: speedDocument.getElementById("codex-speed-result")?.getAttribute("role") || "",
    disabled: speedButton?.disabled || false,
    live: speedContext.CODEX.serviceTier, draft: speedContext.CODEX.draft,
  });
  const speedDiagnostic = (requested, sent, reported, confirmation, capability = "supported", reason = null) => ({
    ok: true, status: 200, latency_ms: 12, target: "codex:manual:model", response_model: "served-model", detail: null,
    tier_diagnostic: {requested_tier: requested, sent_tier: sent, backend_confirmed_tier: reported,
      capability_status: capability, capability_reason: reason, confirmation_status: confirmation,
      confirmation_reason: confirmation === "confirmed" ? null : "Provider evidence does not confirm the selected accelerated tier."},
  });
  const codexSpeedTestStates = {rows: [], guards: [], authRequests: []};
  const diagnosticCases = [
    [null, null, null, "unconfirmed", "supported", null],
    ["fast", "priority", "priority", "confirmed", "supported", null],
    ["ultrafast", "ultrafast", "ultrafast", "confirmed", "supported", null],
    ["ultrafast", "ultrafast", "priority", "different", "supported", "<provider chose another tier>"],
    ["ultrafast", null, null, "unconfirmed", "unknown", "Tier support changed or could not be verified at dispatch; sent Standard without the earlier selection's tier."],
    ["ultrafast", null, "default", "different", "unsupported", "Catalog does not advertise the requested tier."],
    [null, null, "default", "reported", "supported", null],
  ];
  speedInput.value = "codex:manual:model";
  for (const [requested, sent, reported, confirmation, capability, reason] of diagnosticCases) {
    const start = speedRequests.length;
    speedSelect.value = requested || "standard";
    speedSelect.dispatchEvent({type: "change"});
    const selectionRequests = speedRequests.slice(start);
    speedResponses.push({ok: true, status: 200, body: speedDiagnostic(requested, sent, reported, confirmation, capability, reason)});
    speedButton?.click();
    await flushCodex();
    codexSpeedTestStates.rows.push({requested, sent, reported, confirmation, capability, reason,
      selectionRequests, result: captureSpeedTest(), requests: speedRequests.slice(start)});
  }
  speedResponses.length = 0;
  speedResponses.push({ok: true, status: 200, body: {ok: true, status: 200, latency_ms: 1, response_model: "ordinary-model"}});
  speedSelect.value = "ultrafast";
  speedSelect.dispatchEvent({type: "change"});
  const ordinaryStart = speedRequests.length;
  speedDocument.getElementById("ct-btn").click();
  await flushCodex();
  codexSpeedTestStates.ordinary = {requests: speedRequests.slice(ordinaryStart), toast: speedDocument.getElementById("toast")?.innerHTML || ""};
  for (const target of ["", "kimi:not-codex", "codex:"]) {
    speedInput.value = target;
    const start = speedRequests.length;
    speedButton?.click();
    await flushCodex();
    codexSpeedTestStates.guards.push({target, requests: speedRequests.slice(start), result: captureSpeedTest()});
  }
  speedInput.value = "codex:manual:model";
  const pendingSpeed = createDeferred();
  speedResponses.push(pendingSpeed.promise);
  const pendingSpeedStart = speedRequests.length;
  speedButton?.click();
  const pendingSpeedState = captureSpeedTest();
  speedSelect.value = "fast";
  speedSelect.dispatchEvent({type: "change"});
  speedButton?.click();
  if (typeof speedContext.runCodexSpeedTest === "function") speedContext.runCodexSpeedTest();
  pendingSpeed.resolve({ok: true, status: 200, body: speedDiagnostic("ultrafast", "ultrafast", null, "unconfirmed")});
  await flushCodex();
  codexSpeedTestStates.pending = {before: pendingSpeedState, after: captureSpeedTest(), requests: speedRequests.slice(pendingSpeedStart)};
  if (typeof speedContext.runCodexSpeedTest === "function") {
    speedContext.jfetch = speedRealJfetch;
    speedContext.localToken = "speed-test-token";
    speedContext.fetch = async (url, options) => {
      codexSpeedTestStates.authRequests.push({url, options});
      return {ok: true, status: 200, json: async () => speedDiagnostic("fast", "priority", "priority", "confirmed")};
    };
    await speedContext.runCodexSpeedTest();
  }

  const codexSpeedFailureStates = {rows: [], blocked: [], locked: null};
  speedContext.jfetch = (url, options) => {
    speedRequests.push({url, options: options || null});
    return Promise.resolve(speedResponses.shift() || {ok: false, status: 500, body: {}});
  };
  speedContext.renderCodexState(codexEnvelope("ultrafast").body);
  const validSpeedBody = speedDiagnostic("ultrafast", "ultrafast", "ultrafast", "confirmed");
  const badDiagnostics = [
    ["null-diagnostic", null], ["array-diagnostic", []],
    ["invalid-requested", {...validSpeedBody.tier_diagnostic, requested_tier: "priority"}],
    ["echo-mismatch", {...validSpeedBody.tier_diagnostic, requested_tier: "fast"}],
    ["missing-wire", Object.fromEntries(Object.entries(validSpeedBody.tier_diagnostic).filter(([key]) => key !== "sent_tier"))],
    ["invalid-wire", {...validSpeedBody.tier_diagnostic, sent_tier: "warp"}],
    ["invalid-report", {...validSpeedBody.tier_diagnostic, backend_confirmed_tier: {tier: "ultrafast"}}],
    ["invalid-confirmation", {...validSpeedBody.tier_diagnostic, confirmation_status: "success"}],
    ["false-confirmed", {...validSpeedBody.tier_diagnostic, backend_confirmed_tier: "priority"}],
    ["false-reported", {...validSpeedBody.tier_diagnostic, confirmation_status: "reported"}],
    ["invalid-capability", {...validSpeedBody.tier_diagnostic, capability_status: "ready"}],
    ["invalid-capability-reason", {...validSpeedBody.tier_diagnostic, capability_reason: {error: "bad"}}],
    ["invalid-confirmation-reason", {...validSpeedBody.tier_diagnostic, confirmation_reason: []}],
  ];
  const speedOutcomes = [
    ["legacy", {ok: true, status: 200, body: {ok: true, status: 200, response_model: "legacy-model"}}],
    ["missing-body", {ok: true, status: 200, body: {}}],
    ["http", {ok: false, status: 401, body: {error: {message: "<auth failed>"}}}],
    ["failed-response", {ok: true, status: 200, body: {...validSpeedBody, ok: false, status: 502, detail: "<stream incomplete>"}}],
    ["network", null],
    ...badDiagnostics.map(([name, diagnostic]) => [name, {ok: true, status: 200, body: {...validSpeedBody, tier_diagnostic: diagnostic}}]),
  ];
  for (const [outcome, response] of speedOutcomes) {
    speedResponses.length = 0;
    const start = speedRequests.length;
    const deferred = createDeferred();
    speedResponses.push(deferred.promise);
    speedButton?.click();
    if (outcome === "network") deferred.reject(new Error("speed request disconnected"));
    else deferred.resolve(response);
    await flushCodex();
    codexSpeedFailureStates.rows.push({outcome, result: captureSpeedTest(), requests: speedRequests.slice(start)});
  }
  for (const settingState of ["unloaded", "saving", "loading"]) {
    speedContext.renderCodexState(codexEnvelope("ultrafast").body);
    if (settingState === "unloaded") speedContext.CODEX.hasLoaded = false;
    if (settingState === "saving") speedContext.CODEX.isSaving = true;
    if (settingState === "loading") speedContext.CODEX.isLoading = true;
    speedContext.renderCodex();
    const start = speedRequests.length;
    if (typeof speedContext.runCodexSpeedTest === "function") await speedContext.runCodexSpeedTest();
    codexSpeedFailureStates.blocked.push({settingState, disabled: speedButton?.disabled || false, requests: speedRequests.slice(start)});
    speedContext.CODEX.isSaving = false;
    speedContext.CODEX.isLoading = false;
  }
  speedContext.renderCodexState(codexEnvelope("ultrafast", true).body);
  const lockedTestStart = speedRequests.length;
  speedResponses.push({ok: true, status: 200, body: validSpeedBody});
  speedButton?.click();
  await flushCodex();
  codexSpeedFailureStates.locked = {result: captureSpeedTest(), requests: speedRequests.slice(lockedTestStart)};

  const codexBootRaceStates = [];
  for (const outcome of ["saved", "conflict", "reload-error"]) {
    const raceDocument = new FakeDocument(html);
    const raceContext = Object.assign({}, context, {document: raceDocument});
    vm.createContext(raceContext);
    vm.runInContext(source, raceContext, {filename: javascriptPath});
    const slowCatalog = createDeferred();
    const raceRequests = [];
    let raceTier = null;
    let raceLocked = false;
    let settingsReads = 0;
    raceContext.jfetch = (url, options) => {
      raceRequests.push({url, options: options || null});
      if (url === "/api/hello") return Promise.resolve({ok: true, body: {local_auth_required: false}});
      if (url === "/health") return Promise.resolve({ok: true, body: {status: "ok", providers: {codex: {status: "ok"}}}});
      if (url === "/admin/providers/kimi/models") return slowCatalog.promise;
      if (url === "/admin/settings/mapping") return Promise.resolve({ok: true, body: {model_map: {}, custom_providers: []}});
      if (url === "/admin/providers/codex/capabilities") return Promise.resolve(codexCapabilitiesEnvelope(raceTier));
      if (url === "/admin/settings/codex") {
        if (options) {
          raceTier = JSON.parse(options.body).service_tier;
          if (outcome === "conflict") {
            raceLocked = true;
            return Promise.resolve({ok: false, status: 409, body: {error: {message: "environment lock"}}});
          }
          return Promise.resolve(codexEnvelope(raceTier, raceLocked));
        }
        settingsReads += 1;
        if (settingsReads > 1 && outcome === "reload-error") return Promise.resolve({ok: false, status: 503, body: {}});
        return Promise.resolve(codexEnvelope(raceTier, raceLocked));
      }
      if (url === "/admin/settings/compaction") return Promise.resolve({ok: true, body: {model: null, env_locked: false}});
      if (url === "/admin/providers/claude/pool/routing") return Promise.resolve({ok: true, body: {mode: "disabled", env_locked: false, include_local_login: true}});
      return Promise.resolve({ok: true, body: {models: [], log_level: "INFO"}});
    };
    const captureRace = () => ({value: raceDocument.getElementById("codex-speed").value,
      live: raceContext.CODEX.serviceTier, draft: raceContext.CODEX.draft, locked: raceContext.CODEX.locked,
      hasLoaded: raceContext.CODEX.hasLoaded, disabled: raceDocument.getElementById("codex-speed").disabled,
      status: raceDocument.getElementById("codex-settings-status").textContent,
      capabilities: raceDocument.getElementById("codex-capabilities-status").textContent});
    raceContext.boot();
    await flushCodex();
    raceTier = "fast";
    raceDocument.getElementById("codex-reload").click();
    await flushCodex();
    if (outcome !== "reload-error") {
      const select = raceDocument.getElementById("codex-speed");
      select.value = "ultrafast";
      select.dispatchEvent({type: "change"});
      await raceContext.applyCodex();
      await flushCodex();
    }
    const newer = captureRace();
    slowCatalog.resolve({ok: true, body: {models: []}});
    await flushCodex();
    codexBootRaceStates.push({outcome, newer, final: captureRace(), requests: raceRequests});
  }

  const codexBootFailureStates = [];
  for (const outcome of ["http", "malformed", "network"]) {
    const failureDocument = new FakeDocument(html);
    const failureContext = Object.assign({}, context, {document: failureDocument});
    vm.createContext(failureContext);
    vm.runInContext(source, failureContext, {filename: javascriptPath});
    const failureRequests = [];
    failureContext.jfetch = (url, options) => {
      failureRequests.push({url, options: options || null});
      if (url === "/api/hello") return Promise.resolve({ok: true, body: {local_auth_required: false}});
      if (url === "/health") return Promise.resolve({ok: true, body: {status: "ok", providers: {codex: {status: "ok"}}}});
      if (url === "/admin/settings/mapping") return Promise.resolve({ok: true, body: {model_map: {}, custom_providers: []}});
      if (url === "/admin/providers/codex/capabilities") return Promise.resolve(codexCapabilitiesEnvelope());
      if (url === "/admin/settings/codex") {
        if (outcome === "network") return Promise.reject(new Error("initial Codex settings disconnected"));
        if (outcome === "http") return Promise.resolve({ok: false, status: 503, body: {error: {message: "<initial settings failed>"}}});
        return Promise.resolve({ok: true, status: 200, body: {service_tier: "priority", env_locked: false}});
      }
      if (url === "/admin/settings/compaction") return Promise.resolve({ok: true, body: {model: null, env_locked: false}});
      if (url === "/admin/providers/claude/pool/routing") return Promise.resolve({ok: true, body: {mode: "disabled", env_locked: false, include_local_login: true}});
      return Promise.resolve({ok: true, body: {models: [], log_level: "INFO"}});
    };
    failureContext.boot();
    await flushCodex();
    const failed = {hasLoaded: failureContext.CODEX.hasLoaded, value: failureDocument.getElementById("codex-speed").value,
      disabled: failureDocument.getElementById("codex-speed").disabled,
      testDisabled: failureDocument.getElementById("codex-speed-test").disabled,
      status: failureDocument.getElementById("codex-settings-status").textContent};
    failureContext.jfetch = (url, options) => {
      failureRequests.push({url, options: options || null});
      return Promise.resolve(url === "/admin/settings/codex" ? codexEnvelope("ultrafast") : codexCapabilitiesEnvelope("ultrafast"));
    };
    failureDocument.getElementById("codex-reload").click();
    await flushCodex();
    codexBootFailureStates.push({outcome, failed, recovered: {
      hasLoaded: failureContext.CODEX.hasLoaded, value: failureDocument.getElementById("codex-speed").value,
      disabled: failureDocument.getElementById("codex-speed").disabled,
    }, requests: failureRequests});
  }

  const optionalCatalogBoot = [];
  async function runOptionalCatalogBoot(provider, locked, authentication, outcome) {
    const optionalDocument = new FakeDocument(html);
    const requests = [];
    const decodedCore = [];
    const warnings = [];
    const pendingCatalog = createDeferred();
    const endpoint = `/admin/providers/${provider}/models`;
    const savedMapping = {sonnet: "codex:saved-main", haiku: "kimi:saved-secondary"};
    const coreEndpoints = ["/admin/settings/mapping", "/admin/settings/codex", "/admin/settings/log-level",
      "/admin/settings/compaction", "/admin/providers/claude/pool/routing"];
    let prompts = 0;
    const optionalContext = Object.assign({}, context, {
      document: optionalDocument,
      location: {host: "127.0.0.1:8787", hash: "#map"},
      console: {log() {}, info() {}, debug() {}, error(...values) {warnings.push(["error", ...values]);},
        warn(...values) {warnings.push(["warn", ...values]);}},
      window: Object.assign({}, context.window, {prompt: () => {
        prompts += 1;
        if (authentication === "none") throw new Error("unexpected optional-catalog token prompt");
        return authentication === "retry" && prompts === 1 ? "wrong-catalog-boot-token" : "catalog-boot-token";
      }}),
      fetch: async (url, options) => {
        const hasValidToken = options?.headers?.Authorization === "Bearer catalog-boot-token";
        requests.push({url, method: options?.method || "GET", hasValidToken});
        let status = 200;
        let body;
        if (url === "/api/hello") body = {local_auth_required: authentication !== "none"};
        else if (url.startsWith("/admin/") && authentication !== "none" && !hasValidToken) {
          status = 401;
          body = localTokenRejection("synthetic invalid local token");
        } else if (url === endpoint) return pendingCatalog.promise;
        else if (url === "/admin/settings/mapping") body = {model_map: savedMapping, env_locked: {model_map: locked}, custom_providers: []};
        else if (url === "/admin/settings/codex") body = {service_tier: "fast", env_locked: false};
        else if (url === "/admin/settings/log-level") body = {log_level: "debug", choices: ["debug", "info", "warning", "error"]};
        else if (url === "/admin/settings/compaction") body = {model: null, env_locked: false, last_reroute: null};
        else if (url === "/admin/providers/claude/pool/routing") body = {mode: "disabled", env_locked: false, include_local_login: true};
        else if (url === "/admin/providers/codex/capabilities") body = codexCapabilitiesEnvelope("fast", {
          "saved-main": {supported_tiers: ["standard", "fast"], advertised_tier_ids: ["priority"], status: "supported",
            reason: null, applicable_tier: "fast", wire_tier: "priority"},
        }).body;
        else if (url === "/health") body = {status: "ok", providers: {
          codex: {status: "ok", auth_mode: "api_key"}, kimi: {status: "ok", required: true}, grok: {status: "ok", required: false},
        }};
        else if (url === "/admin/providers/codex/models") body = {models: ["saved-main"]};
        else if (url.endsWith("/models")) body = {models: ["peer-suggestion"]};
        else throw new Error("unexpected optional-catalog boot endpoint: " + url);
        return {ok: status === 200, status, json: async () => {
          if (status === 200 && coreEndpoints.includes(url)) decodedCore.push(url);
          return body;
        }};
      },
    });
    vm.createContext(optionalContext);
    vm.runInContext(source, optionalContext, {filename: javascriptPath});
    const initial = {live: {...optionalContext.DIR.LIVE}, draft: {...optionalContext.DIR.mapping},
      locked: optionalContext.DIR.locked, codexLoaded: optionalContext.CODEX.hasLoaded};
    optionalContext.boot();
    await flushCodex();
    if (!coreEndpoints.every((url) => decodedCore.includes(url))) {
      throw new Error("core JSON responses did not settle before optional catalog failure");
    }
    const decodedBeforeFailure = decodedCore.slice();
    if (outcome === "network") pendingCatalog.reject(new TypeError(`synthetic disconnected ${provider} catalog fetch`));
    else pendingCatalog.resolve({ok: outcome !== "http", status: outcome === "http" ? 503 : 200, json: async () => {
      if (outcome === "malformed") throw new SyntaxError("synthetic malformed optional catalog JSON");
      return {error: {message: "synthetic optional catalog HTTP failure"}};
    }});
    await flushCodex();
    return {provider, locked, authentication, outcome, initial, decodedBeforeFailure, requests, prompts, warnings,
      toast: {html: optionalDocument.getElementById("toast")?.innerHTML || "",
        role: optionalDocument.getElementById("toast")?.getAttribute("role") || ""},
      live: {...optionalContext.DIR.LIVE}, draft: {...optionalContext.DIR.mapping}, isLocked: optionalContext.DIR.locked,
      isLockVisible: optionalDocument.body.classList.contains("env-locked"),
      sources: optionalContext.DIR.sources.map((node) => node.id), targets: optionalContext.DIR.targets.map((node) => node.id),
      codex: {live: optionalContext.CODEX.serviceTier, draft: optionalContext.CODEX.draft,
        hasLoaded: optionalContext.CODEX.hasLoaded, value: optionalDocument.getElementById("codex-speed").value,
        disabled: optionalDocument.getElementById("codex-speed").disabled},
      compaction: optionalContext.COMP.LIVE.model, routing: optionalContext.ROUTING.mode,
      health: optionalDocument.getElementById("health").innerHTML,
      capabilities: optionalDocument.getElementById("codex-capabilities-status").textContent,
      catalogs: {codex: Array.from(optionalContext.CATALOG.codex), kimi: Array.from(optionalContext.CATALOG.kimi), grok: Array.from(optionalContext.CATALOG.grok)},
    };
  }
  for (const provider of ["kimi", "grok"]) {
    for (const locked of [false, true]) {
      for (const authentication of ["none", "valid", "retry"]) {
        for (const outcome of ["network", "http", "malformed"]) {
          optionalCatalogBoot.push(await runOptionalCatalogBoot(provider, locked, authentication, outcome));
        }
      }
    }
  }

  const authErrorScopeStates = {wrappers: [], boots: [], concurrent: null};
  const providerAuthFailure = {error: {message: "<Synthetic Codex credentials unavailable>", type: "authentication_error", param: null, code: null}};
  const providerUpstreamFailure = {error: {message: "<Synthetic upstream catalog rejected>", type: "authentication_error", param: null, code: null}};
  for (const [name, status, body, authRequired, target, promptAnswer] of [
    ["provider-auth-cancel", 401, providerAuthFailure, true, "/admin/providers/codex/models", null],
    ["provider-upstream-cancel", 401, providerUpstreamFailure, true, "/admin/providers/codex/models", null],
    ["provider-auth-reenter", 401, providerAuthFailure, true, "/admin/providers/codex/models", "valid-local-token"],
    ["unclassified-401", 401, {}, true, "/admin/providers/codex/models", null],
    ["no-local-auth", 401, localTokenRejection(), false, "/admin/providers/codex/models", null],
    ["non-admin", 401, localTokenRejection(), true, "/public/catalog", null],
    ["non-401", 403, localTokenRejection(), true, "/admin/providers/codex/models", null],
    ["local-retry", 401, localTokenRejection(), true, "/admin/settings/log-level", "valid-local-token"],
    ["local-cancel", 401, localTokenRejection(), true, "/admin/settings/log-level", null],
    ["local-repeated-rejection", 401, localTokenRejection(), true, "/admin/settings/log-level", "valid-local-token"],
  ]) {
    const scopeDocument = new FakeDocument(html);
    const scopeContext = Object.assign({}, context, {document: scopeDocument});
    vm.createContext(scopeContext);
    vm.runInContext(source, scopeContext, {filename: javascriptPath});
    scopeContext.authRequired = authRequired;
    scopeContext.localToken = name.startsWith("local-") ? "wrong-local-token" : "valid-local-token";
    const scopeRequests = [];
    let prompts = 0;
    scopeContext.window = Object.assign({}, context.window, {prompt: () => {prompts += 1; return promptAnswer;}});
    scopeContext.fetch = async (url, options) => {
      const hasValidToken = options?.headers?.Authorization === "Bearer valid-local-token";
      scopeRequests.push({url, hasValidToken});
      if (url === target) {
        const shouldRecover = name === "local-retry" && hasValidToken;
        return {ok: shouldRecover, status: shouldRecover ? 200 : status, json: async () => shouldRecover ? {loaded: true} : body};
      }
      return {ok: hasValidToken || url === "/health", status: hasValidToken || url === "/health" ? 200 : 401,
        json: async () => hasValidToken || url === "/health" ? {loaded: true} : localTokenRejection()};
    };
    const response = await scopeContext.jfetch(target);
    const requestsAtResponse = scopeRequests.slice();
    const tokenRetained = scopeContext.localToken === "valid-local-token";
    const tokenCleared = scopeContext.localToken === null;
    const subsequent = await scopeContext.rawFetch("/admin/settings/codex");
    await scopeContext.rawFetch("/health");
    authErrorScopeStates.wrappers.push({name, status: response.status, body: response.body, prompts,
      requestsAtResponse, tokenRetained, tokenCleared, subsequentStatus: subsequent.status,
      subsequentRequests: scopeRequests.slice(requestsAtResponse.length)});
  }
  for (const [name, failure] of [["provider-auth", providerAuthFailure], ["provider-upstream", providerUpstreamFailure]]) {
    const bootDocument = new FakeDocument(html);
    const bootContext = Object.assign({}, context, {document: bootDocument, location: {host: "127.0.0.1:8787", hash: "#map"}});
    vm.createContext(bootContext);
    vm.runInContext(source, bootContext, {filename: javascriptPath});
    let prompts = 0;
    const bootRequests = [];
    bootContext.window = Object.assign({}, context.window, {prompt: () => {prompts += 1; return prompts === 1 ? "valid-local-token" : null;}});
    bootContext.fetch = async (url, options) => {
      const hasValidToken = options?.headers?.Authorization === "Bearer valid-local-token";
      bootRequests.push({url, hasValidToken});
      let status = 200;
      let body;
      if (url === "/api/hello") body = {local_auth_required: true};
      else if (url.startsWith("/admin/") && !hasValidToken) {status = 401; body = localTokenRejection();}
      else if (url === "/admin/providers/codex/models") {status = 401; body = failure;}
      else if (url === "/admin/settings/mapping") body = {model_map: {sonnet: "codex:saved-model"}, custom_providers: [], env_locked: {}};
      else if (url === "/admin/settings/codex") body = {service_tier: "fast", env_locked: false};
      else if (url === "/admin/providers/codex/capabilities") body = codexCapabilitiesEnvelope("fast").body;
      else if (url === "/admin/settings/compaction") body = {model: null, env_locked: false};
      else if (url === "/admin/providers/claude/pool/routing") body = {mode: "disabled", env_locked: false, include_local_login: true};
      else if (url === "/health") body = {status: "ok", providers: {codex: {status: "ok", auth_mode: "chatgpt", ...(hasValidToken ? {email: "fixture-identity@example.invalid"} : {})}}};
      else body = {models: [], log_level: "info"};
      return {ok: status === 200, status, json: async () => body};
    };
    bootContext.boot();
    await flushCodex();
    const promptsAfterBoot = prompts;
    const adminResponse = await bootContext.jfetch("/admin/settings/codex");
    await bootContext.jfetch("/health");
    authErrorScopeStates.boots.push({name, promptsAfterBoot, prompts, requests: bootRequests,
      tokenRetained: bootContext.localToken === "valid-local-token", subsequentStatus: adminResponse.status,
      mapping: {...bootContext.DIR.LIVE}, tier: bootContext.CODEX.serviceTier, hasLoaded: bootContext.CODEX.hasLoaded,
      identity: bootDocument.getElementById("codex-statline").innerHTML,
      feedback: bootDocument.getElementById("toast")?.innerHTML || ""});
  }
  const concurrentContext = Object.assign({}, context, {document: new FakeDocument(html)});
  vm.createContext(concurrentContext);
  vm.runInContext(source, concurrentContext, {filename: javascriptPath});
  concurrentContext.authRequired = true;
  concurrentContext.localToken = "wrong-local-token";
  const pendingLocalRejection = createDeferred();
  const concurrentRequests = [];
  let concurrentPrompts = 0;
  let slowRequests = 0;
  concurrentContext.window = Object.assign({}, context.window, {prompt: () => {concurrentPrompts += 1; return "valid-local-token";}});
  concurrentContext.fetch = async (url, options) => {
    const hasValidToken = options?.headers?.Authorization === "Bearer valid-local-token";
    concurrentRequests.push({url, hasValidToken});
    if (url === "/admin/settings/log-level" && ++slowRequests === 1) return pendingLocalRejection.promise;
    return {ok: hasValidToken, status: hasValidToken ? 200 : 401, json: async () => hasValidToken ? {loaded: true} : localTokenRejection()};
  };
  const slow = concurrentContext.jfetch("/admin/settings/log-level");
  const corrected = await concurrentContext.jfetch("/admin/settings/codex");
  pendingLocalRejection.resolve({ok: false, status: 401, json: async () => localTokenRejection()});
  const stale = await slow;
  authErrorScopeStates.concurrent = {prompts: concurrentPrompts, correctedStatus: corrected.status, staleStatus: stale.status,
    tokenRetained: concurrentContext.localToken === "valid-local-token", requests: concurrentRequests};

  const output = {credentialLeak};
  if (!credentialLeak) {
    Object.assign(output, {
      names: {configuredName, connectedName, unusedName, errorName},
      codexCatalogBoot,
      codexSpeedStates,
      codexSettingsStates,
      codexCapabilitiesStates,
      codexMetadataStates,
      codexMetadataFailureStates,
      codexSpeedTestStates,
      codexSpeedFailureStates,
      codexBootRaceStates,
      codexBootFailureStates,
      optionalCatalogBoot,
      healthAuthenticationBoot,
      authErrorScopeStates,
      localHeroStates,
      localHeroLoadOrder,
      routingEnvelopeValidation,
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
      operatorFeedback,
      usageRendering,
      gptProConcurrencyStates,
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
