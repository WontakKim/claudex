"""DOM selectors and page probes used by the gptpro ask runner."""

from __future__ import annotations

# Accessible names are localized; composer and send locators match structure only.
COMPOSER_SELECTOR = (
    'form[data-chatgpt-composer] div.ProseMirror[contenteditable="true"]'
    '[role="textbox"]'
)
SEND_BUTTON_SELECTOR = 'form[data-chatgpt-composer] button[type="submit"]'
MODEL_PICKER_SELECTOR = 'button[data-composer-navigation-target="reasoning"]'
POWER_SELECTOR = '[role="menuitem"][data-reasoning-slider="true"]'
# The composer's trailing slot is voice (type=button with a data-state tooltip
# trigger) when empty, submit when text is present, and stop (type=button without
# data-state) while streaming. Accessible names are localized; match structure.
STOP_BUTTON_SELECTOR = (
    'form[data-chatgpt-composer] '
    'button.size-token-button-composer[type="button"]:not([data-state])'
)
# Message units carry a search-unit key whose prefix ("fallback-turn-N") is
# index-based and changes between renders; only the ":user"/":assistant"
# suffix is stable, so units are addressed by suffix match. The message-ids
# attribute holds one id on a user unit and space-separated ids on an
# assistant unit — probes read it on user units only, where it is the anchor.
USER_MESSAGE_SELECTOR = '[data-chatgpt-search-unit-key$=":user"]'
ASSISTANT_MESSAGE_SELECTOR = '[data-chatgpt-search-unit-key$=":assistant"]'
MESSAGE_ID_ATTRIBUTE = "data-chatgpt-search-message-ids"
MODAL_SELECTOR = '[id*="modal"], [role="dialog"]'
MODAL_BUTTON_TEXTS = ("Got it", "OK")

CHALLENGE_DOM_PROBE_JS = r"""
() => {
  const markers = [];
  try {
    if (document.querySelector(
      '#challenge-form, #challenge-running, #challenge-error-title, #cf-chl-widget'
    )) {
      markers.push('challenge-form');
    }
    if (document.querySelector('iframe[src*="challenges.cloudflare.com"]')) {
      markers.push('turnstile-iframe');
    }
    const title = (document.title || '').toLowerCase();
    if (
      title.includes('just a moment') ||
      title.includes('attention required') ||
      title.includes('verifying') ||
      title.includes('security check')
    ) {
      markers.push(`interstitial-title:${document.title}`);
    }
  } catch (_) {
    return [];
  }
  return markers;
}
"""

TOP_LEVEL_ROLE_PREDICATE_JS = (
    "(node) => !node.parentElement?.closest('[data-chatgpt-search-unit-key]')"
)

TOP_LEVEL_USER_IDS_PROBE_JS = r"""
(args) => {
  const isTopLevel = __TOP_LEVEL_ROLE_PREDICATE__;
  return Array.from(document.querySelectorAll(args.userSelector))
    .filter(isTopLevel)
    .map((node) => node.getAttribute(args.idAttribute))
    .filter((id) => typeof id === 'string' && id.length > 0);
}
""".replace("__TOP_LEVEL_ROLE_PREDICATE__", TOP_LEVEL_ROLE_PREDICATE_JS)

COMPOSER_READBACK_PROBE_JS = r"""
(args) => {
  const composer = document.querySelector(args.selector);
  if (!composer) return null;
  if (typeof composer.value === 'string') return composer.value;
  if (typeof composer.innerText === 'string') return composer.innerText;
  return composer.textContent;
}
"""

DISMISS_MODAL_PROBE_JS = r"""
(args) => {
  const isVisible = (element) => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' &&
      style.visibility !== 'hidden' &&
      Number(style.opacity) !== 0 &&
      rect.width > 0 && rect.height > 0;
  };
  const modal = Array.from(document.querySelectorAll(args.modalSelector))
    .find((element) => isVisible(element));
  if (!modal) return 'none';
  const button = Array.from(modal.querySelectorAll('button')).find((candidate) =>
    isVisible(candidate) && args.buttonTexts.some((text) =>
      text.toLowerCase() ===
        (candidate.innerText || candidate.textContent || '').trim().toLowerCase()
    )
  );
  if (button) {
    button.click();
    return 'clicked';
  }
  return 'escape';
}
"""

VISIBLE_MODAL_PROBE_JS = r"""
(args) => Array.from(document.querySelectorAll(args.modalSelector)).some((element) => {
  const style = getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  return style.display !== 'none' &&
    style.visibility !== 'hidden' &&
    Number(style.opacity) !== 0 &&
    rect.width > 0 && rect.height > 0;
})
"""

SEND_BUTTON_READY_PROBE_JS = r"""
(args) => {
  const button = document.querySelector(args.selector);
  if (!button || button.disabled || button.getAttribute('aria-disabled') === 'true') {
    return false;
  }
  const style = getComputedStyle(button);
  let rect = button.getBoundingClientRect();
  if (
    style.display === 'none' || style.visibility === 'hidden' ||
    style.pointerEvents === 'none' || Number(style.opacity) === 0 ||
    rect.width <= 0 || rect.height <= 0
  ) {
    return false;
  }
  let centerX = rect.left + rect.width / 2;
  let centerY = rect.top + rect.height / 2;
  if (centerX < 0 || centerX >= innerWidth || centerY < 0 || centerY >= innerHeight) {
    button.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    rect = button.getBoundingClientRect();
    centerX = rect.left + rect.width / 2;
    centerY = rect.top + rect.height / 2;
  }
  if (centerX < 0 || centerX >= innerWidth || centerY < 0 || centerY >= innerHeight) {
    return false;
  }
  const topElement = document.elementFromPoint(centerX, centerY);
  return topElement === button || (topElement !== null && button.contains(topElement));
}
"""

USER_ECHO_PROBE_JS = r"""
(args) => {
  const isTopLevel = __TOP_LEVEL_ROLE_PREDICATE__;
  const preIds = new Set(args.preIds);
  const matches = Array.from(document.querySelectorAll(args.userSelector))
    .filter(isTopLevel)
    .filter((node) => {
      const id = node.getAttribute(args.idAttribute);
      const text = node.textContent || '';
      return id && !preIds.has(id) && text.includes(args.nonceMarker);
    })
    .map((node) => node.getAttribute(args.idAttribute));
  return matches.length === 1 ? matches[0] : null;
}
""".replace("__TOP_LEVEL_ROLE_PREDICATE__", TOP_LEVEL_ROLE_PREDICATE_JS)

RELOCK_USER_ECHO_PROBE_JS = r"""
(args) => {
  const isTopLevel = __TOP_LEVEL_ROLE_PREDICATE__;
  const matches = Array.from(document.querySelectorAll(args.userSelector))
    .filter(isTopLevel)
    .filter((node) => {
      const id = node.getAttribute(args.idAttribute);
      return id && (node.textContent || '').includes(args.nonceMarker);
    })
    .map((node) => node.getAttribute(args.idAttribute));
  return matches.length === 1 ? matches[0] : null;
}
""".replace("__TOP_LEVEL_ROLE_PREDICATE__", TOP_LEVEL_ROLE_PREDICATE_JS)

TURN_STATE_PROBE_JS = r"""
(args) => {
  const isTopLevel = __TOP_LEVEL_ROLE_PREDICATE__;
  const users = Array.from(document.querySelectorAll(args.userSelector))
    .filter(isTopLevel);
  const lockedUser = users.find(
    (node) => node.getAttribute(args.idAttribute) === args.lockedUserId
  );
  const hasStop = document.querySelector(args.stopSelector) !== null;
  if (!lockedUser) {
    return {
      anchorPresent: false,
      assistantExists: false,
      assistantTextLength: 0,
      assistantMutationKey: '0:0',
      hasStop,
    };
  }

  const messages = Array.from(document.querySelectorAll(
    `${args.userSelector}, ${args.assistantSelector}`
  )).filter(isTopLevel);
  const assistantNodes = [];
  let isPastLockedUser = false;
  for (const node of messages) {
    if (!isPastLockedUser) {
      isPastLockedUser = node === lockedUser;
      continue;
    }
    if (node.matches(args.userSelector)) break;
    if (node.matches(args.assistantSelector)) assistantNodes.push(node);
  }

  let length = 0;
  let hash = 2166136261;
  for (const node of assistantNodes) {
    const text = node.textContent || '';
    length += text.length;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
  }
  return {
    anchorPresent: true,
    assistantExists: assistantNodes.length > 0,
    assistantTextLength: length,
    assistantMutationKey: `${length}:${hash >>> 0}`,
    hasStop,
  };
}
""".replace("__TOP_LEVEL_ROLE_PREDICATE__", TOP_LEVEL_ROLE_PREDICATE_JS)

PAGE_FETCH_PROBE_JS = r"""
async (args) => {
  if (location.origin !== args.origin) {
    return {
      status: 0,
      headers: {},
      text: '',
      json: null,
      fetchError: `Untrusted page origin ${location.origin}`,
      timedOut: false,
    };
  }
  let timedOut = false;
  const controller = new AbortController();
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, args.timeoutMs);
  try {
    const response = await fetch(args.url, {
      method: 'GET',
      headers: args.headers,
      credentials: 'include',
      signal: controller.signal,
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (_) {
      json = null;
    }
    return {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      text,
      json,
      fetchError: null,
      timedOut: false,
    };
  } catch (error) {
    const name = error && error.name ? String(error.name) : 'Error';
    const message = error && error.message ? String(error.message) : String(error);
    return {
      status: 0,
      headers: {},
      text: '',
      json: null,
      fetchError: timedOut
        ? `Timeout: request exceeded ${args.timeoutMs}ms`
        : `${name}: ${message}`,
      timedOut,
    };
  } finally {
    clearTimeout(timer);
  }
}
"""

# Fetches one response body as base64 with at most args.maxBytes bytes read,
# whether or not the server declares a Content-Length. Redirects are refused
# so credentials and signed URLs never follow a response to another location.
FILE_DOWNLOAD_PROBE_JS = r"""
async (args) => {
  const failure = (fetchError, timedOut) => ({
    status: 0,
    headers: {},
    bodyBase64: null,
    byteLength: 0,
    tooLarge: false,
    redirected: false,
    url: '',
    fetchError,
    timedOut,
  });
  if (location.origin !== args.origin) {
    return failure(`Untrusted page origin ${location.origin}`, false);
  }
  let timedOut = false;
  const controller = new AbortController();
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, args.timeoutMs);
  try {
    const response = await fetch(args.url, {
      method: 'GET',
      headers: args.headers,
      credentials: 'include',
      redirect: 'error',
      signal: controller.signal,
    });
    const result = {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      bodyBase64: null,
      byteLength: 0,
      tooLarge: false,
      redirected: response.redirected,
      url: response.url,
      fetchError: null,
      timedOut: false,
    };
    const declaredLength = Number(response.headers.get('content-length'));
    if (Number.isFinite(declaredLength) && declaredLength > args.maxBytes) {
      result.tooLarge = true;
      controller.abort();
      return result;
    }
    const chunks = [];
    let total = 0;
    if (response.body) {
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        total += value.byteLength;
        if (total > args.maxBytes) {
          result.tooLarge = true;
          result.byteLength = total;
          controller.abort();
          return result;
        }
        chunks.push(value);
      }
    }
    const bytes = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    let binary = '';
    for (let index = 0; index < bytes.length; index += 0x8000) {
      binary += String.fromCharCode.apply(null, bytes.subarray(index, index + 0x8000));
    }
    result.bodyBase64 = btoa(binary);
    result.byteLength = total;
    return result;
  } catch (error) {
    const name = error && error.name ? String(error.name) : 'Error';
    const message = error && error.message ? String(error.message) : String(error);
    return failure(
      timedOut ? `Timeout: request exceeded ${args.timeoutMs}ms` : `${name}: ${message}`,
      timedOut,
    );
  } finally {
    clearTimeout(timer);
  }
}
"""


# The Power menu is rendered in a portal, not inside the composer. IDs are
# checked locally in both directions, never used as persistent selectors.
PRO_CONTROLS_JS = r"""
(form, pickerSelector, powerSelector) => {
  const visible = element => {
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 &&
      element.checkVisibility({opacityProperty: true, visibilityProperty: true}) &&
      !element.closest('[hidden], [inert], [role="dialog"], dialog, nav, aside');
  };
  const enabled = element => !element.matches(':disabled, [aria-disabled="true"], [data-disabled]');
  const matches = (element, selector) => {
    try { return element.matches(selector); } catch (_) { return false; }
  };
  const labelLines = element => element.innerText.split('\n').map(line => line.trim()).filter(Boolean);
  const pickers = [...form.querySelectorAll('button[type="button"][aria-haspopup="menu"]')]
    .filter(element => element.form === form && visible(element) &&
      /^(true|false)$/.test(element.getAttribute('aria-expanded')) &&
      (element.getAttribute('aria-label') === 'Select ChatGPT model' ||
       labelLines(element)[0] === 'Thinking effort'));
  const picker = pickers.length === 1 && matches(pickers[0], pickerSelector) &&
    enabled(pickers[0]) ? pickers[0] : null;
  const expanded = picker?.getAttribute('aria-expanded') === 'true';
  const lines = picker ? labelLines(picker) : [];
  // A closed picker shows its level alone or under a 'Thinking effort' caption.
  const pickerLevel = lines.length === 2 && lines[0] === 'Thinking effort' ? lines[1]
    : lines.length === 1 ? lines[0] : null;
  const controls = expanded && picker.id && picker.getAttribute('aria-controls');
  const menus = controls ? [...document.querySelectorAll('[role="menu"]')]
    .filter(element => element.id === controls && visible(element) &&
      element.getAttribute('aria-labelledby') === picker.id) : [];
  const menu = menus.length === 1 ? menus[0] : null;
  const describePower = element => {
    const sliders = [...element.querySelectorAll('[role="slider"]')];
    const statuses = [...menu.querySelectorAll('[role="status"][id]')].filter(status =>
      status.closest('[role="menu"]') === menu &&
      (element.getAttribute('aria-describedby') || '').split(/\s+/).includes(status.id));
    const keys = (element.getAttribute('aria-keyshortcuts') || '').split(/\s+/);
    if (!keys.includes('ArrowLeft') || !keys.includes('ArrowRight') ||
        sliders.length !== 1 || statuses.length !== 1) return null;
    const slider = sliders[0];
    const numbers = ['aria-valuemin', 'aria-valuemax', 'aria-valuenow'].map(name => slider.getAttribute(name));
    if (!numbers.every(value => /^\d+$/.test(value))) return null;
    const [min, max, position] = numbers.map(Number);
    const status = /^([^,]+), (\d+) of (\d+)\.?$/.exec(statuses[0].textContent.trim());
    if (!status || min > position || position > max ||
        Number(status[2]) !== position - min + 1 || Number(status[3]) !== max - min + 1) return null;
    return {level: status[1].trim(), min, max, position};
  };
  const powers = menu ? [...menu.querySelectorAll('[role="menuitem"][aria-label="Power"]')]
    .filter(element => element.closest('[role="menu"]') === menu && visible(element)) : [];
  const feedback = powers.length === 1 && matches(powers[0], powerSelector) && enabled(powers[0])
    ? describePower(powers[0]) : null;
  return {picker, power: feedback ? powers[0] : null, pickers, powers, menu, expanded, pickerLevel, feedback};
}
"""


# Composers are visible editable roots in forms outside dialogs, navigation,
# message units, and search areas; known composer forms constrain eligibility when
# present, otherwise generic forms qualify. Send buttons qualify only in the form
# of exactly one selector-matched composer that satisfies this same predicate.
LOCATOR_CONTRACT_JS = r"""
(target, composerSelector) => {
  const isVisibleCandidate = (element) => {
    const style = getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return !(
      style.display === 'none' || style.visibility === 'hidden' ||
      Number(style.opacity) === 0 || rect.width <= 0 || rect.height <= 0 ||
      element.closest('[role="dialog"], dialog, nav, aside')
    );
  };
  const isEligibleComposer = (element) => {
    if (!isVisibleCandidate(element)) return false;
    if (
      element.closest('__MESSAGE_UNIT_SELECTOR__, [role="search"], search') ||
      element.getAttribute('type') === 'search'
    ) return false;
    const form = element.closest('form');
    if (form === null) return false;
    if (
      document.querySelector('form[data-chatgpt-composer]') !== null &&
      !form.matches('form[data-chatgpt-composer]')
    ) return false;
    return element.getAttribute('contenteditable') === 'true' ||
      element.tagName === 'TEXTAREA';
  };
  if (target === 'composer') return isEligibleComposer;
  let composerForm = null;
  if (target === 'send') {
    try {
      const composers = document.querySelectorAll(composerSelector);
      if (composers.length === 1 && isEligibleComposer(composers[0])) {
        composerForm = composers[0].closest('form');
      }
    } catch (_) {
      // An invalid composer selector cannot establish form ownership.
    }
  }
  return (element) => target === 'send' && isVisibleCandidate(element) &&
    element.tagName === 'BUTTON' && element.type === 'submit' &&
    composerForm !== null && element.form === composerForm;
}
""".replace("__MESSAGE_UNIT_SELECTOR__", USER_MESSAGE_SELECTOR.split("$=", 1)[0] + "]")

# Match only in the semantic scope, even when an unrelated portal or form has
# identical markup. Rediscovery receives fixed structural selectors, not DOM values.
PRO_LOCATOR_SCOPE_JS = r"""
(args, composerContract) => {
  let composers;
  try { composers = document.querySelectorAll(args.composerSelector); }
  catch (_) { return null; }
  if (composers.length !== 1 || !composerContract(composers[0])) return null;
  const form = composers[0].closest('form');
  const ui = (__CONTROLS__)(form, args.target === 'power' ? args.pickerSelector : '*', '*');
  if (args.target === 'power') {
    return {candidates: ui.powers, control: ui.power,
      selector: '[role="menuitem"][aria-label="Power"][aria-keyshortcuts][aria-describedby]'};
  }
  return {
    candidates: ui.pickers, control: ui.picker,
    selector: 'button[type="button"][aria-haspopup="menu"][aria-expanded]'
  };
}
""".replace("__CONTROLS__", PRO_CONTROLS_JS)

LOCATOR_CHECK_PROBE_JS = r"""
(args) => {
  if (args.target === 'picker' || args.target === 'power') {
    const scope = (__PRO_SCOPE__)(args, __LOCATOR_CONTRACT__('composer'));
    let matches = [];
    let invalidSelector = false;
    try {
      document.querySelector(args.selector);
      matches = scope?.candidates.filter(element => element.matches(args.selector)) || [];
    } catch (_) { invalidSelector = true; }
    return {invalidSelector, matched: matches.length,
      eligible: Number(matches.includes(scope?.control)),
      candidates: scope?.candidates.length > 1 ? scope.candidates.length : Number(!!scope?.control),
      lang: document.documentElement.lang || ''};
  }
  const contract = __LOCATOR_CONTRACT__(args.target, args.composerSelector);
  let invalidSelector = false;
  let matches = [];
  try {
    matches = Array.from(document.querySelectorAll(args.selector));
  } catch (_) {
    invalidSelector = true;
  }
  const pool = args.target === 'composer' ? '[contenteditable="true"], textarea' : 'button';
  return {
    invalidSelector,
    matched: matches.length,
    eligible: matches.filter(contract).length,
    candidates: Array.from(document.querySelectorAll(pool)).filter(contract).length,
    lang: document.documentElement.lang || '',
  };
}
""".replace("__LOCATOR_CONTRACT__", "(" + LOCATOR_CONTRACT_JS + ")").replace(
    "__PRO_SCOPE__", PRO_LOCATOR_SCOPE_JS,
)

# Candidates expose structural metadata and aria-label UI labels, never conversation text,
# input values, or element IDs. Element and form data attributes expose names only
# (excluding data-state), never data-* values.
LOCATOR_CANDIDATES_PROBE_JS = r"""
(args) => {
  if (args.target === 'picker' || args.target === 'power') {
    const scope = (__PRO_SCOPE__)(args, __LOCATOR_CONTRACT__('composer'));
    const candidates = scope?.control?.matches(scope.selector)
      ? [{index: 0, selector: scope.selector, ...(args.target === 'picker'
          ? {tag: 'button', popup: 'menu', inComposer: true}
          : {role: 'menuitem', containsSlider: true, inLinkedMenu: true})}]
      : [];
    return {candidates};
  }
  const contract = __LOCATOR_CONTRACT__(args.target, args.composerSelector);
  const dataNames = (element) => Array.from(element.attributes)
    .map(attribute => attribute.name)
    .filter(name => name.startsWith('data-') && name !== 'data-state');
  const presence = (element) => dataNames(element).map(name => `[${CSS.escape(name)}]`).join('');
  const compile = (element) => {
    const form = element.closest('form') || element.form;
    const scope = form ? 'form' + presence(form) + ' ' : '';
    let base = element.tagName.toLowerCase();
    for (const name of ['role', 'type', 'contenteditable']) {
      if (element.hasAttribute(name)) base += `[${name}="${CSS.escape(element.getAttribute(name))}"]`;
    }
    const stableClasses = Array.from(element.classList)
      .filter(name => !name.includes(':') && !/focus|active|hover|open|closed|selected|disabled|checked/i.test(name))
      .slice(0, 3).map(name => '.' + CSS.escape(name)).join('');
    for (const selector of [scope + base, scope + base + presence(element),
                           scope + base + presence(element) + stableClasses]) {
      const matches = document.querySelectorAll(selector);
      if (matches.length === 1 && matches[0] === element) return selector;
    }
    return null;
  };
  const pool = args.target === 'composer' ? '[contenteditable="true"], textarea' : 'button';
  const fraction = (value, size) => Math.round(value / Math.max(1, size) * 100) / 100;
  const candidates = Array.from(document.querySelectorAll(pool)).filter(contract).map((element, index) => {
    const form = element.closest('form') || element.form;
    const ancestors = [];
    let ancestor = element.parentElement;
    while (ancestor && ancestors.length < 6) {
      const role = ancestor.getAttribute('role');
      ancestors.push(ancestor.tagName.toLowerCase() + (role ? `[role=${role}]` : ''));
      ancestor = ancestor.parentElement;
    }
    const rect = element.getBoundingClientRect();
    return {
      index, selector: compile(element), tag: element.tagName.toLowerCase(),
      type: element.getAttribute('type') || '', role: element.getAttribute('role') || '',
      ariaLabel: (element.getAttribute('aria-label') || '').slice(0, 80),
      dataAttributes: dataNames(element),
      classes: Array.from(element.classList).slice(0, 6),
      formDataAttributes: form ? dataNames(form) : [], ancestry: ancestors.join(' < '),
      box: {x: fraction(rect.x, innerWidth), y: fraction(rect.y, innerHeight),
            width: fraction(rect.width, innerWidth), height: fraction(rect.height, innerHeight)},
      disabled: element.disabled === true || element.getAttribute('aria-disabled') === 'true',
    };
  });
  return {lang: document.documentElement.lang || '', candidates};
}
""".replace("__LOCATOR_CONTRACT__", "(" + LOCATOR_CONTRACT_JS + ")").replace(
    "__PRO_SCOPE__", PRO_LOCATOR_SCOPE_JS,
)
