"""DOM selectors and page probes used by the gptpro ask runner."""

from __future__ import annotations

COMPOSER_SELECTOR = (
    'form[data-chatgpt-composer] div.ProseMirror[contenteditable="true"]'
    '[role="textbox"][aria-label="Ask ChatGPT"]'
)
SEND_BUTTON_SELECTOR = (
    'form[data-chatgpt-composer] button[type="submit"][aria-label="Send"]'
)
STOP_BUTTON_SELECTOR = 'form[data-chatgpt-composer] button[aria-label="Stop"]'
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
