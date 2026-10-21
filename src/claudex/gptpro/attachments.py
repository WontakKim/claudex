"""Plain-text attachment upload support for ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from claudex.gptpro.conversation import is_trusted_origin_url
from claudex.gptpro.selectors import COMPOSER_SELECTOR, LOCATOR_CONTRACT_JS

MAX_ATTACHMENTS_PER_ASK = 10
_MAX_ATTACHMENT_RESPONSE_TASKS = MAX_ATTACHMENTS_PER_ASK * 4
ATTACH_SETTLE_TIMEOUT_SECONDS = 120.0
ATTACH_SETTLE_POLL_SECONDS = 0.25
FILE_CREATE_PATH = "/backend-api/files"
FILE_PROCESSING_PATH = "/backend-api/files/process_upload_stream"
MAX_PROCESSING_RESPONSE_BYTES = 262_144
MAX_TOTAL_ATTACHMENT_BYTES = 1_200_000
_MAX_PRE_SUBMIT_REVALIDATION_SECONDS = 2.0

_PLAIN_TEXT_MIME = "text/plain"

CREATE_ATTACHMENT_FILE_JS = """({ bytesBase64, name, mime }) => {
  const binary = atob(bytesBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new File([bytes], name, { type: mime });
}"""

CREATE_ATTACHMENT_DATA_TRANSFER_JS = """(files) => {
  const dataTransfer = new DataTransfer();
  for (const file of files) dataTransfer.items.add(file);
  return dataTransfer;
}"""

COMPOSER_FORM_CURRENT_JS = (
    "(form, selector) => {"
    "const composers = Array.from(document.querySelectorAll(selector));"
    "if (!form.isConnected || composers.length !== 1 || "
    "composers[0].closest('form') !== form) return false;"
    'const eligible = (' + LOCATOR_CONTRACT_JS + ')("composer", selector);'
    "if (!eligible(composers[0])) return false;"
    "let current = composers[0];"
    "while (current) {"
    "const style = getComputedStyle(current);"
    "if (style.display === 'none' || style.visibility === 'hidden' || "
    "style.visibility === 'collapse' || Number(style.opacity) === 0 || "
    "current.getAttribute('aria-hidden') === 'true') return false;"
    "current = current.parentElement;"
    "}"
    "return true;"
    "}"
)

CAPTURE_COMPOSER_FORM_HANDLE_JS = (
    "(selector) => {"
    "const composers = Array.from(document.querySelectorAll(selector));"
    "if (composers.length !== 1) return null;"
    "const form = composers[0].closest('form');"
    "return form && (" + COMPOSER_FORM_CURRENT_JS + ")(form, selector) ? form : null;"
    "}"
)

DISPATCH_ATTACHMENT_DROP_JS = """(dataTransfer, intendedForm = null) => {
  const preferredSelector = 'form[data-chatgpt-composer]';
  const fallbackSelector = 'main';
  let target;
  if (intendedForm !== null) {
    if (!intendedForm.isConnected) {
      throw new Error('The intended composer form changed before attachment drop.');
    }
    target = intendedForm;
  } else {
    target = document.querySelector(preferredSelector) ??
      document.querySelector(fallbackSelector);
  }
  if (!target) {
    throw new Error(
      `Attachment drop target not found (${preferredSelector} or ${fallbackSelector}).`,
    );
  }

  for (const eventType of ['dragenter', 'dragover', 'drop']) {
    target.dispatchEvent(
      new DragEvent(eventType, {
        bubbles: true,
        cancelable: true,
        composed: true,
        dataTransfer,
      }),
    );
  }
  if (intendedForm !== null) return 'intended-form';
  return target.matches(preferredSelector)
    ? preferredSelector
    : fallbackSelector;
}"""

def _dispatch_attachment_drop_for_form_js(composer_selector: str) -> str:
    selector_literal = json.dumps(composer_selector)
    return (
        "(form, dataTransfer) => {"
        f"if (!({COMPOSER_FORM_CURRENT_JS})(form, {selector_literal})) "
        "throw new Error('The intended composer form changed before attachment drop.');"
        f"return ({DISPATCH_ATTACHMENT_DROP_JS})(dataTransfer, form);"
        "}"
    )


DISPATCH_ATTACHMENT_DROP_FOR_FORM_JS = (
    _dispatch_attachment_drop_for_form_js(COMPOSER_SELECTOR)
)

READ_EXISTING_ATTACHMENT_NAMES_JS = """(intendedForm = null) => {
  const form = intendedForm ?? document.querySelector('form[data-chatgpt-composer]');
  if (!form) return [];
  return [...form.querySelectorAll('button[aria-label]')]
    .map(button => button.getAttribute('aria-label'))
    .filter(label => label.startsWith('Remove '))
    .map(label => label.slice('Remove '.length));
}"""

READ_EXISTING_ATTACHMENT_NAMES_FOR_FORM_JS = (
    f"(form) => ({READ_EXISTING_ATTACHMENT_NAMES_JS})(form)"
)

READ_COMPOSER_ATTACHMENT_STATE_JS = r"""(filenames, intendedForm = null) => {
  const form = intendedForm ?? document.querySelector('form[data-chatgpt-composer]');
  const state = { ready: {}, processing: {}, failed: {}, unknown: {} };
  if (!form) return state;
  const expected = new Set(filenames);
  const attributes = ['title', 'aria-label', 'data-filename'];
  const hasFilenameAttribute = (node, filename) =>
    attributes.some((attribute) => node.getAttribute(attribute) === filename);
  const candidates = Array.from(form.querySelectorAll('*'))
    .filter((node) => node.tagName !== 'BUTTON')
    .filter((node) => !node.closest('[contenteditable], [role="textbox"]'))
    .filter((node) => expected.has((node.textContent || '').trim()) ||
      filenames.some((name) => hasFilenameAttribute(node, name)));
  const labels = candidates.filter((node) =>
    !candidates.some((child) => child !== node && node.contains(child))
  );
  const seen = new Set();
  for (const node of labels) {
    const filename = filenames.find((name) =>
      hasFilenameAttribute(node, name) || (node.textContent || '').trim() === name
    );
    const statusFor = (chip) => {
      const statusText = (chip.textContent || '').replace(filename, '').trim();
      if (/^(upload failed|failed|error)\b/i.test(statusText) ||
          [node, chip].some((element) =>
            /^(failed|error)$/i.test(element.getAttribute('data-state') || '') ||
            element.getAttribute('aria-invalid') === 'true')) return 'failed';
      if (/^(uploading|processing|pending)\b/i.test(statusText) ||
          [node, chip].some((element) =>
            element.getAttribute('aria-busy') === 'true' ||
            /^(uploading|processing|pending)$/i.test(element.getAttribute('data-state') || '')) ||
          chip.querySelector(':scope > [role="progressbar"]')) return 'processing';
      return 'ready';
    };
    let chip = node.parentElement === form ? node : node.parentElement;
    const fallback = hasFilenameAttribute(node, filename) ? chip : null;
    let confirmed = false;
    for (let depth = 0; depth < 2 && chip && chip !== form; depth += 1) {
      const parent = chip.parentElement;
      const canInspectParent = depth === 0 && parent && parent !== form &&
        !labels.some((other) => other !== node && parent.contains(other));
      if (canInspectParent && statusFor(parent) !== 'ready') {
        chip = parent;
        confirmed = true;
        break;
      }
      const hasAction = Array.from(chip.children).some((child) =>
        child.tagName === 'BUTTON' &&
        (child.getAttribute('aria-label') === filename ||
         /remove|delete|close/i.test(child.getAttribute('aria-label') || ''))
      );
      if (hasAction || statusFor(chip) !== 'ready' ||
          hasFilenameAttribute(chip, filename)) {
        confirmed = true;
        break;
      }
      if (!canInspectParent) break;
      chip = parent;
    }
    if (!confirmed) chip = fallback;
    if (!chip) {
      state.unknown[filename] = (state.unknown[filename] || 0) + 1;
      continue;
    }
    if (seen.has(chip)) continue;
    seen.add(chip);
    const category = statusFor(chip);
    state[category][filename] = (state[category][filename] || 0) + 1;
  }
  return state;
}"""

READ_COMPOSER_ATTACHMENT_STATE_FOR_FORM_JS = (
    f"(form, filenames) => ({READ_COMPOSER_ATTACHMENT_STATE_JS})(filenames, form)"
)

CREATE_ATTACHMENT_BINDING_TRACKER_JS = r"""({ form: suppliedForm, composerSelector, names }) => {
  const form = suppliedForm ?? document.querySelectorAll(composerSelector)[0]?.closest('form');
  const args = {
    maxRendererReturnLinks: 512, maxRendererSiblingScan: 128,
    maxRendererOwnershipNodes: 512, maxRendererOwnershipDepth: 64,
    maxRendererContainerAncestors: 128,
  };
  /*
   * Copyright (c) Meta Platforms, Inc. and affiliates.
   * Permission is hereby granted, free of charge, to any person obtaining a copy
   * of this software and associated documentation files (the "Software"), to deal
   * in the Software without restriction, including without limitation the rights
   * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   * copies of the Software, and to permit persons to whom the Software is
   * furnished to do so, subject to the following conditions:
   *
   * The above copyright notice and this permission notice shall be included in
   * all copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   * IN THE SOFTWARE.
   *
   * Current-branch and host-ownership rules follow ReactFiberTreeReflection.js
   * and ReactDOMComponentTree.js; the live app's version is not asserted.
   */
  // Fiber and metadata reads use own data descriptors so application getters are never evaluated.
  const hasOwnValue = (descriptor) =>
    descriptor && Object.prototype.hasOwnProperty.call(descriptor, 'value');
  const HOST_ROOT_TAG = 3;
  const HOST_PORTAL_TAG = 4;
  const HOST_COMPONENT_TAG = 5;
  const HOST_TEXT_TAG = 6;
  const PENDING_RENDER_FLAGS = 2 | 4096;
  const failCurrentness = (state) => {
    const error = new Error('bounded renderer inspection stopped');
    error.currentnessState = state;
    throw error;
  };
  const isObject = (value) => value !== null && typeof value === 'object';
  const readOwnData = (object, key) => {
    if (!isObject(object)) failCurrentness('invalid-structural-object');
    let descriptor;
    try { descriptor = Object.getOwnPropertyDescriptor(object, key); }
    catch (_) { failCurrentness('descriptor-read-error'); }
    if (!descriptor) failCurrentness('missing-own-structural-field');
    if (!hasOwnValue(descriptor)) failCurrentness('accessor-not-read');
    return descriptor.value;
  };
  const readOptionalOwnData = (object, key) => {
    if (!isObject(object)) return { state: 'invalid-object' };
    let descriptor;
    try { descriptor = Object.getOwnPropertyDescriptor(object, key); }
    catch (_) { return { state: 'descriptor-read-error' }; }
    if (!descriptor) return { state: 'absent' };
    if (!hasOwnValue(descriptor)) return { state: 'accessor-not-read' };
    return { state: 'value', value: descriptor.value };
  };
  const readFiberTag = (fiber) => {
    const tag = readOwnData(fiber, 'tag');
    if (!Number.isSafeInteger(tag) || tag < 0) failCurrentness('invalid-fiber-tag');
    return tag;
  };
  const readFiberFlags = (fiber) => {
    const flags = readOwnData(fiber, 'flags');
    if (!Number.isSafeInteger(flags) || flags < 0) failCurrentness('invalid-fiber-flags');
    if ((flags & PENDING_RENDER_FLAGS) !== 0) failCurrentness('placement-or-hydration-pending');
    return flags;
  };
  const formWithinRendererContainer = (form, container) => {
    if (!form || form.isConnected !== true || !isObject(container)) {
      failCurrentness('form-or-container-disconnected');
    }
    const visited = new Set();
    let node = form;
    for (let depth = 0; depth <= args.maxRendererContainerAncestors; depth += 1) {
      if (node === container) return true;
      if (node === null || node === undefined) break;
      if (visited.has(node)) failCurrentness('container-parent-cycle');
      visited.add(node);
      if (depth === args.maxRendererContainerAncestors) break;
      node = node.parentNode;
    }
    failCurrentness('form-outside-renderer-container');
  };
  const verifyRootContext = (rootFiber, form) => {
    if (readFiberTag(rootFiber) !== HOST_ROOT_TAG) failCurrentness('host-root-tag-mismatch');
    if (readOwnData(rootFiber, 'return') !== null) failCurrentness('host-root-has-parent');
    const rootState = readOwnData(rootFiber, 'stateNode');
    if (!isObject(rootState)) failCurrentness('invalid-renderer-root-state');
    const alternate = readOwnData(rootFiber, 'alternate');
    if (alternate !== null) {
      if (!isObject(alternate) || readOwnData(alternate, 'alternate') !== rootFiber ||
          readFiberTag(alternate) !== HOST_ROOT_TAG || readOwnData(alternate, 'stateNode') !== rootState ||
          readOwnData(alternate, 'return') !== null) {
        failCurrentness('malformed-host-root-alternate');
      }
    }
    const currentRoot = readOwnData(rootState, 'current');
    if (currentRoot !== rootFiber && currentRoot !== alternate) {
      failCurrentness('renderer-current-root-mismatch');
    }
    if (readFiberTag(currentRoot) !== HOST_ROOT_TAG ||
        readOwnData(currentRoot, 'stateNode') !== rootState ||
        readOwnData(currentRoot, 'return') !== null) {
      failCurrentness('invalid-current-host-root');
    }
    const containerInfo = readOwnData(rootState, 'containerInfo');
    formWithinRendererContainer(form, containerInfo);
    return { rootFiber, rootState, currentRoot, containerInfo };
  };
  const verifyRootPair = (rootA, rootB, form) => {
    if (rootA === rootB || readFiberTag(rootA) !== HOST_ROOT_TAG ||
        readFiberTag(rootB) !== HOST_ROOT_TAG ||
        readOwnData(rootA, 'alternate') !== rootB ||
        readOwnData(rootB, 'alternate') !== rootA ||
        readOwnData(rootA, 'stateNode') !== readOwnData(rootB, 'stateNode')) {
      failCurrentness('malformed-host-root-pair');
    }
    const context = verifyRootContext(rootA, form);
    if (context.currentRoot !== rootA && context.currentRoot !== rootB) {
      failCurrentness('renderer-current-root-not-in-pair');
    }
    return context;
  };
  const scanChildSetForPair = (firstChild, fiberA, fiberB) => {
    const seen = new Set();
    const matches = [];
    let child = firstChild;
    for (let index = 0; child !== null && index < args.maxRendererSiblingScan; index += 1) {
      if (!isObject(child)) failCurrentness('invalid-sibling-fiber');
      if (seen.has(child)) failCurrentness('sibling-cycle');
      seen.add(child);
      if (child === fiberA || child === fiberB) matches.push(child);
      child = readOwnData(child, 'sibling');
    }
    if (child !== null) failCurrentness('sibling-scan-limit');
    if (matches.length > 1) failCurrentness('multiple-child-memberships');
    return matches.length === 1 ? matches[0] : null;
  };
  const rootContextForMountedFiber = (fiber, form) => {
    const seen = new Set();
    let node = fiber;
    for (let depth = 0; depth <= args.maxRendererReturnLinks; depth += 1) {
      if (!isObject(node) || seen.has(node)) failCurrentness('return-cycle-or-invalid-fiber');
      seen.add(node);
      readFiberFlags(node);
      const tag = readFiberTag(node);
      if (tag === HOST_ROOT_TAG) return verifyRootContext(node, form);
      const parent = readOwnData(node, 'return');
      if (parent === null) failCurrentness('disconnected-renderer-tree');
      if (!isObject(parent)) failCurrentness('invalid-return-fiber');
      node = parent;
    }
    failCurrentness('renderer-return-limit');
  };
  const resolveCurrentFiber = (fiber, form) => {
    try {
      if (!isObject(fiber)) failCurrentness('invalid-candidate-fiber');
      const originalAlternate = readOwnData(fiber, 'alternate');
      if (originalAlternate === null) {
        const seen = new Set();
        let node = fiber;
        for (let depth = 0; depth <= args.maxRendererReturnLinks; depth += 1) {
          if (!isObject(node) || seen.has(node)) failCurrentness('return-cycle-or-invalid-fiber');
          seen.add(node);
          readFiberFlags(node);
          const tag = readFiberTag(node);
          const parent = readOwnData(node, 'return');
          if (tag === HOST_ROOT_TAG) {
            const context = verifyRootContext(node, form);
            if (parent !== null || context.currentRoot !== node) {
              failCurrentness('unpaired-fiber-not-in-current-root');
            }
            return {
              state: 'resolved', candidate_is_current: true, current_fiber: fiber,
              root_state: context.rootState, container_info: context.containerInfo,
              current_root: context.currentRoot, resolution_basis: 'unpaired-root-path',
            };
          }
          if (parent === null || !isObject(parent)) failCurrentness('disconnected-renderer-tree');
          const found = scanChildSetForPair(readOwnData(parent, 'child'), node, node);
          if (found !== node) failCurrentness('unpaired-fiber-not-in-parent-children');
          node = parent;
        }
        failCurrentness('renderer-return-limit');
      }
      if (!isObject(originalAlternate) ||
          readOwnData(originalAlternate, 'alternate') !== fiber) {
        failCurrentness('malformed-candidate-alternate-pair');
      }
      let a = fiber;
      let b = originalAlternate;
      const seenA = new Set();
      const seenB = new Set();
      for (let depth = 0; depth <= args.maxRendererReturnLinks; depth += 1) {
        if (!isObject(a) || !isObject(b) || a === b || seenA.has(a) || seenB.has(b)) {
          failCurrentness('return-cycle-or-invalid-alternate-pair');
        }
        seenA.add(a);
        seenB.add(b);
        readFiberFlags(a);
        readFiberFlags(b);
        if (readOwnData(a, 'alternate') !== b || readOwnData(b, 'alternate') !== a) {
          failCurrentness('non-reciprocal-alternate-pair');
        }
        const parentA = readOwnData(a, 'return');
        const parentBFromA = parentA === null ? null : readOwnData(parentA, 'alternate');
        if (parentA === null) {
          if (readOwnData(b, 'return') !== null) failCurrentness('root-return-mismatch');
          const context = verifyRootPair(a, b, form);
          const candidateIsCurrent = context.currentRoot === a;
          return {
            state: 'resolved', candidate_is_current: candidateIsCurrent,
            current_fiber: candidateIsCurrent ? fiber : originalAlternate,
            root_state: context.rootState, container_info: context.containerInfo,
            current_root: context.currentRoot, resolution_basis: 'root-current-reference',
          };
        }
        if (!isObject(parentA) || parentBFromA === null || !isObject(parentBFromA) ||
            readOwnData(parentBFromA, 'alternate') !== parentA) {
          failCurrentness('unpaired-or-malformed-parent');
        }
        const childrenA = readOwnData(parentA, 'child');
        const childrenB = readOwnData(parentBFromA, 'child');
        if (childrenA === childrenB) {
          const sharedChildMatch = scanChildSetForPair(childrenA, a, b);
          if (sharedChildMatch === null) failCurrentness('shared-child-not-found');
          const contextA = rootContextForMountedFiber(parentA, form);
          const contextB = rootContextForMountedFiber(parentBFromA, form);
          if (contextA.rootState !== contextB.rootState ||
              (contextA.rootFiber !== contextB.rootFiber &&
               (readOwnData(contextA.rootFiber, 'alternate') !== contextB.rootFiber ||
                readOwnData(contextB.rootFiber, 'alternate') !== contextA.rootFiber))) {
            failCurrentness('shared-child-root-mismatch');
          }
          return {
            state: 'resolved', candidate_is_current: sharedChildMatch === a,
            current_fiber: sharedChildMatch === a ? fiber : originalAlternate,
            root_state: contextA.rootState, container_info: contextA.containerInfo,
            current_root: contextA.currentRoot, resolution_basis: 'shared-child-set',
          };
        }
        const parentB = readOwnData(b, 'return');
        if (parentA !== parentB) {
          a = parentA;
          b = parentBFromA;
        } else {
          const inChildrenA = scanChildSetForPair(childrenA, a, b);
          if (inChildrenA !== null) {
            if (inChildrenA === a) {
              a = parentA;
              b = parentBFromA;
            } else {
              b = parentA;
              a = parentBFromA;
            }
          } else {
            const inChildrenB = scanChildSetForPair(childrenB, a, b);
            if (inChildrenB === null) failCurrentness('fiber-not-in-either-parent-child-set');
            if (inChildrenB === a) {
              a = parentBFromA;
              b = parentA;
            } else {
              a = parentA;
              b = parentBFromA;
            }
          }
        }
      }
      failCurrentness('renderer-return-limit');
    } catch (error) {
      return {
        state: 'inconclusive', reason: error && error.currentnessState
          ? error.currentnessState : 'renderer-inspection-error',
      };
    }
  };
  const ownsSelectedHostByChildren = (owner, host) => {
    try {
      if (!isObject(owner) || !isObject(host) || owner === host ||
          readFiberTag(host) !== HOST_COMPONENT_TAG || !isObject(readOwnData(host, 'stateNode'))) {
        failCurrentness('invalid-owner-or-selected-host');
      }
      const selectedHostNode = readOwnData(host, 'stateNode');
      const stack = [{ firstChild: readOwnData(owner, 'child'), parent: owner, depth: 1 }];
      const seen = new Set();
      let visitedCount = 0;
      let hostNodeMatchCount = 0;
      let exactFiberMatchCount = 0;
      while (stack.length > 0) {
        const item = stack.pop();
        if (item.depth > args.maxRendererOwnershipDepth) failCurrentness('ownership-depth-limit');
        const parentAlternate = readOwnData(item.parent, 'alternate');
        const parentFirstChild = readOwnData(item.parent, 'child');
        const sharedChildSet = parentAlternate !== null &&
          readOwnData(parentAlternate, 'child') === parentFirstChild;
        let child = item.firstChild;
        let siblingCount = 0;
        while (child !== null) {
          siblingCount += 1;
          if (siblingCount > args.maxRendererSiblingScan) failCurrentness('ownership-sibling-limit');
          if (!isObject(child) || seen.has(child)) failCurrentness('ownership-cycle-or-shared-node');
          seen.add(child);
          visitedCount += 1;
          if (visitedCount > args.maxRendererOwnershipNodes) failCurrentness('ownership-node-limit');
          const childReturn = readOwnData(child, 'return');
          if (childReturn !== item.parent &&
              !(sharedChildSet && childReturn === parentAlternate)) {
            failCurrentness('ownership-return-child-mismatch');
          }
          const tag = readFiberTag(child);
          readFiberFlags(child);
          const stateNode = readOwnData(child, 'stateNode');
          if (tag === HOST_COMPONENT_TAG && stateNode === selectedHostNode) {
            hostNodeMatchCount += 1;
            if (child === host) exactFiberMatchCount += 1;
          }
          const grandchild = readOwnData(child, 'child');
          const sibling = readOwnData(child, 'sibling');
          if (tag !== HOST_PORTAL_TAG && grandchild !== null) {
            stack.push({ firstChild: grandchild, parent: child, depth: item.depth + 1 });
          }
          child = sibling;
        }
      }
      return {
        state: hostNodeMatchCount === 1 && exactFiberMatchCount === 1 ? 'owned'
          : hostNodeMatchCount === 0 ? 'not-owned' : 'ambiguous',
      };
    } catch (error) {
      return {
        state: 'inconclusive', reason: error && error.currentnessState
          ? error.currentnessState : 'ownership-inspection-error',
      };
    }
  };
  const isValidName = (value) => typeof value === 'string' && value.length > 0 && value.length <= 255 &&
    value !== '.' && value !== '..' && !/[\x00-\x1f\x7f/\\]/.test(value);
  const isPlainData = (value) => isObject(value) && !Array.isArray(value) &&
    [null, Object.prototype].includes(Object.getPrototypeOf(value));
  const isFormCurrent = () => (__COMPOSER_FORM_CURRENT__)(form, composerSelector);
  const isVisible = (element) => {
    if (!element.isConnected || !form.contains(element)) return false;
    const targetRect = element.getBoundingClientRect();
    if (targetRect.width <= 0 || targetRect.height <= 0) return false;
    for (let node = element; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.display === 'none' || ['hidden', 'collapse'].includes(style.visibility) ||
          Number(style.opacity) === 0 || node.getAttribute('aria-hidden') === 'true') return false;
    }
    return true;
  };
  const getCardsInForm = (extraNames) => {
    if (!isFormCurrent()) failCurrentness('composer-form-changed');
    const elements = Array.from(form.querySelectorAll('*'));
    if (elements.length > 2000) failCurrentness('composer-node-limit');
    const safeElements = elements.filter((node) =>
      !['INPUT', 'TEXTAREA', 'SCRIPT', 'STYLE', 'IFRAME', 'OBJECT', 'EMBED'].includes(node.tagName) &&
      (node.tagName === 'BUTTON' || !node.closest('button')) &&
      !node.closest('[contenteditable], [role="textbox"]'));
    const expected = new Set([...names, ...extraNames]);
    // File cards place the filename at child path 1/1 beside an SVG file icon.
    const structuredCards = new Map();
    for (const card of safeElements) {
      const label = card.children[1]?.children[1];
      const icon = card.children[0];
      if (!label || !icon || label.children.length !== 0 || card.contains(composersForForm()[0]) ||
          card.tagName === 'BUTTON' || label.closest('[contenteditable], [role="textbox"]')) continue;
      const iconNodes = [icon, ...icon.querySelectorAll('*')];
      if (iconNodes.length > 100) failCurrentness('file-icon-node-limit');
      if (!iconNodes.some((node) => node.tagName.toLowerCase() === 'svg')) continue;
      const name = (label.textContent || '').trim();
      if (!isValidName(name)) continue;
      expected.add(name);
      structuredCards.set(label, card);
    }
    for (const node of safeElements) {
      for (const attribute of ['title', 'data-filename']) {
        const value = node.getAttribute(attribute);
        if (isValidName(value)) expected.add(value);
      }
      if (node.tagName === 'BUTTON') {
        const label = node.getAttribute('aria-label') || '';
        if (label.startsWith('Remove ') && label !== 'Remove attachment' && isValidName(label.slice(7))) {
          expected.add(label.slice(7));
        }
      }
    }
    const nameFor = (node) => {
      for (const attribute of ['title', 'aria-label', 'data-filename']) {
        const value = node.getAttribute(attribute);
        if (expected.has(value)) return value;
      }
      const text = node.children.length === 0 ? (node.textContent || '').trim() : '';
      return expected.has(text) ? text : null;
    };
    const matching = safeElements.filter((node) => node.tagName !== 'BUTTON' && nameFor(node) !== null);
    const labels = matching.filter((node) => !matching.some((child) => child !== node && node.contains(child)));
    const result = new Map();
    for (const label of labels) {
      const name = nameFor(label);
      const status = (card) => {
        const descendants = Array.from(card.querySelectorAll('*'));
        if (descendants.length > 100) failCurrentness('card-node-limit');
        const scope = [card, ...descendants];
        if (scope.some((node) => /^(failed|error)$/i.test(node.getAttribute('data-state') || '') ||
            node.getAttribute('aria-invalid') === 'true')) return 'failed';
        if (scope.some((node) => /^(uploading|processing|pending)$/i.test(node.getAttribute('data-state') || '') ||
            node.getAttribute('aria-busy') === 'true' || node.getAttribute('role') === 'progressbar')) return 'processing';
        return 'ready';
      };
      const actions = (card) => Array.from(card.children).filter((child) => child.tagName === 'BUTTON' &&
        (child.getAttribute('aria-label') === name || /^(remove|delete|close)(?:\s|$)/i.test(child.getAttribute('aria-label') || '')));
      let card = structuredCards.get(label) ?? (label.parentElement === form ? label : label.parentElement);
      const fallback = card;
      let actionNodes = [];
      let outerState = 'ready';
      for (let depth = 0; depth < 2 && card && card !== form; depth += 1) {
        const parent = card.parentElement;
        const canUseParent = depth === 0 && parent && parent !== form &&
          !parent.contains(composersForForm()[0]) && !labels.some((other) => other !== label && parent.contains(other));
        if (canUseParent) outerState = status(parent);
        actionNodes = actions(card);
        if (actionNodes.length || card.getAttribute('data-filename') === name) break;
        if (!canUseParent) break;
        card = parent;
      }
      if (actionNodes.length === 0) card = fallback;
      if (!card || card === form || card.contains(composersForForm()[0])) failCurrentness('attachment-card-unavailable');
      if (result.has(card)) failCurrentness('ambiguous-card-labels');
      const ownState = status(card);
      const state = ownState === 'failed' || outerState === 'failed' ? 'failed'
        : ownState === 'processing' || outerState === 'processing' ? 'processing' : 'ready';
      const hasReadyAction = actionNodes.some((action) => isVisible(action) && action.getAttribute('disabled') === null &&
        action.getAttribute('aria-disabled') !== 'true');
      result.set(card, { card, name, state: state === 'ready' && !hasReadyAction ? 'unknown' : state,
        visible: isVisible(card) && isVisible(label) });
    }
    if (result.size > 100) failCurrentness('attachment-card-limit');
    const cards = Array.from(result.values());
    // Unclassified non-control content cannot establish complete attachment membership.
    if (safeElements.some((node) => node.tagName !== 'BUTTON' && node.children.length === 0 &&
        !cards.some((item) => item.card.contains(node)) && (node.textContent || '').trim())) {
      failCurrentness('unclassified-composer-content');
    }
    return { cards, elements };
  };
  const composersForForm = () => Array.from(document.querySelectorAll(composerSelector));
  const getAttachmentId = (owner) => {
    const props = readOwnData(owner, 'memoizedProps');
    if (!isPlainData(props)) failCurrentness('attachment-props-not-plain-data');
    const attachment = readOptionalOwnData(props, 'attachment');
    if (attachment.state === 'absent') return null;
    if (attachment.state !== 'value' || !isPlainData(attachment.value)) failCurrentness('attachment-not-plain-data');
    const identifier = readOptionalOwnData(attachment.value, 'id');
    if (identifier.state === 'absent') return null;
    if (identifier.state !== 'value' || typeof identifier.value !== 'string' ||
        !/^file_[A-Za-z0-9_-]{1,251}$/.test(identifier.value)) failCurrentness('attachment-id-invalid');
    return identifier.value;
  };
  const getIdentityForCard = (card) => {
    const wrapper = card.parentElement;
    const enclosure = wrapper && wrapper.parentElement;
    if (!wrapper || !enclosure || wrapper === form || enclosure === form ||
        !form.contains(enclosure)) failCurrentness('attachment-enclosure-unavailable');
    const keys = Object.getOwnPropertyNames(wrapper);
    if (keys.length > 128) failCurrentness('host-key-limit');
    const markers = keys.filter((key) => key.startsWith('__reactFiber$'));
    if (markers.length !== 1) failCurrentness('attachment-marker-unavailable');
    const linked = readOwnData(wrapper, markers[0]);
    if (readFiberTag(linked) !== HOST_COMPONENT_TAG || readOwnData(linked, 'stateNode') !== wrapper) {
      failCurrentness('attachment-marker-host-mismatch');
    }
    const alternate = readOwnData(linked, 'alternate');
    if (alternate !== null && (!isObject(alternate) || readOwnData(alternate, 'alternate') !== linked ||
        readFiberTag(alternate) !== HOST_COMPONENT_TAG || readOwnData(alternate, 'stateNode') !== wrapper)) {
      failCurrentness('attachment-alternate-host-mismatch');
    }
    const resolved = resolveCurrentFiber(linked, form);
    if (resolved.state !== 'resolved' || readFiberTag(resolved.current_fiber) !== HOST_COMPONENT_TAG ||
        readOwnData(resolved.current_fiber, 'stateNode') !== wrapper) failCurrentness('attachment-current-host-unavailable');
    const owners = [];
    const seen = new Set([resolved.current_fiber]);
    let fiber = resolved.current_fiber;
    let enclosed = false;
    for (let depth = 1; depth <= 12; depth += 1) {
      fiber = readOwnData(fiber, 'return');
      if (!isObject(fiber) || seen.has(fiber)) failCurrentness('attachment-return-cycle-or-end');
      seen.add(fiber);
      const tag = readFiberTag(fiber);
      readFiberFlags(fiber);
      const stateNode = readOwnData(fiber, 'stateNode');
      if (tag === HOST_COMPONENT_TAG && stateNode === enclosure) { enclosed = true; break; }
      if ([HOST_ROOT_TAG, HOST_PORTAL_TAG, HOST_COMPONENT_TAG, HOST_TEXT_TAG].includes(tag) || stateNode !== null) {
        failCurrentness('attachment-return-outside-enclosure');
      }
      if (depth === 3 || depth === 4) owners.push(fiber);
    }
    if (!enclosed || owners.length !== 2) failCurrentness('attachment-return-limit');
    const identifiers = new Set();
    for (const owner of owners) {
      const current = resolveCurrentFiber(owner, form);
      if (current.state !== 'resolved' || current.root_state !== resolved.root_state ||
          current.container_info !== resolved.container_info ||
          ownsSelectedHostByChildren(current.current_fiber, resolved.current_fiber).state !== 'owned') {
        failCurrentness('attachment-current-owner-unavailable');
      }
      if ([HOST_ROOT_TAG, HOST_PORTAL_TAG, HOST_COMPONENT_TAG, HOST_TEXT_TAG].includes(readFiberTag(current.current_fiber)) ||
          readOwnData(current.current_fiber, 'stateNode') !== null) failCurrentness('attachment-owner-not-component');
      const identifier = getAttachmentId(current.current_fiber);
      if (identifier !== null) identifiers.add(identifier);
    }
    return identifiers;
  };
  const baselineNodes = new Set();
  const baselineIds = new Set();
  const assignedIds = new WeakMap();
  const cardAliases = new WeakMap();
  const failedIds = new Set();
  let nextCardAlias = 0;
  let conflict = false;
  let isBaselineValid = true;
  try {
    const baseline = getCardsInForm([]);
    for (const element of baseline.elements) baselineNodes.add(element);
    for (const item of baseline.cards) {
      const ids = getIdentityForCard(item.card);
      if (ids.size !== 1) failCurrentness('baseline-attachment-id-unavailable');
      const identifier = ids.values().next().value;
      if (baselineIds.has(identifier)) failCurrentness('baseline-attachment-id-reused');
      baselineIds.add(identifier);
      assignedIds.set(item.card, identifier);
    }
  } catch (_) { isBaselineValid = false; }
  return {
    baseline_valid: isBaselineValid,
    read({ aliases, names: extraNames }) {
      const bindings = [];
      let valid = isBaselineValid;
      try {
        if (!isBaselineValid) failCurrentness('baseline-attachment-unavailable');
        const current = getCardsInForm(extraNames);
        const currentIds = new Set();
        for (const item of current.cards) {
          const ids = getIdentityForCard(item.card);
          if (ids.size > 1) { conflict = true; continue; }
          if (ids.size !== 1) { valid = false; continue; }
          const identifier = ids.values().next().value;
          if (currentIds.has(identifier) || (assignedIds.has(item.card) && assignedIds.get(item.card) !== identifier)) {
            conflict = true;
          }
          currentIds.add(identifier);
          assignedIds.set(item.card, identifier);
          const isNew = !baselineNodes.has(item.card) && !baselineIds.has(identifier);
          if (isNew && item.state === 'failed') failedIds.add(identifier);
          if (failedIds.size > 100) { conflict = true; break; }
          if (!Object.prototype.hasOwnProperty.call(aliases, identifier)) continue;
          if (!cardAliases.has(item.card)) cardAliases.set(item.card, `card-${++nextCardAlias}`);
          bindings.push({ alias: aliases[identifier], card: cardAliases.get(item.card), name: item.name,
            state: item.state, visible: item.visible, is_new: isNew });
        }
      } catch (_) { valid = false; }
      return { valid, conflict, bindings,
        failed_aliases: Array.from(failedIds).filter((identifier) => Object.prototype.hasOwnProperty.call(aliases, identifier))
          .map((identifier) => aliases[identifier]) };
    },
  };
}
""".replace("__COMPOSER_FORM_CURRENT__", COMPOSER_FORM_CURRENT_JS)

READ_ATTACHMENT_BINDINGS_JS = "(tracker, options) => tracker.read(options)"

_monotonic = time.monotonic
_sleep = asyncio.sleep


def _is_valid_display_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 255
        and value not in (".", "..")
        and not any(char in value for char in "/\\")
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


def _is_valid_composer_attachment_state(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    for category in ("ready", "processing", "failed", "unknown"):
        counts = state.get(category, {} if category == "unknown" else None)
        if not isinstance(counts, dict) or any(
            not isinstance(name, str)
            or type(count) is not int
            or count < 0
            for name, count in counts.items()
        ):
            return False
    return True


class AttachmentSettleTimeoutError(TimeoutError):
    """The upload responses or composer attachment chips did not settle."""

    def __init__(
        self, message: str, *, completed_file_create_responses: int,
        ready_attachments: int,
    ) -> None:
        super().__init__(message)
        self.completed_file_create_responses = completed_file_create_responses
        self.ready_attachments = ready_attachments


class AttachmentUploadFailedError(RuntimeError):
    """A newly attached composer chip reports an upload failure."""


class AttachmentObservation:
    """Retain upload callbacks until pre-submit revalidation and cleanup."""

    def __init__(
        self,
        revalidate: Callable[[], Awaitable[str | None]],
        close: Callable[[], Awaitable[None]],
    ) -> None:
        self._revalidate = revalidate
        self._close = close
        self._closed = False

    async def revalidate_before_submit(self) -> str | None:
        if self._closed:
            return "observer-closed"
        return await self._revalidate()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._close()


def _load_descriptors(attachment_paths: Sequence[str]) -> list[dict[str, str]]:
    if len(attachment_paths) > MAX_ATTACHMENTS_PER_ASK:
        raise ValueError(
            f"At most {MAX_ATTACHMENTS_PER_ASK} attachments may be sent in one "
            f"ask; received {len(attachment_paths)}."
        )

    loaded: list[tuple[str, bytes, str]] = []
    total_bytes = 0
    for attachment_path in attachment_paths:
        with Path(attachment_path).open("rb") as attachment_file:
            # One extra byte distinguishes an exact-budget file from overflow.
            data = attachment_file.read(MAX_TOTAL_ATTACHMENT_BYTES - total_bytes + 1)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachments total at least {total_bytes} bytes, exceeding the "
                f"{MAX_TOTAL_ATTACHMENT_BYTES}-byte limit per ask."
            )
        loaded.append((attachment_path, data, Path(attachment_path).name))

    descriptors: list[dict[str, str]] = []
    for attachment_path, data, filename in loaded:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Attachment must be UTF-8 plain text: {attachment_path!r} "
                "(invalid UTF-8)."
            ) from exc
        if b"\x00" in data:
            raise ValueError(
                f"Attachment must be UTF-8 plain text: {attachment_path!r} "
                "(contains NUL bytes)."
            )
        descriptors.append(
            {
                "bytesBase64": base64.b64encode(data).decode("ascii"),
                "name": filename,
                "mime": _PLAIN_TEXT_MIME,
            }
        )
    return descriptors


def _is_completed_file_create_response(response: Any) -> bool:
    try:
        request = response.request
        if request.method != "POST" or not 200 <= response.status < 300:
            return False
        if not is_trusted_origin_url(response.url):
            return False
        pathname = urlsplit(response.url).path.rstrip("/")
        return pathname == FILE_CREATE_PATH
    except (AttributeError, TypeError, ValueError):
        return False


def _is_file_processing_response(response: Any) -> bool:
    try:
        return (
            response.request.method == "POST"
            and is_trusted_origin_url(response.url)
            and urlsplit(response.url).path.rstrip("/") == FILE_PROCESSING_PATH
        )
    except (AttributeError, TypeError, ValueError):
        return False


async def _remove_response_listener(page: Any, listener: Any) -> None:
    try:
        remove_listener = getattr(page, "remove_listener", None)
        if not callable(remove_listener):
            remove_listener = getattr(page, "off", None)
        if not callable(remove_listener):
            return
        result = remove_listener("response", listener)
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


async def _dispose_handle(handle: Any) -> None:
    try:
        await handle.dispose()
    except Exception:
        return


async def _wait_for_response_completion(response: Any) -> None:
    from playwright._impl._network import Response as ImplementationResponse
    from playwright.async_api import Error, Response

    implementation = getattr(response, "_impl_obj", None)
    if not isinstance(implementation, ImplementationResponse):
        if isinstance(response, Response):
            raise TypeError("Unsupported Playwright response implementation")
        await response.finished()
        return

    # Playwright's public finished() leaves a close watcher behind. These futures
    # belong to Playwright (the close future is shared), so never cancel them.
    finished = implementation._finished_future
    closed = implementation._request._target_closed_future()
    done, _ = await asyncio.wait(
        (finished, closed), return_when=asyncio.FIRST_COMPLETED,
    )
    if closed in done:
        closed.result()
        raise Error("Target closed")
    finished.result()


async def attach_files(
    page: Any,
    attachment_paths: Sequence[str],
    *,
    timeout_seconds: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    composer_form_handle: Any | None = None,
    composer_selector: str = COMPOSER_SELECTOR,
    on_settlement: Callable[[AttachmentObservation], None] | None = None,
) -> dict[str, tuple[int, int]]:
    """Upload text files and return per-name (prior, required) ready counts.

    ``on_settlement`` transfers ownership of the live observer to its receiver.
    """
    if not attachment_paths:
        return {}

    descriptors = _load_descriptors(attachment_paths)
    filenames = [descriptor["name"] for descriptor in descriptors]
    expected_counts = Counter(filenames)
    file_handles: list[Any] = []
    data_transfer_handle: Any | None = None
    binding_tracker_handle: Any | None = None
    owned_form_handle: Any | None = None
    last_binding_snapshot: dict[str, Any] | None = None
    listener_tasks: set[asyncio.Task[None]] = set()
    completed_file_create_responses = 0
    receipt_attribution_pending = 0
    receipt_file_ids: list[str] = []
    receipt_aliases: dict[str, str] = {}
    receipt_file_ids_by_name: dict[str, list[str]] = {}
    metadata_failures: list[str] = []
    metadata_failure_overflow = 0
    processing_events: dict[str, dict[str, Any]] = {}
    processing_failure: str | None = None
    processing_pending = 0
    settle_timeout = (
        ATTACH_SETTLE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    deadline = _monotonic() + settle_timeout
    drop_started = False
    ready_attachments = 0
    observer_generation = 0
    settled_observer_generation: int | None = None
    observer_transferred = False

    async def ensure_composer_form_current() -> None:
        if composer_form_handle is None:
            raise RuntimeError("The intended composer form binding is unavailable")
        is_current = await composer_form_handle.evaluate(
            COMPOSER_FORM_CURRENT_JS, composer_selector,
        )
        if is_current is not True:
            raise RuntimeError("The intended composer form changed during attachment upload")

    async def read_existing_names() -> Any:
        await ensure_composer_form_current()
        return await composer_form_handle.evaluate(
            READ_EXISTING_ATTACHMENT_NAMES_FOR_FORM_JS,
        )

    async def read_attachment_state(names: Sequence[str]) -> Any:
        await ensure_composer_form_current()
        return await composer_form_handle.evaluate(
            READ_COMPOSER_ATTACHMENT_STATE_FOR_FORM_JS, list(names),
        )

    async def dispatch_attachment_drop(data_transfer: Any) -> Any:
        await ensure_composer_form_current()
        return await composer_form_handle.evaluate(
            _dispatch_attachment_drop_for_form_js(composer_selector), data_transfer,
        )

    async def read_binding_snapshot(names: Sequence[str]) -> Any:
        await ensure_composer_form_current()
        if binding_tracker_handle is None:
            raise RuntimeError("The composer attachment identity observer is unavailable")
        return await binding_tracker_handle.evaluate(
            READ_ATTACHMENT_BINDINGS_JS,
            {"aliases": dict(receipt_aliases), "names": list(dict.fromkeys([*filenames, *names]))},
        )

    def report_progress() -> None:
        if on_progress is not None:
            on_progress(completed_file_create_responses, ready_attachments)

    def record_metadata_failure(reason: str) -> None:
        nonlocal metadata_failure_overflow
        reason = reason[:128]
        if reason in metadata_failures:
            return
        if len(metadata_failures) < MAX_ATTACHMENTS_PER_ASK:
            metadata_failures.append(reason)
        else:
            metadata_failure_overflow += 1

    report_progress()

    async def record_completed_response(response: Any) -> None:
        nonlocal completed_file_create_responses, processing_failure
        nonlocal receipt_attribution_pending
        try:
            try:
                await _wait_for_response_completion(response)
            except Exception as exc:
                record_metadata_failure(
                    f"response completion failed ({type(exc).__name__})"
                )
                return
            completed_file_create_responses += 1
            report_progress()
            response_json = getattr(response, "json", None)
            if not callable(response_json):
                record_metadata_failure("response JSON unavailable")
                return
            try:
                payload = await asyncio.wait_for(response_json(), timeout=2)
            except Exception as exc:
                record_metadata_failure(type(exc).__name__)
                return
            if not isinstance(payload, dict):
                record_metadata_failure("invalid response JSON")
                return
            file_id = payload.get("file_id")
            if not isinstance(file_id, str) or not re.fullmatch(
                r"file_[A-Za-z0-9_-]{1,251}", file_id
            ):
                record_metadata_failure("file ID unavailable")
                return
            if file_id in receipt_file_ids:
                alias = receipt_aliases[file_id]
                processing_failure = f"duplicate receipt file ID {alias}"
                return
            if len(receipt_file_ids) >= len(filenames):
                record_metadata_failure("surplus receipt IDs")
                return
            receipt_aliases[file_id] = f"upload-{len(receipt_aliases) + 1}"
            receipt_file_ids.append(file_id)
            try:
                request_json = response.request.post_data_json
            except Exception as exc:
                record_metadata_failure(type(exc).__name__)
                return
            filename = (
                request_json.get("file_name")
                if isinstance(request_json, dict) else None
            )
            if not isinstance(filename, str) or filename not in expected_counts:
                record_metadata_failure("request filename unavailable or unexpected")
                return
            ids = receipt_file_ids_by_name.setdefault(filename, [])
            ids.append(file_id)
            if len(ids) > expected_counts[filename]:
                record_metadata_failure("surplus receipt IDs for filename")
        finally:
            receipt_attribution_pending -= 1

    async def record_processing_response(response: Any) -> None:
        nonlocal processing_failure, processing_pending
        try:
            await asyncio.wait_for(
                _wait_for_response_completion(response),
                timeout=max(0.001, deadline - _monotonic())
            )
            if not 200 <= response.status < 300:
                processing_failure = "processing response failed"
                return
            body = await asyncio.wait_for(
                response.text(), timeout=max(0.001, deadline - _monotonic())
            )
            if not isinstance(body, str) or len(body.encode("utf-8")) > MAX_PROCESSING_RESPONSE_BYTES:
                processing_failure = "processing response invalid or oversized"
                return
            for line in body.splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    processing_failure = "processing event invalid"
                    return
                if not isinstance(event, dict):
                    processing_failure = "processing event invalid"
                    return
                file_id = event.get("file_id")
                event_name = event.get("event")
                if not isinstance(file_id, str) or not re.fullmatch(
                    r"file_[A-Za-z0-9_-]{1,251}", file_id
                ) or not isinstance(event_name, str):
                    processing_failure = "processing event identity unavailable"
                    return
                if (
                    file_id not in processing_events
                    and len(processing_events) >= MAX_ATTACHMENTS_PER_ASK
                ):
                    processing_failure = "too many processing file identities"
                    return
                details = processing_events.setdefault(file_id, {})
                if event_name in ("file.indexing.started", "file.indexing.completed"):
                    extra = event.get("extra")
                    if extra is None:
                        extra = {}
                    if not isinstance(extra, dict):
                        processing_failure = "processing name metadata invalid"
                        return
                    has_name = "library_file_name" in extra
                    display_name = extra.get("library_file_name")
                    if has_name:
                        details["name_metadata_observed"] = True
                    if event_name == "file.indexing.started" and "retrieval_indexing_deferred" in extra:
                        deferred = extra["retrieval_indexing_deferred"]
                        if type(deferred) is not bool:
                            details["deferred_name_state"] = "invalid-deferred-flag"
                    else:
                        deferred = False
                    if has_name and not _is_valid_display_name(display_name):
                        if event_name == "file.indexing.started" and deferred is True:
                            details["deferred_name_state"] = "invalid"
                        processing_failure = "processing display name invalid"
                        return
                    if has_name and details.get("observed_display_name", display_name) != display_name:
                        if details.get("deferred_name_state") == "valid":
                            details["deferred_name_state"] = "conflicting"
                        processing_failure = "processing display name conflicted"
                        return
                    if has_name:
                        details["observed_display_name"] = display_name
                    if event_name == "file.indexing.completed" and has_name:
                        details["display_name"] = display_name
                    elif deferred is True and has_name and details.get("deferred_name_state") not in (
                        "invalid", "conflicting", "invalid-deferred-flag",
                    ):
                        details["deferred_display_name"] = display_name
                        details["deferred_name_state"] = "valid"
                elif event_name == "file.processing.completed":
                    details["completed"] = True
                elif event_name.endswith((".failed", ".error")):
                    details["failed"] = True
        except Exception as exc:
            processing_failure = f"processing observation failed ({type(exc).__name__})"
        finally:
            processing_pending -= 1

    def on_response(response: Any) -> None:
        nonlocal processing_pending, receipt_attribution_pending, observer_generation
        if not drop_started:
            return
        if _is_completed_file_create_response(response):
            is_file_create = True
        elif _is_file_processing_response(response):
            is_file_create = False
        else:
            return
        observer_generation += 1
        if len(listener_tasks) >= _MAX_ATTACHMENT_RESPONSE_TASKS:
            record_metadata_failure("attachment response callback limit exceeded")
            return
        if is_file_create:
            receipt_attribution_pending += 1
            task = asyncio.create_task(record_completed_response(response))
        else:
            processing_pending += 1
            task = asyncio.create_task(record_processing_response(response))
        listener_tasks.add(task)
        task.add_done_callback(listener_tasks.discard)

    listener_installed = False

    async def close_observer() -> None:
        nonlocal listener_installed, binding_tracker_handle, owned_form_handle
        if listener_installed:
            await _remove_response_listener(page, on_response)
            listener_installed = False
        tasks = tuple(listener_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        listener_tasks.clear()
        if binding_tracker_handle is not None:
            await _dispose_handle(binding_tracker_handle)
            binding_tracker_handle = None
        if owned_form_handle is not None:
            await _dispose_handle(owned_form_handle)
            owned_form_handle = None

    try:
        if composer_form_handle is None:
            owned_form_handle = await page.evaluate_handle(
                CAPTURE_COMPOSER_FORM_HANDLE_JS, composer_selector,
            )
            composer_form_handle = owned_form_handle.as_element()
            if composer_form_handle is None:
                raise RuntimeError("A unique visible composer form is required before attachment upload")
        existing_names = await read_existing_names()
        if not isinstance(existing_names, list) or len(existing_names) > 100 or any(
            not isinstance(name, str) or not name or len(name) > 255
            for name in existing_names
        ):
            raise RuntimeError("Composer existing attachments probe returned invalid names")
        initial_state = await read_attachment_state(
            list(dict.fromkeys([*filenames, *existing_names]))
        )
        if not _is_valid_composer_attachment_state(initial_state):
            raise RuntimeError("Composer attachment probe returned invalid state")
        last_state = initial_state
        page.on("response", on_response)
        listener_installed = True
        for descriptor in descriptors:
            file_handles.append(
                await page.evaluate_handle(CREATE_ATTACHMENT_FILE_JS, descriptor)
            )
        data_transfer_handle = await page.evaluate_handle(
            CREATE_ATTACHMENT_DATA_TRANSFER_JS, file_handles
        )
        await ensure_composer_form_current()
        binding_tracker_handle = await page.evaluate_handle(
            CREATE_ATTACHMENT_BINDING_TRACKER_JS,
            {"form": composer_form_handle, "composerSelector": composer_selector, "names": filenames},
        )
        if await binding_tracker_handle.evaluate("(tracker) => tracker.baseline_valid") is not True:
            raise RuntimeError("Composer attachment identity baseline could not be verified; no upload attempted")
        drop_started = True
        await dispatch_attachment_drop(data_transfer_handle)

        def initial_count_for(name: str) -> int:
            return sum(
                initial_state.get(category, {}).get(name, 0)
                for category in ("ready", "processing", "failed", "unknown")
            )

        def resolved_file(file_id: str, filename: str) -> tuple[str, bool]:
            nonlocal processing_failure
            details = processing_events.get(file_id)
            if details is None:
                unknown_processing = any(identifier not in receipt_aliases for identifier in processing_events)
                return filename, not unknown_processing
            if details.get("deferred_name_state") in (
                "invalid", "conflicting", "invalid-deferred-flag",
            ):
                processing_failure = "deferred indexing display name invalid or conflicted"
            display_name = details.get("display_name") or details.get("deferred_display_name")
            has_supported_name = display_name is not None
            is_absent_name = not details.get("name_metadata_observed", False)
            return display_name or filename, bool(
                details.get("completed") and not details.get("failed") and
                (has_supported_name or is_absent_name)
            )

        def resolved_names() -> tuple[Counter[str], Counter[str]]:
            names: Counter[str] = Counter()
            completed: Counter[str] = Counter()
            for filename, count in expected_counts.items():
                ids = receipt_file_ids_by_name.get(filename, [])
                for file_id in ids:
                    display_name, is_complete = resolved_file(file_id, filename)
                    names[display_name] += 1
                    if is_complete:
                        completed[display_name] += 1
                unresolved = max(0, count - len(ids))
                if unresolved:
                    names[filename] += unresolved
            return names, completed

        def qualified_ready_count(snapshot: Any) -> int:
            nonlocal processing_failure, last_binding_snapshot
            if not isinstance(snapshot, dict) or type(snapshot.get("valid")) is not bool or \
                    type(snapshot.get("conflict")) is not bool:
                raise RuntimeError("Composer attachment identity probe returned invalid state")
            bindings = snapshot.get("bindings")
            failed_aliases = snapshot.get("failed_aliases")
            if not isinstance(bindings, list) or len(bindings) > 100 or \
                    not isinstance(failed_aliases, list) or len(failed_aliases) > MAX_ATTACHMENTS_PER_ASK or \
                    any(not isinstance(alias, str) for alias in failed_aliases):
                raise RuntimeError("Composer attachment identity probe returned invalid records")
            seen_cards: set[str] = set()
            seen_aliases: set[str] = set()
            for binding in bindings:
                if not isinstance(binding, dict) or not all(
                    isinstance(binding.get(key), str) and 0 < len(binding[key]) <= 255
                    for key in ("alias", "card", "name", "state")
                ) or binding["state"] not in ("ready", "processing", "failed", "unknown") or \
                        type(binding.get("visible")) is not bool or type(binding.get("is_new")) is not bool:
                    raise RuntimeError("Composer attachment identity probe returned invalid card")
                if binding["alias"] in seen_aliases or binding["card"] in seen_cards:
                    processing_failure = "current attachment ID or card assignment conflicted"
                seen_aliases.add(binding["alias"])
                seen_cards.add(binding["card"])
            last_binding_snapshot = snapshot
            if snapshot["conflict"]:
                processing_failure = "current attachment ID or card assignment conflicted"
            if failed_aliases:
                processing_failure = "composer reports an attachment upload failure"
            if not snapshot["valid"] or processing_failure:
                return 0
            required: dict[str, str] = {}
            for filename in expected_counts:
                for file_id in receipt_file_ids_by_name.get(filename, []):
                    display_name, is_complete = resolved_file(file_id, filename)
                    if is_complete:
                        required[receipt_aliases[file_id]] = display_name
            # ID, name, visibility and readiness must belong to the same per-card evaluation.
            return sum(
                binding["alias"] in required and binding["name"] == required[binding["alias"]] and
                binding["visible"] and binding["is_new"] and binding["state"] == "ready"
                for binding in bindings
            )

        def has_exact_receipt_attribution() -> bool:
            if (
                receipt_attribution_pending
                or metadata_failures
                or processing_failure
                or completed_file_create_responses != len(filenames)
                or len(receipt_file_ids) != len(filenames)
                or len(set(receipt_file_ids)) != len(filenames)
            ):
                return False
            return all(
                len(receipt_file_ids_by_name.get(filename, [])) == count
                for filename, count in expected_counts.items()
            )

        def describe_processing(file_id: str, filename: str) -> str:
            details = processing_events.get(file_id)
            alias = receipt_aliases.get(file_id, "unattributed")
            has_completion = bool(details and details.get("completed"))
            has_indexed_display_name = bool(details and details.get("display_name"))
            deferred_name_state = (
                details.get("deferred_name_state", "not-observed")
                if details is not None else "not-observed"
            )
            deferred_name_source = {
                "valid": "deferred-indexing-started",
                "invalid": "deferred-indexing-started-invalid",
                "conflicting": "deferred-indexing-started-conflicting",
                "invalid-deferred-flag": "deferred-indexing-started-invalid-flag",
            }.get(deferred_name_state, "none")
            resolved_name, _ = resolved_file(file_id, filename)
            has_bound_card = bool(
                last_binding_snapshot and last_binding_snapshot["valid"] and not last_binding_snapshot["conflict"] and any(
                    binding["alias"] == alias and binding["name"] == resolved_name and binding["is_new"] and binding["visible"]
                    for binding in last_binding_snapshot["bindings"]
                )
            )
            name_source = (
                "indexing-completed" if has_indexed_display_name
                else "deferred-indexing-started" if deferred_name_state == "valid"
                else "requested-name-candidate" if details is None or not details.get("name_metadata_observed")
                else "unsupported-or-invalid-metadata"
            )
            missing = []
            if not has_completion:
                missing.append("completion")
            if not has_indexed_display_name:
                missing_name = {
                    "valid": "verified card binding for deferred name",
                    "invalid": "valid deferred name metadata",
                    "conflicting": "non-conflicting deferred name metadata",
                    "invalid-deferred-flag": "valid deferred indexing flag",
                }.get(deferred_name_state, "indexed display name")
                if not (has_bound_card and has_completion and (
                    deferred_name_state == "valid" or not details.get("name_metadata_observed", False)
                )):
                    missing.append(missing_name)
            missing_conditions = (
                ", ".join(missing) or "none"
                if details is not None else "not required without observation"
            )
            return (
                f"processing {alias}: observed={'yes' if details is not None else 'no'}, "
                f"completion={'yes' if has_completion else 'no'}, "
                f"indexed display name={'yes' if has_indexed_display_name else 'no'}, "
                f"missing={missing_conditions}, "
                f"name source candidate={deferred_name_source}, "
                f"resolved name source={name_source}, "
                f"card binding={'verified' if has_bound_card else 'unverified'}"
            )

        def describe_observations() -> str:
            required_names, _ = resolved_names()
            observed_ready_attachments = sum(
                min(count, max(0, last_state["ready"].get(name, 0) - initial_count_for(name)))
                for name, count in required_names.items()
            )
            return (
                f"{observed_ready_attachments}/{len(filenames)} observed composer-ready "
                f"attachments; {processing_pending} pending processing responses"
            )

        def describe_files() -> str:
            descriptions = []
            for descriptor in descriptors:
                filename = descriptor["name"]
                ids = receipt_file_ids_by_name.get(filename, [])
                display_names = [
                    resolved_file(file_id, filename)[0]
                    for file_id in ids
                ] or [filename]
                statuses = []
                for display_name in display_names:
                    if last_state["failed"].get(display_name, 0) > initial_state[
                        "failed"
                    ].get(display_name, 0):
                        status = "failed (composer reports an upload error)"
                    elif last_state["processing"].get(display_name, 0) > initial_state[
                        "processing"
                    ].get(display_name, 0):
                        status = "processing (composer reports upload in progress)"
                    elif last_state["ready"].get(display_name, 0) > initial_count_for(
                        display_name
                    ):
                        status = "ready (composer has a file-specific action)"
                    elif last_state.get("unknown", {}).get(display_name, 0):
                        status = "unknown (filename visible without a ready action)"
                    else:
                        status = "unknown (filename not visible in active composer)"
                    if expected_counts[filename] > 1 and 0 < last_state["ready"].get(
                        display_name, 0
                    ) - initial_count_for(display_name) < expected_counts[filename]:
                        status = (
                            f"ambiguous ({last_state['ready'][display_name] - initial_count_for(display_name)}/"
                            f"{expected_counts[filename]} ready)"
                        )
                    statuses.append(status)
                receipt = (
                    receipt_aliases.get(ids[0], "unattributed") if len(ids) == 1
                    else "unavailable (not uniquely attributable)"
                )
                display = (
                    f"; composer name(s) {', '.join(repr(name) for name in display_names)}"
                    if display_names != [filename] else ""
                )
                processing = "; ".join(describe_processing(file_id, filename) for file_id in ids)
                descriptions.append(
                    f"{filename!r}: {statuses[0]}; receipt file ID {receipt}{display}; "
                    f"{processing or 'processing unavailable (no attributable receipt file ID)'}"
                )
            return "; ".join(descriptions)

        def describe_receipts() -> str:
            attributed = {
                file_id for ids in receipt_file_ids_by_name.values()
                for file_id in ids
            }
            unmatched = [
                receipt_aliases[file_id]
                for file_id in receipt_file_ids if file_id not in attributed
            ]
            failures = (
                f"; receipt metadata unavailable ({', '.join(metadata_failures)})"
                if metadata_failures else ""
            )
            if metadata_failure_overflow:
                failures += "; additional receipt metadata failures omitted"
            if metadata_failures or processing_failure:
                attribution = "failed"
            elif receipt_attribution_pending:
                attribution = "pending"
            elif has_exact_receipt_attribution():
                attribution = "complete"
            else:
                attribution = "unknown"
            pending = (
                f"; {receipt_attribution_pending} receipt attribution callbacks pending"
                if receipt_attribution_pending else ""
            )
            return (
                f"receipt attribution={attribution}; receipt file IDs "
                f"(unattributed to filenames): "
                f"{', '.join(unmatched) if unmatched else 'none'}{failures}{pending}"
            )

        def create_timeout_error() -> AttachmentSettleTimeoutError:
            expected = ", ".join(repr(filename) for filename in filenames)
            return AttachmentSettleTimeoutError(
                f"Attachments did not settle within {settle_timeout:g} seconds: "
                f"{completed_file_create_responses}/{len(filenames)} completed "
                f"POST {FILE_CREATE_PATH} responses; {ready_attachments}/"
                f"{len(filenames)} processing-qualified ready attachments; "
                f"{describe_observations()}; expected filename "
                f"chips for {expected}. File states: {describe_files()}; "
                f"{describe_receipts()}; no send click attempted by attachment upload.",
                completed_file_create_responses=completed_file_create_responses,
                ready_attachments=ready_attachments,
            )

        async def revalidate_observation() -> str | None:
            if not observer_transferred or not listener_installed:
                return "observer-unavailable"
            try:
                await asyncio.wait_for(
                    ensure_composer_form_current(),
                    timeout=_MAX_PRE_SUBMIT_REVALIDATION_SECONDS,
                )
            except Exception:
                return "composer-form-changed"
            tasks = tuple(listener_tasks)
            if tasks:
                done, pending = await asyncio.wait(
                    tasks, timeout=_MAX_PRE_SUBMIT_REVALIDATION_SECONDS,
                )
                if pending:
                    return "callbacks-pending"
                await asyncio.gather(*done, return_exceptions=True)
            if receipt_attribution_pending:
                return "receipt-attribution-pending"
            if processing_pending:
                return "processing-pending"
            required_names, completed_names = resolved_names()
            if metadata_failures:
                return "receipt-metadata-failed"
            if processing_failure:
                return "processing-failed"
            attributed_ids = {
                file_id for ids in receipt_file_ids_by_name.values() for file_id in ids
            }
            if any(processing_events.get(file_id, {}).get("failed") for file_id in attributed_ids):
                return "processing-failed"
            if not has_exact_receipt_attribution():
                return "receipt-attribution-incomplete"
            if sum(completed_names.values()) != len(filenames):
                return "processing-incomplete"
            if observer_generation != settled_observer_generation:
                return "protocol-evidence-changed"
            try:
                state = await asyncio.wait_for(
                    read_attachment_state(list(required_names)),
                    timeout=_MAX_PRE_SUBMIT_REVALIDATION_SECONDS,
                )
            except Exception:
                return "attachment-state-unavailable"
            if not _is_valid_composer_attachment_state(state):
                return "attachment-state-invalid"
            if observer_generation != settled_observer_generation:
                return "protocol-evidence-changed"
            if any(
                state["failed"].get(name, 0) > initial_state["failed"].get(name, 0)
                for name in required_names
            ):
                return "composer-reports-failure"
            try:
                bound = await asyncio.wait_for(
                    read_binding_snapshot(list(required_names)),
                    timeout=_MAX_PRE_SUBMIT_REVALIDATION_SECONDS,
                )
                ready_count = qualified_ready_count(bound)
            except Exception:
                return "attachment-binding-unavailable"
            if processing_failure:
                return "attachment-binding-failed"
            if observer_generation != settled_observer_generation:
                return "protocol-evidence-changed"
            return None if ready_count == len(filenames) else "attachments-not-ready"

        while True:
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(0)
            required_names, completed_names = resolved_names()
            try:
                state = await asyncio.wait_for(
                    read_attachment_state(list(required_names)),
                    timeout=remaining,
                )
            except TimeoutError:
                break
            if not _is_valid_composer_attachment_state(state):
                raise RuntimeError("Composer attachment probe returned invalid state")
            await asyncio.sleep(0)
            last_state = state
            current_names, current_completed = resolved_names()
            if required_names != current_names or completed_names != current_completed:
                continue
            attributed_ids = {
                file_id for ids in receipt_file_ids_by_name.values()
                for file_id in ids
            }
            failed_ids = [
                file_id for file_id in attributed_ids
                if processing_events.get(file_id, {}).get("failed")
            ]
            if processing_failure or failed_ids:
                failed_aliases = sorted(
                    receipt_aliases.get(file_id, "unattributed")
                    for file_id in failed_ids
                )
                reason = processing_failure or (
                    "processing failed for " + ", ".join(failed_aliases)
                )
                raise AttachmentUploadFailedError(
                    f"Attachment {reason}. File states: {describe_files()}; "
                    f"{describe_observations()}; {describe_receipts()}; "
                    "no send click attempted by attachment upload."
                )
            for name in required_names:
                if state["failed"].get(name, 0) > initial_state[
                    "failed"
                ].get(name, 0):
                    raise AttachmentUploadFailedError(
                        f"Composer attachment upload failed for {name!r}. "
                        f"File states: {describe_files()}; {describe_observations()}; "
                        f"{describe_receipts()}; no send click attempted by attachment upload."
                    )
            try:
                bound = await asyncio.wait_for(
                    read_binding_snapshot(list(required_names)),
                    timeout=max(0.001, deadline - _monotonic()),
                )
            except TimeoutError:
                break
            ready_attachments = qualified_ready_count(bound)
            if processing_failure:
                raise AttachmentUploadFailedError(
                    f"Attachment {processing_failure}; no send click attempted by attachment upload."
                )
            if (required_names, completed_names) != resolved_names():
                continue
            report_progress()
            if (
                has_exact_receipt_attribution()
                and processing_pending == 0
                and ready_attachments == len(filenames)
            ):
                if on_settlement is not None:
                    settled_observer_generation = observer_generation
                    on_settlement(AttachmentObservation(
                        revalidate_observation, close_observer,
                    ))
                    observer_transferred = True
                return {
                    name: (initial_count_for(name), count)
                    for name, count in required_names.items()
                }
            remaining = deadline - _monotonic()
            if remaining <= 0:
                break
            await _sleep(min(ATTACH_SETTLE_POLL_SECONDS, remaining))

        raise create_timeout_error()
    finally:
        if not observer_transferred:
            await close_observer()
        if data_transfer_handle is not None:
            await _dispose_handle(data_transfer_handle)
        for file_handle in file_handles:
            await _dispose_handle(file_handle)
