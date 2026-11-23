"""Ask against a synthetic, portalled ChatGPT composer in Chromium."""

import asyncio
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from playwright.async_api import async_playwright

from claudex.gptpro import ask, locators


_CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174000"
_PAGE = """
<html lang="en"><main>
<form data-chatgpt-composer>
  <div class="ProseMirror" contenteditable="true" role="textbox"> </div>
  <button id="picker" type="button" aria-label="Select ChatGPT model" aria-haspopup="menu"
          aria-expanded="false" data-composer-navigation-target="reasoning"
          data-selected-reasoning-effort="medium">
    <span class="level"></span>
  </button>
  <button type="submit">Send</button>
</form>
</main>
<div id="menu" role="menu" aria-labelledby="picker" hidden>
  <button role="menuitem" aria-label="Select model">Models</button>
  <div role="menuitemradio" aria-checked="true">GPT-6</div>
  <div role="menuitem" aria-label="Power" aria-keyshortcuts="ArrowLeft ArrowRight"
       aria-describedby="feedback" tabindex="0" data-reasoning-slider="true">
    <span role="slider" aria-valuemin="0"></span>Power
  </div>
  <span role="status" id="feedback"></span>
</div>
<script>
const config = CONFIG;
const levels = config.levels ?? ['Instant', 'Medium', 'High', 'Extra High', 'Pro'];
let level = config.level ?? 1;
window.actions = [];
const picker = () => document.getElementById('picker');
const menu = document.getElementById('menu');
if (['picker', 'both'].includes(config.drift)) {
  picker().removeAttribute('data-composer-navigation-target');
  picker().className = 'synthetic-account-identifier';
  picker().title = 'synthetic-private-metadata';
  picker().setAttribute('data-synthetic-private-metadata', 'synthetic-private-value');
  const unrelated = picker().cloneNode(true);
  unrelated.id = 'unrelated-picker';
  document.body.append(unrelated);
}
const replaceForm = () => {
  const form = picker().closest('form');
  form.replaceWith(form.cloneNode(true));
};
// A closed picker holds its level alone and gains the caption while its menu is open;
// the 'captioned' variant also keeps the caption while closed. Both forms are supported.
const drawCaption = () => {
  picker().querySelector('.caption')?.remove();
  if (config.closedLabel !== 'captioned' && picker().getAttribute('aria-expanded') !== 'true') return;
  const caption = document.createElement('span');
  caption.className = 'caption';
  caption.append('Thinking effort', document.createElement('br'));
  picker().prepend(caption);
};
const draw = () => {
  picker().querySelector('.level').textContent = config.pickerAlwaysShowsPro ? 'Pro' : levels[level];
  const slider = menu.querySelector('[role=slider]');
  slider.setAttribute('aria-valuemax', levels.length - 1);
  slider.setAttribute('aria-valuenow', level);
  document.getElementById('feedback').textContent = `${levels[level]}, ${level+1} of ${levels.length}.`;
};
// A model badge before the level gives the closed picker a label such as '6 Pro'.
if (config.modelBadge) {
  picker().querySelector('.level').before(Object.assign(
    document.createElement('span'), {className: 'badge', textContent: config.modelBadge}), ' ');
}
draw();
drawCaption();
if (['power', 'both'].includes(config.drift)) {
  menu.querySelector('[aria-label=Power]').removeAttribute('data-reasoning-slider');
  const unrelated = menu.cloneNode(true);
  unrelated.id = 'unrelated-menu';
  unrelated.setAttribute('aria-labelledby', 'unrelated-picker');
  unrelated.querySelector('[role=status]').id = 'unrelated-feedback';
  unrelated.querySelector('[aria-label=Power]').setAttribute('aria-describedby', 'unrelated-feedback');
  unrelated.hidden = false;
  document.body.append(unrelated);
}
const power = menu.querySelector('[aria-label=Power]');
if (config.problem === 'missing-picker') picker().hidden = true;
if (config.delayedPicker) picker().hidden = true;
window.hydratePicker = () => setTimeout(() => { picker().hidden = false; }, 200);
if (config.problem === 'disabled-picker') picker().disabled = true;
if (config.problem === 'ambiguous-picker') picker().after(picker().cloneNode(true));
if (config.problem === 'wrong-portal') menu.setAttribute('aria-labelledby', 'unrelated-picker');
if (config.problem === 'ambiguous-power') {
  const competing = power.cloneNode(true);
  competing.removeAttribute('aria-describedby');
  power.after(competing);
}
if (config.problem === 'disabled-power') power.setAttribute('aria-disabled', 'true');
const upgrade = document.createElement('button');
upgrade.type = 'button';
upgrade.textContent = 'Upgrade';
upgrade.onclick = () => actions.push(['upgrade']);
menu.append(upgrade);
window.changeProof = () => {
  if (config.boundary === 'rebadged') picker().querySelector('.badge').textContent = '7';
  else if (config.boundary !== 'unreadable') {
    level = 1;
    draw();
  } else picker().querySelector('.level').remove();
};
document.addEventListener('click', event => {
  if (event.target.closest('button') !== picker()) return;
  actions.push(['open', levels[level]]);
  picker().setAttribute('aria-expanded', 'true');
  picker().setAttribute('aria-controls', menu.id);
  menu.hidden = false;
  drawCaption();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !menu.hidden) {
    menu.hidden = true;
    picker().setAttribute('aria-expanded', 'false');
    picker().removeAttribute('aria-controls');
    if (config.revertOnClose) level = 1;
    draw();
    drawCaption();
    if (config.replaceForm === 'close') replaceForm();
  }
  if (event.key === 'ArrowRight' && event.target === menu.querySelector('[aria-label=Power]')) {
    actions.push(['key', levels[level]]);
    if (config.stalledKey) return;
    level = Math.min(level + 1, levels.length - 1);
    draw();
    if (config.replaceForm === 'key') replaceForm();
  }
});
document.addEventListener('input', () => actions.push(['input', levels[level]]));
document.addEventListener('submit', event => {
  event.preventDefault();
  actions.push(['send', levels[level]]);
  const echo = document.createElement('div');
  echo.dataset.chatgptSearchUnitKey = 'synthetic:user';
  echo.dataset.chatgptSearchMessageIds = 'synthetic-echo';
  echo.textContent = event.target.querySelector('[contenteditable]').innerText;
  document.querySelector('main').prepend(echo);
});
</script>
"""


async def _run_ask(
    tmp_path: Path, config: dict, *, conversation_id=None, expect_failure=False,
    healer=None, attachment=False, book=None,
):
    async with async_playwright() as playwright:
        executable = Path(playwright.chromium.executable_path)
        if not executable.is_file() and os.environ.get('GPTPRO_REQUIRE_BROWSER') == '1':
            pytest.fail('Chromium is required; run python -m playwright install --with-deps chromium')
        if not executable.is_file():
            installed = sorted((Path.home() / "Library/Caches/ms-playwright").glob(
                "chromium-*/chrome-mac*/Google Chrome for Testing.app/"
                "Contents/MacOS/Google Chrome for Testing"
            ))
            if not installed:
                pytest.skip("No local Chromium executable available")
            executable = installed[-1]
        browser = await playwright.chromium.launch(executable_path=str(executable), headless=True)
        try:
            page = await browser.new_page()
            await page.route('**/*', lambda route: route.fulfill(
                content_type='text/html', body=_PAGE.replace('CONFIG', json.dumps(config)),
            ))
            book = book or locators.LocatorBook(tmp_path / 'locators.json', healer=healer)
            execution = ask._AskExecution(
                page, 'Synthetic prompt', None, timeout_seconds=35, conversation_id=conversation_id,
                locator_book=book, attachment_paths=('synthetic.txt',) if attachment else (),
            )
            execution._monitor_completion = AsyncMock(return_value='synthetic response')
            if config.get('delayedPicker'):
                wait_for_composer = execution._wait_for_composer

                async def hydrate_after_form_is_bound():
                    await wait_for_composer()
                    await page.evaluate('() => { hydratePicker(); }')

                execution._wait_for_composer = hydrate_after_form_is_bound
            if config.get('boundary'):
                revalidate = execution._revalidate_attachment_observation

                async def change_after_revalidation():
                    await revalidate()
                    await page.evaluate('changeProof()')

                execution._revalidate_attachment_observation = change_after_revalidation
            if config.get('beforeInput'):
                baseline_users = execution._stable_pre_submit_user_ids

                async def change_after_baseline_users():
                    users = await baseline_users()
                    await page.evaluate('changeProof()')
                    return users

                execution._stable_pre_submit_user_ids = change_after_baseline_users
            if config.get('invalidateHealed'):
                healer.after_hint = lambda: page.evaluate(
                    "document.querySelector('[aria-label=Power]').setAttribute('aria-disabled', 'true')"
                )
            async def attach_files(*args, on_settlement, **kwargs):
                await page.evaluate("actions.push(['attach', levels[level]])")
                on_settlement(ask.attachments.AttachmentObservation(
                    AsyncMock(return_value=None), AsyncMock(),
                ))
                return {'synthetic.txt': (0, 1)}

            upload = patch.object(ask.attachments, 'attach_files', attach_files) if attachment else nullcontext()
            with upload:
                if expect_failure:
                    with pytest.raises(ask.GptProAskError) as error:
                        await execution.run()
                    expected = 'submit_failed' if config.get('boundary') else 'locator_unresolved'
                    assert error.value.failure == expected
                    assert execution.evidence.submission == 'not_attempted'
                    assert not book.pending
                else:
                    outcome = await execution.run()
                    assert outcome.text == 'synthetic response'
                    assert execution.evidence.submission == 'confirmed'
            return await page.evaluate('actions')
        finally:
            await browser.close()


@pytest.mark.parametrize('replacement', ['close', 'key'])
def test_medium_is_promoted_before_first_input_on_replaced_composer(tmp_path, replacement):
    actions = asyncio.run(_run_ask(tmp_path, {'replaceForm': replacement}))
    assert actions[:4] == [
        ['open', 'Medium'], ['key', 'Medium'], ['key', 'High'], ['key', 'Extra High'],
    ]
    assert actions[4:-1] and all(action == ['input', 'Pro'] for action in actions[4:-1])
    assert actions[-1] == ['send', 'Pro']


@pytest.mark.parametrize('closed_label', ['level', 'captioned'])
def test_pro_is_verified_on_a_closed_picker_with_or_without_its_caption(tmp_path, closed_label):
    actions = asyncio.run(_run_ask(tmp_path, {'closedLabel': closed_label}))
    assert actions[:4] == [
        ['open', 'Medium'], ['key', 'Medium'], ['key', 'High'], ['key', 'Extra High'],
    ]
    assert actions[4:-1] and all(action == ['input', 'Pro'] for action in actions[4:-1])
    assert actions[-1] == ['send', 'Pro']


def test_a_picker_label_of_pro_is_not_proof_while_power_reports_another_level(tmp_path):
    actions = asyncio.run(_run_ask(tmp_path, {'pickerAlwaysShowsPro': True}))
    assert actions[:4] == [
        ['open', 'Medium'], ['key', 'Medium'], ['key', 'High'], ['key', 'Extra High'],
    ]
    assert actions[-1] == ['send', 'Pro']


@pytest.mark.parametrize('model_badge', [None, '6'])
def test_power_that_reverts_when_the_menu_closes_never_receives_input(tmp_path, model_badge):
    healer = _LabelHealer()
    config = {'revertOnClose': True, 'modelBadge': model_badge}
    actions = asyncio.run(_run_ask(tmp_path, config, expect_failure=True, healer=healer))
    assert actions[:4] == [
        ['open', 'Medium'], ['key', 'Medium'], ['key', 'High'], ['key', 'Extra High'],
    ]
    assert all(action[0] in ('open', 'key') for action in actions)
    assert healer.calls == []


class _LabelHealer:
    def __init__(self, shows_level='yes'):
        self.calls = []
        self.shows_level = shows_level

    async def complete(self, system, prompt, timeout_seconds):
        self.calls.append(prompt)
        return json.dumps({'shows_level': self.shows_level})


def test_an_unrecognized_picker_label_is_judged_once_and_reused(tmp_path):
    healer = _LabelHealer()
    book = locators.LocatorBook(tmp_path / 'locators.json', healer=healer)
    for _ in range(2):
        actions = asyncio.run(_run_ask(tmp_path, {'level': 4, 'modelBadge': '6'}, book=book))
        assert actions[0] == ['open', 'Pro']
        assert actions[1:-1] and all(action == ['input', 'Pro'] for action in actions[1:-1])
        assert actions[-1] == ['send', 'Pro']
    assert len(healer.calls) == 1
    assert json.loads(healer.calls[0].splitlines()[-1]) == {'level': 'Pro', 'label_lines': ['6 Pro']}
    assert 'Synthetic prompt' not in healer.calls[0]


@pytest.mark.parametrize('shows_level', ['no', 'uncertain', None])
def test_an_unrecognized_picker_label_never_receives_input_without_a_confirming_judgment(
    tmp_path, shows_level,
):
    healer = _LabelHealer(shows_level) if shows_level else None
    actions = asyncio.run(_run_ask(
        tmp_path, {'level': 4, 'modelBadge': '6'}, expect_failure=True, healer=healer,
    ))
    assert actions and all(action == ['open', 'Pro'] for action in actions)
    assert healer is None or len(healer.calls) == 1


@pytest.mark.parametrize('boundary', ['changed', 'rebadged'])
def test_a_judged_picker_label_that_changes_before_send_aborts_without_another_judgment(
    tmp_path, boundary,
):
    healer = _LabelHealer()
    config = {'level': 4, 'modelBadge': '6', 'boundary': boundary}
    actions = asyncio.run(_run_ask(
        tmp_path, config, conversation_id=_CONVERSATION_ID, expect_failure=True, healer=healer,
    ))
    assert actions[0] == ['open', 'Pro']
    assert actions[1:] and all(action == ['input', 'Pro'] for action in actions[1:])
    assert len(healer.calls) == 1


@pytest.mark.parametrize('boundary', ['changed', 'unreadable'])
def test_send_aborts_without_reselection_or_refill_after_final_check(tmp_path, boundary):
    actions = asyncio.run(_run_ask(tmp_path, {'level': 4, 'boundary': boundary},
                                   conversation_id=_CONVERSATION_ID, expect_failure=True))
    assert actions[0] == ['open', 'Pro']
    assert actions[1:] and all(action == ['input', 'Pro'] for action in actions[1:])


@pytest.mark.parametrize('conversation_id', [None, _CONVERSATION_ID])
def test_already_pro_is_confirmed_in_the_menu_and_not_changed(tmp_path, conversation_id):
    actions = asyncio.run(_run_ask(tmp_path, {'level': 4}, conversation_id=conversation_id))
    assert actions[0] == ['open', 'Pro']
    assert actions[1:-1] and all(action == ['input', 'Pro'] for action in actions[1:-1])
    assert actions[-1] == ['send', 'Pro']


class _StructuralHealer:
    def __init__(self, *, after_hint=None):
        self.calls = []
        self.after_hint = after_hint

    async def complete(self, system, prompt, timeout_seconds):
        self.calls.append(prompt)
        candidates = json.loads(prompt.split('\n', 1)[1])
        assert len(candidates) == 1
        assert 'synthetic-' not in prompt and 'Synthetic prompt' not in prompt
        if self.after_hint:
            await self.after_hint()
        return '{"status":"selected","candidate":0}'


def test_picker_drift_heals_only_the_composer_control_after_proof(tmp_path):
    healer = _StructuralHealer()
    actions = asyncio.run(_run_ask(tmp_path, {'drift': 'picker'}, healer=healer))
    assert len(healer.calls) == 1
    assert actions[0] == ['open', 'Medium'] and actions[-1] == ['send', 'Pro']
    assert locators.LocatorBook(tmp_path / 'locators.json').selector_for('en', 'picker') != (
        locators.SEED_SELECTORS['picker']
    )


@pytest.mark.parametrize('drift', ['power', 'both'])
def test_power_drift_heals_in_the_linked_menu_ignoring_other_portals(tmp_path, drift):
    async def assert_not_cached():
        assert not (tmp_path / 'locators.json').exists()

    healer = _StructuralHealer(after_hint=assert_not_cached)
    actions = asyncio.run(_run_ask(tmp_path, {'drift': drift}, healer=healer))
    assert len(healer.calls) == (2 if drift == 'both' else 1)
    assert actions[0] == ['open', 'Medium'] and actions[-1] == ['send', 'Pro']
    assert locators.LocatorBook(tmp_path / 'locators.json').selector_for('en', 'power') != (
        locators.SEED_SELECTORS['power']
    )


def test_power_change_after_setup_aborts_before_first_input(tmp_path):
    actions = asyncio.run(_run_ask(tmp_path, {'level': 4, 'beforeInput': True}, expect_failure=True))
    assert actions == [['open', 'Pro']]


@pytest.mark.parametrize('problem', [
    'missing-picker', 'disabled-picker', 'ambiguous-picker', 'wrong-portal',
    'ambiguous-power', 'disabled-power',
])
def test_missing_ambiguous_disabled_or_unowned_control_never_sends(tmp_path, problem):
    healer = _StructuralHealer()
    actions = asyncio.run(_run_ask(tmp_path, {'problem': problem}, expect_failure=True, healer=healer))
    assert actions == ([['open', 'Medium']] if problem.endswith('power') or problem == 'wrong-portal' else [])
    assert healer.calls == []


def test_no_healer_means_no_unverified_click(tmp_path):
    actions = asyncio.run(_run_ask(tmp_path, {'drift': 'picker'}, expect_failure=True))
    assert actions == []


def test_pro_is_not_inferred_from_the_highest_position_or_gpt_model(tmp_path):
    actions = asyncio.run(_run_ask(tmp_path, {'levels': ['Instant', 'Medium', 'High']}, expect_failure=True))
    assert actions == [['open', 'Medium'], ['key', 'Medium']]


def test_pro_is_accepted_when_not_the_highest_position(tmp_path):
    config = {'levels': ['Instant', 'Medium', 'Pro', 'Another level']}
    actions = asyncio.run(_run_ask(tmp_path, config, conversation_id=_CONVERSATION_ID))
    assert actions[:2] == [['open', 'Medium'], ['key', 'Medium']]
    assert actions[2:-1] and all(action == ['input', 'Pro'] for action in actions[2:-1])
    assert actions[-1] == ['send', 'Pro']


def test_a_healer_hint_is_not_authority_to_use_a_changed_control(tmp_path):
    healer = _StructuralHealer()
    config = {'drift': 'power', 'invalidateHealed': True}
    actions = asyncio.run(_run_ask(tmp_path, config, expect_failure=True, healer=healer))
    assert actions == [['open', 'Medium']]
    assert len(healer.calls) == 1
    assert locators.LocatorBook(tmp_path / 'locators.json').selector_for('en', 'power') == (
        locators.SEED_SELECTORS['power']
    )


def test_a_healed_power_control_that_does_not_move_is_not_cached_and_backs_off(tmp_path):
    healer = _StructuralHealer()
    config = {'drift': 'power', 'stalledKey': True}
    actions = asyncio.run(_run_ask(tmp_path, config, expect_failure=True, healer=healer))
    assert actions[0] == ['open', 'Medium']
    key_attempts = actions[1:]
    assert 1 <= len(key_attempts) <= 10
    assert all(action == ['key', 'Medium'] for action in key_attempts)
    book = locators.LocatorBook(tmp_path / 'locators.json')
    assert book.selector_for('en', 'power') == locators.SEED_SELECTORS['power']
    assert book.refusal('en', 'power') is not None

    actions = asyncio.run(_run_ask(tmp_path, config, expect_failure=True, healer=healer))
    assert actions == [['open', 'Medium']]
    assert len(healer.calls) == 1


def test_picker_can_hydrate_after_the_composer(tmp_path):
    actions = asyncio.run(_run_ask(tmp_path, {'level': 4, 'delayedPicker': True}))
    assert actions[0] == ['open', 'Pro']
    assert actions[1:-1] and all(action == ['input', 'Pro'] for action in actions[1:-1])
    assert actions[-1] == ['send', 'Pro']


@pytest.mark.parametrize('is_available', [True, False])
def test_attachments_wait_for_pro_selection(tmp_path, is_available):
    config = {} if is_available else {'levels': ['Instant', 'Medium']}
    actions = asyncio.run(_run_ask(tmp_path, config, attachment=True, expect_failure=not is_available))
    if not is_available:
        assert actions == [['open', 'Medium']]
        return
    assert actions[:5] == [
        ['open', 'Medium'], ['key', 'Medium'], ['key', 'High'], ['key', 'Extra High'], ['attach', 'Pro'],
    ]
    assert actions[5:-1] and all(action == ['input', 'Pro'] for action in actions[5:-1])
    assert actions[-1] == ['send', 'Pro']
