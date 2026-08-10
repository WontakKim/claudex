## Images:
![Workspaces in the Anthropic API Console | Claude by Anthropic](https://tse4.mm.bing.net/th/id/OIP.u9ayz9igufJyHl4nyDIMqwHaDF?r=0&pid=Api)
![Tailscale : le guide complet du VPN Mesh (WireGuard)](https://images.openai.com/static-rsc-4/S5Z6qyTowb9nrY5nPvMS436kRLW3XfGBsHw_GLaFcWNwfDn1gtV1THBDuRcHEYcmLpSZ9dqXXavGlJJCJxcyfZZbt22srqbZQiZ-X3n-L9xHxuhc3Cc-w7TxIbUGVb1ykz1Zh2Gum1GJjrA0Qc4fZeN6rGnZVNBoqJUzJp1v5zA?purpose=inline)
![Integrating v0 with existing GitHub repositories and Vercel projects - Help - Vercel Community](https://images.openai.com/static-rsc-4/5vQiVW9UFwqGYQBZhjPRsSB_6IrTW-FuiZoeDqy0OdVvaqdTjjpKRn9KN4ctLZhInqqBynlmiH5IWLUzj1FuleIrfEKXXJ6D9FAQ3MgApXv7jxUT5Wnyg1AyEnmC9X08tMe_Xj3vRhtWO0uc6j_TlcvDz6MwoIzRMOJH5wgA64g?purpose=inline)
![Vercel Billing UI: Real Screenshots & UX Patterns | SaaSUI](https://tse3.mm.bing.net/th/id/OIP.tEjtO4SV1ISuX0Dsn4skhgHaD4?r=0&pid=Api)


## 결론

현재 설계의 큰 방향은 좋습니다. 특히 **계정별 상세를 접어서 목록 밀도를 유지하고, 펼쳤을 때 사용량·메타데이터·액션을 한곳에 보여주는 구조**, 그리고 **여러 행을 동시에 열 수 있게 바꾸는 결정**은 이 화면의 작업 성격에 맞습니다.

다만 접힘 행이 지금처럼 **이메일 + 플랜 라벨**만 보여주는 것은 부족합니다. `PRO`와 `MAX`는 계정의 등급이지, 현재 서빙 가능한지 또는 인증에 문제가 있는지를 알려주는 운영 상태가 아닙니다. 조사한 제품들을 유형으로 나누면 다음과 같습니다.

- AWS SSO·1Password 같은 **계정 선택기**는 이름과 선택 상태만 보여줘도 됩니다.
- Tailscale 같은 **운영 리소스 목록**은 상태와 최근성을 접힌 목록에 남깁니다.
- Vercel·Google Cloud 같은 **quota 화면**은 한계치와 사용량의 관계가 핵심일 때 게이지를 씁니다.
- OpenAI·Anthropic·GitHub는 **계정/키 관리 목록과 상세 사용량 분석을 분리**합니다.

Claudex Gateway는 단순 계정 선택기가 아니라 **운영 계정 목록 + quota 상세**에 가깝습니다. 따라서 접힘 행은 최소한 “현재 작동 가능한가”, “현재 서빙 중인가”, “어느 한도가 가장 위험한가”를 답해야 합니다.

---

## 1. 유사 제품의 실제 화면 패턴

### Tailscale — Machines

Tailscale의 Machines 화면은 가장 가까운 **운영 인벤토리형 UI**입니다. 목록에서 머신 이름·관리 주체·주소·클라이언트 버전·최근 접속 정보를 다루고, 현재 연결됨/연결되지 않음과 마지막 접속 시점을 바로 필터링할 수 있습니다. 만료·승인 필요·업데이트 필요 같은 상태도 목록을 탐색하는 주요 기준입니다. 즉, 상세를 열기 전에 “이 리소스가 현재 정상인가”와 “언제 마지막으로 확인됐는가”를 판단하게 합니다. ([tailscale.com](https://tailscale.com/docs/features/access-control/device-management/how-to/filter))

**Claudex에 적용할 점:** `서빙 중`, `재로그인 필요`, `마지막 상태 확인` 중 중요한 신호는 펼침 안에만 숨기지 않아야 합니다.

### OpenAI Platform — Organization API keys / Project members / Usage

OpenAI의 API 키 목록은 `Name`, 마스킹된 키, `Created`, `Last used`, `Project access`, `Created by`, `Permissions`를 열로 노출하고 편집·삭제 액션을 붙입니다. 키의 생명주기와 권한 판단에 필요한 정보는 목록에 남기지만, 키별 사용량 분석은 별도의 Usage 화면에서 프로젝트 선택, 기간·세부 간격, 그룹 기준으로 탐색합니다. 프로젝트 멤버 목록 역시 역할 변경과 제거를 해당 관리 화면에서 처리합니다. ([images.ctfassets.net](https://images.ctfassets.net/j22is2dtoxu1/intercom-img-e0b10057112edb4f7ffd81c0/3b9fb08bcc636eaefee1a7f6c5391ea2/Frame_12.png))

**Claudex에 적용할 점:** 접힘 행에는 인증·활성·최근성처럼 계정 운영에 필요한 정보를 두고, 세 개 quota의 상세 해석은 펼친 영역에 두는 것이 자연스럽습니다.

### Anthropic Console — Workspaces / API Keys / Limits / Usage

Anthropic Console은 Workspace 목록에서 항목을 선택한 뒤 상세 페이지의 `API Keys`, `Limits` 등의 탭으로 들어갑니다. Usage와 Cost 보고서는 별도 탐색 화면이며 Workspace·모델·API 키로 필터링하고, 시간대별 차트와 현재 rate limit 대비 사용량을 분석합니다. Workspace별 spend limit과 notification도 Limits 안에서 설정합니다. 다시 말해 Workspace 목록 하나에 키·멤버·여러 quota 그래프를 모두 밀어 넣지 않고, **목록 → 범위별 상세 → 분석 화면**으로 깊이를 나눕니다. ([support.anthropic.com](https://support.anthropic.com/en/articles/9534590-cost-and-usage-reporting-in-console))

**Claudex에 적용할 점:** 현재의 펼침 영역은 적절하지만, 접힌 행에 세 개의 큰 바를 모두 올리지는 않는 편이 낫습니다. 대신 가장 제약이 큰 한도를 한 줄로 요약해야 합니다.

### GitHub — Copilot Access / Billing & Licensing

GitHub Copilot의 Access 화면 상단에는 할당된 seat 수와 예상 월 비용 요약이 있고, 아래 사용자 목록은 마지막 Copilot 사용 시점으로 정렬할 수 있습니다. 더 자세한 활동은 보고서로 내려받습니다. 전체 사용량과 비용은 별도의 Billing/Usage 화면에서 필터·그룹·기간을 선택하고 차트와 breakdown table로 분석합니다. ([docs.github.com](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-organization/review-activity/review-user-activity-data))

**Claudex에 적용할 점:** 목록에는 `마지막 인증`보다 운영 판단에 더 직접적인 `마지막 사용량 갱신` 또는 `최근 정상 확인`을 노출할 가치가 있습니다. 상세 사용량 전체는 펼친 영역에 유지하면 됩니다.

### Vercel — Usage / AI Gateway Budgets

Vercel은 일반 사용량을 팀·프로젝트 범위의 Usage 화면에서 다루고, AI Gateway Budgets처럼 “현재 지출 / 정해진 한도”가 객체 자체의 핵심인 화면에서는 각 budget에 사용량 바를 붙입니다. 즉, progress bar는 **분모가 명확하고 한계 접근 여부가 즉시 의사결정을 바꾸는 경우**에 사용됩니다. ([vercel.com](https://vercel.com/docs/ai-gateway/observability-and-spend/budgets?utm_source=chatgpt.com))

**Claudex에 적용할 점:** 세션·주간·Fable처럼 상한과 reset이 명확한 값은 펼친 상세에서 게이지를 쓰기 좋은 데이터입니다. 접힘 행에는 바 세 개가 아니라 “가장 위험한 한도 하나”만 요약하는 편이 낫습니다.

### Stripe API Keys / Cloudflare Account API Tokens

Stripe의 API 키 관리 흐름은 생성·권한 제한·회전·삭제 같은 credential 생명주기에 집중합니다. Cloudflare Account API Token도 이름·권한·선택적 만료일을 지정하고 관리합니다. 이런 credential 관리 화면은 상세 사용량보다 권한·만료·비활성화 같은 이상 상태와 액션을 우선합니다. ([docs.stripe.com](https://docs.stripe.com/keys))

**Claudex에 적용할 점:** `재로그인 필요`는 사용량보다 우선순위가 높은 credential 오류입니다. quota와 같은 수준으로 취급하지 말고, 접힘 행에서 즉시 보여주고 펼침의 주 액션을 `다시 로그인`으로 바꿔야 합니다.

### AWS access portal / 1Password 다중 계정

AWS access portal은 사용자가 AWS 계정을 고른 다음 그 계정에서 사용할 역할을 선택합니다. 1Password의 다중 계정 기능도 특정 계정으로 전환하거나 여러 계정의 항목을 함께 보는 것이 목적입니다. 이들은 운영 상태를 관리하는 화면이 아니라 **컨텍스트 선택기**이므로 이름·역할·선택 상태 중심의 간결한 항목이 적합합니다. ([docs.aws.amazon.com](https://docs.aws.amazon.com/singlesignon/latest/userguide/manage-your-accounts.html))

이 패턴을 Claudex 등록 계정에 그대로 적용하면 안 됩니다. Claudex 행은 단순 선택지가 아니라 인증이 깨질 수 있고, 실제 서빙 트래픽을 담당하며, quota 소진에 따라 교체해야 하는 운영 리소스이기 때문입니다.

### Docker — Members와 Usage의 분리

Docker 역시 멤버·seat·license·제품 접근 관리는 조직 관리 영역에서 다루지만, 실제 사용량은 제품에 따라 Insights, Usage, Build minutes, Billing 등 별도 화면으로 나눕니다. ([docs.docker.com](https://docs.docker.com/admin/organization/manage/manage-products/?utm_source=chatgpt.com))

---

## 2. 접힘/펼침 정보 분배에 대한 판단

현재 접힘 행은 사실상 “이름 + 한 가지 지표”도 아닙니다. `MAX`, `PRO`는 지표가 아니라 **정적 분류값**입니다.

접힘 행이 최소한 다음 네 질문에 답하는 것이 좋습니다.

1. 어느 계정인가?
2. 현재 이 계정으로 서빙 중인가?
3. 지금 다시 인증해야 하는가?
4. 어느 quota가 가장 먼저 문제가 될 가능성이 높은가?

따라서 권장 접힘 행 구성은 다음과 같습니다.

```text
[셰브론] 이메일              플랜    운영 상태           가장 제약적인 사용량
```

예시는 다음과 같습니다.

```text
▾ ai-platform@wrtn.io   [PRO]  ● 서빙 중        세션 잔여 3% · 2시간 34분 후
▸ wontak@wrtn.io        [MAX]  ⚠ 재로그인 필요
▸ dev-test@wrtn.io      [—]                     주간 잔여 62% · 일요일 21:00
```

여기서 중요한 점은 다음과 같습니다.

- 정상이고 서빙하지 않는 계정에는 상태 문구를 굳이 넣지 않습니다.
- `서빙 중`과 `재로그인 필요` 같은 **예외 또는 선택 상태만 표시**합니다.
- quota는 세 개를 모두 넣지 않고, 현재 가장 제약이 큰 window 하나만 표시합니다.
- 오류가 발생한 계정에서는 오래된 quota 숫자보다 `재로그인 필요`를 우선합니다.
- 같은 행에서 오류와 정상을 동시에 표현하지 않습니다.

특히 활성 계정의 인증이 깨졌다면 다음처럼 표현해야 합니다.

```text
⚠ 서빙 불가 · 재로그인 필요
```

`● 서빙 중`과 `⚠ 재로그인 필요`를 동시에 보여주면 “설정상 active인가, 실제 요청을 처리하고 있는가”가 모호해집니다. 상태 우선순위를 다음처럼 파생하는 것이 좋습니다.

```text
재로그인 필요/서빙 불가 > 서빙 중 > 정상 대기
```

---

## 3. 독립 펼침과 아코디언 중 무엇이 맞는가

이 화면에서는 **독립 state의 멀티 오픈 방식**이 더 적합합니다.

사용자가 두 계정의 세션·주간 quota를 비교하거나, 한 계정의 조직·인증 시각을 본 상태에서 다른 계정을 확인할 수 있기 때문입니다. 하나를 열면 기존 항목이 닫히는 전통적 단일 아코디언은 계정 간 비교에 불필요한 반복 조작을 만듭니다.

WAI-ARIA의 accordion 패턴도 단일 오픈과 멀티 오픈을 모두 허용합니다. 각 헤더는 실제 `button`이어야 하고, `aria-expanded`와 `aria-controls`로 패널 상태를 전달해야 합니다. Enter와 Space로 펼침을 토글할 수 있어야 합니다. ([w3.org](https://www.w3.org/WAI/ARIA/apg/patterns/accordion/))

구현은 다음 형태가 적절합니다.

```html
<h3>
  <button
    aria-expanded="false"
    aria-controls="account-panel-ai-platform"
  >
    <!-- 셰브론, 이메일, 플랜, 상태, quota 요약 -->
  </button>
</h3>

<div id="account-panel-ai-platform" hidden>
  <!-- 사용량, 메타데이터, 액션 -->
</div>
```

실무상 주의할 점은 다음과 같습니다.

- 현재처럼 접힘 행 안에 별도 액션이 없다면 **행 전체를 하나의 펼침 버튼**으로 만들어도 좋습니다.
- 이후 kebab 메뉴나 삭제 버튼을 접힘 행에 추가한다면 펼침 버튼 안에 중첩하지 말고 형제 요소로 둡니다.
- 셰브론 자체에는 별도 포커스를 주지 않고 장식 요소로 처리합니다.
- 여러 패널이 열릴 수 있고 계정 수가 많아질 가능성이 있다면 모든 패널에 `role="region"`을 붙이지 않는 편이 낫습니다. W3C도 동시에 열릴 수 있는 패널이 약 6개를 넘을 때 landmark 남발을 피하라고 안내합니다. ([w3.org](https://www.w3.org/WAI/ARIA/apg/patterns/accordion/))
- 계정이 크게 늘어나면 멀티 오픈을 단일 아코디언으로 바꾸기보다, 목록 + 우측 상세 패널 구조로 전환하는 편이 낫습니다.

---

## 4. 상태 칩 없이 active/error를 보여주는 미니멀 패턴

가장 적합한 방식은 **컨테이너 없는 아이콘 + 짧은 텍스트**입니다.

### 권장

```text
● 서빙 중
⚠ 재로그인 필요
```

- `서빙 중`: accent 파란 점 또는 작은 radio 형태 + 텍스트
- `재로그인 필요`: warn/err 삼각형 + 텍스트
- 정상 대기: 아무 표시 없음
- 상태 확인 실패: `? 상태 확인 불가` 또는 cloud-off 아이콘 + 텍스트

이렇게 하면 pill이 늘어나는 시각적 혼잡 없이도 상태를 즉시 파악할 수 있습니다.

활성 행에는 추가로 다음 중 하나를 보조 신호로 사용할 수 있습니다.

- 2px accent 좌측선
- 아주 옅은 accent 배경
- 이메일 앞의 작은 filled radio

단, 이것들은 보조 표현이어야 합니다. 색만으로 상태를 전달해서는 안 되며, 눈에 보이는 텍스트 또는 형태 차이가 함께 있어야 합니다. ([w3.org](https://www.w3.org/WAI/WCAG21/Understanding/use-of-color.html?utm_source=chatgpt.com))

점만 표시하고 hover tooltip에서만 `서빙 중`이라고 설명하는 패턴은 권하지 않습니다. 데스크톱 마우스 사용자에게는 작동해도 키보드·터치 사용자와 색 구분이 어려운 사용자에게 상태 전달이 약합니다.

### 공간이 매우 부족한 경우

텍스트를 완전히 없애야 한다면 최소한 모양을 다르게 해야 합니다.

```text
●  활성
▲  오류
```

아이콘에는 접근 가능한 이름을 제공해야 하지만, 가능하면 현재 화면 폭에서는 `서빙 중`, `재로그인 필요`라는 짧은 가시 텍스트를 유지하는 편이 낫습니다.

---

## 5. 인라인 quota 게이지 사용 기준

조사한 제품에서는 다음 분리가 반복됩니다.

- Vercel Budget처럼 한도 대비 현재 값이 객체의 핵심이면 목록 또는 카드에 바를 표시합니다.
- Google Cloud Quotas는 표에 현재 사용률과 사용량을 보여주고, 많이 사용한 quota를 우선 정렬하며, 개별 quota를 열어 시간대별 차트를 봅니다.
- AWS Service Quotas는 quota 이름·적용값·기본값·utilization을 표와 상세 화면에 제공합니다.
- Anthropic과 OpenAI는 API 키나 멤버 행 안이 아니라 별도 Usage 화면에서 시계열 차트와 필터를 제공합니다. ([docs.cloud.google.com](https://docs.cloud.google.com/docs/quotas/view-manage))

Claudex에서는 다음 선택이 적절합니다.

| 표현 | 적합한 경우 | 이 화면에서의 사용 |
|---|---|---|
| 풀-위드 바 | 상한이 명확하고 임계 접근 여부가 중요할 때 | 펼친 계정의 세션·주간·Fable |
| 숫자 + reset | 여러 항목을 빠르게 비교할 때 | 접힌 행의 가장 위험한 window |
| 스파크라인 | 증가 속도나 시간 추세 자체가 의사결정을 바꿀 때 | 현재는 사용하지 않음 |
| 차트 | 기간별 사용 패턴·forecast를 분석할 때 | 향후 별도 사용량 화면이 생길 때 |

세션·주간 quota는 reset 순간 값이 급변하는 sawtooth 데이터입니다. 작은 스파크라인은 reset 경계를 설명하지 않으면 오히려 오독 가능성이 큽니다. 현재는 **비율 + 남은 시간**이 더 직접적입니다.

### 반드시 명확히 할 의미

지금 와이어프레임의 `97%`는 사용한 비율인지 남은 비율인지 문맥을 읽어야 알 수 있습니다. 다음처럼 명시해야 합니다.

```text
세션                              97% 사용
█████████████████████████████░
15:00 리셋 · 2시간 34분 후
```

접힘 행에서는 반대로 사용자의 다음 행동에 더 직접적인 잔여량을 보여줄 수 있습니다.

```text
세션 잔여 3% · 2시간 34분 후
```

단, 동일 화면에서 기준을 섞는 것이 혼란스럽다면 접힘과 펼침 모두 `97% 사용`으로 통일하고, 색과 reset 문구로 위험도를 보조해도 됩니다. 핵심은 숫자만 `97%`로 두지 않는 것입니다.

quota 바는 작업 진행률이 아니라 알려진 범위 안의 정적 측정값이므로 접근성 의미상 `progressbar`보다 `meter`가 더 잘 맞습니다. `aria-valuetext`에는 비율뿐 아니라 reset 정보까지 넣을 수 있습니다. ([w3.org](https://www.w3.org/WAI/ARIA/apg/practices/range-related-properties/?utm_source=chatgpt.com))

```html
<div
  role="meter"
  aria-label="세션 사용량"
  aria-valuemin="0"
  aria-valuemax="100"
  aria-valuenow="97"
  aria-valuetext="97퍼센트 사용, 2시간 34분 후 리셋"
>
</div>
```

사용량을 가져오지 못했다면 `0%` 바를 표시하지 말고 다음처럼 명확히 구분해야 합니다.

```text
사용량을 가져오지 못함
마지막 정상 확인 18분 전
```

`0%`는 정상적으로 사용하지 않았다는 의미이고, “알 수 없음”과 같지 않습니다.

---

## 6. 우선순위별 개선 제안

### P0 — 접힘 행에 운영 상태를 노출

가장 먼저 고쳐야 할 부분입니다.

현재:

```text
▶ ai-platform@wrtn.io ........................ PRO
```

권장:

```text
▸ ai-platform@wrtn.io   [PRO]   ● 서빙 중      세션 잔여 3% · 2시간 34분 후
▸ wontak@wrtn.io        [MAX]   ⚠ 재로그인 필요
▸ dev-test@wrtn.io      [—]                    주간 잔여 62% · 일요일 21:00
```

구체적인 규칙은 다음과 같습니다.

- active 계정만 `● 서빙 중`
- 인증 오류만 `⚠ 재로그인 필요`
- 정상 inactive 계정은 상태 생략
- 오류 상태가 active보다 우선
- 오류 시 오래된 quota 요약은 숨기거나 `사용량 확인 불가`로 대체
- plan pill은 그대로 유지하되 status 역할을 맡기지 않음

이 변경만으로 사용자는 모든 행을 펼치지 않고도 failover 또는 재인증 필요성을 판단할 수 있습니다.

### P0 — 로컬 Claude Code 로그인과 게이트웨이 등록 계정의 범위를 더 강하게 분리

현재는 동일한 카드 안에서 로컬 계정과 등록 계정이 비슷한 시각 언어를 사용하고, 실제 예시에서는 `ai-platform@wrtn.io`가 양쪽에 모두 나타납니다. 설명문을 끝까지 읽지 않으면 “이 로컬 로그인도 게이트웨이가 사용한다”고 오해하기 쉽습니다.

별도 카드까지 만들 필요는 없지만 섹션 제목과 범위 설명을 다음처럼 바꾸는 것이 좋습니다.

```text
이 머신의 Claude Code 로그인                              [새로고침]
ai-platform@wrtn.io  [PRO]  Wrtn AI Platform
게이트웨이 서빙에는 사용되지 않습니다 · 2분 전 확인

세션 ...
주간 ...

─────────────────────────────────────────────────────────

게이트웨이 등록 계정 3                                  [계정 추가]
...
```

`로컬 CLAUDE`보다는 `이 머신의 Claude Code 로그인`이 역할을 더 정확히 설명합니다.

또한 `(갱신)`의 실제 기능에 따라 명칭을 구분해야 합니다.

- 단순히 상태와 사용량을 다시 읽음: `새로고침`
- OAuth 인증을 다시 수행함: `다시 로그인`
- 토큰 만료를 연장함: `로그인 갱신`

하나의 `갱신` 표현으로 세 동작을 포괄하지 않는 편이 좋습니다.

### P1 — 접힘에는 한 개의 risk summary, 펼침에는 세 개의 전체 게이지

펼침 영역의 세 개 풀-위드 게이지는 유지해도 좋습니다. 대신 `사용량 N분 전 기준`을 게이지 밑에 고립시키지 말고 사용량 섹션 제목으로 올리는 편이 관계가 명확합니다.

```text
사용량 · 2분 전 업데이트

세션                              97% 사용
█████████████████████████████░
15:00 리셋 · 2시간 34분 후

주간                              88% 사용
██████████████████████████░░░░
일요일 21:00 리셋 · 5일 6시간 후

Fable                             42% 사용
████████████░░░░░░░░░░░░░░░░░
...
```

접힘 행에 표시할 하나의 summary는 다음 기준으로 선택합니다.

1. 오류 상태가 있으면 quota 대신 오류를 표시
2. 유효한 데이터 중 소진율이 가장 높은 window 선택
3. 해당 window 이름, 잔여 또는 사용 비율, reset까지 표시
4. 데이터가 polling SLA보다 오래됐으면 `12분 전 데이터`를 함께 표시

향후 사용 속도 forecast가 생기면 단순 최고 비율 대신 `예상 소진까지 48분` 같은 예측 신호를 우선할 수 있습니다. 현재 데이터만으로는 최고 사용률 요약이 가장 예측 가능하고 설명하기 쉽습니다.

### P1 — 멀티 오픈 disclosure를 그대로 채택하고 키보드·포커스 모델을 확정

행별 독립 state로 변경하는 계획은 유지하는 것이 맞습니다.

구현 기준은 다음처럼 잡는 것이 좋습니다.

- 행 전체가 하나의 native button
- Enter/Space로 열고 닫기
- `aria-expanded`, `aria-controls`
- 열림 후 포커스를 내부로 강제 이동하지 않음
- 다시 접을 때 포커스는 행 버튼에 유지
- 내부 액션을 Tab 순서로 자연스럽게 탐색
- 시각적으로 보이는 focus ring 유지
- 계정 정렬이나 갱신으로 행이 이동하더라도 열린 상태는 index가 아니라 account ID 기준으로 보존

현재 계정 수가 3개라면 `모두 펼치기/모두 접기`는 필요하지 않습니다. 목록이 커졌을 때만 보조 액션으로 추가하는 편이 낫습니다.

### P2 — 계정 상태에 따라 주 액션을 하나만 강조

펼친 행의 주 액션을 상태 기반으로 바꾸는 방향은 맞습니다. 다만 버튼 위계를 다음처럼 고정하면 더 명확합니다.

| 계정 상태 | 강조 액션 | 보조 액션 |
|---|---|---|
| 정상 inactive | `이 계정으로 서빙` | 제거 |
| 정상 active | `서빙 해제`를 neutral/secondary | 제거 |
| 재인증 필요 | `다시 로그인` | 제거 |
| active이지만 인증 실패 | `다시 로그인` | `서빙 대상에서 제외` |

`서빙 해제`는 새로운 작업을 시작하는 primary action이라기보다 현재 상태를 해제하는 동작이므로 반드시 accent-filled 버튼일 필요는 없습니다.

`제거`는 Stripe·Cloudflare의 credential lifecycle 액션처럼 overflow menu로 이동시키거나, 현재 위치를 유지하더라도 확인 dialog를 거쳐야 합니다. 특히 active 계정을 제거할 때는 단순히 “제거하시겠습니까?”가 아니라 다음 결과를 명시해야 합니다.

```text
이 계정은 현재 게이트웨이 서빙 계정입니다.
제거하면 서빙이 중단됩니다.

[취소] [서빙 해제 및 제거]
```

### P2 — OAuth 모달을 단순 입력 폼이 아니라 상태 흐름으로 표현

현재의 인증 URL + 코드 입력 + 180초 countdown은 기본 요소가 갖춰져 있습니다. 여기에 다음 상태를 명시적으로 구분하면 실패 복구가 쉬워집니다.

```text
브라우저 인증 대기
→ 코드 확인 중
→ 계정 확인됨
→ 중복 계정 교체 확인
→ 등록 완료
```

구체적으로는 다음을 적용하는 것이 좋습니다.

- URL 텍스트만 두지 말고 `브라우저에서 인증 열기`를 primary로 제공
- 인증 코드는 별도 `복사` 버튼 제공
- countdown 외에 절대 만료 시각도 보조 표시
- 만료 시 입력을 그대로 두고 `새 코드 받기` 제공
- countdown을 스크린 리더에 매초 announce하지 않고 60초·30초·만료 등 단계에서만 알림
- 중복 확인에는 기존 계정 이메일, plan, 현재 서빙 여부를 표시
- 중복 교체가 active assignment까지 바꾸는지 명시
- 성공 후 새 계정 행을 잠시 강조하고, 자동으로 펼칠지는 사용자가 확인해야 할 후속 작업이 있을 때만 적용

---

## 권장 최종 형태

```text
Claude 계정
게이트웨이에 등록된 계정과 이 머신의 Claude Code 로그인을 관리합니다.

이 머신의 Claude Code 로그인                              [새로고침]
ai-platform@wrtn.io  [PRO]  Wrtn AI Platform
게이트웨이 서빙에는 사용되지 않습니다 · 2분 전 확인

세션                              97% 사용
█████████████████████████████░
15:00 리셋 · 2시간 34분 후

주간                              88% 사용
██████████████████████████░░░░
일요일 21:00 리셋 · 5일 6시간 후

──────────────────────────────────────────────────────────

게이트웨이 등록 계정 3                                  [계정 추가]

▾ ai-platform@wrtn.io  [PRO]  ● 서빙 중    세션 잔여 3% · 2시간 34분 후

  사용량 · 2분 전 업데이트

  세션                            97% 사용
  █████████████████████████████░
  15:00 리셋 · 2시간 34분 후

  주간                            88% 사용
  ██████████████████████████░░░░
  일요일 21:00 리셋 · 5일 6시간 후

  Fable                           42% 사용
  ████████████░░░░░░░░░░░░░░░░░
  ...

  플랜          PRO
  조직          Wrtn AI Platform
  추가일        2026-08-01
  마지막 인증   2026-08-08 01:42

  [서빙 해제]                                      [⋯ 계정 메뉴]

──────────────────────────────────────────────────────────

▸ wontak@wrtn.io       [MAX]  ⚠ 재로그인 필요

──────────────────────────────────────────────────────────

▸ dev-test@wrtn.io     [—]                주간 잔여 62% · 일요일 21:00
```

최종적으로 유지해야 할 원칙은 하나입니다.

> **접힘 행에는 판단에 필요한 상태를, 펼침 영역에는 원인과 조치를 둔다.**

현재 설계는 사용량과 조치를 펼침 안에 두는 부분은 잘 되어 있습니다. 여기에 `서빙 중/재로그인 필요`와 한 개의 quota 위험 요약만 접힘 행으로 끌어올리면, 제품 사례들과도 일치하면서 Claudex 특유의 다중 Claude 계정 운영 문제를 훨씬 빠르게 해결할 수 있습니다.
