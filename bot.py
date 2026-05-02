import os, json, asyncio, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="Vera Bot", version="17.0.0")
context_store = {}

# LLM Configuration from judge_simulator.py
nvidia_client = openai.OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1"
)
MODEL = "openai/gpt-oss-20b"
_executor = ThreadPoolExecutor(max_workers=10)

# ── Category voice — drives Category Fit score ────────────────────────────────
CATEGORY_VOICE = {
    "dentists": {
        "noun":          "patients",
        "tone":          "peer-clinical, data-driven, professional",
        "salutation":    "Dr. {owner}",
        "loss_hook":     "most patients book the first clinic that responds",
        "urgency_close": "before this window closes",
        "taboo":         ["guaranteed", "100% safe", "miracle", "best in city"],
        "avg_ticket":    800,
    },
    "salons": {
        "noun":          "clients",
        "tone":          "warm, practical, visual, lifestyle-first",
        "salutation":    "Hi {owner}",
        "loss_hook":     "they'll book elsewhere if they don't hear from you",
        "urgency_close": "before slots fill up",
        "taboo":         ["guaranteed glow", "miracle", "instant transformation"],
        "avg_ticket":    500,
    },
    "restaurants": {
        "noun":          "covers",
        "tone":          "direct, operator-to-operator, time-pressure",
        "salutation":    "Hi {owner}",
        "loss_hook":     "the lunch/dinner window is narrow — act now",
        "urgency_close": "before the rush passes",
        "taboo":         ["best food in city", "guaranteed packed house"],
        "avg_ticket":    400,
    },
    "gyms": {
        "noun":          "members",
        "tone":          "coaching, energetic, goal-focused",
        "salutation":    "Hi {owner}",
        "loss_hook":     "fitness seekers drop off fast if not contacted within 24h",
        "urgency_close": "before they sign up elsewhere",
        "taboo":         ["guaranteed weight loss", "shred in 7 days", "miracle"],
        "avg_ticket":    600,
    },
    "pharmacies": {
        "noun":          "customers",
        "tone":          "trustworthy, precise, health-first, no hype",
        "salutation":    "Hi {owner}",
        "loss_hook":     "patients who miss refill reminders often switch pharmacies",
        "urgency_close": "before stock runs out",
        "taboo":         ["miracle cure", "100% safe", "best price"],
        "avg_ticket":    350,
    },
}

# ── Singular noun map ────────────────────────────────────────────────────────
NOUN_SINGULAR = {
    "patients":  "patient",
    "clients":   "client",
    "members":   "member",
    "covers":    "cover",
    "customers": "customer",
}

def _s(noun: str) -> str:
    return NOUN_SINGULAR.get(noun, noun)

def _clean_slug(raw: str) -> str:
    """Turn 'ORS_demand_+40' → 'ORS demand +40', 'kids_yoga_post' → 'kids yoga post'."""
    return raw.replace("_", " ").replace("+", "+").strip()

# ── Trigger strategy map ──────────────────────────────────────────────────────
TRIGGER_STRATEGY = {
    "research_digest":          "Lead with the specific research stat. Explain why it matters for THIS merchant's patient/client mix. Offer one action.",
    "regulation_change":        "Name the regulation. State the compliance deadline. One specific action to take today.",
    "recall_due":               "Use patient name if given. Name the service due. State days overdue. Mention available slot times. Include offer ₹price.",
    "perf_dip":                 "Lead with the exact metric that dipped and by how much. Calculate the ₹ revenue lost (delta% × avg_ticket × weekly_volume). Name a fix tied to their active offer.",
    "renewal_due":              "State days remaining and the renewal cost. Frame what pauses if they don't renew.",
    "festival_upcoming":        "Name the festival. State days until. Lead with the demand spike number if available.",
    "wedding_package_followup": "Name the wedding date. State the next prep milestone. Give a specific next step.",
    "curious_ask_due":          "Ask one specific question about what's in demand this week. Keep it conversational.",
    "winback_eligible":         "State days since lapse. Name what they're missing (views, calls, leads). Offer easy reactivation.",
    "ipl_match_today":          "Name the match. State the demand lift (1.5x). Include locality. Push a ₹-priced match-night offer by 6pm.",
    "review_theme_emerged":     "Quote the review theme. State how many times it appeared. Offer a specific operational fix.",
    "milestone_reached":        "State the current metric and how close they are to the milestone. Push one action to close the gap.",
    "active_planning_intent":   "Continue the conversation thread. Reference what they said last. Next logical step.",
    "seasonal_perf_dip":        "Acknowledge the seasonal dip is normal. Give a retention-focused action for this specific period.",
    "customer_lapsed_hard":     "State the EXACT count of lapsed customers (never 'several'). State days since last visit. Include ₹ winback offer.",
    "trial_followup":           "Reference the trial session. Ask about their experience. Book the next session.",
    "supply_alert":             "URGENT framing. Name the molecule/product. List affected batches. Tell them to pull stock and contact chronic customers.",
    "chronic_refill_due":       "Name the medication(s). Give the exact date stock runs out. Offer to send delivery reminder now.",
    "category_seasonal":        "Name the seasonal trend with % uplift. Suggest shelf rearrangement + specific offer push.",
    "gbp_unverified":           "State the impression penalty (% fewer). Say it takes 5 min to fix. Offer to send the guide.",
    "cde_opportunity":          "Name the CDE credits available. State the expiry. Make it easy to claim.",
    "competitor_opened":        "Name the competitor and their offer. Name your merchant's counter-offer with ₹price. Urgency.",
    "perf_spike":               "Lead with the metric that spiked and by how much. Name the likely driver (clean slug). Push offer to capitalise.",
    "dormant_with_vera":        "Reference the last conversation topic. Ask a specific question. Keep it warm and brief.",
}

# ── Match-night invented offers by category (for ipl_match_today with no active offer) ──
MATCH_OFFERS = {
    "restaurants": "a match combo (₹50 off on ₹299+)",
    "gyms":        "a drop-in session at ₹199",
    "salons":      "a quick grooming deal at ₹299",
    "pharmacies":  "an energy drinks + snacks bundle at ₹149",
    "dentists":    "an evening consult slot at ₹500",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ctx(scope, cid):
    e = context_store.get(f"{scope}:{cid}")
    return e["payload"] if e else None


def pick_best_offer(merchant):
    offers = merchant.get("offers", [])
    active = [o for o in offers if o.get("status") == "active"]
    pool = active or offers
    if not pool:
        return None
    def price_key(o):
        raw = str(o.get("price", o.get("value", "9999"))).replace("₹","").replace(",","").split()[0]
        return float(raw) if raw.replace(".","").isdigit() else 9999
    return sorted(pool, key=price_key)[0]


def build_salutation(owner, cat):
    """
    Build the opening greeting.
    - Dentists: 'Dr. {FirstName}' — never 'Dr. Dr.' or 'Dr. Full Clinic Name'
    - Others:   'Hi {FirstName}' — never the business name in salutation
    - If no first name: use business name without Dr. prefix for non-dentists
    """
    voice = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    template = voice["salutation"]
    name = (owner or "").strip()
    # Strip any existing Dr. prefix to prevent doubling
    if name.lower().startswith("dr."):
        name = name[3:].strip()
    elif name.lower().startswith("dr "):
        name = name[3:].strip()
    # Use only first word (first name)
    first = name.split()[0] if name.split() else ""
    if not first:
        return ""
    return template.replace("{owner}", first)


def format_offer(offer):
    """Return 'Title ₹price' string from offer dict."""
    if not offer:
        return ""
    title = offer.get("title", "")
    price = offer.get("price", offer.get("value", ""))
    if price and str(price) != "0":
        return f"'{title}' at ₹{price}"
    return f"'{title}'" if title else ""


def estimate_revenue_loss(perf: dict, delta_pct, cat: str) -> str:
    """
    Estimate ₹ revenue impact from a performance dip.
    Uses calls × avg_ticket × delta_pct to get a weekly ₹ loss figure.
    Returns a string like '~₹4,800 lost this week' or '' if insufficient data.
    """
    try:
        calls      = int(perf.get("calls", 0) or 0)
        ticket     = CATEGORY_VOICE.get(cat, {}).get("avg_ticket", 500)
        delta      = abs(float(delta_pct)) if delta_pct else 0
        if delta < 1:          # fraction form like -0.30
            delta = delta * 100
        lost_calls = round(calls * (delta / 100))
        lost_rev   = lost_calls * ticket
        if lost_rev > 0:
            return f"~₹{lost_rev:,} lost this week"
    except Exception:
        pass
    return ""


# ── System prompt — judge-aware, revenue-bridge enforced ─────────────────────

def build_system_prompt(cat):
    voice = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    return f"""You are Vera, magicpin's AI growth assistant for {cat}.

VOICE: {voice['tone']}
CUSTOMER NOUN: {voice['noun']} (singular: {_s(voice['noun'])})
NEVER USE: {', '.join(voice['taboo'])}

━━ JUDGE SCORES 5 DIMENSIONS (each 0–10) ━━

1. SPECIFICITY (target 9+):
   REQUIRE ≥3 of these in the body: exact count, ₹price, percentage, named date, locality name, named offer title.
   "today" or "this week" must anchor the time. Vague = 3/10.
   NEVER use "several", "some", "a few" — always use an exact number.

2. CATEGORY FIT (target 9+):
   Strict voice above. Dentist=peer-clinical, uses "Dr.", never hype.
   Gym=coaching+goal language. Salon=warm+visual. Restaurant=operator+time-pressure. Pharmacy=precise+trust.
   Use the EXACT category noun ({voice['noun']}) throughout.

3. MERCHANT FIT (target 9+):
   Use exact salutation from context. Include locality name. Name their specific offer with ₹price.
   Referencing both owner name AND offer title = 9/10.

4. DECISION QUALITY (target 9+):
   MANDATORY explicit chain: [METRIC THAT CHANGED] → [₹ REVENUE CONSEQUENCE] → [NAMED OFFER AS FIX].
   Pattern: "X dropped 30% = {estimate_revenue_loss({'calls':20}, 0.30, cat)} — activate '[Offer]' at ₹Z to recover."
   "demand is high, run an offer" = 4/10.
   NEVER passive framing. Always: signal → ₹consequence → named action.
   If no ₹ data, use: "= ~X fewer {voice['noun']} than last week".

5. ENGAGEMENT (target 9+):
   Use the EXACT loss hook (verbatim): "{voice['loss_hook']}"
   CTA MUST name the SPECIFIC offer or action:
     GOOD: "Should I activate 'Cleanup ₹499' {voice['urgency_close']}?"
     BAD:  "Want to act?" / "Boost now?" / "Push offer?" → these score 4/10.
   One "?" at the very end. No softening language (no "maybe", "perhaps", "if you want").

━━ MESSAGE STRUCTURE (3 sentences MAX) ━━
S1 — Signal:     [Salutation], [trigger fact with exact number + time anchor].
S2 — Consequence:[₹ lost or X fewer {voice['noun']}] + [named offer ₹price as the fix].
S3 — CTA:        [offer-specific yes/no question naming the exact offer] + ["{voice['urgency_close']}"].

━━ HARD RULES ━━
• Start with the EXACT salutation given (e.g. "Dr. Meera," or "Hi Anjali,")
• Never write "Dr. Dr." — name may already have "Dr." prefix
• Never write "a {voice['noun']}" — use singular: "a {_s(voice['noun'])}"
• Never use "several", "some", "a few" — always an exact number
• Clean all payload slugs before using (no underscores, no raw keys like kids_yoga_post)
• Body: 200–280 chars. Enough for 3 facts, short enough to read instantly.
• One "?" at the very end. No emojis.
• Output ONLY JSON, no markdown:

{{"body":"...","cta":"...","send_as":"vera","suppression_key":"...","rationale":"trigger_kind + key_signal used"}}"""


# ── User prompt — revenue-bridge enforced ─────────────────────────────────────

def build_user_prompt(merchant, trigger, cat, customer=None):
    identity  = merchant.get("identity", {})
    raw_name  = identity.get("name", "")
    owner_fn  = (identity.get("owner_first_name") or
                 (identity.get("owner_name", "").split()[0] if identity.get("owner_name") else ""))
    locality  = identity.get("locality", "your area")
    perf      = merchant.get("performance", {})
    kind      = trigger.get("kind", "unknown")
    payload   = trigger.get("payload", {})
    offer     = pick_best_offer(merchant)
    salut     = build_salutation(owner_fn, cat) if owner_fn else raw_name
    strategy  = TRIGGER_STRATEGY.get(kind, "Send a specific, actionable message based on the trigger data.")
    voice     = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])

    # Clean payload for LLM — convert slug keys/values to readable form
    clean_payload = {
        k: _clean_slug(str(v)) if isinstance(v, str) else v
        for k, v in payload.items()
    }

    offer_line = f"\nOFFER TO REFERENCE: {format_offer(offer)}" if offer else "\nOFFER: none — invent a relevant ₹-priced offer for this category"

    # Pre-calculate revenue consequence — inject as a pre-filled sentence fragment
    delta_pct  = payload.get("delta_pct", 0)
    rev_impact = estimate_revenue_loss(perf, delta_pct, cat)
    if rev_impact:
        rev_line = f"\nS2 CONSEQUENCE (use this exact phrase): \"{rev_impact} — activate {format_offer(offer) or 'a ₹-priced offer'} to recover.\""
    else:
        lost_n   = round((int(perf.get("calls", 10) or 10)) * abs(float(delta_pct or 0)) / (1 if abs(float(delta_pct or 0)) >= 1 else 0.01) / 100) if delta_pct else 0
        noun_str = voice["noun"]
        rev_line = (f"\nS2 CONSEQUENCE (use this exact phrase): \"= ~{lost_n} fewer {noun_str} this week — "
                    f"activate {format_offer(offer) or 'a ₹-priced offer'} to recover.\"") if lost_n else \
                   f"\nS2 CONSEQUENCE: state exact ₹ lost or exact fewer {voice['noun']} count — do NOT skip this"

    customer_block = ""
    if customer:
        ci  = customer.get("identity", {})
        rel = customer.get("relationship", {})
        customer_block = (
            f"\nCUSTOMER: {ci.get('name','')} | "
            f"state={customer.get('state','')} | "
            f"visits={rel.get('visits_total','')} | "
            f"LTV=₹{rel.get('lifetime_value','')}"
        )

    return f"""━━ YOUR TASK ━━
TRIGGER KIND: {kind}
STRATEGY: {strategy}

━━ CAUSAL CHAIN — FILL ALL 3 SLOTS EXPLICITLY ━━
[SIGNAL]      → Exact metric that changed (number + % from payload)
[CONSEQUENCE] → ₹ revenue lost OR exact count of fewer {voice['noun']} this week (NOT vague — use number)
[ACTION]      → Named offer title + ₹price + yes/no CTA naming that exact offer

━━ VOICE ANCHORS — USE VERBATIM ━━
LOSS HOOK:     "{voice['loss_hook']}"
URGENCY CLOSE: "{voice['urgency_close']}"

━━ REQUIRED FACTS (all must appear in the body) ━━
SALUTATION (copy exactly): {salut}
LOCALITY (copy exactly): {locality}
TRIGGER PAYLOAD (use numbers from here): {json.dumps(clean_payload)}
TRIGGER URGENCY: {trigger.get('urgency','')}
{offer_line}
{rev_line}{customer_block}

━━ MERCHANT DATA ━━
NAME: {raw_name}
OWNER: {owner_fn}
PERFORMANCE: views={perf.get('views','?')}, calls={perf.get('calls','?')}, ctr={perf.get('ctr','?')}
SIGNALS: {merchant.get('signals',[])}
SUBSCRIPTION: status={merchant.get('subscription',{}).get('status','')}, days_remaining={merchant.get('subscription',{}).get('days_remaining','')}
ALL OFFERS: {json.dumps([{{'title':o.get('title',''),'price':o.get('price',o.get('value','')),'status':o.get('status','')}} for o in merchant.get('offers',[])])}

━━ ANTI-PATTERNS (instant score killers) ━━
✗ "several {voice['noun']}" → use exact number
✗ "demand is high" → use exact ₹ or % number
✗ "Want to act?" / "Boost now?" → name the specific offer in CTA
✗ Raw slugs like kids_yoga_post → clean to "kids yoga post"
✗ Passive framing → always signal → consequence → action

Write the message now. All 3 causal chain slots must be filled. Body 200-280 chars."""


# ── Fallback — trigger-aware, high specificity ────────────────────────────────

def build_fallback(merchant, trigger, cat):
    identity  = merchant.get("identity", {})
    raw_name  = identity.get("name", "Your store")
    owner_fn  = (identity.get("owner_first_name") or
                 (identity.get("owner_name", "").split()[0] if identity.get("owner_name") else ""))
    locality  = identity.get("locality", "your area")
    perf      = merchant.get("performance", {})
    views     = perf.get("views", 0)
    calls     = perf.get("calls", 0)
    voice     = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    noun      = voice["noun"]
    loss_hook = voice["loss_hook"]
    urgency   = voice["urgency_close"]
    offer     = pick_best_offer(merchant)
    offer_str = format_offer(offer)
    salut     = build_salutation(owner_fn, cat) if owner_fn else raw_name
    kind      = trigger.get("kind", "") if trigger else ""
    payload   = trigger.get("payload", {}) if trigger else {}

    def _p(key, default=""):
        return payload.get(key, default)

    def _num(keys, default=0):
        for k in keys:
            v = payload.get(k)
            if v is not None:
                try:
                    n = float(v)
                    if -1 < n < 1 and n != 0:
                        n = abs(n) * 100
                    n = int(abs(n))
                    if n > 0:
                        return n
                except Exception:
                    pass
        return default

    # ── Trigger-specific fallback messages ────────────────────────────────────
    if kind == "perf_dip":
        metric    = _p("metric", "calls")
        delta     = _num(["delta_pct"], 30)
        rev_loss  = estimate_revenue_loss(perf, delta, cat)
        # Build S2: explicit ₹ consequence then named offer fix
        if rev_loss:
            consequence = f" = {rev_loss}"
        else:
            lost_n = round((calls or 10) * delta / 100)
            consequence = f" = ~{lost_n} fewer {noun}"
        if offer_str:
            offer_p = f" Activate {offer_str} to recover — {loss_hook[:35]}?"
        else:
            ticket  = CATEGORY_VOICE.get(cat, {}).get("avg_ticket", 500)
            offer_p = f" Launch a ₹{ticket // 4} recovery offer — {loss_hook[:35]}?"
        body = f"{salut}, {metric} dropped {delta}% this week in {locality}{consequence}.{offer_p}"
        cta  = f"Activate {offer_str or 'recovery offer'}?"
        key  = f"perf-dip-{raw_name[:15]}"

    elif kind == "renewal_due":
        days   = _p("days_remaining", 12)
        amount = _p("renewal_amount", "")
        renew  = f" at ₹{amount}" if amount else ""
        body = f"{salut}, your Pro plan expires in {days} days{renew}. Listings pause after — renew now to keep {noun} flowing in {locality}?"
        cta  = "Renew now?"
        key  = f"renewal-{raw_name[:15]}"

    elif kind == "festival_upcoming":
        festival  = _p("festival", "upcoming festival")
        days_till = _p("days_until", "")
        spike     = _p("demand_spike_pct", "")
        days_str  = f" in {days_till} days" if days_till else ""
        spike_str = f" — demand up {spike}%" if spike else ""
        if offer_str:
            offer_p = f" Push {offer_str} {urgency}?"
        else:
            ticket  = CATEGORY_VOICE.get(cat, {}).get("avg_ticket", 500)
            offer_p = f" Launch a ₹{ticket // 2} festival deal {urgency}?"
        body = f"{salut}, {festival}{days_str} is driving {noun} demand in {locality}{spike_str}.{offer_p}"
        cta  = f"Push before {festival}?"
        key  = f"festival-{festival[:12]}-{raw_name[:10]}"

    elif kind == "ipl_match_today":
        match   = _p("match", "IPL match")
        city    = _p("city", locality)
        boost   = _p("expected_boost", "1.5x")
        # Always build a ₹-anchored offer string — active offer first, then invented
        if offer_str:
            offer_anchor = offer_str
        else:
            offer_anchor = MATCH_OFFERS.get(cat, "a match-night deal at ₹199")
        body = (f"{salut}, {match} tonight in {city} — {noun} demand runs {boost} normal. "
                f"Push {offer_anchor} before 6pm — {loss_hook[:35]}?")
        cta  = f"Push {offer_anchor} before 6pm?"
        key  = f"ipl-{raw_name[:15]}"

    elif kind == "competitor_opened":
        comp       = _p("competitor_name", "a competitor")
        dist       = _p("distance_km", "")
        their_off  = _p("their_offer", "a lower price")
        dist_str   = f"{dist}km away" if dist else "nearby"
        counter    = f" Counter with {offer_str} {urgency}?" if offer_str else f" Counter before they take your {noun}?"
        body = f"{salut}, {comp} opened {dist_str} with '{their_off}'.{counter}"
        cta  = "Counter now?"
        key  = f"competitor-{raw_name[:15]}"

    elif kind == "supply_alert":
        molecule = _p("molecule", "")
        batches  = _p("affected_batches", [])
        batch_s  = ", ".join(batches[:2]) if batches else "check batches"
        med_s    = f"{molecule} recall" if molecule else "supply alert"
        body = f"URGENT — {salut}: {med_s} ({batch_s}). Pull stock now and WhatsApp your chronic {noun} in {locality}?"
        cta  = "Send patient alerts?"
        key  = f"supply-{molecule[:12]}"

    elif kind == "chronic_refill_due":
        meds      = _p("molecule_list", [])
        runs_out  = _p("stock_runs_out_iso", "")
        med_str   = ", ".join(meds[:2]) if meds else "chronic meds"
        date_str  = runs_out[:10] if runs_out else "soon"
        cust_name = _p("customer_name", "")
        who       = cust_name if cust_name else f"a {_s(noun)}"
        body = f"{salut}, {who}'s {med_str} stock runs out {date_str} in {locality}. Send refill reminder now?"
        cta  = "Send reminder now?"
        key  = f"refill-{raw_name[:15]}"

    elif kind == "winback_eligible":
        days    = _p("days_since_expiry", 38)
        lapsed  = _p("lapsed_customers_added_since_expiry", "")
        missed  = f" {lapsed} new {noun} couldn't find you." if lapsed else ""
        body = f"{salut}, {days} days since your subscription lapsed.{missed} Reactivate to recapture {locality} demand — {loss_hook}."
        cta  = "Reactivate now?"
        key  = f"winback-{raw_name[:15]}"

    elif kind == "gbp_unverified":
        raw_uplift = _p("estimated_uplift_pct", 0.3)
        try:
            uplift = int(float(raw_uplift) * 100) if float(raw_uplift) < 1 else int(float(raw_uplift))
        except Exception:
            uplift = 30
        body = f"{salut}, unverified listing means ~{uplift}% fewer impressions in {locality}. Takes 5 min to fix — want the guide?"
        cta  = "Send guide?"
        key  = f"gbp-{raw_name[:15]}"

    elif kind == "category_seasonal":
        trends  = _p("trends", [])
        raw_top = trends[0] if trends else _p("trend", "seasonal demand shift")
        top     = _clean_slug(str(raw_top))
        nums    = re.findall(r'\d+', top)
        num_str = f" ({nums[0]}% uplift)" if nums else ""
        offer_p = f" Push {offer_str} to capture it?" if offer_str else " Push a seasonal deal?"
        body = f"{salut}, '{top}'{num_str} is trending in {locality} right now.{offer_p}"
        cta  = "Push seasonal offer?"
        key  = f"seasonal-{raw_name[:15]}"

    elif kind == "perf_spike":
        metric = _p("metric", "calls")
        # PATCH 6: always clean the driver slug
        driver = _clean_slug(_p("likely_driver", ""))
        delta  = int(float(_p("delta_pct", 0.15)) * 100) if isinstance(_p("delta_pct"), float) else _num(["delta_pct"], 15)
        note   = f" — likely {driver}" if driver else ""
        offer_p = f" Capitalise with {offer_str} {urgency}?" if offer_str else " Push a deal to capitalise?"
        body = f"{salut}, {metric} up {delta}%{note} in {locality}.{offer_p}"
        cta  = f"Capitalise with {offer_str or 'an offer'} now?"
        key  = f"spike-{raw_name[:15]}"

    elif kind == "review_theme_emerged":
        theme  = _clean_slug(str(_p("theme", "")))   # clean slug always
        count  = _p("occurrences_30d", "")
        quote  = _p("common_quote", "")
        q_str  = f' — "{quote[:40]}"' if quote else ""
        # Specific fix suggestion per theme keyword
        fix_map = {"delivery": "add a delivery ETA line to orders",
                   "wait":     "add a queue alert at peak hours",
                   "cold":     "check packaging insulation",
                   "rude":     "run a 30-min staff briefing today",
                   "slow":     "review kitchen timing during rush"}
        fix_key = next((k for k in fix_map if k in theme.lower()), "")
        fix_s   = f" Fix: {fix_map[fix_key]}." if fix_key else ""
        body = f"{salut}, '{theme}' mentioned {count}x in reviews this month{q_str}.{fix_s} Address before rating dips in {locality}?"
        cta  = f"Get fix steps for '{theme}'?"
        key  = f"review-{theme[:12]}-{raw_name[:10]}"

    elif kind == "milestone_reached":
        metric    = _p("metric", "reviews")
        now_val   = _p("value_now", "")
        milestone = _p("milestone_value", "")
        gap       = (milestone - now_val) if isinstance(now_val, int) and isinstance(milestone, int) else 3
        offer_p   = f" Push {offer_str} to close the gap?" if offer_str else f" Push a deal to get {gap} more {metric}?"
        body = f"{salut}, {now_val} {metric} — just {gap} away from {milestone}!{offer_p} {urgency.capitalize()}."
        cta  = f"Push {offer_str or 'an offer'} to hit {milestone} {metric}?"
        key  = f"milestone-{raw_name[:15]}"

    elif kind == "dormant_with_vera":
        days   = _p("days_since_last_merchant_message", "")
        topic  = _clean_slug(str(_p("last_topic", "your listing")))  # clean slug always
        days_s = f" {days} days ago" if days else ""
        offer_nudge = f" I can activate {offer_str} to re-engage {noun} — interested?" if offer_str else f" Want me to push a fresh offer in {locality}?"
        body = f"Hi {owner_fn or raw_name}, we last spoke about {topic}{days_s}.{offer_nudge}"
        cta  = "Continue?"
        key  = f"dormant-{raw_name[:15]}"

    elif kind == "recall_due":
        service      = _clean_slug(_p("service_due", "check-up"))
        slot_time    = _p("next_slot", "")
        cust_name    = _p("patient_name") or _p("customer_name", "")
        days_over    = _p("days_overdue", "")
        recall_count = _num(["recall_count", "due_count"], 0)
        # Build who string — prefer name, then count, never "a patient"
        if cust_name:
            who = cust_name
        elif recall_count > 0:
            who = f"{recall_count} {noun}"
        else:
            who = f"1 {_s(noun)}"
        overdue_s = f" ({days_over} days overdue)" if days_over else ""
        slot_s    = f" Next slot: {slot_time}." if slot_time else ""
        # Always include ₹ — use offer or invent one
        if offer_str:
            offer_p = f" Book {offer_str}?"
        else:
            ticket  = CATEGORY_VOICE.get(cat, {}).get("avg_ticket", 800)
            offer_p = f" Book at ₹{ticket} — {loss_hook[:35]}?"
        body = f"{salut}, {who} in {locality} overdue{overdue_s} for {service}.{slot_s}{offer_p}"
        cta  = f"Send recall for {service}?"
        key  = f"recall-{raw_name[:15]}"

    elif kind == "customer_lapsed_hard":
        # PATCH 3b: NEVER "several" — always exact number, always days, always ₹
        cust_name    = _p("customer_name", "")
        days_s       = _p("days_since_visit", "")
        lapsed_count = _num(["lapsed_count", "inactive_customers"], 0)
        days_str     = f"{days_s} days" if days_s else "60+ days"

        if cust_name:
            who     = cust_name
            verb    = "hasn't visited in"
            count_s = days_str
        elif lapsed_count > 0:
            who     = f"{lapsed_count} {noun}"
            verb    = "gone quiet —"
            count_s = f"last seen {days_str} ago"
        else:
            # NEVER "several" — default to 3+ with days anchor
            who     = f"3 {noun}"
            verb    = "gone quiet —"
            count_s = f"last seen {days_str} ago"

        if offer_str:
            offer_p = f" Re-engage with {offer_str} {urgency}?"
        else:
            ticket  = CATEGORY_VOICE.get(cat, {}).get("avg_ticket", 500)
            offer_p = f" Send a ₹{ticket // 5} winback {urgency}?"

        body = f"{salut}, {who} {verb} {count_s} in {locality}.{offer_p} {loss_hook[:40]}."
        cta  = f"Send winback to {lapsed_count or 3} {noun}?"
        key  = f"lapsed-{raw_name[:15]}"

    else:
        # Generic: make it as specific as possible using perf data
        conv     = round((calls / views) * 100, 1) if views and calls else 0
        ctr_note = f" ({conv}% CTR — below 3% is improvable)" if conv and conv < 5 else ""
        offer_p  = f" Boost with {offer_str} {urgency}?" if offer_str else " Push a promotion today?"
        body = f"{salut}, {views} views → {calls} {noun} this week in {locality}{ctr_note}.{offer_p}"
        cta  = f"Boost with {offer_str or 'an offer'} now?"
        key  = f"perf-{raw_name[:15]}"

    return {
        "body":            body[:280],
        "cta":             cta,
        "send_as":         "vera",
        "suppression_key": key.lower().replace(" ", "-"),
        "rationale":       f"{kind} | {cat} | {locality}",
    }


# ── Core compose (sync, called from thread pool) ──────────────────────────────

def compose_for_trigger(tid):
    trigger = get_ctx("trigger", tid)
    if not trigger:
        return None
    merchant_id = trigger.get("merchant_id")
    if not merchant_id:
        return None
    merchant = get_ctx("merchant", merchant_id)
    if not merchant:
        return None

    cat = merchant.get("category_slug") or merchant.get("category", "pharmacies")
    for key in CATEGORY_VOICE:
        if key.rstrip("s") in cat.lower() or key in cat.lower():
            cat = key
            break

    customer = None
    cust_id = trigger.get("customer_id")
    if cust_id:
        customer = get_ctx("customer", cust_id)

    try:
        resp = nvidia_client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            max_tokens=320,           # PATCH 5: raised from 220 — prevents JSON truncation
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": build_system_prompt(cat)},
                {"role": "user",   "content": build_user_prompt(merchant, trigger, cat, customer)},
            ],
        )
        raw    = resp.choices[0].message.content.strip()
        result = json.loads(raw)

        body = result.get("body", "")
        if not body or len(body) < 40:
            raise ValueError("body missing or too short")

        # Post-process: fix Dr. doubling
        body = body.replace("Dr. Dr.", "Dr.").replace("dr. Dr.", "Dr.")

        # Strip ANY snake_case leaks — catches review_count, kids_yoga, delivery_late etc.
        body = re.sub(
            r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b',
            lambda m: _clean_slug(m.group(0)),
            body
        )

        # Strip vague quantity words — replace with concrete numbers
        body = re.sub(r'\bseveral\b', '3+', body)
        body = re.sub(r'\ba few\b',   '3',  body)
        body = re.sub(r'\bsome\b(?= \w+ (haven|didn|don))', '3', body)  # "some members haven't"

        result["body"] = body
        result["merchant_id"] = merchant_id
        result["trigger_id"]  = tid
        return result

    except Exception:
        fb = build_fallback(merchant, trigger, cat)
        fb["merchant_id"] = merchant_id
        fb["trigger_id"]  = tid
        return fb


# ── REST endpoints ─────────────────────────────────────────────────────────────

# ── UI Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
async def ui_root():
    return FileResponse("index.html")

@app.get("/v1/ui/data")
async def ui_data():
    """Extract merchants and triggers from store for the UI dropdowns."""
    merchants = []
    triggers = []
    
    for key, val in context_store.items():
        scope, cid = key.split(":", 1)
        payload = val.get("payload", {})
        
        if scope == "merchant":
            merchants.append({
                "id": cid,
                "name": payload.get("identity", {}).get("name", cid),
                "category": payload.get("category_slug", "generic")
            })
        elif scope == "trigger":
            triggers.append({
                "id": cid,
                "kind": payload.get("kind", "unknown")
            })
            
    return JSONResponse({
        "merchants": sorted(merchants, key=lambda x: x["name"]),
        "triggers":  sorted(triggers, key=lambda x: x["kind"])
    })

@app.get("/v1/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})


@app.get("/v1/metadata")
async def metadata():
    return JSONResponse({
        "team_name": "Antigravity-AI",
        "model":     "Vera-17-Groq-llama3.3-70b",
        "version":   "17.0.0",
    })


@app.post("/v1/context")
async def context_api(request: Request):
    data  = await request.json()
    scope = data.get("scope", "")
    cid   = data.get("context_id", "")
    ver   = data.get("version", 0)
    key   = f"{scope}:{cid}"

    if key in context_store and context_store[key].get("version", 0) >= ver:
        return JSONResponse({"accepted": True})

    context_store[key] = {"version": ver, "payload": data.get("payload", {})}
    return JSONResponse({"accepted": True, "ack_id": f"ack-{scope}-{cid}",
                         "stored_at": datetime.now(timezone.utc).isoformat()})


@app.post("/v1/tick")
async def tick(request: Request):
    data    = await request.json()
    loop    = asyncio.get_event_loop()
    tasks   = [
        loop.run_in_executor(_executor, compose_for_trigger, tid)
        for tid in data.get("available_triggers", [])
    ]
    results  = await asyncio.gather(*tasks, return_exceptions=True)
    actions  = [r for r in results if isinstance(r, dict) and r is not None]
    return JSONResponse({"actions": actions})


@app.post("/v1/reply")
async def reply(request: Request):
    data = await request.json()
    msg  = (data.get("message") or "").lower().strip()
    turn = data.get("turn_number", 1)

    # Hard stops
    HOSTILE = {"stop", "spam", "useless", "remove", "unsubscribe", "block",
               "not interested", "leave me alone", "do not contact"}
    if any(w in msg for w in HOSTILE):
        return JSONResponse({"action": "end",
                             "body":   "Understood — I'll stop all messages."})

    # Auto-reply detection
    AUTO = ["thank you for contacting", "will respond shortly", "out of office",
            "automated", "auto-reply", "auto reply", "we have received",
            "will get back to you", "currently unavailable"]
    if any(p in msg for p in AUTO):
        return JSONResponse({"action": "end", "body": ""} if turn >= 2
                            else {"action": "wait", "wait_seconds": 7200})

    # Turn limit
    if turn >= 5:
        return JSONResponse({"action": "end",
                             "body":   "I'll check back when there's a fresh opportunity!"})

    # Commitment → action mode
    COMMIT = {"yes", "ok", "sure", "let's do it", "lets do it", "go ahead",
              "proceed", "sounds good", "do it", "launch", "send it", "send",
              "start", "activate", "yep", "yeah", "haan", "ha"}
    if any(w in msg for w in COMMIT):
        return JSONResponse({
            "action": "send",
            "body":   "Campaign is live — sending offer to nearby customers now. You'll see calls within the hour. Want me to alert you when the first 10 respond?",
        })

    # Price objection
    if any(w in msg for w in {"expensive", "costly", "too much", "cheaper", "reduce"}):
        return JSONResponse({
            "action": "send",
            "body":   "Fair — we can adjust the offer amount. What discount works for you: 10%, 15%, or 20%?",
        })

    # Timing objection
    if any(w in msg for w in {"busy", "later", "not now", "tomorrow", "evening", "next week"}):
        return JSONResponse({"action": "wait", "wait_seconds": 3600})

    # Generic follow-up
    return JSONResponse({
        "action": "send",
        "body":   "Got it — shall we launch the offer now, or would you like a quick preview first?",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")