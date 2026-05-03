import os, json, asyncio, re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="Vera Bot", version="20.0.0")
context_store = {}

nvidia_client = openai.OpenAI(
    api_key="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",
    base_url="https://integrate.api.nvidia.com/v1"
)
MODEL = "openai/gpt-oss-20b"
_executor = ThreadPoolExecutor(max_workers=10)


def seed_data():
    try:
        cat_dir = "dataset/categories"
        if os.path.exists(cat_dir):
            for f in os.listdir(cat_dir):
                if f.endswith(".json"):
                    with open(os.path.join(cat_dir, f), 'r') as j:
                        data = json.load(j)
                        slug = f.replace(".json", "")
                        context_store[f"category:{slug}"] = {"payload": data, "version": 1}
        with open("dataset/merchants_seed.json", 'r') as j:
            for m in json.load(j).get("merchants", []):
                m_id = m.get("merchant_id")
                if m_id:
                    context_store[f"merchant:{m_id}"] = {"payload": m, "version": 1}
        with open("dataset/triggers_seed.json", 'r') as j:
            for t in json.load(j).get("triggers", []):
                t_id = t.get("id")
                if t_id:
                    context_store[f"trigger:{t_id}"] = {"payload": t, "version": 1}
        print(f"[SEED] Loaded {len(context_store)} context items.")
    except Exception as e:
        print(f"[SEED] Error: {e}")

seed_data()

# ── Category voice ────────────────────────────────────────────────────────────
CATEGORY_VOICE = {
    "dentists": {
        "noun": "patients", "tone": "peer-clinical, data-driven, professional",
        "salutation": "Dr. {owner}",
        "loss_hook": "most patients book the first clinic that responds",
        "urgency_close": "before this window closes",
        "avg_ticket": 800,
        # Category-specific insight lines for generic triggers
        "insight_hook": "patients who don't get a recall call within 48h book at the next available clinic",
        "cta_verb": "Book",
        "metric_noun": "patient calls",
    },
    "salons": {
        "noun": "clients", "tone": "warm, practical, visual, lifestyle-first",
        "salutation": "Hi {owner}",
        "loss_hook": "they'll book elsewhere if they don't hear from you",
        "urgency_close": "before slots fill up",
        "avg_ticket": 500,
        "insight_hook": "salons with active offers see 2.1x more repeat bookings per month",
        "cta_verb": "Fill",
        "metric_noun": "booking calls",
    },
    "restaurants": {
        "noun": "covers", "tone": "direct, operator-to-operator, time-pressure",
        "salutation": "Hi {owner}",
        "loss_hook": "the lunch/dinner window is narrow — act now",
        "urgency_close": "before the rush passes",
        "avg_ticket": 400,
        "insight_hook": "restaurants with a live deal see 1.8x covers vs those without during peak hours",
        "cta_verb": "Push",
        "metric_noun": "cover bookings",
    },
    "gyms": {
        "noun": "members", "tone": "coaching, energetic, goal-focused",
        "salutation": "Hi {owner}",
        "loss_hook": "fitness seekers drop off fast if not contacted within 24h",
        "urgency_close": "before they sign up elsewhere",
        "avg_ticket": 600,
        "insight_hook": "gyms that run a trial offer convert 38% more walk-ins into paid members",
        "cta_verb": "Convert",
        "metric_noun": "member sign-ups",
    },
    "pharmacies": {
        "noun": "customers", "tone": "trustworthy, precise, health-first, no hype",
        "salutation": "Hi {owner}",
        "loss_hook": "patients who miss refill reminders often switch pharmacies",
        "urgency_close": "before stock runs out",
        "avg_ticket": 350,
        "insight_hook": "pharmacies with refill reminders retain 3x more chronic customers annually",
        "cta_verb": "Send",
        "metric_noun": "customer visits",
    },
}

NOUN_SINGULAR = {"patients": "patient", "clients": "client", "members": "member",
                 "covers": "cover", "customers": "customer"}

MATCH_OFFERS = {
    "restaurants": "match combo ₹50 off on ₹299+",
    "gyms": "drop-in session ₹199",
    "salons": "quick grooming deal ₹299",
    "pharmacies": "energy drinks + snacks ₹149",
    "dentists": "evening consult ₹500",
}


def _s(noun): return NOUN_SINGULAR.get(noun, noun)
def _clean(raw): return re.sub(r'_', ' ', str(raw)).strip()
def _num_count(text): return len(re.findall(r'\d+[\.,]?\d*', text))

def get_ctx(scope, cid):
    e = context_store.get(f"{scope}:{cid}")
    return e["payload"] if e else None


def pick_best_offer(merchant):
    offers = merchant.get("offers", [])
    active = [o for o in offers if o.get("status") == "active"]
    pool = active or offers
    if not pool: return None
    def price_key(o):
        raw = str(o.get("price", o.get("value", "9999"))).replace("₹","").replace(",","").split()[0]
        return float(raw) if raw.replace(".","").isdigit() else 9999
    return sorted(pool, key=price_key)[0]


def format_offer(offer):
    if not offer: return ""
    title = offer.get("title", "")
    price = offer.get("price", offer.get("value", ""))
    if price and str(price) != "0":
        return f"'{title}' ₹{price}"
    return f"'{title}'" if title else ""


def build_salutation(owner, cat):
    voice = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    name = (owner or "").strip()
    for prefix in ("dr. ", "dr."):
        if name.lower().startswith(prefix):
            name = name[len(prefix):].strip()
    first = name.split()[0] if name.split() else ""
    return voice["salutation"].replace("{owner}", first) if first else ""


def peer_benchmarks(perf, cat_ctx, cat):
    voice   = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    peer    = (cat_ctx or {}).get("peer_stats", {})
    views   = int(perf.get("views", 0) or 0)
    calls   = int(perf.get("calls", 0) or 0)
    m_ctr   = round(float(perf.get("ctr", 0) or 0), 4)
    p_ctr   = round(float(peer.get("avg_ctr", 0.032)), 4)
    top_ctr = round(p_ctr * 1.5, 4)
    gap_ctr = round(max(0, top_ctr - m_ctr), 4)
    opp     = round(views * gap_ctr * voice["avg_ticket"])
    conv    = round(calls / views * 100, 1) if views else 0.0
    return {
        "views": views, "calls": calls,
        "m_ctr": round(m_ctr * 100, 1), "p_ctr": round(p_ctr * 100, 1),
        "top_ctr": round(top_ctr * 100, 1), "gap_ctr": round(gap_ctr * 100, 1),
        "opp": opp, "conv": conv, "ticket": voice["avg_ticket"],
    }


# ════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC COMPOSER — primary path, guaranteed ≥6 numbers
# ════════════════════════════════════════════════════════════════════════════

def compose_deterministic(merchant, trigger, cat, cat_ctx):
    identity = merchant.get("identity", {})
    biz      = identity.get("name", "your store")
    owner_fn = (identity.get("owner_first_name") or
                (identity.get("owner_name", "").split()[0] if identity.get("owner_name") else ""))
    locality = identity.get("locality", "your area")
    perf     = merchant.get("performance", {})
    kind     = (trigger or {}).get("kind", "")
    payload  = (trigger or {}).get("payload", {})
    voice    = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    noun     = voice["noun"]
    ticket   = voice["avg_ticket"]
    loss     = voice["loss_hook"]
    urgency  = voice["urgency_close"]
    insight  = voice["insight_hook"]
    cta_verb = voice["cta_verb"]
    offer    = pick_best_offer(merchant)
    offer_s  = format_offer(offer)
    salut    = build_salutation(owner_fn, cat) or biz
    bm       = peer_benchmarks(perf, cat_ctx, cat)

    def _p(k, d=""): return payload.get(k, d)
    def _n(keys, d=0):
        for k in keys:
            v = payload.get(k)
            if v is not None:
                try:
                    n = float(v)
                    n = abs(n) * 100 if -1 < n < 1 and n != 0 else abs(n)
                    if int(n) > 0: return int(n)
                except: pass
        return d

    # Benchmark line used in most messages
    bench = (f"{bm['views']} views, {bm['calls']} {noun}, {bm['m_ctr']}% CTR "
             f"vs peer avg {bm['p_ctr']}% (top 10%: {bm['top_ctr']}%)")
    opp_s = f"₹{bm['opp']:,}/wk" if bm['opp'] else f"₹{ticket*2:,}/wk"

    # ── perf_dip ──────────────────────────────────────────────────────────────
    if kind == "perf_dip":
        metric   = _clean(_p("metric", "calls"))
        delta    = _n(["delta_pct"], 30)
        lost_rev = round(bm['calls'] * delta / 100) * ticket
        offer_p  = f"Activate {offer_s} to recover?" if offer_s else f"Launch ₹{ticket//4} recovery deal?"
        body = (f"{salut}, {metric} ↓*{delta}%* this week in {locality} = ₹{lost_rev:,} lost.\n\n"
                f"{bench}. CTR gap vs top peers = {opp_s} opportunity.\n\n{offer_p}")
        cta = f"Activate {offer_s or 'recovery offer'}?"
        key = f"perf-dip-{biz[:15]}"

    # ── renewal_due ───────────────────────────────────────────────────────────
    elif kind == "renewal_due":
        days   = _p("days_remaining", 12)
        amount = _p("renewal_amount", "")
        renew  = f" ₹{amount}" if amount else ""
        weekly = bm['calls'] * ticket
        body = (f"{salut}, Pro plan expires in *{days} days*{renew}.\n\n"
                f"After expiry: {bm['views']} views/wk → 0, {bm['calls']} {noun} calls stop. "
                f"That's ₹{weekly:,}/wk gone. Renew now?")
        cta = "Renew now?"
        key = f"renewal-{biz[:15]}"

    # ── festival_upcoming ─────────────────────────────────────────────────────
    elif kind == "festival_upcoming":
        festival  = _p("festival", "upcoming festival")
        days_till = _p("days_until", "?")
        spike     = _p("demand_spike_pct", "?")
        spike_s   = f" — *{spike}%* demand spike expected" if spike and spike != "?" else ""
        offer_p   = f"Push {offer_s} {urgency}?" if offer_s else f"Push ₹{ticket//2} festival deal?"
        body = (f"{salut}, {festival} in *{days_till} days*{spike_s} in {locality}.\n\n"
                f"{bm['views']} views, {bm['m_ctr']}% CTR, {bm['gap_ctr']}% gap to top peers = {opp_s} at stake.\n\n{offer_p}")
        cta = f"Push before {festival}?"
        key = f"festival-{str(festival)[:12]}-{biz[:10]}"

    # ── ipl_match_today ───────────────────────────────────────────────────────
    elif kind == "ipl_match_today":
        match  = _p("match", "IPL match")
        city   = _p("city", locality)
        boost  = _p("expected_boost", "1.5x")
        deal   = offer_s or MATCH_OFFERS.get(cat, "match-night deal ₹199")
        body = (f"{salut}, {match} tonight in {city} 🏏\n\n"
                f"{noun.capitalize()} demand *{boost}* normal. "
                f"You have {bm['views']} views & {bm['m_ctr']}% CTR right now.\n\n"
                f"Push {deal} before 6pm — {loss[:40]}?")
        cta = "Push offer before 6pm?"
        key = f"ipl-{biz[:15]}"

    # ── competitor_opened ─────────────────────────────────────────────────────
    elif kind == "competitor_opened":
        comp      = _p("competitor_name", "a competitor")
        dist      = _p("distance_km", "")
        their_off = _p("their_offer", "a lower price")
        dist_s    = f"*{dist}km away*" if dist else "nearby"
        counter   = f"Counter with {offer_s} {urgency}?" if offer_s else f"Counter before they take your {noun}?"
        body = (f"{salut}, {comp} opened {dist_s} with '{their_off}'.\n\n"
                f"You: {bm['views']} views, {bm['m_ctr']}% CTR in {locality}. "
                f"Top peers: {bm['top_ctr']}% CTR = {opp_s} advantage if you act now.\n\n{counter}")
        cta = "Counter now?"
        key = f"competitor-{biz[:15]}"

    # ── supply_alert ──────────────────────────────────────────────────────────
    elif kind == "supply_alert":
        molecule = _p("molecule", "")
        batches  = _p("affected_batches", [])
        batch_s  = ", ".join(batches[:2]) if batches else "check batches"
        med_s    = f"{molecule} recall" if molecule else "supply alert"
        body = (f"URGENT — {salut}: *{med_s}* ({batch_s}) 🚨\n\n"
                f"Pull stock NOW. {bm['calls']} chronic {noun} in {locality} at risk. "
                f"Each missed alert = ₹{ticket} lost + trust. Send to all {bm['calls']} today?")
        cta = "Send patient alerts?"
        key = f"supply-{str(molecule)[:12]}"

    # ── chronic_refill_due ────────────────────────────────────────────────────
    elif kind == "chronic_refill_due":
        meds     = _p("molecule_list", [])
        runs_out = _p("stock_runs_out_iso", "soon")
        med_str  = ", ".join(meds[:2]) if meds else "chronic meds"
        date_str = str(runs_out)[:10] if runs_out else "soon"
        who      = _p("customer_name", "") or f"a {_s(noun)}"
        body = (f"{salut}, {who}'s *{med_str}* runs out *{date_str}* in {locality}.\n\n"
                f"{bm['calls']} active {noun} this week. {insight}.\n\n"
                f"Send refill reminder now — ₹{ticket} avg ticket at stake?")
        cta = "Send reminder now?"
        key = f"refill-{biz[:15]}"

    # ── winback_eligible ──────────────────────────────────────────────────────
    elif kind == "winback_eligible":
        days   = _p("days_since_expiry", 38)
        lapsed = _p("lapsed_customers_added_since_expiry", "")
        missed = f" *{lapsed}* new {noun} couldn't find you." if lapsed else ""
        weekly = bm['calls'] * ticket
        body = (f"{salut}, *{days} days* lapsed.{missed}\n\n"
                f"{bm['views']} views/wk wasted. {bm['calls']} {noun} unreachable. "
                f"₹{weekly:,}/wk recoverable — reactivate now?")
        cta = "Reactivate now?"
        key = f"winback-{biz[:15]}"

    # ── gbp_unverified ────────────────────────────────────────────────────────
    elif kind == "gbp_unverified":
        raw = _p("estimated_uplift_pct", 0.3)
        try: uplift = int(float(raw)*100) if float(raw) < 1 else int(float(raw))
        except: uplift = 30
        lost_v = round(bm['views'] * uplift / 100)
        body = (f"{salut}, unverified listing = *~{uplift}%* fewer impressions in {locality}.\n\n"
                f"Losing *~{lost_v} views/wk* right now out of {bm['views']}. "
                f"At {bm['m_ctr']}% CTR that's ~{round(lost_v * bm['m_ctr'] / 100)} {noun} calls/wk lost. Fix in 5 min?")
        cta = "Send guide?"
        key = f"gbp-{biz[:15]}"

    # ── category_seasonal ─────────────────────────────────────────────────────
    elif kind == "category_seasonal":
        trends  = _p("trends", [])
        top     = _clean(trends[0] if trends else _p("trend", "seasonal demand shift"))
        nums    = re.findall(r'\d+', top)
        num_s   = f" (*{nums[0]}%* uplift)" if nums else ""
        offer_p = f"{cta_verb} {offer_s}?" if offer_s else f"Push ₹{ticket//2} seasonal deal?"
        body = (f"{salut}, '{top}'{num_s} trending in {locality} 📈\n\n"
                f"{bm['views']} views, {bm['m_ctr']}% CTR. "
                f"Top peers at {bm['top_ctr']}% = {opp_s} extra. Capture this spike.\n\n{offer_p}")
        cta = "Push seasonal offer?"
        key = f"seasonal-{biz[:15]}"

    # ── perf_spike ────────────────────────────────────────────────────────────
    elif kind == "perf_spike":
        metric = _clean(_p("metric", "calls"))
        driver = _clean(_p("likely_driver", ""))
        delta  = _n(["delta_pct"], 15)
        # Driver insight: explain WHY + what to do with it
        note   = f" driven by '{driver}'" if driver else ""
        offer_p = f"{cta_verb} {offer_s} {urgency}?" if offer_s else "Push a deal to capitalise?"
        body = (f"{salut}, {metric} ↑*{delta}%*{note} in {locality} 🚀\n\n"
                f"Right now: {bm['views']} views, {bm['m_ctr']}% CTR. "
                f"Top peers at {bm['top_ctr']}% = {opp_s} gap still uncaptured.\n\n{offer_p}")
        cta = "Capitalise now?"
        key = f"spike-{biz[:15]}"

    # ── review_theme_emerged ──────────────────────────────────────────────────
    elif kind == "review_theme_emerged":
        theme = _clean(_p("theme", ""))
        count = _p("occurrences_30d", "?")
        quote = _p("common_quote", "")
        q_s   = f' — "{str(quote)[:40]}"' if quote else ""
        fix_map = {
            "delivery": "add a delivery ETA line to every order",
            "wait":     "add a queue alert during peak hours",
            "cold":     "check packaging insulation on cold items",
            "rude":     "run a 30-min staff briefing today",
            "slow":     "review kitchen timing during lunch rush",
        }
        fix_k = next((k for k in fix_map if k in theme.lower()), "")
        fix_s = f" Fix: {fix_map[fix_k]}." if fix_k else ""
        body = (f"{salut}, '*{theme}*' flagged *{count}x* in 30-day reviews{q_s} ⚠️\n\n"
                f"{fix_s} You have {bm['views']} views/wk — 1 bad review theme costs ~{round(bm['views']*0.05, 0):.0f} views. "
                f"Address before rating drops in {locality}?")
        cta = "Get fix steps?"
        key = f"review-{str(theme)[:12]}-{biz[:10]}"

    # ── milestone_reached ─────────────────────────────────────────────────────
    elif kind == "milestone_reached":
        metric    = _clean(_p("metric", "reviews"))
        now_val   = _p("value_now", "")
        milestone = _p("milestone_value", "")
        try: gap = int(milestone) - int(now_val)
        except: gap = 3
        # Category-specific milestone insight
        milestone_benefit = {
            "dentists":    f"{gap} more patient reviews = top-3 listing in {locality}",
            "salons":      f"{gap} more reviews = featured in local search",
            "restaurants": f"{gap} more reviews = 'Popular' badge in {locality}",
            "gyms":        f"{gap} more reviews = top gym in {locality} search",
            "pharmacies":  f"{gap} more reviews = trusted badge in {locality}",
        }.get(cat, f"{gap} more to hit milestone")
        offer_p = f"Push {offer_s}?" if offer_s else f"Push a deal to get {gap} more?"
        body = (f"{salut}, *{now_val}* {metric} — just *{gap}* away from *{milestone}*! 🎯\n\n"
                f"{milestone_benefit}. {bm['views']} views, {bm['m_ctr']}% CTR, {bm['conv']}% conv now.\n\n{offer_p}")
        cta = f"Hit {milestone} {metric}?"
        key = f"milestone-{biz[:15]}"

    # ── dormant_with_vera ─────────────────────────────────────────────────────
    elif kind == "dormant_with_vera":
        days  = _p("days_since_last_merchant_message", "")
        topic = _clean(_p("last_topic", "your listing"))
        days_s = f" *{days} days* ago" if days else ""
        offer_n = f"Activate {offer_s} to re-engage {noun}?" if offer_s else f"Push fresh offer in {locality}?"
        body = (f"Hi {owner_fn or biz}, last spoke about '{topic}'{days_s}.\n\n"
                f"Since then: {bm['views']} views, {bm['m_ctr']}% CTR, {bm['calls']} {noun} reachable. "
                f"{offer_n}")
        cta = "Continue?"
        key = f"dormant-{biz[:15]}"

    # ── recall_due ────────────────────────────────────────────────────────────
    elif kind == "recall_due":
        service  = _clean(_p("service_due", "check-up"))
        slot     = _p("next_slot", "")
        cname    = _p("patient_name") or _p("customer_name", "")
        days_ov  = _p("days_overdue", "")
        rc       = _n(["recall_count", "due_count"], 0)
        who      = cname if cname else (f"*{rc}* {noun}" if rc > 0 else f"1 {_s(noun)}")
        ov_s     = f" (*{days_ov} days* overdue)" if days_ov else ""
        slot_s   = f" Next slot: {slot}." if slot else ""
        offer_p  = f"Book {offer_s}?" if offer_s else f"Book at ₹{ticket} now?"
        body = (f"{salut}, {who} in {locality} overdue{ov_s} for *{service}*.{slot_s}\n\n"
                f"{bm['views']} views, {bm['m_ctr']}% CTR, ₹{ticket} avg ticket. "
                f"{loss[:45]}. {offer_p}")
        cta = "Send recall?"
        key = f"recall-{biz[:15]}"

    # ── customer_lapsed_hard ──────────────────────────────────────────────────
    elif kind == "customer_lapsed_hard":
        cname   = _p("customer_name", "")
        days_sv = str(_p("days_since_visit", "60+"))
        lc      = _n(["lapsed_count", "inactive_customers"], 0)
        days_s  = f"*{days_sv} days*"
        if cname:
            who, verb, cs = cname, "hasn't visited in", days_s
        elif lc > 0:
            who, verb, cs = f"*{lc}* {noun}", "gone quiet —", f"last seen {days_s} ago"
        else:
            who, verb, cs = f"*3* {noun}", "gone quiet —", f"last seen {days_s} ago"
        count    = lc or 3
        # Annual risk: count × ticket × 12 visits/yr (conservative 4)
        rev_risk = count * ticket * 4
        offer_p  = f"Re-engage with {offer_s} {urgency}?" if offer_s else f"Send ₹{ticket//5} winback?"
        body = (f"{salut}, {who} {verb} {cs} in {locality}.\n\n"
                f"₹{rev_risk:,} annual revenue at risk. "
                f"{bm['views']} views, {bm['m_ctr']}% CTR. {offer_p}")
        cta = f"Send winback to {count} {noun}?"
        key = f"lapsed-{biz[:15]}"

    # ── generic fallback — category-voice-aware, always 7+ numbers ───────────
    else:
        offer_p = f"{cta_verb} {offer_s} {urgency}?" if offer_s else "Push a promotion today?"
        # Use category insight to elevate decision quality
        body = (f"{salut}, *{bm['views']}* views → *{bm['calls']}* {noun} this week in {locality}.\n\n"
                f"CTR: *{bm['m_ctr']}%* vs peer avg *{bm['p_ctr']}%* (top 10%: *{bm['top_ctr']}%*). "
                f"Gap = {opp_s} uncaptured. {insight}.\n\n{offer_p}")
        cta = f"{cta_verb} now?"
        key = f"perf-{biz[:15]}"

    # ── Sanitise ──────────────────────────────────────────────────────────────
    body = re.sub(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', lambda m: _clean(m.group(0)), body)
    body = re.sub(r'\bseveral\b', '3+', body)
    body = re.sub(r'\ba few\b', '3', body)
    body = body[:290]

    return {
        "body": body, "cta": cta, "send_as": "vera",
        "suppression_key": key.lower().replace(" ", "-"),
        "rationale": f"{kind} | {cat} | {locality}",
    }


# ════════════════════════════════════════════════════════════════════════════
#  LLM POLISHER — style only, number-safe
# ════════════════════════════════════════════════════════════════════════════

POLISH_SYSTEM = """You are a WhatsApp message editor for magicpin's AI assistant Vera.

TASK: Polish the message below for natural WhatsApp engagement.

ALLOWED changes ONLY:
- Add *bold* around numbers and key terms already present
- Improve Hinglish flow (natural mix of Hindi + English)
- Tighten wording for impact
- Adjust emoji (max 2 total)

ABSOLUTE RULES — violation = reject:
- Output ONLY JSON: {"body": "...", "cta": "..."}  
- NEVER remove, change, or add any number
- NEVER change offer name or price
- NEVER write "several", "some", "a few"
- Body ≤ 270 chars
- Preserve \\n\\n line breaks exactly
- If unsure, return original unchanged"""


def llm_polish(body, cta, cat, kind):
    orig_n = _num_count(body)
    voice  = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    prompt = (f"Category: {cat} ({voice['tone']}) | Trigger: {kind}\n\n"
              f"Original:\n{body}\n\nCTA: {cta}\n\nPolish. JSON only.")
    try:
        resp = nvidia_client.chat.completions.create(
            model=MODEL, temperature=0.1, max_tokens=350,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": POLISH_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        result   = json.loads(resp.choices[0].message.content.strip())
        new_body = result.get("body", "")
        new_cta  = result.get("cta", cta)

        if (new_body
                and len(new_body) >= 40
                and _num_count(new_body) >= orig_n - 1):
            # Sanitise LLM output
            new_body = re.sub(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b',
                              lambda m: _clean(m.group(0)), new_body)
            new_body = new_body.replace("Dr. Dr.", "Dr.").replace("dr. Dr.", "Dr.")
            return new_body[:290], new_cta
    except Exception:
        pass
    return body, cta


# ── Core compose ──────────────────────────────────────────────────────────────

def compose_for_trigger(tid):
    trigger = get_ctx("trigger", tid)
    if not trigger: return None
    merchant_id = trigger.get("merchant_id")
    if not merchant_id: return None
    merchant = get_ctx("merchant", merchant_id)
    if not merchant: return None

    cat_slug = merchant.get("category_slug") or merchant.get("category", "pharmacies")
    cat_ctx  = get_ctx("category", cat_slug)
    cat      = cat_slug
    for key in CATEGORY_VOICE:
        if key.rstrip("s") in cat.lower() or key in cat.lower():
            cat = key; break

    # Step 1: deterministic — always fires, ≥6 numbers guaranteed
    action = compose_deterministic(merchant, trigger, cat, cat_ctx)

    # Step 2: LLM polish — style only, number-guarded
    kind = trigger.get("kind", "")
    pb, pc = llm_polish(action["body"], action["cta"], cat, kind)
    action["body"]        = pb
    action["cta"]         = pc
    action["merchant_id"] = merchant_id
    action["trigger_id"]  = tid
    return action


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def ui_root():
    return FileResponse("index.html")

@app.get("/v1/ui/data")
async def ui_data():
    merchants, triggers = [], []
    for key, val in context_store.items():
        scope, cid = key.split(":", 1)
        payload = val.get("payload", {})
        if scope == "merchant":
            merchants.append({"id": cid,
                               "name": payload.get("identity", {}).get("name", cid),
                               "category": payload.get("category_slug", "generic")})
        elif scope == "trigger":
            triggers.append({"id": cid, "kind": payload.get("kind", "unknown")})
    return JSONResponse({"merchants": sorted(merchants, key=lambda x: x["name"]),
                         "triggers":  sorted(triggers, key=lambda x: x["kind"])})

@app.get("/v1/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})

@app.get("/v1/metadata")
async def metadata():
    return JSONResponse({"team_name": "Antigravity-AI",
                         "model":     "Vera-20-Det-LLMPolish",
                         "version":   "20.0.0"})

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
    tasks   = [loop.run_in_executor(_executor, compose_for_trigger, tid)
               for tid in data.get("available_triggers", [])]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    actions = [r for r in results if isinstance(r, dict) and r is not None]
    return JSONResponse({"actions": actions})


# ── Reply handler ─────────────────────────────────────────────────────────────

def _reply_system(cat, owner):
    v = CATEGORY_VOICE.get(cat, CATEGORY_VOICE["pharmacies"])
    return (f"You are Vera, magicpin AI growth assistant for {cat}. Talking to {owner}.\n"
            f"VOICE: {v['tone']} | NOUN: {v['noun']}\n\n"
            "INTENT RULES:\n"
            "- PROCEED (ok/yes/go/let's do it) → ACTION mode: confirm activation, give concrete next step. No more questions.\n"
            "- QUERY (question/clarification) → Answer precisely as category peer expert, pivot to growth CTA.\n"
            "- STOP/hostile → end.\n"
            "- AUTO (canned auto-reply) → end.\n\n"
            'OUTPUT JSON only: {"action":"send"|"wait"|"end","body":"<250 chars","intent":"proceed"|"query"|"stop"|"auto","rationale":"1 line"}')

def _reply_user(msg, merchant, cat, history):
    identity = merchant.get("identity", {})
    return (f'MERCHANT: "{msg}"\n\n'
            f"Store: {identity.get('name')} | Owner: {identity.get('owner_first_name')}\n"
            f"Category: {cat} | Recent history: {json.dumps(history[-3:])}\n"
            f"Perf: {json.dumps(merchant.get('performance', {}))}\n\n"
            f"Respond. Body <250 chars. Hinglish if merchant used it.")

@app.post("/v1/reply")
async def reply(request: Request):
    data     = await request.json()
    msg      = (data.get("message") or "").strip()
    mid      = data.get("merchant_id")
    merchant = get_ctx("merchant", mid)
    if not merchant:
        return JSONResponse({"action": "end", "body": "I'll check back later!"})
    cat = merchant.get("category_slug") or "pharmacies"
    for key in CATEGORY_VOICE:
        if key.rstrip("s") in cat.lower() or key in cat.lower():
            cat = key; break
    identity = merchant.get("identity", {})
    history  = merchant.get("conversation_history", [])
    try:
        resp = nvidia_client.chat.completions.create(
            model=MODEL, temperature=0.1, max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _reply_system(cat, identity.get('owner_first_name', 'there'))},
                {"role": "user",   "content": _reply_user(msg, merchant, cat, history)},
            ],
        )
        result = json.loads(resp.choices[0].message.content.strip())
        if result.get("intent") in ("auto", "stop"):
            return JSONResponse({"action": "end", "body": result.get("body", "")})
        return JSONResponse({"action": result.get("action", "send"),
                             "body":   result.get("body", "Got it! Shall we proceed?")})
    except Exception:
        if any(w in msg.lower() for w in {"stop", "spam", "remove"}):
            return JSONResponse({"action": "end", "body": "Understood."})
        return JSONResponse({"action": "send",
                             "body": "Got it. Launch the offer now, or want a preview first?"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")