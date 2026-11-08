import html
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(
    page_title="NETFLIX | AI Discovery Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Design System CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: #070707;
        background-image: radial-gradient(circle at 50% 0%, #150508 0%, #070707 65%);
        color: #ffffff;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    @keyframes fadeInApp {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stApp > div {
        animation: fadeInApp 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* --- TOP NAVIGATION BAR --- */
    .netflix-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 0 1.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 2rem;
    }
    .nav-wordmark {
        display: inline-flex;
        align-items: flex-start;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.2rem;
        line-height: 0.85;
        user-select: none;
        filter: drop-shadow(0 4px 16px rgba(229, 9, 20, 0.4));
    }
    .nav-n-svg {
        display: inline-flex;
        height: 2.6rem;
        width: 1.5rem;
        margin-right: 3px;
        transform-origin: top center;
        transform: scaleY(1.12) skewY(-3.5deg);
    }
    .nav-n-svg svg {
        width: 100%;
        height: 100%;
        overflow: visible;
        shape-rendering: geometricPrecision;
    }
    .nav-arch-char {
        display: inline-block;
        color: #E50914;
        letter-spacing: 2.5px;
        transform-origin: top center;
        text-shadow: -2px 0 0 #b20710;
    }
    .nav-e { transform: scaleY(1.06) skewY(-2deg); }
    .nav-t { transform: scaleY(1.00) skewY(-0.8deg); }
    .nav-f { transform: scaleY(0.98) skewY(0.5deg); }
    .nav-l { transform: scaleY(1.01) skewY(1.5deg); }
    .nav-i { transform: scaleY(1.06) skewY(2.5deg); }
    .nav-x { transform: scaleY(1.14) skewY(3.5deg); }
    .netflix-sub {
        color: #888888;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* --- SEARCH & CONTROLS UI --- */
    .stSelectbox label {
        color: #d4d4d4 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #121212 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #E50914 !important;
        box-shadow: 0 0 12px rgba(229, 9, 20, 0.2);
    }

    /* --- HERO SPOTLIGHT CONTAINER --- */
    .hero-box {
        background: linear-gradient(135deg, rgba(20, 20, 20, 0.62) 0%, rgba(8, 8, 8, 0.80) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 5px solid #E50914;
        border-radius: 14px;
        padding: 32px;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0 24px 50px rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.6rem;
        font-weight: 400;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        color: #ffffff;
        line-height: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }
    .hero-plot {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #d1d1d1;
        margin: 20px 0;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 16px;
    }
    .badge {
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.08);
        color: #e5e5e5;
        border: 1px solid rgba(255, 255, 255, 0.12);
        letter-spacing: 0.5px;
    }
    .badge-red {
        background: #E50914;
        color: #ffffff;
        border: none;
        box-shadow: 0 2px 10px rgba(229, 9, 20, 0.4);
    }
    .badge-hd {
        background: transparent;
        border: 1.5px solid #777777;
        color: #cccccc;
        font-size: 0.72rem;
        padding: 3px 8px;
    }

    .genre-pill {
        display: inline-block;
        background: rgba(229, 9, 20, 0.1);
        color: #ff7b82;
        border: 1px solid rgba(229, 9, 20, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .genre-pill:hover {
        background: rgba(229, 9, 20, 0.25);
        color: #ffffff;
    }

    /* --- HERO BACKDROP --- */
    [data-testid="stMainBlockContainer"], .block-container { position: relative; z-index: 1; }
    .hero-backdrop {
        position: fixed;
        inset: 0;
        overflow: hidden;
        z-index: -1;
        pointer-events: none;
    }
    .hero-backdrop-img {
        position: absolute;
        inset: -90px;
        background-size: cover;
        background-position: center 25%;
        filter: blur(64px) saturate(1.7) brightness(0.8);
        opacity: 0.5;
        transform: scale(1.15);
    }
    .hero-backdrop::after {
        content: '';
        position: absolute; inset: 0;
        background:
            linear-gradient(to bottom, rgba(7,7,7,0) 0%, rgba(7,7,7,0.35) 55%, rgba(7,7,7,0.85) 100%),
            radial-gradient(ellipse 90% 80% at 50% 25%, rgba(7,7,7,0) 0%, rgba(7,7,7,0.7) 100%);
    }
    .hero-poster {
        position: relative;
        height: 440px;
        max-width: 300px;
        margin: 10px auto 0 auto;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.9), 0 0 30px rgba(229, 9, 20, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.15);
        background-color: #121212;
    }
    .hero-poster img {
        position: absolute; inset: 0;
        width: 100%; height: 100%;
        object-fit: cover; object-position: center top;
        color: transparent;
    }
    .hero-poster .poster-fallback { font-size: 2.6rem; }

    /* --- POSTER FALLBACK (used when no poster is found) --- */
    .poster-fallback {
        position: absolute; inset: 0;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        padding: 14px; text-align: center;
        background: linear-gradient(160deg, #2a0508 0%, #120305 55%, #0a0a0a 100%);
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 1.5px; color: #ffffff;
        font-size: 1.5rem; line-height: 1.05;
    }
    .poster-fallback::before {
        content: ''; width: 34px; height: 4px;
        background: #E50914; border-radius: 2px; margin-bottom: 12px;
    }

    /* --- RECOMMENDATION CARDS (poster + hover reveal) --- */
    .rec-card {
        position: relative;
        background: #111111;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.07);
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.35s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.35s ease;
        margin-bottom: 12px;
    }
    .rec-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 38px rgba(229, 9, 20, 0.28);
        border-color: #E50914;
    }
    .rec-poster-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 2 / 3;
        overflow: hidden;
        background: #1a1a1a;
    }
    .rec-poster-wrap img {
        position: absolute; inset: 0;
        width: 100%; height: 100%;
        object-fit: cover;
        color: transparent;
    }
    .rec-pill {
        position: absolute; top: 10px; right: 10px; z-index: 3;
        background: rgba(0, 0, 0, 0.72);
        backdrop-filter: blur(6px);
        color: #46d369; font-weight: 700; font-size: 0.74rem;
        padding: 3px 8px; border-radius: 6px;
        border: 1px solid rgba(70, 211, 105, 0.35);
    }
    .rec-overlay {
        position: absolute; left: 0; right: 0; bottom: 0; z-index: 2;
        padding: 46px 12px 12px 12px;
        background: linear-gradient(to top, rgba(0,0,0,0.97) 0%, rgba(0,0,0,0.88) 62%, rgba(0,0,0,0) 100%);
        transform: translateY(100%);
        opacity: 0;
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease;
    }
    .rec-card:hover .rec-overlay { transform: translateY(0); opacity: 1; }
    .rec-why-label {
        font-size: 0.66rem; letter-spacing: 1px; text-transform: uppercase;
        color: #a0a0a0; font-weight: 700; margin-bottom: 6px;
    }
    .why-chip {
        display: inline-block;
        font-size: 0.7rem; font-weight: 600;
        padding: 3px 8px; border-radius: 12px;
        margin: 0 4px 4px 0;
        background: rgba(229, 9, 20, 0.14);
        border: 1px solid rgba(229, 9, 20, 0.35);
        color: #ff9a9f;
    }
    .why-chip.person { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.18); color: #e5e5e5; }
    .why-chip.theme { background: rgba(70,211,105,0.10); border-color: rgba(70,211,105,0.30); color: #7ee29a; }
    .rec-snippet {
        font-size: 0.74rem; color: #bdbdbd; line-height: 1.4; margin-top: 4px;
        display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    }
    .rec-body {
        padding: 12px 14px;
        background: linear-gradient(180deg, #121212 0%, #0a0a0a 100%);
    }
    .rec-title {
        font-size: 0.96rem; font-weight: 700; color: #ffffff;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        margin-bottom: 3px;
    }
    .rec-meta {
        color: #a0a0a0; font-size: 0.78rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* --- LOADING SKELETONS --- */
    @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
    .skel {
        background: linear-gradient(90deg, #141414 25%, #232323 50%, #141414 75%);
        background-size: 800px 100%;
        animation: shimmer 1.4s infinite linear;
    }
    .skel-card { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.06); background: #111; margin-bottom: 12px; }
    .skel-poster { aspect-ratio: 2 / 3; }
    .skel-line { height: 12px; border-radius: 6px; margin: 12px 14px 0 14px; }
    .skel-line.short { width: 55%; margin-bottom: 14px; }
    @media (prefers-reduced-motion: reduce) { .skel { animation: none; } }

    /* --- ACTION BUTTONS --- */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(229, 9, 20, 0.08) 100%);
        color: #ffffff;
        border: 1px solid #E50914;
        border-radius: 10px;
        padding: 0.65rem 1.8rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.15);
    }
    div.stButton > button:hover {
        background: #E50914;
        color: #ffffff;
        box-shadow: 0 0 25px rgba(229, 9, 20, 0.6);
        transform: translateY(-3px);
        border-color: #ff1f2a;
    }
    </style>
""", unsafe_allow_html=True)

# 1B. Intro splash animation (plays once per session)
if "intro_played" not in st.session_state:
    st.session_state.intro_played = True
    st.markdown("""
        <style>
        @keyframes wrapper-cinema-zoom {
            0% { opacity: 0; transform: scale(0.75); }
            18% { opacity: 1; transform: scale(1); }
            60% { opacity: 1; transform: scale(1.08); }
            88% { opacity: 1; transform: scale(22); }
            100% { opacity: 0; transform: scale(65); }
        }
        @keyframes n-straighten {
            0%, 28% { transform: scaleY(1.12) skewY(-3.5deg); }
            54%, 100% { transform: scaleY(1.0) skewY(0deg); }
        }
        @keyframes collapse-box-width {
            0%, 26% { max-width: 560px; opacity: 1; }
            52% { max-width: 0px; opacity: 1; }
            54%, 100% { max-width: 0px; opacity: 0; }
        }
        @keyframes slide-into-n {
            0%, 26% { transform: translateX(0%); }
            52%, 100% { transform: translateX(-115%); }
        }
        @keyframes overlay-fadeout {
            0%, 78% { opacity: 1; visibility: visible; }
            100% { opacity: 0; visibility: hidden; }
        }
        .netflix-splash-overlay {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            background: radial-gradient(circle at center, #190003 0%, #000000 72%);
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            pointer-events: none;
            animation: overlay-fadeout 3.8s cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        .netflix-wordmark {
            display: inline-flex;
            align-items: flex-start;
            justify-content: center;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 9.5rem;
            line-height: 0.85;
            user-select: none;
            transform-origin: center center;
            animation: wrapper-cinema-zoom 3.7s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
        .netflix-n-svg-wrap {
            position: relative;
            z-index: 5;
            display: inline-flex;
            align-items: flex-start;
            height: 7.75rem;
            width: 4.5rem;
            margin-right: 4px;
            transform-origin: top center;
            animation: n-straighten 3.7s cubic-bezier(0.65, 0, 0.35, 1) forwards;
        }
        .netflix-n-svg-wrap svg {
            width: 100%;
            height: 100%;
            overflow: visible;
            shape-rendering: geometricPrecision;
        }
        .netflix-rest-mask {
            position: relative;
            z-index: 2;
            display: inline-block;
            max-width: 560px;
            overflow: hidden;
            clip-path: inset(-30px 0px -40px 0px);
            white-space: nowrap;
            animation: collapse-box-width 3.7s cubic-bezier(0.65, 0, 0.35, 1) forwards;
        }
        .netflix-rest-letters {
            display: inline-flex;
            align-items: flex-start;
            padding-left: 2px;
            animation: slide-into-n 3.7s cubic-bezier(0.65, 0, 0.35, 1) forwards;
        }
        .arch-char {
            display: inline-block;
            color: #E50914;
            letter-spacing: 5px;
            transform-origin: top center;
            text-shadow: -3px 0 0 #b20710, 0 10px 25px rgba(229, 9, 20, 0.45);
        }
        .arch-e { transform: scaleY(1.06) skewY(-2deg); }
        .arch-t { transform: scaleY(1.00) skewY(-0.8deg); }
        .arch-f { transform: scaleY(0.98) skewY(0.5deg); }
        .arch-l { transform: scaleY(1.01) skewY(1.5deg); }
        .arch-i { transform: scaleY(1.06) skewY(2.5deg); }
        .arch-x { transform: scaleY(1.14) skewY(3.5deg); }
        </style>
        <div class="netflix-splash-overlay">
            <div class="netflix-wordmark">
                <span class="netflix-n-svg-wrap">
                    <svg viewBox="0 0 100 180" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="n-leg-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="#b80710" />
                                <stop offset="100%" stop-color="#7a0309" />
                            </linearGradient>
                            <linearGradient id="n-ribbon-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#ff1f2a" />
                                <stop offset="50%" stop-color="#E50914" />
                                <stop offset="100%" stop-color="#c20711" />
                            </linearGradient>
                        </defs>
                        <rect x="0" y="0" width="29" height="180" fill="url(#n-leg-grad)" />
                        <rect x="71" y="0" width="29" height="180" fill="url(#n-leg-grad)" />
                        <polygon points="0,0 29,0 100,180 71,180" fill="url(#n-ribbon-grad)" />
                    </svg>
                </span><span class="netflix-rest-mask"><span class="netflix-rest-letters"><span class="arch-char arch-e">E</span><span class="arch-char arch-t">T</span><span class="arch-char arch-f">F</span><span class="arch-char arch-l">L</span><span class="arch-char arch-i">I</span><span class="arch-char arch-x">X</span></span></span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# 3. Load saved artifacts (movies table + TF-IDF feature matrix)
@st.cache_resource
def load_data():
    with open('artifacts/movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('artifacts/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return movies.reset_index(drop=True), tfidf_matrix


movies, tfidf_matrix = load_data()


def esc(value):
    """HTML-escape any value before embedding it in unsafe_allow_html markup."""
    return html.escape(str(value), quote=True)


def set_hero(i):
    """Callback: make a recommendation the new hero title."""
    st.session_state.selected_idx = i
    st.session_state.scroll_top = True


def poster_html(poster, title, lazy=True):
    """Designed fallback underneath; the image (if any) sits on top and hides itself if it fails."""
    fallback = f'<div class="poster-fallback"><span>{esc(title)}</span></div>'
    if not poster:
        return fallback
    loading = 'loading="lazy"' if lazy else ''
    return fallback + f'<img src="{esc(poster)}" alt="{esc(title)}" {loading} />'


SKELETON_CARD = (
    '<div class="skel-card"><div class="skel skel-poster"></div>'
    '<div class="skel skel-line"></div><div class="skel skel-line short"></div></div>'
)

EXTRA_STOP = {"series", "show", "shows", "movie", "film", "story", "life", "world",
              "finds", "find", "takes", "must", "becomes", "help", "helps", "years", "their", "while"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP


def _norm_genre(g):
    g = g.strip().lower()
    g = re.sub(r'^tv\s+', '', g)
    g = re.sub(r'\s+(movies|tv shows)$', '', g)
    return g


def _split(text):
    return [p.strip() for p in str(text).split(',') if p.strip()]


def _words(text):
    return {w for w in re.findall(r"[a-z]{5,}", str(text).lower()) if w not in STOP_WORDS}


def explain_match(a, b):
    """What do titles a (selected) and b (recommended) actually have in common?"""
    ra, rb = movies.iloc[a], movies.iloc[b]

    ga = {_norm_genre(g) for g in _split(ra['listed_in'])}
    genres = [_norm_genre(g).title() for g in _split(rb['listed_in']) if _norm_genre(g) in ga]

    da = {d.lower() for d in _split(ra['director'])}
    directors = [d for d in _split(rb['director']) if d.lower() in da]

    ca = {c.lower() for c in _split(ra['cast'])}
    cast = [c for c in _split(rb['cast']) if c.lower() in ca][:2]

    shared = _words(ra['description']) & _words(rb['description'])
    themes = [w.capitalize() for w in sorted(shared, key=lambda w: (-len(w), w))[:2]]

    return {"genres": genres, "directors": directors, "cast": cast, "themes": themes}


def why_chips_html(reasons, limit=5):
    chips = []
    chips += [f'<span class="why-chip">{esc(g)}</span>' for g in reasons["genres"][:2]]
    chips += [f'<span class="why-chip person">🎬 {esc(d)}</span>' for d in reasons["directors"][:1]]
    chips += [f'<span class="why-chip person">🌟 {esc(c)}</span>' for c in reasons["cast"]]
    chips += [f'<span class="why-chip theme">💬 {esc(t)}</span>' for t in reasons["themes"]]
    if not chips:
        chips = ['<span class="why-chip theme">Similar plot &amp; tone</span>']
    return "".join(chips[:limit])


# 4. Multi-source poster fetcher
@st.cache_data(show_spinner=False)
def fetch_poster(title, media_type="Movie", release_year=None, cache_v=4):
    clean_title = title.strip()
    headers = {"User-Agent": "Mozilla/5.0"}

    if media_type == "TV Show":
        try:
            url = f"https://api.tvmaze.com/singlesearch/shows?q={quote(clean_title)}"
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code == 200:
                data = res.json()
                if data and data.get("image"):
                    poster = data["image"].get("original") or data["image"].get("medium")
                    if poster:
                        return poster
        except Exception:
            pass

    queries = [f"{clean_title} {release_year}", clean_title] if release_year else [clean_title]
    for query_text in queries:
        try:
            alnum = "".join(c for c in clean_title.lower() if c.isalnum())
            first_char = alnum[0] if alnum else "a"
            imdb_url = f"https://v2.sg.media-imdb.com/suggestion/{first_char}/{quote(query_text)}.json"
            res = requests.get(imdb_url, headers=headers, timeout=3)
            if res.status_code == 200:
                items = res.json().get("d", [])
                for item in items:
                    item_id = str(item.get("id", ""))
                    qid = str(item.get("qid", "")).lower()
                    if item_id.startswith("tt") and qid in ["movie", "feature", "tvseries", "tvminiseries", "tvmovie", "tvspecial", ""]:
                        if "i" in item and "imageUrl" in item["i"]:
                            img_url = item["i"]["imageUrl"]
                            if "._V1_" in img_url:
                                img_url = img_url.split("._V1_")[0] + "._V1_UX800_.jpg"
                            return img_url
        except Exception:
            pass

    try:
        url = f"https://api.tvmaze.com/singlesearch/shows?q={quote(clean_title)}"
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if data and data.get("image"):
                poster = data["image"].get("original") or data["image"].get("medium")
                if poster:
                    return poster
    except Exception:
        pass

    return None  # the designed CSS fallback is drawn instead


# 5. Recommendation logic (similarity computed on demand from the TF-IDF matrix)
@st.cache_data(show_spinner=False)
def get_ranked_indices(idx):
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    order = sim.argsort()[::-1]
    order = order[order != idx]
    return order, sim[order]


def get_recommendations(idx, filter_mode="All Titles", top_n=5):
    order, scores = get_ranked_indices(idx)

    picked = []
    for movie_idx, score in zip(order, scores):
        row_type = movies.at[movie_idx, 'type']
        if filter_mode == "Movies Only" and row_type != "Movie":
            continue
        if filter_mode == "TV Shows Only" and row_type != "TV Show":
            continue
        picked.append((int(movie_idx), float(score)))
        if len(picked) == top_n:
            break

    # Fetch posters in parallel (cached after first fetch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        posters = list(pool.map(
            lambda p: fetch_poster(
                movies.at[p[0], 'title'],
                movies.at[p[0], 'type'],
                movies.at[p[0], 'release_year'],
            ),
            picked,
        ))

    results = []
    for (movie_idx, score), poster in zip(picked, posters):
        row = movies.iloc[movie_idx]
        results.append({
            "idx": movie_idx,
            "title": row['title'],
            "type": row['type'],
            "year": row['release_year'],
            "rating": row['rating'] if row['rating'] else "NR",
            "duration": row['duration'],
            "genres": row['listed_in'],
            "description": row['description'],
            "match": round(score * 100, 1),
            "poster": poster,
            "reasons": explain_match(idx, movie_idx),
        })
    return results


# 6. Top navigation header
st.markdown(f"""
    <div class="netflix-nav">
        <div style="display: flex; align-items: center;">
            <div class="nav-wordmark">
                <span class="nav-n-svg">
                    <svg viewBox="0 0 100 180" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="nav-leg-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="#b80710" />
                                <stop offset="100%" stop-color="#7a0309" />
                            </linearGradient>
                            <linearGradient id="nav-ribbon-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#ff1f2a" />
                                <stop offset="50%" stop-color="#E50914" />
                                <stop offset="100%" stop-color="#c20711" />
                            </linearGradient>
                        </defs>
                        <rect x="0" y="0" width="29" height="180" fill="url(#nav-leg-grad)" />
                        <rect x="71" y="0" width="29" height="180" fill="url(#nav-leg-grad)" />
                        <polygon points="0,0 29,0 100,180 71,180" fill="url(#nav-ribbon-grad)" />
                    </svg>
                </span><span class="nav-arch-char nav-e">E</span><span class="nav-arch-char nav-t">T</span><span class="nav-arch-char nav-f">F</span><span class="nav-arch-char nav-l">L</span><span class="nav-arch-char nav-i">I</span><span class="nav-arch-char nav-x">X</span>
            </div>
            <span style="margin-left: 18px; color: #777; font-size: 0.95rem; margin-top: 8px;">AI Discovery Engine</span>
        </div>
        <div class="netflix-sub">{len(movies):,} Titles Indexed</div>
    </div>
""", unsafe_allow_html=True)

# 7. Search bar + inline type filter
search_col, filter_col = st.columns([3, 1.2], gap="medium")

stranger = movies.index[movies['title'] == 'Stranger Things']
default_idx = int(stranger[0]) if len(stranger) else 0

# Options are row indices, so duplicate titles stay distinct (labelled with year)
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = default_idx

with search_col:
    selected_idx = st.selectbox(
        "🔍 Search or select a Movie or TV Show:",
        options=list(range(len(movies))),
        key="selected_idx",
        format_func=lambda i: f"{movies.at[i, 'title']} ({movies.at[i, 'release_year']})",
    )

with filter_col:
    filter_type = st.selectbox(
        "🎞️ Filter Recommendations:",
        ["All Titles", "Movies Only", "TV Shows Only"],
        index=0
    )

if "rec_limit" not in st.session_state:
    st.session_state.rec_limit = 5
if "last_idx" not in st.session_state or st.session_state.last_idx != selected_idx:
    st.session_state.rec_limit = 5
    st.session_state.last_idx = selected_idx

# 8. Hero spotlight
selected_row = movies.iloc[selected_idx]
selected_title = selected_row['title']
hero_poster = fetch_poster(selected_title, selected_row['type'], selected_row['release_year'])

genre_pills_html = "".join(
    f'<span class="genre-pill">{esc(g.strip())}</span>'
    for g in str(selected_row['listed_in']).split(",")
)
director_text = esc(selected_row['director']) if selected_row['director'] else "Not Listed"
cast_text = esc(selected_row['cast']) if selected_row['cast'] else "Not Listed"
hero_rating = esc(selected_row['rating']) if selected_row['rating'] else "NR"

hero_col1, hero_col2 = st.columns([1, 2.8], gap="large")

if hero_poster:
    st.markdown(
        f"""<div class="hero-backdrop"><div class="hero-backdrop-img" style="background-image:url('{esc(hero_poster)}')"></div></div>""",
        unsafe_allow_html=True
    )

with hero_col1:
    st.markdown(
        f"""<div class="hero-poster">{poster_html(hero_poster, selected_title, lazy=False)}</div>""",
        unsafe_allow_html=True
    )

with hero_col2:
    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-title">{esc(selected_title)}</div>
            <div class="badge-row">
                <span class="badge badge-red">{esc(selected_row['type'].upper())}</span>
                <span class="badge">{esc(selected_row['release_year'])}</span>
                <span class="badge">{hero_rating}</span>
                <span class="badge">{esc(selected_row['duration'])}</span>
                <span class="badge badge-hd">HD</span>
            </div>
            <div style="margin-bottom: 16px;">{genre_pills_html}</div>
            <div class="hero-plot">{esc(selected_row['description'])}</div>
            <div style="font-size: 0.9rem; color: #b3b3b3; line-height: 1.8;">
                <div><strong style="color:#fff;">🎬 Director:</strong> {director_text}</div>
                <div><strong style="color:#fff;">🌟 Cast:</strong> {cast_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# 9. Recommendations grid
st.markdown(
    f"### 🔥 More Like *{selected_title}* "
    f"<span style='font-size:0.9rem; color:#888; font-weight:400;'>(Showing Top {st.session_state.rec_limit})</span>",
    unsafe_allow_html=True
)

placeholder = st.empty()
with placeholder.container():
    for skel_col in st.columns(5, gap="medium"):
        skel_col.markdown(SKELETON_CARD, unsafe_allow_html=True)

recommendations = get_recommendations(
    selected_idx,
    filter_mode=filter_type,
    top_n=st.session_state.rec_limit
)
placeholder.empty()

for row_start in range(0, len(recommendations), 5):
    row_items = recommendations[row_start:row_start + 5]
    cols = st.columns(5, gap="medium")
    for col, rec in zip(cols, row_items):
        with col:
            st.markdown(
                f"""
                <div class="rec-card" title="{esc(rec['title'])}">
                    <div class="rec-poster-wrap">
                        {poster_html(rec['poster'], rec['title'])}
                        <div class="rec-pill">{rec['match']}% Match</div>
                        <div class="rec-overlay">
                            <div class="rec-why-label">Why this match</div>
                            <div>{why_chips_html(rec['reasons'])}</div>
                            <div class="rec-snippet">{esc(rec['description'])}</div>
                        </div>
                    </div>
                    <div class="rec-body">
                        <div class="rec-title">{esc(rec['title'])}</div>
                        <div class="rec-meta">{esc(rec['year'])} • {esc(rec['type'])} • {esc(rec['rating'])} • {esc(rec['duration'])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            with st.popover("📖 Plot & Details", use_container_width=True):
                st.markdown(f"**{rec['title']}** ({rec['year']})")
                st.caption(f"{rec['type']} • {rec['duration']} • {rec['genres']}")
                st.write(rec["description"])
                r = rec["reasons"]
                st.markdown("**Why this match**")
                if r["genres"]:
                    st.write("Shared genres: " + ", ".join(r["genres"]))
                if r["directors"]:
                    st.write("Same director: " + ", ".join(r["directors"]))
                if r["cast"]:
                    st.write("Shared cast: " + ", ".join(r["cast"]))
                if r["themes"]:
                    st.write("Shared themes: " + ", ".join(r["themes"]))
                if not any(r.values()):
                    st.write("Similar plot and tone based on the descriptions.")
            st.button(
                "▶ View",
                key=f"view_{rec['idx']}_{row_start}",
                use_container_width=True,
                on_click=set_hero,
                args=(rec['idx'],),
            )

# 10. "See More" / "Show Less"
st.write("")
btn_col1, btn_col2, btn_col3 = st.columns([2, 1.5, 2])

with btn_col2:
    if st.session_state.rec_limit < 20:
        if st.button("See More ❯❯", use_container_width=True):
            st.session_state.rec_limit += 5
            st.rerun()
    if st.session_state.rec_limit > 5:
        if st.button("Show Less ❮❮", use_container_width=True):
            st.session_state.rec_limit = 5
            st.rerun()

# 11. Scroll back to the hero after swapping titles
import streamlit.components.v1 as components

if st.session_state.get("scroll_top"):
    components.html(
        """<script>
        const d = window.parent.document;
        (d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main')).scrollTo({top: 0, behavior: 'smooth'});
        </script>""",
        height=0,
    )
    st.session_state.scroll_top = False