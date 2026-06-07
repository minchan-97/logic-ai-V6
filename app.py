"""
matrix_guardrail_app.py — GasCode MatrixGuardrail + LLM 통합 Streamlit 앱
=========================================================================

GasCode 핵심 명제 검증:
  "행렬화된 LLM은 환각 탐지가 빠르고 쉽다"

작동 방식:
  1. 도메인 코퍼스 업로드 → skip-gram 임베딩 + 마르코프 학습
  2. 질문 입력 → LLM 호출 (OpenAI-compatible)
  3. MatrixGuardrail 실시간 판정 (0.04ms/건)
  4. 판정 결과 + 위치별 진단 표시
"""
import streamlit as st
import numpy as np
from collections import defaultdict, Counter
import time
import json
import requests
import io


# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="MatrixGuardrail",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Pretendard:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}
.metric-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
}
.status-pass    { color: #34d399; }
.status-warning { color: #fbbf24; }
.status-critical{ color: #fb923c; }
.status-fatal   { color: #f87171; }
.token-ok  { background: #064e3b; color: #6ee7b7; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; font-family: monospace; font-size: 0.8em; }
.token-bad { background: #7f1d1d; color: #fca5a5; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; font-family: monospace; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)


# ── MatrixGuardrailEngine ──────────────────────────────────
class MatrixGuardrailEngine:
    def __init__(self, lambda_1=0.6, lambda_2=0.3, lambda_3=0.1, alpha=0.01):
        self.l1 = lambda_1; self.l2 = lambda_2; self.l3 = lambda_3; self.alpha = alpha
        self.unigram_counts = Counter()
        self.bigram_counts  = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.total_tokens = 0
        self.word2idx = {}; self.vocab_size = 0; self.word_embeddings = None
        self.is_trained = False
        self.corpus_name = ""

    def train(self, corpus_text, embedding_dim=32, epochs=30, window=2, lr=0.05):
        tokens = corpus_text.strip().split()
        self.total_tokens = len(tokens)

        for i, t in enumerate(tokens):
            self.unigram_counts[t] += 1
            if i >= 1: self.bigram_counts[tokens[i-1]][t] += 1
            if i >= 2: self.trigram_counts[(tokens[i-2], tokens[i-1])][t] += 1

        words = [w for w, c in Counter(tokens).items() if c >= 1]
        self.word2idx   = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(words)

        rng   = np.random.default_rng(42)
        W_in  = (rng.random((self.vocab_size, embedding_dim)) - 0.5) / embedding_dim
        W_out = (rng.random((self.vocab_size, embedding_dim)) - 0.5) / embedding_dim

        pairs = []
        for i, w in enumerate(tokens):
            if w not in self.word2idx: continue
            c = self.word2idx[w]
            for j in range(max(0, i-window), min(len(tokens), i+window+1)):
                if j == i or tokens[j] not in self.word2idx: continue
                pairs.append((c, self.word2idx[tokens[j]]))

        def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        for _ in range(epochs):
            rng.shuffle(pairs)
            for center, ctx in pairs:
                negs  = rng.integers(0, self.vocab_size, size=5)
                v_c   = W_in[center]; v_pos = W_out[ctx]
                p_pos = sig(np.dot(v_c, v_pos))
                grad_c = (p_pos - 1.0) * v_pos
                W_out[ctx] -= lr * (p_pos - 1.0) * v_c
                for ng in negs:
                    v_neg  = W_out[ng]; p_neg = sig(np.dot(v_c, v_neg))
                    grad_c += p_neg * v_neg; W_out[ng] -= lr * p_neg * v_c
                W_in[center] -= lr * grad_c

        norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-12
        self.word_embeddings = W_in / norms
        self.is_trained = True

    def get_vec(self, word):
        if word in self.word2idx: return self.word_embeddings[self.word2idx[word]]
        return None

    def score_jm(self, tokens):
        total_lp = 0.0; K = len(tokens)
        if K < 2: return 0.0, []
        per = []
        for t in range(2, K):
            wc = tokens[t]; wp = tokens[t-1]; wpp = tokens[t-2]
            p1 = (self.unigram_counts[wc] + self.alpha) / (self.total_tokens + self.alpha * self.vocab_size)
            cp = self.unigram_counts[wp]
            p2 = (self.bigram_counts[wp][wc] / cp) if cp > 0 else 0.0
            cpp = self.trigram_counts[(wpp, wp)][wc]
            p3 = (cpp / self.bigram_counts[wpp][wp]) if self.bigram_counts[wpp][wp] > 0 else 0.0
            p_jm = self.l1 * p3 + self.l2 * p2 + self.l3 * p1
            lp = float(np.log(p_jm + 1e-12))
            total_lp += lp
            in_graph = (p2 > 0 or p3 > 0)
            per.append({"token": wc, "logp": lp, "in_graph": in_graph})
        return total_lp / (K - 1), per

    def score_mismatch(self, tokens):
        vecs = [self.get_vec(w) for w in tokens if self.get_vec(w) is not None]
        if len(vecs) < 2: return 0.5
        sims = [float(np.dot(vecs[i], vecs[i+1])) for i in range(len(vecs)-1)]
        return float((1.0 - np.mean(sims)) / 2.0)

    def evaluate(self, text, logp_thr=-4.5, mis_thr=0.40):
        t0 = time.perf_counter()
        tokens = text.strip().split()
        avg_logp, per = self.score_jm(tokens)
        mismatch      = self.score_mismatch(tokens)
        elapsed_ms    = (time.perf_counter() - t0) * 1000

        mf = avg_logp < logp_thr; ef = mismatch > mis_thr
        if   not mf and not ef: status = "PASS"
        elif mf  and not ef:    status = "WARNING"
        elif not mf and ef:     status = "CRITICAL"
        else:                   status = "FATAL"

        return {
            "status": status, "avg_logp": avg_logp,
            "mismatch": mismatch, "elapsed_ms": elapsed_ms,
            "per_token": per, "tokens": tokens,
        }


# ── LLM 호출 ──────────────────────────────────────────────
def call_llm(prompt, system_prompt, api_url, api_key, model, temperature, max_tokens):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data


# ── 세션 초기화 ────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = MatrixGuardrailEngine()
if "history" not in st.session_state:
    st.session_state.history = []


# ── UI ────────────────────────────────────────────────────
st.markdown("## 🔬 MatrixGuardrail")
st.caption("GasCode 핵심 명제 검증 — 행렬화 LLM은 환각 탐지가 빠르고 쉽다")

sidebar, main = st.columns([1, 2], gap="large")

with sidebar:
    st.markdown("### ⚙️ 설정")

    # API 설정
    with st.expander("API 설정", expanded=True):
        api_url  = st.text_input("Chat API URL", value="https://api.openai.com/v1/chat/completions")
        api_key  = st.text_input("API Key", type="password")
        model    = st.text_input("Model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens  = st.slider("Max tokens", 128, 2048, 512, 64)

    # 가드레일 설정
    with st.expander("🔬 MatrixGuardrail 설정", expanded=True):
        uploaded = st.file_uploader(
            "도메인 코퍼스 (.txt / .pdf / .docx)",
            type=["txt", "pdf", "docx"],
        )

        emb_dim   = st.slider("임베딩 차원", 16, 128, 32, 16)
        epochs    = st.slider("학습 epochs", 10, 60, 30, 10)
        logp_thr  = st.slider("logP 임계값", -10.0, 0.0, -4.5, 0.5)
        mis_thr   = st.slider("mismatch 임계값", 0.1, 0.8, 0.4, 0.05)
        l1        = st.slider("λ1 (trigram)", 0.0, 1.0, 0.6, 0.1)
        l2        = st.slider("λ2 (bigram)",  0.0, 1.0, 0.3, 0.1)
        l3        = st.slider("λ3 (unigram)", 0.0, 1.0, 0.1, 0.1)

        engine = st.session_state.engine
        engine.l1 = l1; engine.l2 = l2; engine.l3 = l3

        if uploaded and st.button("📚 가드레일 학습", use_container_width=True):
            with st.spinner("skip-gram 학습 중..."):
                try:
                    name = uploaded.name.lower()
                    if name.endswith(".txt"):
                        text = uploaded.read().decode("utf-8", errors="ignore")
                    elif name.endswith(".pdf"):
                        import pypdf
                        reader = pypdf.PdfReader(uploaded)
                        text = "\n".join(p.extract_text() or "" for p in reader.pages)
                    elif name.endswith(".docx"):
                        import docx as docx_lib
                        doc = docx_lib.Document(io.BytesIO(uploaded.read()))
                        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                    else:
                        text = ""

                    if text.strip():
                        t0 = time.perf_counter()
                        engine.train(text, embedding_dim=emb_dim, epochs=epochs)
                        engine.corpus_name = uploaded.name
                        elapsed = (time.perf_counter() - t0) * 1000
                        st.success(
                            f"학습 완료\n"
                            f"어휘 {engine.vocab_size}개 · 토큰 {engine.total_tokens}개\n"
                            f"({elapsed:.0f}ms)"
                        )
                    else:
                        st.error("텍스트를 추출하지 못했습니다.")
                except Exception as e:
                    st.error(f"학습 실패: {e}")

        if engine.is_trained:
            st.success(f"✓ 학습됨: {engine.corpus_name}")

    # 시스템 프롬프트
    with st.expander("시스템 프롬프트"):
        system_prompt = st.text_area(
            "System Prompt",
            value="너는 정확하고 신뢰할 수 있는 답변을 제공하는 AI 어시스턴트다. 불확실하면 단정하지 말고 보수적으로 답하라.",
            height=100,
        )


with main:
    st.markdown("### 💬 질문")
    prompt = st.text_area("질문을 입력하세요", height=80, placeholder="예: 주민등록등본은 어디서 발급받나요?")

    run = st.button("🚀 실행", type="primary", use_container_width=True)

    if run and prompt:
        if not api_key:
            st.error("API Key를 입력해주세요.")
        else:
            with st.spinner("LLM 호출 중..."):
                try:
                    # 코퍼스 주입
                    final_system = system_prompt
                    if engine.is_trained and hasattr(engine, 'corpus_name'):
                        pass  # 코퍼스 텍스트 직접 저장은 별도 구현

                    answer, raw = call_llm(
                        prompt, final_system,
                        api_url, api_key, model, temperature, max_tokens,
                    )

                    # MatrixGuardrail 판정
                    if engine.is_trained:
                        result = engine.evaluate(answer, logp_thr=logp_thr, mis_thr=mis_thr)
                    else:
                        result = None

                    # 히스토리 저장
                    st.session_state.history.append({
                        "prompt": prompt, "answer": answer,
                        "result": result, "raw": raw,
                    })

                except Exception as e:
                    st.error(f"오류: {e}")

    # 결과 표시
    if st.session_state.history:
        latest = st.session_state.history[-1]
        answer = latest["answer"]
        result = latest["result"]

        st.markdown("---")
        st.markdown("#### 📝 답변")
        st.write(answer)

        if result:
            st.markdown("---")
            st.markdown("#### 🔬 MatrixGuardrail 판정")

            status = result["status"]
            color_map = {
                "PASS": "status-pass", "WARNING": "status-warning",
                "CRITICAL": "status-critical", "FATAL": "status-fatal",
            }
            icon_map = {
                "PASS": "🟢", "WARNING": "🟡", "CRITICAL": "🟠", "FATAL": "🔴",
            }

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("판정", f"{icon_map[status]} {status}")
            with c2:
                st.metric("avg logP", f"{result['avg_logp']:+.3f}")
            with c3:
                st.metric("mismatch", f"{result['mismatch']:.3f}")
            with c4:
                st.metric("처리 시간", f"{result['elapsed_ms']:.2f}ms")

            # 판정 설명
            desc_map = {
                "PASS":     "두 신호 모두 도메인 안. 자연스러운 응답.",
                "WARNING":  "마르코프 이탈 감지. 도메인 밖 표현 등장.",
                "CRITICAL": "의미 불일치 감지. 문맥 왜곡 가능성.",
                "FATAL":    "두 신호 동시 이탈. 도메인 완전 이탈.",
            }
            if status == "PASS":
                st.success(desc_map[status])
            elif status == "WARNING":
                st.warning(desc_map[status])
            else:
                st.error(desc_map[status])

            st.caption("⚠️ 이건 표면 자연스러움 + 의미 일관성 신호입니다. 환각 확정 판정이 아니에요.")

            # 토큰별 진단
            if result["per_token"]:
                with st.expander("📍 토큰별 진단", expanded=False):
                    html_parts = []
                    for p in result["per_token"]:
                        cls = "token-ok" if p["in_graph"] else "token-bad"
                        mark = "✓" if p["in_graph"] else "✗"
                        html_parts.append(
                            f'<span class="{cls}">{mark} {p["token"]} ({p["logp"]:.1f})</span>'
                        )
                    st.markdown(" ".join(html_parts), unsafe_allow_html=True)

                    in_count  = sum(1 for p in result["per_token"] if p["in_graph"])
                    out_count = len(result["per_token"]) - in_count
                    st.caption(f"그래프 안: {in_count}개 / 밖: {out_count}개 / 전체: {len(result['per_token'])}개")

        # Raw API
        with st.expander("Raw API Response"):
            st.json(latest["raw"])

    # 히스토리
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("#### 📋 히스토리")
        for i, h in enumerate(reversed(st.session_state.history[:-1])):
            r = h["result"]
            status_str = r["status"] if r else "N/A"
            with st.expander(f"[{len(st.session_state.history)-1-i}] {h['prompt'][:40]}... → {status_str}"):
                st.write(h["answer"])
                if r:
                    st.caption(f"logP: {r['avg_logp']:+.3f} | mismatch: {r['mismatch']:.3f} | {r['elapsed_ms']:.2f}ms")
