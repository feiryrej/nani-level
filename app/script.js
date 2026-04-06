const API = "http://127.0.0.1:8000";

const LEVEL_LABELS = {
  easy: { jp: "簡単", icon: "🌸" },
  intermediate: { jp: "中級", icon: "⛩️" },
  hard: { jp: "難しい", icon: "⚡" },
};

const JLPT_CLASS = {
  1: "hard",
  2: "hard",
  3: "intermediate",
  4: "easy",
  5: "easy",
};

function setExample(text) {
  document.getElementById("jp-input").value = text;
  document.getElementById("jp-input").focus();
}

async function classify() {
  const text = document.getElementById("jp-input").value.trim();
  if (!text) {
    showError(
      "テキストを入力してください。 (Please enter some Japanese text.)",
    );
    return;
  }

  const btn = document.getElementById("submit-btn");
  btn.classList.add("loading");
  btn.innerHTML = '<span class="spinner"></span> 判定中…';

  document.getElementById("result").innerHTML = "";

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    showResult(data);
  } catch (e) {
    if (
      e.message.includes("fetch") ||
      e.message.includes("NetworkError") ||
      e.name === "TypeError"
    ) {
      showError(
        "Cannot connect to the API server. Make sure you ran:<br><code>uvicorn app:app --reload</code>",
      );
    } else {
      showError(e.message);
    }
  } finally {
    btn.classList.remove("loading");
    btn.innerHTML = "判定する &nbsp;→";
  }
}

function showResult(data) {
  const { label, confidence, probabilities, matched_vocab, char_count } = data;
  const meta = LEVEL_LABELS[label];

  // probability bars
  const bars = ["easy", "intermediate", "hard"]
    .map(
      (cls) => `
    <div class="prob-bar-row">
      <div class="prob-bar-name">${cls.charAt(0).toUpperCase() + cls.slice(1)}</div>
      <div class="prob-bar-outer">
        <div class="prob-bar-inner ${cls}" style="width:0%" data-pct="${probabilities[cls]}"></div>
      </div>
      <div class="prob-pct">${probabilities[cls]}%</div>
    </div>
  `,
    )
    .join("");

  // vocab chips
  let vocabHtml;
  if (matched_vocab && matched_vocab.length > 0) {
    vocabHtml = matched_vocab
      .map((v) => {
        const cls = JLPT_CLASS[v.jlpt_level] || "hard";
        return `<span class="vocab-chip ${cls}">${v.word}<span class="chip-level">N${6 - v.jlpt_level}</span></span>`;
      })
      .join("");
  } else {
    vocabHtml =
      '<span class="no-vocab">No JLPT vocabulary matched (text may be very simple or uses non-dictionary forms)</span>';
  }

  document.getElementById("result").innerHTML = `
    <div class="result-card">
      <div class="difficulty-badge">
        <div class="badge-icon">${meta.icon}</div>
        <div class="badge-text">
          <div class="badge-label ${label}">${label.charAt(0).toUpperCase() + label.slice(1)}</div>
          <div class="badge-jp">${meta.jp} &nbsp;·&nbsp; ${char_count} characters</div>
        </div>
        <div class="badge-conf">
          <div class="conf-num">${confidence}%</div>
          <div class="conf-label">confidence</div>
        </div>
      </div>

      <div class="prob-section">
        <div class="prob-title">Probability Distribution</div>
        ${bars}
      </div>

      <div class="vocab-section">
        <div class="vocab-title">Matched JLPT Vocabulary</div>
        <div class="vocab-chips">${vocabHtml}</div>
      </div>
    </div>
  `;

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-inner").forEach((bar) => {
      bar.style.width = bar.dataset.pct + "%";
    });
  });
}

function showError(msg) {
  document.getElementById("result").innerHTML = `
    <div class="error-card">⚠️ &nbsp;${msg}</div>
  `;
}

// Allow Ctrl+Enter to submit
document.getElementById("jp-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.ctrlKey) classify();
});
