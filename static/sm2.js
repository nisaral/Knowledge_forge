/* SM-2 spaced repetition (SuperMemo 2) + localStorage deck persistence.
 * Plain script (CSP-safe, no modules): exposes globalThis.SM2. Pure functions are tested via node. */
(function (root) {
  "use strict";
  var DAY = 86400000;

  function newState(now) {
    return { ef: 2.5, interval: 0, reps: 0, due: now || Date.now(), lapses: 0, last: null };
  }

  /* quality: 0..5 (0-2 = fail, 3 = hard, 4 = good, 5 = easy). Returns a new state. */
  function review(state, quality, now) {
    now = now || Date.now();
    var q = Math.max(0, Math.min(5, Math.round(Number(quality))));
    var s = Object.assign(newState(now), state || {});
    if (q < 3) {
      s.reps = 0;
      s.interval = 1;
      s.lapses += 1;
    } else {
      s.reps += 1;
      if (s.reps === 1) s.interval = 1;
      else if (s.reps === 2) s.interval = 6;
      else s.interval = Math.round(s.interval * s.ef);
    }
    s.ef = Math.max(1.3, s.ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)));
    s.ef = Math.round(s.ef * 1000) / 1000;
    s.due = now + s.interval * DAY;
    s.last = now;
    return s;
  }

  function isDue(state, now) {
    return !state || state.due <= (now || Date.now());
  }

  /* Order: due cards first (most overdue first), then new cards. */
  function dueCards(cards, states, now) {
    now = now || Date.now();
    return cards
      .filter(function (c) { return isDue(states[c.id], now); })
      .sort(function (a, b) {
        var sa = states[a.id], sb = states[b.id];
        return (sa ? sa.due : Infinity) - (sb ? sb.due : Infinity);
      });
  }

  function csvCell(v) {
    v = String(v == null ? "" : v);
    return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }

  /* Anki-importable CSV: Front,Back,Tags (with a header comment Anki understands). */
  function toAnkiCSV(cards) {
    var lines = ["#separator:Comma", "#html:false", "#tags column:3"];
    cards.forEach(function (c) {
      var back = c.answer + (c.sources && c.sources.length ? "\n\nSource: " + c.sources.map(function (s) { return s.source; }).join("; ") : "");
      var tags = (c.tags || []).concat(c.difficulty ? [c.difficulty] : []).map(function (t) { return String(t).replace(/\s+/g, "_"); });
      lines.push([csvCell(c.question), csvCell(back), csvCell(tags.join(" "))].join(","));
    });
    return lines.join("\n") + "\n";
  }

  var KEY = "kf.flashcards.v1";
  function load(storage) {
    try { return JSON.parse((storage || root.localStorage).getItem(KEY)) || { decks: {}, states: {} }; }
    catch (e) { return { decks: {}, states: {} }; }
  }
  function save(data, storage) {
    try { (storage || root.localStorage).setItem(KEY, JSON.stringify(data)); return true; }
    catch (e) { return false; }
  }

  root.SM2 = { newState: newState, review: review, isDue: isDue, dueCards: dueCards, toAnkiCSV: toAnkiCSV,
               load: load, save: save, DAY: DAY };
})(typeof globalThis !== "undefined" ? globalThis : this);
