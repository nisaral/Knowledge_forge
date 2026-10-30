// Node check for static/sm2.js (run by tests/test_create.py::test_sm2_js).
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";

const src = fs.readFileSync(new URL("../static/sm2.js", import.meta.url), "utf8");
vm.runInThisContext(src);
const S = globalThis.SM2;
const t0 = 1_700_000_000_000;
let s = S.newState(t0);
s = S.review(s, 4, t0); assert.equal(s.interval, 1); assert.equal(s.reps, 1); assert.equal(s.ef, 2.5);
s = S.review(s, 5, t0); assert.equal(s.interval, 6); assert.equal(s.ef, 2.6);
s = S.review(s, 4, t0); assert.equal(s.interval, 16); assert.equal(s.due, t0 + 16 * S.DAY);
s = S.review(s, 1, t0); assert.equal(s.interval, 1); assert.equal(s.reps, 0); assert.equal(s.lapses, 1);
assert.ok(s.ef < 2.6 && s.ef >= 1.3);
let low = S.newState(t0); for (let i = 0; i < 20; i++) low = S.review(low, 0, t0); assert.equal(low.ef, 1.3);
const cards = [{ id: "a" }, { id: "b" }, { id: "c" }];
const states = { a: { due: t0 + S.DAY }, b: { due: t0 - 5 } };
assert.deepEqual(S.dueCards(cards, states, t0).map((c) => c.id), ["b", "c"]);
const csv = S.toAnkiCSV([{ question: 'Say "hi", ok?', answer: "Yes", tags: ["greet ing"], difficulty: "easy",
  sources: [{ source: "a.txt" }] }]);
assert.ok(csv.startsWith("#separator:Comma"));
assert.ok(csv.includes('"Say ""hi"", ok?"') && csv.includes("greet_ing easy") && csv.includes("Source: a.txt"));
const mem = { v: null, getItem() { return this.v; }, setItem(k, v) { this.v = v; } };
S.save({ decks: { d: 1 }, states: {} }, mem); assert.deepEqual(S.load(mem).decks, { d: 1 });
console.log("sm2 ok");
