const assert = require("assert");
global.window = undefined;
const { buildResult, iou } = require(require("path").join(__dirname, "../docs/static/js/browser-ai.js"));
const dims = { width: 640, height: 480 };
const preds = [
  { class: "mouse", score: 0.31, bbox: [100, 100, 60, 40] },
  { class: "keyboard", score: 0.55, bbox: [50, 300, 400, 120] },
  { class: "remote", score: 0.28, bbox: [102, 101, 60, 40] },   // same box as mouse, weaker
  { class: "person", score: 0.9, bbox: [0, 0, 640, 480] },
  { class: "cup", score: 0.2, bbox: [500, 50, 40, 60] },        // below slider 0.35
  { class: "banana", score: 0.6, bbox: [300, 10, 50, 100] },
];
// object mode, slider 0.35: person excluded, mouse/keyboard kept (focus cap 0.30), cup dropped
let r = buildResult(preds, dims, "object", 0.35, 12.3);
assert.deepStrictEqual(r.detections.map(d => d.label).sort(), ["Banana", "Computer Mouse", "Keyboard"].sort());
assert.strictEqual(r.detections.find(d => d.label === "Computer Mouse").reference_key, "mouse");
assert.ok(r.detections.every(d => d.bbox.every(v => v >= 0 && v <= 1)));
assert.strictEqual(r.meta.people, 0); assert.strictEqual(r.meta.objects, 2); assert.strictEqual(r.meta.fruits, 1);
assert.strictEqual(r.mode, "object"); assert.strictEqual(r.saved, 0);
// slider 0.9 must not hide a 0.31 mouse
r = buildResult(preds, dims, "object", 0.9, 1);
assert.ok(r.detections.some(d => d.label === "Computer Mouse"));
assert.ok(!r.detections.some(d => d.label === "Banana"));
// person mode
r = buildResult(preds, dims, "person", 0.35, 1);
assert.deepStrictEqual(r.detections.map(d => d.label), ["Person"]); assert.strictEqual(r.meta.people, 1);
// fruit mode
r = buildResult(preds, dims, "fruit", 0.35, 1);
assert.deepStrictEqual(r.detections.map(d => d.label), ["Banana"]);
// gesture unsupported
r = buildResult(preds, dims, "gesture", 0.35, 1);
assert.strictEqual(r.detections.length, 0); assert.ok(/Python backend/.test(r.notes[0]));
// empty
r = buildResult([], dims, "object", 0.35, 1);
assert.ok(r.notes[0].startsWith("No object detected")); assert.ok(r.description.length > 0);
assert.ok(iou([0,0,1,1],[0,0,1,1]) === 1);
console.log("browser-ai tests OK");
